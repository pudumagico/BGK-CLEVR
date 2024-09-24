from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from pathlib import Path
from ultralytics.utils.loss import (
    E2EDetectLoss,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

import torch
from utils import associate_anchors_to_subgrid, best_class_per_subgrid, decode_boxes

import torch
from torchvision.ops import nms


def reverse_forward(target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx, anc_points):
    """
    Reverse the task-aligned assignment to recover predictions (approximated).
    
    Args:
        target_labels (Tensor): shape(bs, num_total_anchors)
        target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
        target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
        fg_mask (Tensor): shape(bs, num_total_anchors)
        target_gt_idx (Tensor): shape(bs, num_total_anchors)
        anc_points (Tensor): shape(num_total_anchors, 2)
    
    Returns:
        pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
        pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
    """
    # Recover predicted bounding boxes (pd_bboxes)
    # Assuming target_bboxes were aligned with the anchor points, reverse the process
    # Ancillary points must be used to transform back the bboxes to anchor-relative bboxes.
    # This is an approximation as not all information can be perfectly recovered.
    
    # Get the predicted bounding boxes by subtracting anchor points
    pd_bboxes = target_bboxes + anc_points.unsqueeze(0).expand_as(target_bboxes)
    
    # For predicted scores (pd_scores), we try to reverse the normalization
    # Normalize the target scores back into predicted scores
    norm_align_metric = target_scores.sum(dim=-1, keepdim=True)  # reverse the normalization
    pd_scores = target_scores / (norm_align_metric + 1e-6)  # Approximate predicted scores

    return pd_scores, pd_bboxes


def find_best_class_for_grid(tensor_BAPC, combined_grid, N):
    B, AP, C = tensor_BAPC.shape  # Extract the dimensions of the input tensor
    
    # Create an output tensor of shape B x N x N to store the best class for each grid cell
    best_class_tensor = torch.zeros(B, N, N, dtype=torch.long)
    
    # Iterate over each batch
    for b in range(B):
        # For each grid cell in the combined grid
        for i in range(N):
            for j in range(N):
                grid_cell_indices = combined_grid[i][j]  # Get the indices for this grid cell
                
                if len(grid_cell_indices) == 0:
                    # If no anchors are in this cell, we can skip it or set it to a default class (e.g., class 0)
                    best_class_tensor[b, i, j] = 0  # Default class 0 if no anchors
                    continue
                
                # Get the scores for the anchors in this grid cell for the current batch b
                # tensor_BAPC[b, :, :] gives the anchor point scores for batch b
                # tensor_BAPC[b, grid_cell_indices, :] gives the scores for the selected indices
                anchor_scores = tensor_BAPC[b, grid_cell_indices, :]  # Shape will be (num_anchors_in_cell, C)
                
                # Average the scores over all anchor points in this cell (mean over dimension 0)
                mean_scores = torch.mean(anchor_scores, dim=0)  # Shape will be (C)
                
                # Find the class with the maximum average score
                best_class = torch.argmax(mean_scores)  # This gives the index of the best class
                
                # Store the best class index in the output tensor
                best_class_tensor[b, i, j] = best_class
    
    return best_class_tensor

def group_indices_by_grid_combined(grouped_anchors, N):
    # Step 1: Find the overall max_x and max_y across all strides
    all_anchor_points = []
    
    # Gather all the anchor points across all strides
    for stride, data in grouped_anchors.items():
        all_anchor_points.append(data['anchor_points'])
    
    # Stack them together to get one large tensor of all anchor points
    all_anchor_points = torch.cat(all_anchor_points, dim=0)
    
    # Find the maximum x and y values across all strides
    max_x, max_y = torch.max(all_anchor_points, dim=0).values
    max_x, max_y = max_x.item(), max_y.item()

    # Step 2: Create a combined NxN grid
    combined_grid = [[[] for _ in range(N)] for _ in range(N)]
    
    # Step 3: Group the indices from all strides into this combined grid
    for stride, data in grouped_anchors.items():
        anchor_points = data['anchor_points']
        indices = data['indices']

        # Define the step size for the grid based on max_x and max_y (global max values)
        step_x = max_x / N
        step_y = max_y / N

        # Iterate over each anchor and its corresponding index
        for i, (x, y) in enumerate(anchor_points):
            # Determine the grid cell based on the anchor's (x, y) position
            grid_x = min(int(x / step_x), N - 1)  # Ensure the value is within bounds
            grid_y = min(int(y / step_y), N - 1)

            # Append the index to the corresponding combined grid cell
            combined_grid[grid_x][grid_y].append(indices[i].item())

    return combined_grid


def group_indices_by_grid(anchor_points, indices, N, max_x, max_y):
    # Create a list of lists to hold the indices for each grid cell
    grid = [[[] for _ in range(N)] for _ in range(N)]
    
    # Define the step size for the grid based on max_x and max_y
    step_x = max_x / N
    step_y = max_y / N
    
    # Iterate over each anchor and its corresponding index
    for i, (x, y) in enumerate(anchor_points):
        # Determine the grid cell based on the anchor's (x, y) position
        grid_x = min(int(x / step_x), N - 1)  # Ensure the value is within bounds
        grid_y = min(int(y / step_y), N - 1)
        
        # Append the index to the corresponding grid cell
        grid[grid_x][grid_y].append(indices[i].item())
    
    return grid


class CustomLoss(v8DetectionLoss):

    def __call__(self, preds, batch):

        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy

        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
 
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        ################################################


        ################################################


        target_scores_sum = max(target_scores.sum(), 1)

        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        ##########################################################
        print()
        # print(anchor_points.shape)
        # print(stride_tensor.shape)
        # print(anchor_points[0], anchor_points[1599])
        # print(anchor_points[1600], anchor_points[1999])
        # print(anchor_points[2000], anchor_points[2099])
        
        unique_strides = torch.unique(stride_tensor)
        grouped_anchors = {}
        # Group the anchors by stride
        for stride in unique_strides:
            mask = stride_tensor.squeeze() == stride  # Create a mask for the current stride
            indices = torch.nonzero(mask).squeeze()  # Get the indices of the masked anchor points
            grouped_anchors[stride.item()] = {
                'anchor_points': anchor_points[mask],
                'indices': indices
            }

        max_coordinates = {}

        for stride, data in grouped_anchors.items():
            anchor_points = data['anchor_points']  # Should be of shape (num_anchors, 2)
            max_x, max_y = torch.max(anchor_points, dim=0).values            
            max_coordinates[stride] = (max_x.item(), max_y.item())

        N = 3  # User-defined grid size

        combined_grid = group_indices_by_grid_combined(grouped_anchors, N)

        best_class_tensor = find_best_class_for_grid(pred_scores.sigmoid(), combined_grid, N)
        print(best_class_tensor[0])
        print(batch)
        # print(torch.Tensor(batch['cls'][0:9]).view(3,3))

        # exit()

        ##########################################################




        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)    

class MyCustomModel(DetectionModel):
    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return CustomLoss(self)


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = MyCustomModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)

        if weights:
            model.load(weights)
        return model


# callback to upload model weights
def log_model(trainer):
    """Logs the path of the last model weight used by the trainer."""
    last_weight_path = trainer.last
    print(last_weight_path)


data_path = str(Path('test_dataset/data.yaml').resolve())

from pprint import pprint

trainer = CustomTrainer(cfg='yolo_cfg.yaml' ,overrides={'data':data_path, 'epochs':10, 'imgsz':320, 'batch':32, 'name':'yolov8_custom', 'model': 'yolov8l.pt'})
trainer.add_callback("on_train_epoch_end", log_model)  # Adds to existing callback
trainer.train()
