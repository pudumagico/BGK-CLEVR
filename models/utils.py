import torch

def decode_boxes(anchor_points, strides, predictions):
    # anchor_points: tensor of shape (AP, 2), where AP is the number of anchor points
    # strides: tensor of shape (AP, 1), stride for each anchor point
    # predictions: tensor of shape (B, AP, C), where B is batch size, AP is anchor points, and C includes both class scores and box deltas
    
    # We assume the last 4 channels in C are the bounding box deltas (dx, dy, dw, dh)
    # Extract the bounding box deltas: (dx, dy, dw, dh)
    box_deltas = predictions[:, :, -4:]  # shape (B, AP, 4)
    
    # Split the deltas into dx, dy, dw, dh
    dx = box_deltas[:, :, 0]
    dy = box_deltas[:, :, 1]
    dw = box_deltas[:, :, 2]
    dh = box_deltas[:, :, 3]

    # Anchor points (x_anchor, y_anchor)
    x_anchor = anchor_points[:, 0]  # shape (AP,)
    y_anchor = anchor_points[:, 1]  # shape (AP,)

    # Stride for each anchor
    stride = strides.squeeze()  # shape (AP,)

    # Repeat anchor points and stride for the batch dimension
    x_anchor = x_anchor.unsqueeze(0).repeat(predictions.size(0), 1)  # (B, AP)
    y_anchor = y_anchor.unsqueeze(0).repeat(predictions.size(0), 1)  # (B, AP)
    stride = stride.unsqueeze(0).repeat(predictions.size(0), 1)      # (B, AP)

    # Compute center of the bounding box
    x_center = x_anchor + dx * stride
    y_center = y_anchor + dy * stride

    # Compute width and height of the bounding box
    w_box = torch.exp(dw) * stride
    h_box = torch.exp(dh) * stride

    # Convert center (x_center, y_center) to corners (x_min, y_min, x_max, y_max)
    x_min = x_center - w_box / 2
    y_min = y_center - h_box / 2
    x_max = x_center + w_box / 2
    y_max = y_center + h_box / 2

    # Stack the box coordinates together
    boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)  # shape (B, AP, 4)

    return boxes


def best_class_per_subgrid(subgrid_anchors, predictions):
    """
    Chooses the best class for each subgrid based on prediction scores.
    
    Args:
    - subgrid_anchors: A nested list of shape (n, n), where each element contains a list of
                       anchor indices belonging to that subgrid.
    - predictions: Tensor of shape (B, AP, C) containing prediction scores for each anchor and class.
    
    Returns:
    - best_classes: Tensor of shape (B, n, n) containing the best class index for each subgrid in each batch.
    """
    B, AP, C = predictions.shape
    n = len(subgrid_anchors)  # Assuming subgrid_anchors is of shape (n, n)
    
    # Initialize the tensor to hold the best class for each subgrid in each batch
    best_classes = torch.zeros((B, n, n), dtype=torch.long)
    
    # Loop through each subgrid
    for i in range(n):
        for j in range(n):
            # Get the list of anchor indices for the subgrid
            anchor_indices = subgrid_anchors[i][j]
            
            # If the subgrid has no anchors, skip
            if len(anchor_indices) == 0:
                continue
            
            # Extract the predictions for the anchors in this subgrid, for each batch
            subgrid_predictions = predictions[:, anchor_indices, :]
            
            # Find the maximum prediction score across all anchors in this subgrid, for each batch
            max_scores, best_class_indices = torch.max(subgrid_predictions.mean(dim=1), dim=1)
            
            # Store the best class for each batch
            best_classes[:, i, j] = best_class_indices
    
    return best_classes

def associate_anchors_to_subgrid(anchors, image_size, n):
    """
    Associates each anchor point with a specific subgrid in an n x n grid.
    
    Args:
    - anchors: Tensor of shape (M, 2) containing the x, y coordinates of the anchors.
    - image_size: Tuple (height, width) representing the size of the image.
    - n: Number of subgrids along each axis (so there will be n x n subgrids).
    
    Returns:
    - subgrid_anchors: Tensor of shape (n, n), where each element contains a list of indices of anchors
                       belonging to that subgrid.
    """
    # Get height and width of the image
    height, width = image_size
    
    # Calculate the size of each subgrid
    grid_height = height / n
    grid_width = width / n
    
    # Initialize a tensor to hold the indices for each subgrid
    subgrid_anchors = [[[] for _ in range(n)] for _ in range(n)]
    
    # Loop through each anchor point
    for i, (x, y) in enumerate(anchors):
        # Calculate which subgrid the anchor point belongs to
        subgrid_x = min(int(x // grid_width), n - 1)
        subgrid_y = min(int(y // grid_height), n - 1)
        
        # Append the anchor index to the corresponding subgrid
        subgrid_anchors[subgrid_y][subgrid_x].append(i)
    
    # Convert the subgrid_anchors to a tensor of lists
    return subgrid_anchors

def find_anchor_grid_positions(anchor_points, img_height, img_width):
    # Grid dimensions
    cell_height = img_height / 3
    cell_width = img_width / 3

    # Anchor positions
    x_coords = anchor_points[:, 0]
    y_coords = anchor_points[:, 1]

    # Calculate grid indices
    grid_rows = torch.floor(y_coords / cell_height).long()
    grid_cols = torch.floor(x_coords / cell_width).long()

    # Ensure indices are within the valid range [0, 2]
    grid_rows = torch.clamp(grid_rows, 0, 2)
    grid_cols = torch.clamp(grid_cols, 0, 2)

    return grid_rows, grid_cols

# Example usage
# Assuming anchor_points is a tensor of shape (N, 2) with (x, y) coordinates
# Example: torch.tensor([[50, 30], [150, 250], [300, 350]]) for a 300x300 image
# anchor_points = torch.tensor([[50, 30], [150, 250], [300, 350]], dtype=torch.float32)
# img_height = 300
# img_width = 300

# grid_rows, grid_cols = find_anchor_grid_positions(anchor_points, img_height, img_width)
# print("Grid row indices:", grid_rows)
# print("Grid column indices:", grid_cols)
