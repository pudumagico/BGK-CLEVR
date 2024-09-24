import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

import os

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

class CustomDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None, max_boxes=10):
        self.data = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.max_boxes = max_boxes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = f"{self.img_dir}/{self.data.iloc[idx, 0]}"
        image = Image.open(img_path).convert("RGB")
        boxes = ast.literal_eval(self.data.iloc[idx, 1])
        labels = ast.literal_eval(self.data.iloc[idx, 2])
        binary_label = int(self.data.iloc[idx, 3])
        
        if self.transform:
            image = self.transform(image)
        
        # Pad or truncate the number of boxes and labels to max_boxes
        if len(boxes) > self.max_boxes:
            boxes = boxes[:self.max_boxes]
            labels = labels[:self.max_boxes]
        else:
            boxes.extend([[0, 0, 0, 0]] * (self.max_boxes - len(boxes)))
            labels.extend([0] * (self.max_boxes - len(labels)))
        
        # Convert boxes to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
        # Convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Convert binary label to tensor
        binary_label = torch.tensor(binary_label, dtype=torch.long)
        
        return image, boxes, labels, binary_label

# Example usage
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


dataset = CustomDataset(annotation_file='/home/nhiguera/Research/cclevr/test_dataset/0/annotations.csv', img_dir='/home/nhiguera/Research/cclevr/test_dataset/0', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class CustomVGG11(nn.Module):
    def __init__(self, num_classes, num_object_classes, max_boxes=10):
        super(CustomVGG11, self).__init__()
        self.max_boxes = max_boxes
        # Load pre-trained VGG-11
        self.vgg11 = models.vgg11(pretrained=True)
        
        # Modify the classifier to remove the last layer
        self.vgg11.classifier = nn.Sequential(*list(self.vgg11.classifier.children())[:-1])
        
        # Add new layers for object detection
        self.bbox_regression = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4 * self.max_boxes)  # 4 coordinates for each bounding box
        )
        
        self.classification = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_object_classes * self.max_boxes)  # Class for each bounding box
        )
        
        # Add a layer for binary classification
        self.binary_classification = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        features = self.vgg11(x)
        bbox_pred = self.bbox_regression(features).view(-1, self.max_boxes, 4)
        class_pred = self.classification(features).view(-1, self.max_boxes, 10)
        binary_pred = self.binary_classification(features)
        
        return bbox_pred, class_pred, binary_pred

# Example usage
num_classes = 2  # For binary classification
num_object_classes = 10  # Adjust based on your dataset
max_boxes = 10  # Fixed number of bounding boxes

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Dataset and Dataloader
dataset = CustomDataset(annotation_file='/home/nhiguera/Research/cclevr/test_dataset/0/annotations.csv', img_dir='/home/nhiguera/Research/cclevr/test_dataset/0', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

# Initialize the model, loss functions, and optimizer
model = CustomVGG11(num_classes=2, num_object_classes=10, max_boxes=10)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

bbox_loss_fn = nn.MSELoss()
class_loss_fn = nn.CrossEntropyLoss()
binary_loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, boxes, labels, binary_labels in dataloader:
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        binary_labels = binary_labels.to(device)

        
        optimizer.zero_grad()
        
        bbox_pred, class_pred, binary_pred = model(images)

        print(bbox_pred, '\n',class_pred,'\n', binary_pred)
        exit()
        
        
        # Adjusting the shape of class_pred and labels
        class_pred = class_pred.view(-1, 10)
        labels = labels.view(-1)
        
        bbox_loss = bbox_loss_fn(bbox_pred, boxes)
        class_loss = class_loss_fn(class_pred, labels)
        binary_loss = binary_loss_fn(binary_pred, binary_labels)
        
        loss = bbox_loss + class_loss + binary_loss
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

print("Training complete.")