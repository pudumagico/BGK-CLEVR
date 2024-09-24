import torch
import torch.nn as nn
import torchvision.models as models

class MultiTaskVGG(nn.Module):
    def __init__(self, num_classes):
        super(MultiTaskVGG, self).__init__()
        # Load the pretrained VGG model
        vgg = models.vgg16(pretrained=True)
        self.backbone = vgg.features  # Use VGG features as the backbone
        
        # Object detection head
        self.detector = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes + 1, kernel_size=1)  # num_classes + 1 for background
        )
        
        # Classification head
        self.classifier_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        
        # Binary classification head
        self.binary_classifier_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.binary_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        
        # Object detection
        detections = self.detector(features)
        
        # Classification
        pooled_features_class = self.classifier_pool(features)
        flat_features_class = pooled_features_class.view(pooled_features_class.size(0), -1)
        class_preds = self.classifier(flat_features_class)
        
        # Binary classification
        pooled_features_binary = self.binary_classifier_pool(features)
        flat_features_binary = pooled_features_binary.view(pooled_features_binary.size(0), -1)
        binary_preds = self.binary_classifier(flat_features_binary)
        
        return detections, class_preds, binary_preds

# Instantiate and use the model
model = MultiTaskVGG(num_classes=10)
# input_image = torch.randn(1, 3, 224, 224)
# detections, class_preds, binary_preds = model(input_image)

# print("Detections:", detections)
# print("Class Predictions:", class_preds)
# print("Binary Predictions:", binary_preds)


import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import ast

class CustomDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        self.data = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform

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
        
        # Convert boxes to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
        # Convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Convert binary label to tensor
        binary_label = torch.tensor(binary_label, dtype=torch.long)
        
        return image, boxes, labels, binary_label

# Example usage
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = CustomDataset(annotation_file='/home/nhiguera/Research/cclevr/test_dataset/0/annotations.csv', img_dir='/home/nhiguera/Research/cclevr/test_dataset/0', transform=transform)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

dataset = CustomDataset(annotation_file='/home/nhiguera/Research/cclevr/test_dataset/0/annotations.csv', img_dir='/home/nhiguera/Research/cclevr/test_dataset/0', transform=transform)
val_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

import torch.nn.functional as F

detection_loss_fn = nn.CrossEntropyLoss()  # Adjust based on your detection loss requirements
classification_loss_fn = nn.CrossEntropyLoss()
binary_classification_loss_fn = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, boxes, labels, binary_labels in train_loader:
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        binary_labels = binary_labels.to(device)

        optimizer.zero_grad()
        
        detections, class_preds, binary_preds = model(images)
        
        # Calculate losses
        # You might need to reshape and handle detections based on your output format
        detection_loss = detection_loss_fn(detections, boxes)
        classification_loss = classification_loss_fn(class_preds, labels)
        binary_classification_loss = binary_classification_loss_fn(binary_preds, binary_labels)
        
        loss = detection_loss + classification_loss + binary_classification_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}")

    # Validation step (optional, similar to training loop but without optimizer step)
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for images, boxes, labels, binary_labels in val_loader:
            images = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)
            binary_labels = binary_labels.to(device)

            detections, class_preds, binary_preds = model(images)

            detection_loss = detection_loss_fn(detections, boxes)
            classification_loss = classification_loss_fn(class_preds, labels)
            binary_classification_loss = binary_classification_loss_fn(binary_preds, binary_labels)

            loss = detection_loss + classification_loss + binary_classification_loss
            val_loss += loss.item()

        print(f"Validation Loss: {val_loss/len(val_loader)}")
