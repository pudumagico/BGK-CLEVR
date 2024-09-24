import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import ast

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, images, transform=None):
        self.data_path =images
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_path + self.data_frame.iloc[idx, 0]
        image = Image.open(img_name)
        bounding_boxes = ast.literal_eval(self.data_frame.iloc[idx, 1])
        labels = ast.literal_eval(self.data_frame.iloc[idx, 2])
        binary_label = self.data_frame.iloc[idx, 3]

        if self.transform:
            image = self.transform(image)

        return image, torch.Tensor(bounding_boxes), torch.Tensor(labels), torch.Tensor(binary_label)

# Define the model
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove the final classification layer
        self.obj_detect_fc = nn.Linear(2048, 4)  # For bounding box regression (x, y, w, h)
        self.obj_class_fc = nn.Linear(2048, 21)  # Assuming 20 classes + background
        self.binary_class_fc = nn.Linear(2048, 1)  # For binary classification

    def forward(self, x):
        features = self.backbone(x)
        bbox = self.obj_detect_fc(features)
        obj_class = self.obj_class_fc(features)
        binary_class = torch.sigmoid(self.binary_class_fc(features))
        return bbox, obj_class, binary_class

# Hyperparameters
batch_size = 4
learning_rate = 0.001
num_epochs = 10

# Data transformations
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset
dataset = CustomDataset(csv_file='/home/nhiguera/Research/cclevr/test_dataset/0/annotations.csv', images = '/home/nhiguera/Research/cclevr/test_dataset/0/', transform=data_transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Initialize model, loss functions, and optimizer
model = CustomModel()
criterion_bbox = nn.MSELoss()
criterion_obj_class = nn.CrossEntropyLoss()
criterion_binary_class = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, bboxes, obj_labels, binary_labels in data_loader:
        optimizer.zero_grad()

        bboxes_pred, obj_class_pred, binary_class_pred = model(images)

        loss_bbox = criterion_bbox(bboxes_pred, bboxes)
        loss_obj_class = criterion_obj_class(obj_class_pred, obj_labels)
        loss_binary_class = criterion_binary_class(binary_class_pred, binary_labels.float().unsqueeze(1))

        loss = loss_bbox + loss_obj_class + loss_binary_class
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader)}')

print('Training complete')
