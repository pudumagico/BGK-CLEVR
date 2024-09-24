import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import ast

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes):
        super(MultiTaskModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.binary_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self.object_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes),
                nn.Softmax(dim=1)
            ) for _ in range(9)
        ])

    def forward(self, x):
        features = self.backbone(x)
        binary_out = self.binary_head(features)
        object_outs = []
        for i in range(3):
            for j in range(3):
                grid_cell = features[:, :, i*features.size(2)//3:(i+1)*features.size(2)//3, j*features.size(3)//3:(j+1)*features.size(3)//3]
                object_out = self.object_heads[i*3 + j](grid_cell)
                object_outs.append(object_out)
        return binary_out, object_outs

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.image_folder = os.path.dirname(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.data.iloc[idx, 0])
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        binary_label = torch.tensor(self.data.iloc[idx, 3], dtype=torch.float32)
        object_labels = ast.literal_eval(self.data.iloc[idx, 2])
        object_labels = torch.tensor(object_labels, dtype=torch.long)
        
        return image, binary_label, object_labels

def train_model(model, dataloader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for images, binary_labels, object_labels in dataloader:
            optimizer.zero_grad()
            binary_out, object_outs = model(images)
            binary_loss = binary_loss_fn(binary_out, binary_labels.unsqueeze(1))
            object_loss = sum([object_loss_fn(object_out, object_labels[:, i]) for i, object_out in enumerate(object_outs)])
            total_loss = binary_loss + object_loss
            total_loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item()}')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CustomDataset(csv_file='/home/nhiguera/Research/cclevr/test_dataset/0/annotations.csv', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

num_classes = 25  # Adjust based on your actual number of classes
model = MultiTaskModel(num_classes)
binary_loss_fn = nn.BCELoss()
object_loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, dataloader, num_epochs=10)
