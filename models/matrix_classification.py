import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define a modified VGG11 model
class ModifiedVGG11(nn.Module):
    def __init__(self, input_channels=1, pretrained=True):
        super(ModifiedVGG11, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


positive_tensor = torch.load('./test_dataset/1/matrices.pt')
negative_tensor = torch.load('./test_dataset/0/matrices.pt')

positive_labels = torch.ones(len(positive_tensor), 1).float()
negative_labels = torch.zeros(len(negative_tensor), 1).float()

# Combine positive and negative tensors and labels
X = torch.cat((positive_tensor, negative_tensor), dim=0)
y = torch.cat((positive_labels, negative_labels), dim=0)

# Convert tensors to numpy for train_test_split
X_np = X.numpy()
y_np = y.numpy()

# Split the data into training and validation sets
X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

# Convert numpy arrays back to tensors
X_train = torch.tensor(X_train_np)
X_val = torch.tensor(X_val_np)
y_train = torch.tensor(y_train_np).float()
y_val = torch.tensor(y_val_np).float()

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ModifiedVGG11().to(device)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    preds = outputs.round()  # Round predictions to 0 or 1
    correct = (preds == labels).float()  # Convert to float for division
    acc = correct.sum() / len(correct)
    return acc

# Function to print label distribution
def print_label_distribution(labels, dataset_name, epoch):
    unique, counts = torch.unique(labels, return_counts=True)
    distribution = dict(zip(unique.tolist(), counts.tolist()))
    print(f"Label distribution in {dataset_name} dataset at epoch {epoch}: {distribution}")

# Print label distribution
print_label_distribution(y_train, "training", 0)
print_label_distribution(y_val, "validation", 0)

# Lists to store accuracy for plotting
train_accuracies = []
val_accuracies = []

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    all_train_labels = []

    for inputs, labels in train_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        all_train_labels.append(labels)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_acc += calculate_accuracy(outputs, labels) * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_acc / len(train_loader.dataset)
    train_accuracies.append(epoch_acc.item())  # Store training accuracy
    all_train_labels = torch.cat(all_train_labels)
    print_label_distribution(all_train_labels, "training", epoch+1)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

    # Validation step
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    all_val_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            all_val_labels.append(labels)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            val_acc += calculate_accuracy(outputs, labels) * inputs.size(0)

    val_loss /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)
    val_accuracies.append(val_acc.item())  # Store validation accuracy
    all_val_labels = torch.cat(all_val_labels)
    print_label_distribution(all_val_labels, "validation", epoch+1)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

print('Training complete')

# Plotting accuracy over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_over_epochs.png')  # Save the plot to a file
plt.show()
