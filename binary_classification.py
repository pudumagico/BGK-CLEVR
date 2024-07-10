import matplotlib.pyplot as plt
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms, models

def calculate_accuracy(outputs, labels):
    # Assuming a binary classification task
    preds = torch.round(torch.sigmoid(outputs))  # Convert outputs to 0 or 1
    corrects = (preds == labels).sum().float()
    accuracy = corrects / labels.numel()
    return accuracy.item()

model = models.vgg11(weights=False)

# num_ftrs = model.fc.in_features
# model.fc = nn.Sequential(
#     nn.Linear(num_ftrs, 1),
#     nn.Sigmoid()
# )
# num_ftrs = model.classifier[1].in_features
# model.classifier = nn.Sequential(
#     nn.Dropout(0.2),
#     nn.Linear(num_ftrs, 1)  # For binary classification
# )

num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 1) 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root='/home/nhiguera/Research/cclevr/datasets/no_adj_same_color/train', transform=transform, )
train_dataset, _ = torch.utils.data.random_split(train_dataset, [0.05, 0.95])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)


# labels_map = {
#     0: "Negative",
#     1: "Positive"
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
#     img, label = train_dataset[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     img = img.swapaxes(0,1)
#     img = img.swapaxes(1,2)
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

# exit()

validation_dataset = datasets.ImageFolder(root='/home/nhiguera/Research/cclevr/datasets/no_adj_same_color/val', transform=transform)
val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=True)

# val, _ = torch.utils.data.random_split(train_loader, [0.01, 0.99])

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 2


model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0  # To keep track of the number of correct predictions
    train_pred_counts = Counter()  # Counter for training predictions
    
    # Variables to store the first batch data
    first_batch_inputs_train, first_batch_labels_train, first_batch_predictions_train = None, None, None
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to('cuda'), labels.to('cuda').float().view(-1, 1)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)        
        running_corrects += (torch.round(torch.sigmoid(outputs)) == labels).sum().item()

        predictions = torch.round(torch.sigmoid(outputs.detach())).cpu().numpy().flatten()
        train_pred_counts.update(predictions)
        
        # Store the first batch data
        if i == 0:
            first_batch_inputs_train = inputs.cpu()
            first_batch_labels_train = labels.cpu()
            first_batch_predictions_train = torch.round(torch.sigmoid(outputs)).cpu()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    print(f'Training Prediction Distribution: {train_pred_counts}')
    
    # Print the first batch of training examples
    # print(f'First Batch Training Labels: {first_batch_labels_train.numpy().flatten()}')
    # print(f'First Batch Training Predictions: {first_batch_predictions_train.detach().numpy().flatten()}')

    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    val_pred_counts = Counter()  # Counter for validation predictions
    
    # Variables to store the first batch data
    first_batch_inputs_val, first_batch_labels_val, first_batch_predictions_val = None, None, None
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to('cuda'), labels.to('cuda').float().view(-1, 1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # print(outputs)
            
            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += (torch.round(torch.sigmoid(outputs)) == labels).sum().item()

            predictions = torch.round(torch.sigmoid(outputs.detach())).cpu().numpy().flatten()
            val_pred_counts.update(predictions)
            
            # Store the first batch data
            if i == 0:
                first_batch_inputs_val = inputs.cpu()
                first_batch_labels_val = labels.cpu()
                first_batch_predictions_val = torch.round(torch.sigmoid(outputs)).cpu()

    val_loss = val_running_loss / len(val_loader.dataset)
    val_acc = val_running_corrects / len(val_loader.dataset)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
    print(f'Validation Prediction Distribution: {val_pred_counts}')
    
    # Print the first batch of validation examples
    # print(f'First Batch Validation Labels: {first_batch_labels_val.numpy().flatten()}')
    # print(f'First Batch Validation Predictions: {first_batch_predictions_val.detach().numpy().flatten()}')

# Plot and save the training and validation loss over epochs
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.savefig('loss_over_epochs.png')

# Plot and save the training and validation accuracy over epochs
plt.figure()
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.savefig('accuracy_vgg11.png')