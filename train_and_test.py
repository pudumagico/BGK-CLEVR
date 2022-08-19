import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Resize

from dataset import Dataset
import models

N_EPOCHS = 10
BATCH_SZ = 16
LEARN_RATE = 0.001

def main(dataset):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    full_dataset = Dataset(f'{dataset}/ann.csv', f'{dataset}/images', transform=Resize(256))

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # img = train_features[0].squeeze()
    # label = train_labels[0]
    # plt.imshow(img.permute(1, 2, 0), cmap="gray")
    # plt.show()
    # print(f"Label: {label}")

    model =  models.NeuralNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

    for epoch in range(N_EPOCHS):
        
        print(epoch)

        for images, labels in train_dataloader:
            images = images.to(device).float()
            labels = labels.to(device).type(torch.LongTensor)

            optimizer.zero_grad()

            outputs = model.forward(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images = images.to(device).float()
            labels = labels.to(device).type(torch.LongTensor)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-o', '--output_log', type=str, required=True)
    
    args = parser.parse_args()

    main(args.dataset)