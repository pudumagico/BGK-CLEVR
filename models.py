import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 50),   # 6 is the output channel size; 5 is the kernel size; 1 (channel) 28 28 -> 6 24 24
            nn.MaxPool2d(2, 2),   # kernel size 2; stride size 2; 6 24 24 -> 6 12 12
            nn.ReLU(True),        # inplace=True means that it will modify the input directly thus save memory
            nn.Conv2d(6, 16, 50),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),   # 16 8 8 -> 16 4 4
            nn.ReLU(True) 
        )
        self.classifier = nn.Sequential(
            nn.Linear(11664, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 11664)
        x = self.classifier(x)
        return x

# # Twice as many filters and 4x hidden layer units
# class WideNeuralNet(nn.Module):
#     def __init__(self):
#         super(WideNeuralNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 6, 3),   # 6 is the output channel size; 3 is the kernel size; 1 (channel) 28 28 -> 6 26 26
#             nn.MaxPool2d(2, 2),   # kernel size 2; stride size 2; 6 26 26 -> 6 13 13
#             nn.ReLU(True),        # inplace=True means that it will modify the input directly thus save memory
#             nn.Conv2d(6, 16, 2),  # 6 13 13 -> 16 12 12
#             nn.MaxPool2d(2, 2),   # 16 12 12 -> 16 6 6
#             nn.ReLU(True) 
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(16 * 6 * 6, 480),
#             nn.ReLU(),
#             nn.Linear(480, 336),
#             nn.ReLU(),
#             nn.Linear(336, 10),
#             nn.Softmax(1)
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = x.view(-1, 16 * 6 * 6)
#         x = self.classifier(x)
#         return x

# # Twice convolutional layers with 2 filter layers for each new one
# class LongNeuralNet(nn.Module):
#     def __init__(self):
#         super(LongNeuralNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 6, 3),   # 6 is the output channel size; 5 is the kernel size; 1 (channel) 28 28 -> 6 26 26
#             nn.MaxPool2d(2, 2),   # kernel size 2; stride size 2; 6 26 26 -> 6 13 13
#             nn.ReLU(True),
#             nn.Conv2d(6, 6, 2),   # 6 12 12 -> 6 12 12
#             nn.MaxPool2d(2, 2, padding=1),   # 6 12 12 -> 6 6 6
#             nn.ReLU(True),        
#             nn.Conv2d(6, 16, 3),  # 6 6 6 -> 16 4 4
#             nn.MaxPool2d(2, 2),   # 16 4 4 -> 16 2 2
#             nn.ReLU(True),
#             nn.Conv2d(16, 16, 2),   # 16 2 2 -> 16 1 1
#             nn.MaxPool2d(2, 2, padding=1),   # kernel size 2; stride size 2; 6 24 24 -> 6 12 12
#             nn.ReLU(True),  
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(16 * 1 * 1, 120),
#             nn.ReLU(),
#             nn.Linear(120, 84),
#             nn.ReLU(),
#             nn.Linear(84, 10),
#             nn.Softmax(1)
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = x.view(-1, 16 * 1 * 1)
#         x = self.classifier(x)
#         return 