# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 15:27:37 2025

@author: taske
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

# TODO: Add data augmentation (hint: flip, crop, maybe others)
transform_train = transforms.Compose([
    # e.g., transforms.RandomHorizontalFlip(), ...
    transforms.ToTensor()
])
transform_test = transforms.ToTensor()

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# TODO: Define a CNN with at least two conv layers, pooling, and two linear layers
class StudentCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentCNN, self).__init__()
        # Fill in conv, pool, linear layers

    def forward(self, x):
        # Implement forward pass
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StudentCNN().to(device)

# TODO: Create CrossEntropyLoss with label smoothing
criterion = nn.CrossEntropyLoss( # look up smoothing keyword )

optimizer = optim.Adam(model.parameters(), lr=0.001)

def compute_accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    return (preds == labels).float().mean().item()

num_epochs = 5
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(num_epochs):
    # TODO: Implement full training loop (forward, loss, backward, optimizer step)
    # Track and store train/val loss & accuracy
    pass

# TODO: Plot train/val loss & accuracy curves using matplotlib
# Hint: Two subplots: one for loss, one for accuracy

# TODO: Show some misclassified samples and compare predictions
# Hint: Collect indices where prediction != label and plot them
