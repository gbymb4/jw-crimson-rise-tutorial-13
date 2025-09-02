# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 13:20:44 2025

@author: taske
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

print("Session 13 Homework: Remember the Basics!")
print("=" * 40)

# =============================================================================
# PART 1: DATA AUGMENTATION - What does it do?
# =============================================================================

print("PART 1: Data Augmentation")

# TODO 1: Add basic augmentations to training data
# Remember: Augmentation = making slightly different versions of the same image
transform_train = transforms.Compose([
    # TODO: Add a transform that randomly flips images horizontally
    # TODO: Add a transform that does random cropping with some padding
    transforms.ToTensor()
])

transform_test = transforms.ToTensor()  # NO augmentation on test data!

# Load data
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Key Point: Training data gets augmented, test data does NOT")

# =============================================================================
# PART 2: BASIC CNN MODEL
# =============================================================================

print("\nPART 2: Simple CNN")

class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Create first conv layer - input channels, output channels, kernel size
        # TODO: Create second conv layer 
        # TODO: Create pooling layer
        # TODO: Create first fully connected layer - calculate input size carefully!
        # TODO: Create final layer for classification
        pass

    def forward(self, x):
        # TODO: Apply conv layers with activations and pooling
        # TODO: Flatten the tensor
        # TODO: Apply fully connected layers
        return x

model = BasicCNN()
print("Remember: CIFAR10 has 10 classes, images are 32x32")

# =============================================================================
# PART 3: LABEL SMOOTHING - What is it?
# =============================================================================

print("\nPART 3: Label Smoothing")

# TODO: Create loss function with label smoothing
# Remember: label smoothing makes the model less overconfident
# criterion = ???

# Compare: What's the difference?
print("Without smoothing: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] - 100% confident")
print("With smoothing 0.1: [0.01, 0.01, 0.91, 0.01, 0.01, ...] - 91% confident")

# =============================================================================
# PART 4: BASIC TRAINING
# =============================================================================

print("\nPART 4: Training")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_one_epoch():
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # TODO: Complete the training step
        # Step 1: Clear gradients
        # Step 2: Forward pass through model
        # Step 3: Calculate loss
        # Step 4: Backward pass
        # Step 5: Update parameters
        
        break  # Just do one batch for homework
    
    return total_loss

# TODO: Run training for 2-3 epochs
print("TODO: Run train_one_epoch() a few times")

# =============================================================================
# REVIEW QUESTIONS - Answer these to check your understanding
# =============================================================================

print("\nREVIEW: Do you remember?")
print("1. What does RandomHorizontalFlip() do to an image?")
print("2. Why don't we augment test data?")
print("3. What does label_smoothing=0.1 mean?")
print("4. In CIFAR10, how many output neurons does your final layer need?")

print("\n" + "=" * 40)
print("BASICS CHECKLIST:")
print("□ Added augmentation transforms to training data")
print("□ Built CNN with correct input/output sizes")
print("□ Used CrossEntropyLoss with label_smoothing")
print("□ Can explain what each technique does")
print("=" * 40)