# **Deep Learning with PyTorch – Session 13**

### **Advanced Regularization: Data Augmentation & Label Smoothing**

---

## **Session Objective**

By the end of this session, the student will:

* Understand the motivation and mechanics of **two advanced regularization methods**:

  1. **Data Augmentation**: Improving generalization by diversifying inputs.
  2. **Label Smoothing**: Preventing overconfidence by softening labels.
* Implement these methods in PyTorch.
* Analyze their impact through **plots**, **metrics**, and **qualitative inspection** (e.g., misclassified samples).

---

## **Instructional Content**

### **Why More Regularization?**

Even with dropout, weight decay (L2), batchnorm, and gradient clipping, models can still overfit or be overconfident.

* *Dropout* → Adds noise inside the network.
* *Weight decay* → Penalizes large weights.
* *BatchNorm* → Stabilizes activations.
* *Data augmentation* → Adds noise in the **input space**, encouraging robustness.
* *Label smoothing* → Regularizes the **output distribution** to avoid extreme confidence.

### **What is Data Augmentation?**

* **Goal**: Artificially expand the training dataset using realistic transformations like flips, crops, or color changes.
* **Why it helps**: Each epoch, the model sees slightly altered versions of the same images, making it harder to memorize exact inputs.
* **Examples**: Random horizontal flips, random crops with padding.

### **What is Label Smoothing?**

* **Goal**: Prevent the model from becoming overly confident by softening one-hot target labels.
* **Mechanics**: Instead of targets being exactly `[0,0,1,0,...]`, distribute a small ε amount over all classes:
  For class y: `1-ε` on true class, `ε/k` on others (`k` = number of classes).
* **Benefits**: Helps generalization, prevents overconfident predictions, improves calibration and robustness to noisy labels.
* **In PyTorch**: `nn.CrossEntropyLoss(label_smoothing=0.1)`

---

## **Session Timeline (1 Hour)**

| Time      | Activity                                                        |
| --------- | --------------------------------------------------------------- |
| 0:00–0:05 | **Check-in + Recap**: Review previous regularizers.             |
| 0:05–0:20 | **Instructional Content**: Explain augmentation & smoothing.    |
| 0:20–0:35 | **Guided Example**: Instructor-led implementation and analysis. |
| 0:35–0:55 | **Solo Exercise**: Student implements with vague TODOs.         |
| 0:55–1:00 | **Discussion**: Analyze plots, effects, and intuition.          |

---

## **Guided Example (Instructor Script)**

This **complete script** demonstrates augmentation and label smoothing, plus **plots and analysis**.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 1. Data Augmentation
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()
])
transform_test = transforms.ToTensor()

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. Model Definition
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)          # 32x32 -> 16x16
        self.fc1   = nn.Linear(64*16*16, 128)    # <-- was 64*8*8
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)                  # (B, 16384)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)

# 3. Label Smoothing Loss
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Helper: Accuracy computation
def compute_accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    return (preds == labels).float().mean().item()

# 4. Training Loop with Metrics Tracking
num_epochs = 5
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(num_epochs):
    model.train()
    epoch_loss, epoch_acc = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += compute_accuracy(outputs, labels)
    train_losses.append(epoch_loss / len(train_loader))
    train_accs.append(epoch_acc / len(train_loader))

    # Validation
    model.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += compute_accuracy(outputs, labels)
    val_losses.append(val_loss / len(test_loader))
    val_accs.append(val_acc / len(test_loader))

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_losses[-1]:.4f}, "
          f"Train Acc: {train_accs[-1]*100:.2f}%, "
          f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]*100:.2f}%")

# 5. Plot Loss and Accuracy Curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1,2,2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curves')
plt.show()

# 6. Display Misclassified Samples
model.eval()
mis_images, mis_preds, true_labels = [], [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        mis_idx = (preds != labels).nonzero(as_tuple=True)[0]
        for idx in mis_idx[:2]:  # take a few misclassified examples
            mis_images.append(images[idx].cpu())
            mis_preds.append(preds[idx].cpu().item())
            true_labels.append(labels[idx].cpu().item())
        if len(mis_images) >= 5:
            break

classes = train_dataset.classes
plt.figure(figsize=(10,2))
for i in range(len(mis_images)):
    img = mis_images[i].permute(1,2,0).numpy()
    plt.subplot(1,5,i+1)
    plt.imshow(img)
    plt.title(f"P:{classes[mis_preds[i]]}\nT:{classes[true_labels[i]]}")
    plt.axis('off')
plt.suptitle('Some Misclassified Examples')
plt.show()
```

---

## **Solo Exercise (Student Script with TODOs)**

A partially complete script with vague hints to encourage problem-solving:

```python
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
```

---

## **Discussion & Reflection Prompts**

* How did augmentation affect the train/val gap?
* Did label smoothing alter prediction confidence?
* Compare results with earlier sessions: which regularizers are most effective?
* How would you combine these techniques in a real project?
