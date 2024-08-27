import os
import sys
import torchvision.transforms as transforms
from torchvision.datasets import GTSRB
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import seaborn as sns

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
from repvgg import *

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3418, 0.3126, 0.3216], std=[0.2769, 0.2646, 0.2706])
])

# Create datasets
train_dataset = GTSRB(root='/home/QARepVGG/QARepVGG/data/GTSRB', split='train', transform=transform, download=True)
test_dataset = GTSRB(root='/home/QARepVGG/QARepVGG/data/GTSRB', split='test', transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=4)
print(test_dataset)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = create_QARepVGGBlockV2_A0(deploy=False).to(device)
model = create_GTSRB(deploy=True).to(device)
#print(model)
summary(model, (3, 224, 224))
input()
# To change the output dimension
model.linear = nn.Sequential(
    nn.Linear(model.linear.in_features, 43).to(device))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 10

# Lists to store accuracy and confusion matrix data for plotting
train_accuracies = []
test_accuracies = []
confusion_matrices = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_train = 0
    correct_train = 0
    all_labels = []
    all_preds = []
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += targets.size(0)
        correct_train += (predicted == targets).sum().item()

        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    train_acc = 100 * correct_train / total_train
    train_accuracies.append(train_acc)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss / len(train_loader):.4f}, Training Accuracy: {train_acc:.2f}%')
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            all_labels.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    test_acc = 100 * correct / total
    test_accuracies.append(test_acc)
    print(f'Accuracy of the model on the test images: {test_acc:.2f}%')
    

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=False, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

# Training loop
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
        all_labels.extend(targets.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# Calculate confusion matrix
cm = confusion_matrix(all_labels, all_preds, labels=list(range(43)))  # Change 43 to the number of classes

# Plot confusion matrix
plot_confusion_matrix(cm, classes=list(range(43)))

print('Finished Training')
torch.save(model.state_dict(), "GTSRB_light.pth")

# Plotting accuracies
epochs = list(range(1, num_epochs + 1))
plt.plot(epochs, train_accuracies, 'b', label='Training Accuracy')
plt.plot(epochs, test_accuracies, 'r', label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Test Accuracy Over Epochs')
plt.savefig('accuracy_plot.png')
plt.show()
