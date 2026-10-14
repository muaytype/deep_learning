import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

# --- 1. Data Pipeline ---
# ResNet expects images to be normalized using the exact mean and std dev of its original ImageNet training data.
transformation = transforms.Compose([
    transforms.Resize((224, 224)), # ResNet expects 224x224, not 200x200
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_path = "realwaste-main/RealWaste/"
realwaste_dataset = datasets.ImageFolder(root=dataset_path, transform=transformation)
train_size = int(0.7 * len(realwaste_dataset))
test_size = len(realwaste_dataset) - train_size
train_dataset, test_dataset = random_split(realwaste_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# --- 2. Transfer Learning Architecture ---
# Download the pre-trained model weights
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Freeze all foundational layers to save memory and compute time
for param in model.parameters():
    param.requires_grad = False

# Extract the input size of the original final layer
num_ftrs = model.fc.in_features

# Replace the final layer. New layers have requires_grad=True by default.
model.fc = nn.Linear(num_ftrs, 9)

# --- 3. Hardware Routing ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Executing mathematical operations on: {device}")

# --- 4. Optimizer Target ---
# CRITICAL: We only pass the parameters of the new final layer to the Adam optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# --- 5. Training Loop ---
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    print(f"--- End of Epoch {epoch+1} | Average Loss: {running_loss/len(train_loader):.4f} ---")
