import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

transformation = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor() 
])

dataset_path = "/storage/slurm/home/68070701603@cpe.kmutt.ac.th/deep_learning/src/lab6/realwaste-main/RealWaste"
realwaste_dataset = datasets.ImageFolder(root=dataset_path, transform=transformation)

print(f"Classes found: {realwaste_dataset.classes}")
print(f"Total images loaded: {len(realwaste_dataset)}")

train_loader = DataLoader(realwaste_dataset, batch_size=64, shuffle=True)
images, labels = next(iter(train_loader))

print(f"Batch Tensor Shape: {images.shape}")
print(f"Labels Tensor Shape: {labels.shape}")

class VanillaCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(VanillaCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(640000, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        out = self.fc(x)
        return out

model = VanillaCNN(num_classes=9)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"DEBUG LR: {optimizer.param_groups[0]['lr']}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Executing mathematical operations on: {device}")

epochs = 5

images, labels = next(iter(train_loader))
print(f"DEBUG DATA: Max pixel={images.max().item()}, Min pixel={images.min().item()}")

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

        if (batch_idx + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print(f"--- End of Epoch {epoch+1} | Average Loss: {running_loss/len(train_loader):.4f} ---")

print("Vanilla CNN Training Complete.")
