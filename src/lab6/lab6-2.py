import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

transformation = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor()
])

dataset_path = "/storage/slurm/home/68070701603@cpe.kmutt.ac.th/deep_learning/src/lab6/realwaste-main/RealWaste"
realwaste_dataset = datasets.ImageFolder(root=dataset_path, transform=transformation)
train_size = int(0.7 * len(realwaste_dataset))
test_size = len(realwaste_dataset) - train_size
train_dataset, test_dataset = random_split(realwaste_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class AdvancedCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(AdvancedCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(80000, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        out = self.fc(x)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdvancedCNN(num_classes=9).to(device)
print(model)
print(f"Executing mathematical operations on: {device}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

        if (batch_idx + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print(f"--- End of Epoch {epoch+1} | Average Loss: {running_loss/len(train_loader):.4f} ---")
