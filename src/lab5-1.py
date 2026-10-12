import torch
import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('data/UTF-8_cardio.csv', sep=';')

X_raw = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
y_raw = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32).view(-1, 1)

X_scaled = (X_raw - X_raw.mean(dim=0)) / X_raw.std(dim=0)

n_samples = X_scaled.size(0)
n_train = int(0.7 * n_samples)

torch.manual_seed(42)
indices = torch.randperm(n_samples)

X_train, y_train = X_scaled[indices[:n_train]], y_raw[indices[:n_train]]
X_test, y_test = X_scaled[indices[n_train:]], y_raw[indices[n_train:]]

n_features = X_train.size(1)

W1 = torch.randn(n_features, 6, dtype=torch.float32) * 0.01
b1 = torch.zeros(1, 6, dtype=torch.float32)

# Output Layer (2): 6 Units -> 1 Unit
W2 = torch.randn(6, 1, dtype=torch.float32) * 0.01
b2 = torch.zeros(1, 1, dtype=torch.float32)


print("--- Deep Neural Network Training ---")



alpha = 0.01 
epochs = 1000
loss_history = []

for epoch in range(epochs):
    # --- Forward Pass ---
    Z1 = torch.matmul(X_train, W1) + b1
    A1 = torch.relu(Z1) 
    
    Z2 = torch.matmul(A1, W2) + b2
    y_hat = torch.sigmoid(Z2)
    
    y_hat_safe = torch.clamp(y_hat, min=1e-9, max=1.0 - 1e-9)
    
    loss = -torch.mean(y_train * torch.log(y_hat_safe) + (1 - y_train) * torch.log(1 - y_hat_safe))
    loss_history.append(loss.item())
    
    dZ2 = y_hat - y_train
    dW2 = (1.0 / X_train.size(0)) * torch.matmul(A1.T, dZ2)
    db2 = (1.0 / X_train.size(0)) * torch.sum(dZ2, dim=0, keepdim=True)
    
    dA1 = torch.matmul(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0).float() 
    dW1 = (1.0 / X_train.size(0)) * torch.matmul(X_train.T, dZ1)
    db1 = (1.0 / X_train.size(0)) * torch.sum(dZ1, dim=0, keepdim=True)
    
    W2 -= alpha * dW2
    b2 -= alpha * db2
    W1 -= alpha * dW1
    b1 -= alpha * db1

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4} | BCE Loss: {loss.item():.4f}")
    
print("Training Complete. No NaNs encountered.")

output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(output_dir, "baseline_loss_history.png")


plt.figure(figsize=(8, 5))
plt.plot(loss_history, color='blue', label='Training Loss')
plt.title('Loss History Across Iterations')
plt.xlabel('Epoch')
plt.ylabel('BCE Loss')
plt.legend()
plt.savefig(file_path)
print("Saved lab5_baseline_loss_history.png to cluster storage.")

with torch.no_grad():
    Z1_test = torch.matmul(X_test, W1) + b1
    A1_test = torch.relu(Z1_test)
    Z2_test = torch.matmul(A1_test, W2) + b2
    y_hat_test = torch.sigmoid(Z2_test)
    
    predictions = (y_hat_test >= 0.5).float()
    accuracy = torch.mean((predictions == y_test).float())
    print(f"\nFinal Test Accuracy: {accuracy.item() * 100:.2f}%")
