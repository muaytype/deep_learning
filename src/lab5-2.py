import torch
import pandas as pd
import time

device = torch.device('cuda')

df = pd.read_csv('UTF-8_cardio.csv', sep=';')
df = df.dropna()
if 'id' in df.columns: 
    df = df.drop(columns=['id'])

X_raw = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
y_raw = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32).view(-1, 1)

X_scaled = (X_raw - X_raw.mean(dim=0)) / (X_raw.std(dim=0) + 1e-8)
n_features = X_scaled.size(1) 

n_train = int(0.7 * X_scaled.size(0))
torch.manual_seed(42)
indices = torch.randperm(X_scaled.size(0))

X_train = X_scaled[indices[:n_train]].to(device)
y_train = y_raw[indices[:n_train]].to(device)
X_test = X_scaled[indices[n_train:]].to(device)
y_test = y_raw[indices[n_train:]].to(device)

W1 = torch.randn(n_features, 6, dtype=torch.float32, requires_grad=True, device=device)
b1 = torch.zeros(1, 6, dtype=torch.float32, requires_grad=True, device=device)
W2 = torch.randn(6, 1, dtype=torch.float32, requires_grad=True, device=device)
b2 = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True, device=device)

optimizer = torch.optim.Adam([W1, b1, W2, b2], lr=0.001)

epochs = 1000
noise_std = 0.05 
loss_history = []
start_time = time.time()

for epoch in range(epochs):
    # CATEGORY II: Pure Noise Injection
    noise = torch.randn_like(X_train, device=device) * noise_std
    X_train_noisy = X_train + noise
    
    Z1 = torch.matmul(X_train_noisy, W1) + b1
    A1 = torch.relu(Z1) 
    Z2 = torch.matmul(A1, W2) + b2
    y_hat = torch.sigmoid(Z2)
    
    y_hat_safe = torch.clamp(y_hat, min=1e-7, max=1.0 - 1e-7)
    loss = -torch.mean(y_train * torch.log(y_hat_safe) + (1 - y_train) * torch.log(1 - y_hat_safe))
    loss_history.append(loss.item())
    
    optimizer.zero_grad() 
    loss.backward()       
    optimizer.step()      
    
    if epoch == 0 or (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch:4} | BCE Loss: {loss.item():.4f}")

print(f"Training completed in {time.time() - start_time:.2f} seconds.")

with torch.no_grad():
    Z1_test = torch.matmul(X_test, W1) + b1
    A1_test = torch.relu(Z1_test)
    Z2_test = torch.matmul(A1_test, W2) + b2
    y_hat_test = torch.sigmoid(Z2_test)
    
    test_preds = (y_hat_test >= 0.5).float()
    test_acc = torch.mean((test_preds == y_test).float()) * 100
    print(f"\nAdvanced Test Accuracy (Noise Only): {test_acc.item():.2f}%")
