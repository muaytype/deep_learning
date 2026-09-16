import pandas as pd
import numpy as np

df = pd.read_csv('data/UTF-8_insurance_charges.csv')

X_raw = df[['age', 'gender', 'bmi']].values
y_raw = df['charges'].values.reshape(-1, 1)

X_mean = np.mean(X_raw, axis=0)
X_std = np.std(X_raw, axis=0)
X_scaled = (X_raw - X_mean) / X_std

y_mean = np.mean(y_raw, axis=0)
y_std = np.std(y_raw, axis=0)
y_scaled = (y_raw - y_mean) / y_std

X = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

m, n = X.shape 
w = np.zeros((n, 1)) 
learning_rates = [0.1, 0.01, 0.001]
iterations = 100

print("--- Learning Rate Experiment ---")

for alpha in learning_rates:
    
    w = np.zeros((X.shape[1], 1))
    
    for i in range(iterations):
        y_hat = X @ w
        error = y_hat - y_scaled
        gradient = (1 / m) * (X.T @ error)
        
        w = w - alpha * gradient
        
        if i == iterations - 1:
            final_loss = (1 / (2 * m)) * np.sum(error ** 2)
            
    print(f"Learning Rate {alpha}: Final Loss = {final_loss:.4f}")

print("Final Scaled Weight Vector (w):\n", w)
