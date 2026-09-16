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

alpha = 0.01
iterations = 100
m, n = X.shape
w = np.zeros((n, 1))

for i in range(iterations):
    y_hat = X @ w
    error = y_hat - y_scaled
    gradient = (1 / m) * (X.T @ error)
    w = w - alpha * gradient

print("--- Training Complete ---")
print("Optimal Weights:\n", w)

X_new_raw = np.array([
    [31, 0, 22],
    [43, 1, 19],
    [28, 1, 33],
    [50, 0, 28]
])

X_new_scaled = (X_new_raw - X_mean) / X_std

X_new_final = np.c_[np.ones(X_new_scaled.shape[0]), X_new_scaled]

y_pred_scaled = X_new_final @ w

y_pred_actual = (y_pred_scaled * y_std) + y_mean

print("Predicted Charges:\n", y_pred_actual)
