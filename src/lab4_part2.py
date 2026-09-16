import pandas as pd
import numpy as np

df_iris = pd.read_csv('data/UTF-8_iris_2class_train.csv')

df_iris['class'] = df_iris['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1})

X_iris_raw = df_iris[['sepal length', 'sepal width', 'petal length', 'petal width']].values
y_iris = df_iris['class'].values.reshape(-1, 1)

X_iris_mean = np.mean(X_iris_raw, axis=0)
X_iris_std = np.std(X_iris_raw, axis=0)
X_iris_scaled = (X_iris_raw - X_iris_mean) / X_iris_std

X_iris = np.c_[np.ones(X_iris_scaled.shape[0]), X_iris_scaled]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

m, n = X_iris.shape
w_iris = np.zeros((n, 1)) # Initialize weights as a vector of zeros
alpha_iris = 0.1 # Learning rate for classification[cite: 8]
iterations = 100 # Total training steps

loss_history_iris = []

print("--- Classification Training ---")

for i in range(iterations):
    z = X_iris @ w_iris
    
    y_hat = sigmoid(z)
    
    error = y_hat - y_iris
    
    epsilon = 1e-15
    loss = -(1 / m) * np.sum(y_iris * np.log(y_hat + epsilon) + (1 - y_iris) * np.log(1 - y_hat + epsilon))
    loss_history_iris.append(loss)
    
    gradient = (1 / m) * (X_iris.T @ error)
    
    w_iris = w_iris - alpha_iris * gradient

print(f"Final Classification Loss after {iterations} steps: {loss_history_iris[-1]:.4f}")
print("Final Weight Vector (w_iris):\n", w_iris)

df_test = pd.read_csv('data/UTF-8_iris_2class_test.csv')

df_test['class'] = df_test['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1})
y_test = df_test['class'].values.reshape(-1, 1)

X_test_raw = df_test[['sepal length', 'sepal width', 'petal length', 'petal width']].values

X_test_scaled = (X_test_raw - X_iris_mean) / X_iris_std

X_test = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]

z_test = X_test @ w_iris
y_test_prob = sigmoid(z_test)

y_test_pred = (y_test_prob >= 0.5).astype(int)

TP = np.sum((y_test_pred == 1) & (y_test == 1))
TN = np.sum((y_test_pred == 0) & (y_test == 0))
FP = np.sum((y_test_pred == 1) & (y_test == 0))
FN = np.sum((y_test_pred == 0) & (y_test == 1))

accuracy = (TP + TN) / len(y_test)

print("--- Final Exam: Test Evaluation ---")
print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"Accuracy: {accuracy * 100:.2f}%")


m, n = X_iris.shape
w_mb = np.zeros((n, 1))
alpha_mb = 0.1 
epochs = 100 
batch_size = 16 # Our chunk size

loss_history_mb = []

batch_sizes = [4, 16, 64]
epochs = 100

print("\n--- Mini-Batch Size Experiment ---")
for batch_size in batch_sizes:
    w_mb = np.zeros((X_iris.shape[1], 1)) 
    
    for epoch in range(epochs):
        permutation = np.random.permutation(X_iris.shape[0])
        X_sh = X_iris[permutation]
        y_sh = y_iris[permutation]
        
        for i in range(0, X_iris.shape[0], batch_size):
            X_b = X_sh[i:i + batch_size]
            y_b = y_sh[i:i + batch_size]
            
            y_hat = sigmoid(X_b @ w_mb)
            gradient = (1 / X_b.shape[0]) * (X_b.T @ (y_hat - y_b))
            w_mb = w_mb - alpha_mb * gradient
            
    final_preds = sigmoid(X_iris @ w_mb)
    final_loss = -(1 / X_iris.shape[0]) * np.sum(y_iris * np.log(final_preds + 1e-15) + (1 - y_iris) * np.log(1 - final_preds + 1e-15))
    
    print(f"Batch Size {batch_size}: Final Loss = {final_loss:.4f}")
