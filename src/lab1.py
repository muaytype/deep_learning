import numpy as np
import matplotlib.pyplot as plt

N = 5 
X = np.random.randn(N, 2)
w = np.random.randn(2, 1)
b = 0.5

z = X @ w + b

print(f"Shape of Data (X): {X.shape}")
print(f"Shape of Weights (w): {w.shape}")
print(f"Shape of Scores (z): {z.shape}\n")


# Apply the threshold (z >= 0) to get binary predictions
y_hat = (z >= 0).astype(int)
print(f"Raw Scores (z):\n{z}")
print(f"Initial Predictions (y_hat):\n{y_hat}\n")


w_rotated = np.array([[1.0], [0.0]])
b_translated = -2.0

z_transformed = X @ w_rotated + b_translated
y_hat_transformed = (z_transformed >= 0).astype(int)

print(f"New Weights (Horizontal):\n{w_rotated}")
print(f"New Bias: {b_translated}")
print(f"Transformed Predictions:\n{y_hat_transformed}\n")


y_true = np.array([[1], [1], [0], [0], [1]])

TP = np.sum((y_hat_transformed == 1) & (y_true == 1))
TN = np.sum((y_hat_transformed == 0) & (y_true == 0))
FP = np.sum((y_hat_transformed == 1) & (y_true == 0))
FN = np.sum((y_hat_transformed == 0) & (y_true == 1))

accuracy = (TP + TN) / N

print(f"True Positives (TP):  {TP}")
print(f"True Negatives (TN):  {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"Overall Accuracy:     {accuracy * 100}%\n")


plt.figure(figsize=(8, 6))

plt.scatter(X[:, 0], X[:, 1], c=y_hat.flatten(), cmap='bwr', edgecolor='k', s=100, label="Data Points")

x_min, x_max = -3, 3
y_min, y_max = -3, 3
xx1, xx2 = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

grid_points = np.c_[xx1.ravel(), xx2.ravel()]
Z = (grid_points @ w + b).reshape(xx1.shape)

plt.contour(xx1, xx2, Z, levels=[0], colors='k', linestyles='dashed', linewidths=2)

if w[1, 0] != 0:
    x1_origin, x2_origin = 0, -b / w[1, 0]
else:
    x1_origin, x2_origin = -b / w[0, 0], 0

plt.quiver(x1_origin, x2_origin, w[0, 0], w[1, 0], angles='xy', scale_units='xy', scale=1, 
           color='green', label="Weight Vector (w)")

plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Linear Classifier Decision Boundary")
plt.grid(True, linestyle=':', alpha=0.7)

# plt.show()
plt.savefig("boundary.png")
