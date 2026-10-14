import matplotlib.pyplot as plt
import os

# 1. Hardcode the average loss values from your terminal outputs
epochs = [1, 2, 3, 4, 5]
vanilla_loss = [17.5883, 1.6355, 1.1298, 0.8409, 0.4979]
advanced_loss = [12.8017, 3.0423, 2.5569, 2.1567, 1.8216]

# 2. Set up the canvas
plt.figure(figsize=(10, 6))
plt.plot(epochs, vanilla_loss, marker='o', linestyle='-', color='red', label='Vanilla CNN (Task 1)')
plt.plot(epochs, advanced_loss, marker='s', linestyle='--', color='blue', label='Advanced CNN (Task 2)')

# 3. Label the axes and add a grid for readability
plt.title('Training Loss Comparison: Vanilla vs. Advanced CNN')
plt.xlabel('Epoch')
plt.ylabel('Average Cross Entropy Loss')
plt.xticks(epochs)
plt.grid(True, alpha=0.5)
plt.legend()

# 4. HPC Headless Save - Ensure the directory exists
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(output_dir, "loss_comparison.png")

plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Graph successfully saved to {file_path}")
