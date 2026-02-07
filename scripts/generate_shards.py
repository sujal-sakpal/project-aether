import numpy as np
import os

os.makedirs("weights/worker_a", exist_ok=True)
os.makedirs("weights/worker_b", exist_ok=True)

# 1. Weights (Keeping the clean 1x and 5x math)
weight_a = np.eye(5, dtype=np.float32)
weight_b = (np.eye(5) * 5.0).astype(np.float32)

# 2. Bias Vectors (Clean addition)
# Worker A adds 10 to every element
bias_a = np.array([10, 10, 10, 10, 10], dtype=np.float32)
# Worker B adds 100 to every element
bias_b = np.array([100, 100, 100, 100, 100], dtype=np.float32)

# 3. Save everything
np.save("weights/worker_a/layer.npy", weight_a)
np.save("weights/worker_a/bias.npy", bias_a)
np.save("weights/worker_b/layer.npy", weight_b)
np.save("weights/worker_b/bias.npy", bias_b)

print("✅ Weights AND Biases Generated!")