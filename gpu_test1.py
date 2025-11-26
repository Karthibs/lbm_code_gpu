import numpy as np
import cupy as cp
import time

# -----------------------------
# Example: large matrix multiplication
# -----------------------------

# Matrix size
N = 5000

# Generate random matrices on CPU
A_cpu = np.random.rand(N, N).astype(np.float32)
B_cpu = np.random.rand(N, N).astype(np.float32)

# -----------------------------
# CPU timing using NumPy
# -----------------------------
start = time.time()
C_cpu = np.dot(A_cpu, B_cpu)
end = time.time()
print(f"CPU time: {end - start:.4f} seconds")

# -----------------------------
# GPU timing using CuPy
# -----------------------------
# Transfer matrices to GPU
A_gpu = cp.array(A_cpu)
B_gpu = cp.array(B_cpu)

# Warm-up (optional, first GPU call can be slower)
#_ = cp.dot(A_gpu, B_gpu)

# Measure GPU time
start = time.time()
C_gpu = cp.dot(A_gpu, B_gpu)
# Make sure GPU finishes computation
cp.cuda.Stream.null.synchronize()
end = time.time()
print(f"GPU time: {end - start:.4f} seconds")