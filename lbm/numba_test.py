import time
import numpy as np

from numba import cuda

# -------------------------
# GPU Kernel (Numba CUDA)
# -------------------------
@cuda.jit
def vector_add(a, b, c):
    i = cuda.grid(1)
    if i < a.size:
        c[i] = a[i] + b[i]

# -------------------------
# Benchmark Parameters
# -------------------------
N = 50_000_000  # 50 million elements
print(f"Array size: {N:,}")

# Prepare host arrays
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)

# -------------------------
# NumPy (CPU)
# -------------------------
t0 = time.time()
c_cpu = a + b
t_cpu = time.time() - t0
print(f"NumPy CPU time: {t_cpu:.4f} sec")

# -------------------------
# Numba CUDA (GPU)
# -------------------------

# Copy data to GPU
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.device_array_like(a)

# Kernel launch configuration
threads_per_block = 256
blocks = (N + threads_per_block - 1) // threads_per_block

# Warm-up (important for fair timing)
vector_add[blocks, threads_per_block](d_a, d_b, d_c)
cuda.synchronize()

# Timed GPU execution
t0 = time.time()
vector_add[blocks, threads_per_block](d_a, d_b, d_c)
cuda.synchronize()
t_gpu = time.time() - t0

print(f"Numba CUDA GPU time: {t_gpu:.4f} sec")

# Copy back result
c_gpu = d_c.copy_to_host()

# -------------------------
# Validate correctness
# -------------------------
if np.allclose(c_cpu, c_gpu):
    print("Output check: OK ✔")
else:
    print("Output check: FAILED ❌")
