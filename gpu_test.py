import cupy as cp

def main():
    print("=== CuPy GPU Test ===")

    # Check for CUDA availability
    try:
        num_devices = cp.cuda.runtime.getDeviceCount()
        print(f"Number of CUDA devices detected: {num_devices}")
    except cp.cuda.runtime.CUDARuntimeError as e:
        print("CUDA runtime error:", e)
        print("→ No CUDA GPU detected or CUDA is not installed correctly.")
        return

    # Print device info
    for i in range(num_devices):
        props = cp.cuda.runtime.getDeviceProperties(i)
        name = props["name"].decode("utf-8")
        print(f"Device {i}: {name}")
        print(f"  Compute capability: {props['major']}.{props['minor']}")
        print(f"  Total memory: {props['totalGlobalMem'] / 1e9:.2f} GB")

    # Run a simple computation
    try:
        print("\nRunning a small GPU computation...")
        x = cp.arange(10**7, dtype=cp.float32)
        y = cp.sin(x) * cp.cos(x)
        cp.cuda.Stream.null.synchronize()
        print("Computation succeeded.")
    except Exception as e:
        print("GPU computation failed:", e)
        return

    print("\n✓ GPU test successful! CuPy and CUDA are working correctly.")

if __name__ == "__main__":
    main()