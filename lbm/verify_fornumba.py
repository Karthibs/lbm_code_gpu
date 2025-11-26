import numba
from numba import cuda

print(f"Numba Version: {numba.__version__}")

try:
    # Check if CUDA is available first
    if cuda.is_available():
        # Get the version of the CUDA Runtime linked to Numba
        # This returns a tuple like (12, 0)
        ver = cuda.runtime.get_version()
        print(f"CUDA Runtime Version (Numba): {ver[0]}.{ver[1]}")
        
        # Get the driver version
        driver_ver = cuda.driver.driver.get_version()
        print(f"CUDA Driver Version: {driver_ver[0]}.{driver_ver[1]}")
    else:
        print("CUDA is NOT available to Numba.")
except Exception as e:
    print(f"Error checking CUDA: {e}")