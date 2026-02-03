# CUDA Kernels Package
# Import compiled extensions after setup.py install

try:
    from cuda_kernels import vector_add
    from cuda_kernels import relu
    from cuda_kernels import matmul
    from cuda_kernels import memory_patterns
except ImportError:
    print("CUDA kernels not built yet. Run: cd kernels && python setup.py install")
