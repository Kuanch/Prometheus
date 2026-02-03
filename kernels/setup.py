"""
PyTorch C++ Extension Build Script for Custom CUDA Kernels

Usage:
    cd /workspace/kernels
    python setup.py install

Or for development (rebuild on import):
    python setup.py develop
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define CUDA extensions
ext_modules = [
    CUDAExtension(
        name='cuda_kernels.vector_add',
        sources=[os.path.join(script_dir, 'vector_add.cu')],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-lineinfo',  # Enable line info for profiling
            ]
        }
    ),
    CUDAExtension(
        name='cuda_kernels.relu',
        sources=[os.path.join(script_dir, 'relu.cu')],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-lineinfo',
            ]
        }
    ),
    CUDAExtension(
        name='cuda_kernels.matmul',
        sources=[os.path.join(script_dir, 'matmul.cu')],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-lineinfo',
            ]
        }
    ),
    CUDAExtension(
        name='cuda_kernels.memory_patterns',
        sources=[os.path.join(script_dir, 'memory_patterns.cu')],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-lineinfo',
            ]
        }
    ),
]

setup(
    name='cuda_kernels',
    version='0.1.0',
    description='Custom CUDA kernels for GPU architecture learning',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=['cuda_kernels'],
    package_dir={'cuda_kernels': '.'},
)
