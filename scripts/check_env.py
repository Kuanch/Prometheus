#!/usr/bin/env python3
"""
Environment Check Script for GPU Architecture Learning Project

Verifies that all dependencies are correctly installed:
- Python version
- PyTorch installation
- CUDA availability
- GPU properties
- Custom CUDA kernels
- Basic GPU operation test
"""

import sys
import os

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_header(text):
    print(f"\n{BLUE}{BOLD}{'='*60}{RESET}")
    print(f"{BLUE}{BOLD} {text}{RESET}")
    print(f"{BLUE}{BOLD}{'='*60}{RESET}")


def check_pass(name, detail=""):
    detail_str = f" ({detail})" if detail else ""
    print(f"  {GREEN}✓{RESET} {name}{detail_str}")
    return True


def check_fail(name, detail=""):
    detail_str = f" ({detail})" if detail else ""
    print(f"  {RED}✗{RESET} {name}{detail_str}")
    return False


def check_warn(name, detail=""):
    detail_str = f" ({detail})" if detail else ""
    print(f"  {YELLOW}⚠{RESET} {name}{detail_str}")
    return True


def check_python():
    """Check Python version"""
    print_header("Python Environment")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major >= 3 and version.minor >= 8:
        check_pass("Python version", version_str)
        return True
    else:
        check_fail("Python version", f"{version_str} (requires 3.8+)")
        return False


def check_pytorch():
    """Check PyTorch installation"""
    print_header("PyTorch")

    try:
        import torch
        check_pass("PyTorch installed", torch.__version__)
    except ImportError:
        check_fail("PyTorch not installed")
        return False

    # Check CUDA availability
    if torch.cuda.is_available():
        check_pass("CUDA available", f"CUDA {torch.version.cuda}")
    else:
        check_fail("CUDA not available")
        return False

    # Check cuDNN
    if torch.backends.cudnn.is_available():
        check_pass("cuDNN available", f"v{torch.backends.cudnn.version()}")
    else:
        check_warn("cuDNN not available")

    return True


def check_gpu():
    """Check GPU properties"""
    print_header("GPU Information")

    try:
        import torch

        if not torch.cuda.is_available():
            check_fail("No GPU detected")
            return False

        device_count = torch.cuda.device_count()
        check_pass(f"GPU count", str(device_count))

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\n  {BOLD}GPU {i}: {props.name}{RESET}")
            print(f"    Compute Capability: {props.major}.{props.minor}")
            print(f"    Total Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"    SM Count: {props.multi_processor_count}")
            print(f"    Max Threads/Block: {props.max_threads_per_block}")

        return True

    except Exception as e:
        check_fail("GPU check failed", str(e))
        return False


def check_cuda_kernels():
    """Check if custom CUDA kernels are built"""
    print_header("Custom CUDA Kernels")

    kernels = ['vector_add', 'relu', 'matmul', 'memory_patterns']
    all_good = True

    for kernel in kernels:
        try:
            module = __import__(f'cuda_kernels.{kernel}', fromlist=[kernel])
            check_pass(f"cuda_kernels.{kernel}")
        except ImportError as e:
            check_fail(f"cuda_kernels.{kernel}", "not built")
            all_good = False

    if not all_good:
        print(f"\n  {YELLOW}To build kernels:{RESET}")
        print(f"    pip install -e /workspace/kernels")

    return all_good


def check_gpu_operation():
    """Run a simple GPU operation to verify functionality"""
    print_header("GPU Operation Test")

    try:
        import torch

        # Create tensors on GPU
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')

        # Matrix multiply
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

        check_pass("Matrix multiply on GPU")

        # Test memory transfer
        cpu_tensor = torch.randn(1000, 1000)
        gpu_tensor = cpu_tensor.cuda()
        back_to_cpu = gpu_tensor.cpu()

        if torch.allclose(cpu_tensor, back_to_cpu):
            check_pass("CPU ↔ GPU transfer")
        else:
            check_fail("CPU ↔ GPU transfer", "data mismatch")
            return False

        # Test custom kernel if available
        try:
            from cuda_kernels import vector_add
            x = torch.randn(10000, device='cuda')
            y = torch.randn(10000, device='cuda')
            z = vector_add.vector_add(x, y)
            ref = x + y
            if torch.allclose(z, ref):
                check_pass("Custom CUDA kernel execution")
            else:
                check_fail("Custom CUDA kernel", "incorrect result")
                return False
        except ImportError:
            check_warn("Custom kernel test skipped", "not built")

        return True

    except Exception as e:
        check_fail("GPU operation failed", str(e))
        return False


def check_profilers():
    """Check if Nsight tools are available"""
    print_header("Profiling Tools")

    import shutil

    nsys = shutil.which('nsys')
    if nsys:
        check_pass("Nsight Systems (nsys)", nsys)
    else:
        check_warn("Nsight Systems not in PATH")

    ncu = shutil.which('ncu')
    if ncu:
        check_pass("Nsight Compute (ncu)", ncu)
    else:
        check_warn("Nsight Compute not in PATH")

    return True


def check_workspace():
    """Check workspace structure"""
    print_header("Workspace Structure")

    workspace = os.environ.get('WORKSPACE', '/workspace')

    expected_dirs = ['experiments', 'kernels', 'utils', 'scripts', 'output']
    expected_files = ['Dockerfile', 'docker-compose.yml', 'requirements.txt']

    for d in expected_dirs:
        path = os.path.join(workspace, d)
        if os.path.isdir(path):
            check_pass(f"Directory: {d}/")
        else:
            check_warn(f"Directory: {d}/", "not found")

    for f in expected_files:
        path = os.path.join(workspace, f)
        if os.path.isfile(path):
            check_pass(f"File: {f}")
        else:
            check_warn(f"File: {f}", "not found")

    return True


def main():
    print(f"\n{BOLD}GPU Architecture Learning - Environment Check{RESET}")
    print(f"{'='*60}")

    results = {}

    results['python'] = check_python()
    results['pytorch'] = check_pytorch()
    results['gpu'] = check_gpu()
    results['kernels'] = check_cuda_kernels()
    results['operation'] = check_gpu_operation()
    results['profilers'] = check_profilers()
    results['workspace'] = check_workspace()

    # Summary
    print_header("Summary")

    critical = ['python', 'pytorch', 'gpu', 'operation']
    optional = ['kernels', 'profilers', 'workspace']

    critical_pass = all(results[k] for k in critical)
    optional_pass = all(results[k] for k in optional)

    if critical_pass and optional_pass:
        print(f"\n  {GREEN}{BOLD}All checks passed!{RESET}")
        print(f"  Environment is ready for experiments.")
        return 0
    elif critical_pass:
        print(f"\n  {YELLOW}{BOLD}Core environment OK, some optional components missing.{RESET}")
        print(f"  You can run most experiments.")
        return 0
    else:
        print(f"\n  {RED}{BOLD}Critical checks failed.{RESET}")
        print(f"  Please fix the issues above before running experiments.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
