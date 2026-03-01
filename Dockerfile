# GPU Architecture Learning Environment
# Base: NVIDIA PyTorch container with CUDA toolkit, cuDNN, and Nsight tools

FROM nvcr.io/nvidia/pytorch:25.04-py3

# Install additional tools for kernel development
RUN apt-get update && apt-get install -y --no-install-recommends \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set environment for PyTorch CUDA extensions (RTX 4060 = Ada Lovelace SM89)
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV CUDA_HOME=/usr/local/cuda

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set working directory
WORKDIR /workspace

# Default command
CMD ["bash"]
