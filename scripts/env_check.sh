#!/bin/bash
# Environment Check Script for GPU Architecture Learning Project
#
# Usage:
#   ./scripts/env_check.sh          # Auto-detect environment
#   ./scripts/env_check.sh docker   # Force Docker checks
#   ./scripts/env_check.sh host     # Force host checks

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}${BOLD}============================================================${NC}"
    echo -e "${BLUE}${BOLD} $1${NC}"
    echo -e "${BLUE}${BOLD}============================================================${NC}"
}

check_pass() {
    echo -e "  ${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "  ${RED}✗${NC} $1"
}

check_warn() {
    echo -e "  ${YELLOW}⚠${NC} $1"
}

# Detect if running inside Docker
detect_environment() {
    if [ -f /.dockerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
        echo "docker"
    else
        echo "host"
    fi
}

# Check Docker environment (inside container)
check_docker_env() {
    print_header "Docker Environment Checks"

    local all_pass=true

    # Check NVIDIA driver
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            check_pass "NVIDIA driver accessible"
            nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | while read line; do
                echo "       $line"
            done
        else
            check_fail "NVIDIA driver not working"
            all_pass=false
        fi
    else
        check_fail "nvidia-smi not found"
        all_pass=false
    fi

    # Check CUDA toolkit
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
        check_pass "CUDA toolkit ($cuda_version)"
    else
        check_warn "nvcc not found (CUDA toolkit may not be in PATH)"
    fi

    # Check Python
    if command -v python &> /dev/null; then
        py_version=$(python --version 2>&1)
        check_pass "Python ($py_version)"
    else
        check_fail "Python not found"
        all_pass=false
    fi

    # Check workspace mount
    if [ -d "/workspace" ]; then
        check_pass "Workspace mounted at /workspace"
    else
        check_warn "Workspace not mounted (expected at /workspace)"
    fi

    # Check output directory
    if [ -d "/output" ]; then
        check_pass "Output directory mounted at /output"
        if [ -w "/output" ]; then
            check_pass "Output directory is writable"
        else
            check_warn "Output directory is not writable"
        fi
    else
        check_warn "Output directory not mounted (expected at /output)"
    fi

    # Run Python checks
    print_header "Running Python Environment Checks"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "$SCRIPT_DIR/check_env.py" ]; then
        python "$SCRIPT_DIR/check_env.py"
    elif [ -f "/workspace/scripts/check_env.py" ]; then
        python /workspace/scripts/check_env.py
    else
        check_warn "check_env.py not found, skipping Python checks"
    fi

    if [ "$all_pass" = true ]; then
        return 0
    else
        return 1
    fi
}

# Check host environment (outside container)
check_host_env() {
    print_header "Host Environment Checks"

    local all_pass=true

    # Check Docker
    if command -v docker &> /dev/null; then
        docker_version=$(docker --version | cut -d' ' -f3 | tr -d ',')
        check_pass "Docker installed ($docker_version)"
    else
        check_fail "Docker not installed"
        echo "       Install: https://docs.docker.com/get-docker/"
        all_pass=false
    fi

    # Check Docker Compose
    if command -v docker-compose &> /dev/null; then
        compose_version=$(docker-compose --version | cut -d' ' -f4 | tr -d ',')
        check_pass "Docker Compose ($compose_version)"
    elif docker compose version &> /dev/null; then
        compose_version=$(docker compose version | cut -d' ' -f4)
        check_pass "Docker Compose (plugin: $compose_version)"
    else
        check_fail "Docker Compose not installed"
        all_pass=false
    fi

    # Check NVIDIA Container Toolkit
    if command -v nvidia-container-toolkit &> /dev/null || \
       docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
        check_pass "NVIDIA Container Toolkit"
    else
        check_warn "NVIDIA Container Toolkit may not be installed"
        echo "       Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi

    # Check if nvidia-smi works on host
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            check_pass "NVIDIA driver (host)"
            gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
            echo "       GPU: $gpu_name"
        else
            check_fail "NVIDIA driver not working"
            all_pass=false
        fi
    else
        check_warn "nvidia-smi not found on host"
    fi

    # Check project files
    print_header "Project Files"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

    if [ -f "$PROJECT_DIR/Dockerfile" ]; then
        check_pass "Dockerfile"
    else
        check_fail "Dockerfile not found"
        all_pass=false
    fi

    if [ -f "$PROJECT_DIR/docker-compose.yml" ]; then
        check_pass "docker-compose.yml"
    else
        check_fail "docker-compose.yml not found"
        all_pass=false
    fi

    # Quick Docker GPU test
    print_header "Docker GPU Access Test"

    echo "  Testing GPU access inside Docker..."
    if docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi &> /dev/null; then
        check_pass "Docker can access GPU"
    else
        check_fail "Docker cannot access GPU"
        echo "       Make sure NVIDIA Container Toolkit is properly configured"
        all_pass=false
    fi

    # Summary
    print_header "Next Steps"

    if [ "$all_pass" = true ]; then
        echo -e "  ${GREEN}${BOLD}Host environment is ready!${NC}"
        echo ""
        echo "  To get started:"
        echo "    1. cd $PROJECT_DIR"
        echo "    2. docker-compose build"
        echo "    3. docker-compose run --rm gpu-lab pip install -e /workspace/kernels"
        echo "    4. docker-compose run --rm gpu-lab ./scripts/env_check.sh"
        echo "    5. docker-compose run --rm gpu-lab python experiments/01_vector_add.py"
    else
        echo -e "  ${RED}${BOLD}Some checks failed. Please fix the issues above.${NC}"
    fi

    if [ "$all_pass" = true ]; then
        return 0
    else
        return 1
    fi
}

# Main
main() {
    echo -e "\n${BOLD}GPU Architecture Learning - Environment Check${NC}"
    echo "============================================================"

    # Determine environment
    if [ "$1" = "docker" ]; then
        ENV_TYPE="docker"
    elif [ "$1" = "host" ]; then
        ENV_TYPE="host"
    else
        ENV_TYPE=$(detect_environment)
    fi

    echo -e "  Detected environment: ${BOLD}$ENV_TYPE${NC}"

    if [ "$ENV_TYPE" = "docker" ]; then
        check_docker_env
    else
        check_host_env
    fi
}

main "$@"
