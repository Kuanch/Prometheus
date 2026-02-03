#!/bin/bash
# GPU Profiling Scripts
# Run these inside the Docker container

set -e

OUTPUT_DIR="/output"
WORKSPACE="/workspace"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

case "$1" in
    # Run all experiments without profiling
    "run-all")
        print_header "Running All Experiments"
        for exp in 01_vector_add 02_relu 03_matmul 04_memory_patterns 05_overhead_analysis; do
            echo -e "${GREEN}Running $exp...${NC}"
            python "$WORKSPACE/experiments/${exp}.py"
            echo ""
        done
        ;;

    # Nsight Systems profiling (timeline view)
    "nsys")
        EXPERIMENT="${2:-01_vector_add}"
        print_header "Nsight Systems Profile: $EXPERIMENT"

        nsys profile \
            --stats=true \
            --force-overwrite=true \
            -o "$OUTPUT_DIR/nsys_${EXPERIMENT}" \
            python "$WORKSPACE/experiments/${EXPERIMENT}.py"

        echo -e "\n${GREEN}Report saved to: $OUTPUT_DIR/nsys_${EXPERIMENT}.nsys-rep${NC}"
        echo "Open in Nsight Systems GUI on Windows"
        ;;

    # Nsight Compute profiling (kernel analysis)
    "ncu")
        EXPERIMENT="${2:-01_vector_add}"
        print_header "Nsight Compute Profile: $EXPERIMENT"

        # Note: ncu requires elevated privileges
        ncu \
            --set full \
            --force-overwrite \
            --export "$OUTPUT_DIR/ncu_${EXPERIMENT}" \
            python "$WORKSPACE/experiments/${EXPERIMENT}.py"

        echo -e "\n${GREEN}Report saved to: $OUTPUT_DIR/ncu_${EXPERIMENT}.ncu-rep${NC}"
        echo "Open in Nsight Compute GUI on Windows"
        ;;

    # Quick nsys profile (less detailed, faster)
    "nsys-quick")
        EXPERIMENT="${2:-01_vector_add}"
        print_header "Quick Nsight Systems Profile: $EXPERIMENT"

        nsys profile \
            --trace=cuda,nvtx \
            --force-overwrite=true \
            -o "$OUTPUT_DIR/nsys_quick_${EXPERIMENT}" \
            python "$WORKSPACE/experiments/${EXPERIMENT}.py"

        echo -e "\n${GREEN}Report saved to: $OUTPUT_DIR/nsys_quick_${EXPERIMENT}.nsys-rep${NC}"
        ;;

    # Profile specific kernel with ncu
    "ncu-kernel")
        KERNEL_NAME="${2:-vector_add_kernel}"
        EXPERIMENT="${3:-01_vector_add}"
        print_header "Nsight Compute: Kernel $KERNEL_NAME"

        ncu \
            --kernel-name "$KERNEL_NAME" \
            --launch-count 10 \
            --set full \
            --force-overwrite \
            --export "$OUTPUT_DIR/ncu_kernel_${KERNEL_NAME}" \
            python "$WORKSPACE/experiments/${EXPERIMENT}.py"

        echo -e "\n${GREEN}Report saved to: $OUTPUT_DIR/ncu_kernel_${KERNEL_NAME}.ncu-rep${NC}"
        ;;

    # Memory analysis with ncu
    "ncu-memory")
        EXPERIMENT="${2:-04_memory_patterns}"
        print_header "Nsight Compute Memory Analysis: $EXPERIMENT"

        ncu \
            --section MemoryWorkloadAnalysis \
            --section MemoryWorkloadAnalysis_Chart \
            --section MemoryWorkloadAnalysis_Tables \
            --force-overwrite \
            --export "$OUTPUT_DIR/ncu_memory_${EXPERIMENT}" \
            python "$WORKSPACE/experiments/${EXPERIMENT}.py"

        echo -e "\n${GREEN}Report saved to: $OUTPUT_DIR/ncu_memory_${EXPERIMENT}.ncu-rep${NC}"
        ;;

    # Print stats from nsys report
    "nsys-stats")
        REPORT="${2:-$OUTPUT_DIR/nsys_01_vector_add.nsys-rep}"
        print_header "Nsight Systems Stats"

        nsys stats "$REPORT"
        ;;

    # List available experiments
    "list")
        print_header "Available Experiments"
        echo "  01_vector_add      - Memory-bound: vector addition"
        echo "  02_relu            - Memory-bound: element-wise ReLU"
        echo "  03_matmul          - Compute-bound: matrix multiplication"
        echo "  04_memory_patterns - Memory access pattern analysis"
        echo "  05_overhead_analysis - Framework overhead measurement"
        ;;

    # Help
    *)
        print_header "GPU Profiling Scripts"
        echo "Usage: ./run_profiler.sh <command> [args]"
        echo ""
        echo "Commands:"
        echo "  run-all                    Run all experiments (no profiling)"
        echo "  nsys [experiment]          Full Nsight Systems profile"
        echo "  nsys-quick [experiment]    Quick Nsight Systems profile"
        echo "  ncu [experiment]           Full Nsight Compute profile"
        echo "  ncu-kernel <name> [exp]    Profile specific kernel"
        echo "  ncu-memory [experiment]    Memory-focused analysis"
        echo "  nsys-stats [report]        Print stats from nsys report"
        echo "  list                       List available experiments"
        echo ""
        echo "Examples:"
        echo "  ./run_profiler.sh run-all"
        echo "  ./run_profiler.sh nsys 01_vector_add"
        echo "  ./run_profiler.sh ncu 03_matmul"
        echo "  ./run_profiler.sh ncu-kernel matmul_tiled_kernel 03_matmul"
        echo ""
        echo "Output files are saved to: $OUTPUT_DIR/"
        echo "View .nsys-rep files in Nsight Systems GUI"
        echo "View .ncu-rep files in Nsight Compute GUI"
        ;;
esac
