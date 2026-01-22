#!/usr/bin/env bash

set -u
set -o pipefail

# =============================================================================
# Autotuner for GEMM Warp Tiling Kernel
# Brute-force search through parameter combinations to find optimal config
# =============================================================================

# Get project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Project files
DEFINITIONS="$PROJECT_ROOT/inc/definitions.hpp"
EXECUTABLE="$PROJECT_ROOT/bin/gemm_warp_tiling"
OUTPUT_DIR="$PROJECT_ROOT/benchmark_results"
OUTPUT="$OUTPUT_DIR/warp_tiling_autotune_results.txt"

# Check that definitions file exists
if [[ ! -f "$DEFINITIONS" ]]; then
    echo "Error: $DEFINITIONS not found"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Parameter ranges to search
# =============================================================================
BKWARP_VALUES=(8 16 32)
BMWARP_VALUES=(64 128 256)
BNWARP_VALUES=(64 128 256)
WM_VALUES=(8 16 32 64)
WN_VALUES=(16 32 64)
WNITER_VALUES=(1 2 4)
TMWARPS_VALUES=(2 4 8)
TNWARPS_VALUES=(2 4 8)

# Fixed values
WARPSIZE=32

# Calculate total configurations
TOTAL_CONFIGS="$(( ${#BKWARP_VALUES[@]} * ${#BMWARP_VALUES[@]} * ${#BNWARP_VALUES[@]} * ${#WM_VALUES[@]} * ${#WN_VALUES[@]} * ${#WNITER_VALUES[@]} * ${#TMWARPS_VALUES[@]} * ${#TNWARPS_VALUES[@]} ))"

echo "============================================="
echo "GEMM Warp Tiling Autotuner"
echo "Total configurations to test: $TOTAL_CONFIGS"
echo "Output file: $OUTPUT"
echo "============================================="

# Clear the output file
echo "# GEMM Warp Tiling Autotune Results" > "$OUTPUT"
echo "# Date: $(date)" >> "$OUTPUT"
echo "" >> "$OUTPUT"

CONFIG_NUM=0
VALID_CONFIGS=0

# Main loop through all parameter combinations
for BKWARP in "${BKWARP_VALUES[@]}"; do
for BMWARP in "${BMWARP_VALUES[@]}"; do
for BNWARP in "${BNWARP_VALUES[@]}"; do
for WM in "${WM_VALUES[@]}"; do
for WN in "${WN_VALUES[@]}"; do
for WNITER in "${WNITER_VALUES[@]}"; do
for TMWARPS in "${TMWARPS_VALUES[@]}"; do
for TNWARPS in "${TNWARPS_VALUES[@]}"; do

CONFIG_NUM=$(( CONFIG_NUM + 1 ))

# Validate constraints
# BN must be divisible by WN, BM must be divisible by WM
if ! (( BNWARP % WN == 0 && BMWARP % WM == 0 )); then
    continue
fi

# Calculate derived values
NUM_WARPS=$(( (BNWARP / WN) * (BMWARP / WM) ))
NUM_THREADS=$(( NUM_WARPS * WARPSIZE ))

# Thread count must be reasonable (max 1024 per block)
if (( NUM_THREADS > 1024 )); then
    continue
fi

# (WM * WN) % (WARPSIZE * TMWARPS * TNWARPS * WNITER) must be 0
if ! (( (WM * WN) % (WARPSIZE * TMWARPS * TNWARPS * WNITER) == 0 )); then
    continue
fi

WMITER=$(( (WM * WN) / (WARPSIZE * TMWARPS * TNWARPS * WNITER) ))

# WM % WMITER must be 0 and WN % WNITER must be 0
if ! (( WM % WMITER == 0 && WN % WNITER == 0 )); then
    continue
fi

# Calculate WSUBN and WSUBM
WSUBN=$(( WN / WNITER ))
WSUBM=$(( WM / WMITER ))

# Shared memory loading constraints
if ! (( (NUM_THREADS * 4) % BKWARP == 0 )); then
    continue
fi
if ! (( (NUM_THREADS * 4) % BNWARP == 0 )); then
    continue
fi

# Shared memory tile loading constraints
# The kernel loads one element per thread, so we need enough threads
# to cover the entire tile in a single pass
# For A tile (BNWARP x BKWARP): need NUM_THREADS >= BNWARP * BKWARP
if (( NUM_THREADS < BNWARP * BKWARP )); then
    continue
fi
# For B tile (BKWARP x BMWARP): need NUM_THREADS >= BKWARP * BMWARP
if (( NUM_THREADS < BKWARP * BMWARP )); then
    continue
fi

# Valid configuration found
VALID_CONFIGS=$(( VALID_CONFIGS + 1 ))

echo ""
echo "[$CONFIG_NUM/$TOTAL_CONFIGS] Testing config #$VALID_CONFIGS:"
echo "  BKWARP=$BKWARP BMWARP=$BMWARP BNWARP=$BNWARP"
echo "  WM=$WM WN=$WN WNITER=$WNITER WMITER=$WMITER"
echo "  TMWARPS=$TMWARPS TNWARPS=$TNWARPS"
echo "  NUM_THREADS=$NUM_THREADS NUM_WARPS=$NUM_WARPS"

# Update definitions.hpp with new parameters
sed -i "s/^#define BKWARP .*/#define BKWARP $BKWARP/" "$DEFINITIONS"
sed -i "s/^#define BMWARP .*/#define BMWARP $BMWARP/" "$DEFINITIONS"
sed -i "s/^#define BNWARP .*/#define BNWARP $BNWARP/" "$DEFINITIONS"
sed -i "s/^#define WM .*/#define WM $WM/" "$DEFINITIONS"
sed -i "s/^#define WN .*/#define WN $WN/" "$DEFINITIONS"
sed -i "s/^#define WNITER .*/#define WNITER $WNITER/" "$DEFINITIONS"
sed -i "s/^#define WMITER .*/#define WMITER $WMITER/" "$DEFINITIONS"
sed -i "s/^#define TMWARPS .*/#define TMWARPS $TMWARPS/" "$DEFINITIONS"
sed -i "s/^#define TNWARPS .*/#define TNWARPS $TNWARPS/" "$DEFINITIONS"
sed -i "s/^#define WSUBN .*/#define WSUBN $WSUBN/" "$DEFINITIONS"
sed -i "s/^#define WSUBM .*/#define WSUBM $WSUBM/" "$DEFINITIONS"

# Rebuild the warp tiling executable only
cd "$PROJECT_ROOT"
if ! make warp -s -j 2>/dev/null; then
    echo "  Build FAILED - skipping"
    echo "CONFIG: BKWARP=$BKWARP BMWARP=$BMWARP BNWARP=$BNWARP WM=$WM WN=$WN WNITER=$WNITER TMWARPS=$TMWARPS TNWARPS=$TNWARPS - BUILD FAILED" >> "$OUTPUT"
    continue
fi

# Run benchmark with timeout
echo "CONFIG: BKWARP=$BKWARP BMWARP=$BMWARP BNWARP=$BNWARP WM=$WM WN=$WN WNITER=$WNITER WMITER=$WMITER TMWARPS=$TMWARPS TNWARPS=$TNWARPS NUM_THREADS=$NUM_THREADS" >> "$OUTPUT"

# Run and capture output + exit status
BENCH_OUTPUT=$(timeout 30 "$EXECUTABLE" 2>&1) || BENCH_EXIT=$?
BENCH_EXIT=${BENCH_EXIT:-0}

echo "$BENCH_OUTPUT" | tee -a "$OUTPUT"

if [[ $BENCH_EXIT -ne 0 ]]; then
    echo "  FAILED (exit code: $BENCH_EXIT)"
    echo "RESULT: FAILED (exit $BENCH_EXIT)" >> "$OUTPUT"
fi
echo "" >> "$OUTPUT"

done
done
done
done
done
done
done
done

# Summary
echo ""
echo "============================================="
echo "Autotuning complete!"
echo "Tested $VALID_CONFIGS valid configurations out of $TOTAL_CONFIGS total"
echo "Results saved to: $OUTPUT"
echo "============================================="
