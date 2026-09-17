#!/bin/bash
# Full correctness verification for the flex flash attention kernel.
#
#   ./run_correctness.sh [extra test_arb_suite.py args...]
#
# Covers the decomposition layer (slice decomposition / merge / tile alignment /
# vbatch layout, mask_mod equivalence, design envelope incl. holey-mask
# rejection, GPU-vs-CPU decomposer equivalence) and kernel numerics (every mask
# type, all headdim buckets, kBlockM=64/128 paths, zone & unaligned boundaries,
# fp16 (incl. targeted fp16 stressors), cached-interface bitwise equality).
# The kernel suite deliberately guards WIDER than the perf sweep: GQA breadth
# (MQA h_k=1, ratio-8, up to h=64, odd head counts, plus fp16 / d=128 /
# padding-d crosses), tile-boundary sequence lengths (63/65/127/129, s_q=1
# and s_k=1), odd non-aligned shapes (17x17, 333x777), extreme headdims
# (d=1/7/255), softmax_scale extremes (1e-6 near-uniform, 8.0 near one-hot),
# and long-sequence numerics up to S=65536 at h=1 — matching perf's longest
# cell, whose fp32 accuracy guard needs the same head budget.  It
# also runs the API-contract cases (dv != head_size, custom softmax_scale,
# non-contiguous inputs, all-False mask, tiny shapes, hand-built BICAUSAL
# bands with left < 0, input-contract rejections for head_dim > 256 and
# non-divisible GQA), the run-to-run nondeterminism gate that pins the noise
# floor below the tolerance system, and the dropout suite (fwd+bwd vs the
# element-exact Philox reference, p=0 invariance of the dropout-free path,
# replay determinism, dropout x GQA/sliding-window crosses, API contract).
#
# Env overrides: GPU, LOG, PYTHON, SUITE_DIR
set -euo pipefail

GPU="${GPU:-0}"
LOG="${LOG:-/tmp/arb_correct.log}"
PYTHON="${PYTHON:-python}"
# scripts/ -> flex_flash_attention/ -> tests/, all derived from this script's own
# location so it can be invoked from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="${SUITE_DIR:-$(dirname "$SCRIPT_DIR")/tests}"

export CUDA_VISIBLE_DEVICES="$GPU"
cd "$SUITE_DIR"
"$PYTHON" test_arb_suite.py --suite decomp,kernel "$@" 2>&1 | tee "$LOG"
