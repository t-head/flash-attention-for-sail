#!/bin/bash
# Full performance + per-cell accuracy sweep for the flex flash attention kernel.
#
#   ./run_perf.sh [extra test_arb_suite.py args...]
#
# 3 seqlens x 3 headdims x 3 head counts x 3 GQA ratios x 10 scenarios =
# 810 cells; every cell times seven columns (sdpa:mem_eff, sdpa:math,
# fa_native, flex_kernel, flex_e2e, ours_kernel, ours_e2e), runs an fp32
# accuracy guard and reports fwd/bwd split + TFLOPS.  The defaults follow
# mainstream LLM configs: seqlens cover the training/prefill range plus a
# long-context point (65536) and one non-aligned real-trace length (25286,
# exercises tail-block handling); headdims are the mainstream 64/128/256
# (the non-bucket round-up padding path is covered by run_correctness.sh);
# head counts 12/16/32/40 cover Gemma/Mistral-class to 70B-class configs;
# GQA ratios 1/2/4 sweep MHA -> light -> Llama-3-style
# (every ratio must divide every head count).  Runtime is overnight or more
# at full scale — run it detached, and trim SCENARIO/SEQ/HDIM for quicker
# passes (e.g. SCENARIO="causal,dense").
#
# Env overrides: GPU, LOG, SEQ, HDIM, BATCH, HEADS, DTYPE, GQA,
#                SCENARIO, SOFTCAP, BACKENDS, FLEX_TMA, PYTHON, SUITE_DIR
set -euo pipefail

GPU="${GPU:-6}"
LOG="${LOG:-./arb_perf_full.log}"
SEQ="${SEQ:-4096,25286}"
HDIM="${HDIM:-64,128,256}"
BATCH="${BATCH:-1}"
HEADS="${HEADS:-12,16,32,40}"
DTYPE="${DTYPE:-bf16}"
GQA="${GQA:-1,2,4}"
# Optional single-value switches: SCENARIO defaults to all 10 (trim with
# e.g. SCENARIO="causal,dense"); BACKENDS defaults to ours+flex (to time
# all groups again: BACKENDS="sdpa,fa,flex,ours").
SCENARIO="${SCENARIO:-dense,causal,sliding_window,sliding_causal,chunked_causal,blockwise,prefix_lm,document,causal_doc,stair}"
# SCENARIO="${SCENARIO:-stair}"
SOFTCAP="${SOFTCAP:-}"
BACKENDS="${BACKENDS:-ours,flex}"
# FLEX_TMA=all|fwd|bwd|off: enable the vendor TMA-descriptor load path in
# torch's flex fwd/bwd kernels (needs the patched torch).  Use a separate
# inductor cache dir per mode so cached compiled kernels never leak across
# A/B runs (nothing is deleted).
FLEX_TMA="${FLEX_TMA:-on}"
if [[ "$FLEX_TMA" != "off" ]]; then
    export FLEX_TMA
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_flex_tma_${FLEX_TMA}}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_flex_tma_${FLEX_TMA}}"
fi
PYTHON="${PYTHON:-python}"
# scripts/ -> flex_flash_attention/ -> tests/, all derived from this script's own
# location so it can be invoked from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="${SUITE_DIR:-$(dirname "$SCRIPT_DIR")/tests}"

export CUDA_VISIBLE_DEVICES="$GPU"
cd "$SUITE_DIR"
OPT=()
# Plain `[[ ... ]] && OPT+=(...)` would abort the script under `set -e` when
# the test is false, so use if-blocks.
if [[ -n "$SCENARIO" ]]; then OPT+=(--scenario "$SCENARIO"); fi
if [[ -n "$SOFTCAP" ]]; then OPT+=(--softcap "$SOFTCAP"); fi
if [[ -n "$BACKENDS" ]]; then OPT+=(--backends "$BACKENDS"); fi
echo "bench start: $(date '+%Y-%m-%dT%H:%M:%S') (FLEX_TMA=$FLEX_TMA)"
"$PYTHON" test_arb_suite.py --suite perf \
    --seq "$SEQ" --hdim "$HDIM" --batch "$BATCH" --heads "$HEADS" \
    --dtype "$DTYPE" --gqa "$GQA" ${OPT[@]+"${OPT[@]}"} \
    --accuracy --split "$@" 2>&1 | tee "$LOG"
echo "bench end:   $(date '+%Y-%m-%dT%H:%M:%S')"
