#!/usr/bin/env bash
# Run the multi-machine mini-Z-Image example as the CPU *coordinator*.
#
# Owns per-block, per-op SLALOM keys for all 7 linears in each of the N
# transformer blocks (q/k/v/o + w1/w3/w2). Reproduces input x deterministically
# and drives N forward rounds. As tensors stream in from the worker, kicks off
# block-b verify futures that wait on block (b-1)'s verified x_in, recompute
# RMSNorm/RoPE/softmax/silu/residuals on CPU, and SLALOM-check every linear.
#
# Start examples/run_zimage_worker.sh on the GPU machine first.
#
# Usage:
#   WORKER_HOST=192.168.1.21 examples/run_zimage_coordinator.sh
#   WORKER_HOST=10.0.0.5 N_LAYERS=24 ROUNDS=20 examples/run_zimage_coordinator.sh
#
# Env knobs:
#   WORKER_HOST  GPU worker address                 (REQUIRED)
#   PORT         GPU worker TCP port                (default: 9103)
#   ROUNDS       forward rounds to run              (default: 20)
#   WARMUP       rounds excluded from the summary   (default: 2)
#   DIM          model dim                          (default: 1536)
#   HEADS        attention heads                    (default: 12)
#   HEAD_DIM     per-head dimension                 (default: 128)
#   FFN_INTER    FFN intermediate size              (default: 4096)
#   N_LAYERS     transformer blocks                 (default: 12)
#   BATCH        batch size                         (default: 2)
#   SEQ          sequence length                    (default: 256)
#   QK_NORM      rms|none                           (default: rms)
#   ROPE_THETA   complex-cis RoPE base              (default: 10000.0)
#   WIRE_DTYPE   fp16|fp32 on-wire activation dtype (default: fp16)
#   THRESHOLD    SLALOM MSE threshold               (default: auto)
#   JSON_REPORT  per-round JSON report path         (default: zimage_run.json; empty disables)
#   LINK_GBPS    nominal link bandwidth (Gbit/s)    (default: unset)
#   PYTHON       python interpreter                 (default: python)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${WORKER_HOST:-}" ]; then
  echo "error: set WORKER_HOST to the GPU worker's address, e.g.:" >&2
  echo "       WORKER_HOST=192.168.1.21 $0" >&2
  exit 2
fi

PORT="${PORT:-9103}"
ROUNDS="${ROUNDS:-20}"
WARMUP="${WARMUP:-2}"
DIM="${DIM:-1536}"
HEADS="${HEADS:-12}"
HEAD_DIM="${HEAD_DIM:-128}"
FFN_INTER="${FFN_INTER:-4096}"
N_LAYERS="${N_LAYERS:-12}"
BATCH="${BATCH:-2}"
SEQ="${SEQ:-256}"
QK_NORM="${QK_NORM:-rms}"
ROPE_THETA="${ROPE_THETA:-10000.0}"
WIRE_DTYPE="${WIRE_DTYPE:-fp16}"
PYTHON="${PYTHON:-python}"
JSON_REPORT="${JSON_REPORT-zimage_run.json}"

ARGS=(examples/multi_machine_zimage.py
      --role coordinator
      --worker-host "${WORKER_HOST}"
      --worker-port "${PORT}"
      --rounds "${ROUNDS}"
      --warmup "${WARMUP}"
      --dim "${DIM}"
      --heads "${HEADS}"
      --head-dim "${HEAD_DIM}"
      --ffn-inter "${FFN_INTER}"
      --n-layers "${N_LAYERS}"
      --batch "${BATCH}"
      --seq "${SEQ}"
      --qk-norm "${QK_NORM}"
      --rope-theta "${ROPE_THETA}"
      --wire-dtype "${WIRE_DTYPE}")
[ -n "${THRESHOLD:-}" ] && ARGS+=(--threshold "${THRESHOLD}")
[ -n "${JSON_REPORT}" ] && ARGS+=(--json-report "${JSON_REPORT}")
[ -n "${LINK_GBPS:-}" ] && ARGS+=(--link-gbps "${LINK_GBPS}")

echo "[coord] repo:    ${REPO_ROOT}"
echo "[coord] worker:  ${WORKER_HOST}:${PORT}"
echo "[coord] zimage:  dim=${DIM} heads=${HEADS} head_dim=${HEAD_DIM} ffn_inter=${FFN_INTER} n_layers=${N_LAYERS} batch=${BATCH} seq=${SEQ} wire=${WIRE_DTYPE}"
echo "[coord] rounds:  ${ROUNDS} (warmup ${WARMUP})   report: ${JSON_REPORT:-<none>}"
echo "[coord] cmd:     ${PYTHON} ${ARGS[*]}"
echo

exec "${PYTHON}" "${ARGS[@]}"
