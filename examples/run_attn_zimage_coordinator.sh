#!/usr/bin/env bash
# Run the multi-machine Z-Image attention example as the CPU *coordinator*.
#
# Owns SLALOM keys for q/k/v/o, reproduces input x deterministically, drives
# N forward rounds, and verifies each linear output. Recomputes per-head
# RMSNorm + complex-cis RoPE + softmax + probs@v on this CPU from verified
# q/k/v before checking the o-projection. Runs entirely on CPU.
#
# Start examples/run_attn_zimage_worker.sh on the GPU machine first.
#
# Usage:
#   WORKER_HOST=192.168.1.21 examples/run_attn_zimage_coordinator.sh
#   WORKER_HOST=10.0.0.5 PORT=9102 ROUNDS=200 examples/run_attn_zimage_coordinator.sh
#
# Env knobs:
#   WORKER_HOST  GPU worker address                 (REQUIRED)
#   PORT         GPU worker TCP port                (default: 9102)
#   ROUNDS       forward rounds to run              (default: 100)
#   WARMUP       rounds excluded from the summary   (default: 10)
#   DIM          model dim                          (default: 1536)
#   HEADS        attention heads                    (default: 12)
#   HEAD_DIM     per-head dimension                 (default: 128; must equal DIM/HEADS)
#   BATCH        batch size                         (default: 2)
#   SEQ          sequence length                    (default: 1024)
#   QK_NORM      rms|none                           (default: rms)
#   ROPE_THETA   complex-cis RoPE base              (default: 10000.0)
#   WIRE_DTYPE   fp16|fp32 on-wire activation dtype (default: fp16)
#   THRESHOLD    SLALOM MSE threshold (q/k/v)       (default: auto)
#   JSON_REPORT  per-round JSON report path         (default: attn_zimage_run.json; empty disables)
#   LINK_GBPS    nominal link bandwidth (Gbit/s)    (default: unset)
#   PIPELINE     1 to enable --pipeline             (default: 0)  -- must match the worker
#   PYTHON       python interpreter                 (default: python)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${WORKER_HOST:-}" ]; then
  echo "error: set WORKER_HOST to the GPU worker's address, e.g.:" >&2
  echo "       WORKER_HOST=192.168.1.21 $0" >&2
  exit 2
fi

PORT="${PORT:-9102}"
ROUNDS="${ROUNDS:-100}"
WARMUP="${WARMUP:-10}"
DIM="${DIM:-1536}"
HEADS="${HEADS:-12}"
HEAD_DIM="${HEAD_DIM:-128}"
BATCH="${BATCH:-2}"
SEQ="${SEQ:-1024}"
QK_NORM="${QK_NORM:-rms}"
ROPE_THETA="${ROPE_THETA:-10000.0}"
WIRE_DTYPE="${WIRE_DTYPE:-fp16}"
PIPELINE="${PIPELINE:-0}"
PYTHON="${PYTHON:-python}"
JSON_REPORT="${JSON_REPORT-attn_zimage_run.json}"

ARGS=(examples/multi_machine_attn_zimage.py
      --role coordinator
      --worker-host "${WORKER_HOST}"
      --worker-port "${PORT}"
      --rounds "${ROUNDS}"
      --warmup "${WARMUP}"
      --dim "${DIM}"
      --heads "${HEADS}"
      --head-dim "${HEAD_DIM}"
      --batch "${BATCH}"
      --seq "${SEQ}"
      --qk-norm "${QK_NORM}"
      --rope-theta "${ROPE_THETA}"
      --wire-dtype "${WIRE_DTYPE}")
[ -n "${THRESHOLD:-}" ] && ARGS+=(--threshold "${THRESHOLD}")
[ -n "${JSON_REPORT}" ] && ARGS+=(--json-report "${JSON_REPORT}")
[ -n "${LINK_GBPS:-}" ] && ARGS+=(--link-gbps "${LINK_GBPS}")
[ "${PIPELINE}" = "1" ] && ARGS+=(--pipeline)

echo "[coord] repo:    ${REPO_ROOT}"
echo "[coord] worker:  ${WORKER_HOST}:${PORT}"
echo "[coord] attn:    Z-Image dim=${DIM} heads=${HEADS} head_dim=${HEAD_DIM} qk_norm=${QK_NORM} batch=${BATCH} seq=${SEQ} wire=${WIRE_DTYPE}"
echo "[coord] rounds:  ${ROUNDS} (warmup ${WARMUP})   pipeline: ${PIPELINE}   report: ${JSON_REPORT:-<none>}"
echo "[coord] cmd:     ${PYTHON} ${ARGS[*]}"
echo

exec "${PYTHON}" "${ARGS[@]}"
