#!/usr/bin/env bash
# Run the multi-machine Llama attention example as the CPU *coordinator*.
#
# Owns the per-projection SLALOM keys (s/s_tilde for q,k,v,o), reproduces the
# input x on CPU from the shared seed, connects to the GPU worker, drives N
# forward rounds, and verifies every linear output the worker returns. The
# attention math (RoPE, qk^T, softmax, probs@v) is recomputed on this CPU
# from the verified q/k/v before the o-projection check. Runs entirely on CPU.
#
# Start examples/run_attn_llama_worker.sh on the GPU machine first, then run
# this with WORKER_HOST pointing at it.
#
# Usage:
#   WORKER_HOST=192.168.1.21 examples/run_attn_llama_coordinator.sh
#   WORKER_HOST=10.0.0.5 PORT=9101 ROUNDS=200 examples/run_attn_llama_coordinator.sh
#   WORKER_HOST=127.0.0.1 PIPELINE=1 examples/run_attn_llama_coordinator.sh
#
# Env knobs:
#   WORKER_HOST  GPU worker address                 (REQUIRED)
#   PORT         GPU worker TCP port                (default: 9101)
#   ROUNDS       forward rounds to run              (default: 100)
#   WARMUP       rounds excluded from the summary   (default: 10)
#   HIDDEN       model hidden size                  (default: 4096)
#   HEADS        attention heads                    (default: 32)
#   KV_HEADS     key/value heads (GQA)              (default: 32; set lower for GQA)
#   HEAD_DIM     per-head dimension                 (default: 128; must equal HIDDEN/HEADS)
#   BATCH        batch size                         (default: 1)
#   SEQ          sequence length                    (default: 512)
#   ROPE_BASE    RoPE base frequency                (default: 500000.0)
#   WIRE_DTYPE   fp16|fp32 on-wire activation dtype (default: fp16)
#   THRESHOLD    SLALOM MSE threshold (q/k/v)       (default: auto: scales with dtype/dims)
#   JSON_REPORT  path for the per-round JSON report (default: attn_llama_run.json; empty disables)
#   LINK_GBPS    nominal link bandwidth (Gbit/s)    (default: unset; adds ideal-vs-measured + link efficiency)
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

PORT="${PORT:-9101}"
ROUNDS="${ROUNDS:-100}"
WARMUP="${WARMUP:-10}"
HIDDEN="${HIDDEN:-4096}"
HEADS="${HEADS:-32}"
KV_HEADS="${KV_HEADS:-32}"
HEAD_DIM="${HEAD_DIM:-128}"
BATCH="${BATCH:-1}"
SEQ="${SEQ:-512}"
ROPE_BASE="${ROPE_BASE:-500000.0}"
WIRE_DTYPE="${WIRE_DTYPE:-fp16}"
PIPELINE="${PIPELINE:-0}"
PYTHON="${PYTHON:-python}"
JSON_REPORT="${JSON_REPORT-attn_llama_run.json}"

ARGS=(examples/multi_machine_attn_llama.py
      --role coordinator
      --worker-host "${WORKER_HOST}"
      --worker-port "${PORT}"
      --rounds "${ROUNDS}"
      --warmup "${WARMUP}"
      --hidden "${HIDDEN}"
      --heads "${HEADS}"
      --kv-heads "${KV_HEADS}"
      --head-dim "${HEAD_DIM}"
      --batch "${BATCH}"
      --seq "${SEQ}"
      --rope-base "${ROPE_BASE}"
      --wire-dtype "${WIRE_DTYPE}")
[ -n "${THRESHOLD:-}" ] && ARGS+=(--threshold "${THRESHOLD}")
[ -n "${JSON_REPORT}" ] && ARGS+=(--json-report "${JSON_REPORT}")
[ -n "${LINK_GBPS:-}" ] && ARGS+=(--link-gbps "${LINK_GBPS}")
[ "${PIPELINE}" = "1" ] && ARGS+=(--pipeline)

echo "[coord] repo:    ${REPO_ROOT}"
echo "[coord] worker:  ${WORKER_HOST}:${PORT}"
echo "[coord] attn:    Llama hidden=${HIDDEN} heads=${HEADS}/kv=${KV_HEADS} head_dim=${HEAD_DIM} batch=${BATCH} seq=${SEQ} wire=${WIRE_DTYPE}"
echo "[coord] rounds:  ${ROUNDS} (warmup ${WARMUP})   pipeline: ${PIPELINE}   report: ${JSON_REPORT:-<none>}"
echo "[coord] cmd:     ${PYTHON} ${ARGS[*]}"
echo

exec "${PYTHON}" "${ARGS[@]}"
