#!/usr/bin/env bash
# Run the multi-machine Llama attention example as the GPU *worker*.
#
# Loads q/k/v/o weights on the GPU (no bias, deterministic seed), then waits
# for one coordinator to connect over TCP, runs the full attention forward
# (RoPE + softmax + attn matmuls + o_proj) per request, and ships only the
# four linear outputs (q,k,v,output) back. Inner attn_out is recomputed on
# the coordinator and never crosses the wire. Holds no SLALOM keys.
#
# Pair this with examples/run_attn_llama_coordinator.sh on the other machine.
#
# Usage:
#   examples/run_attn_llama_worker.sh                  # bind 0.0.0.0:9101, cuda:0
#   PORT=9101 DEVICE=cuda:0 examples/run_attn_llama_worker.sh
#   DEVICE=cpu examples/run_attn_llama_worker.sh       # if CUDA isn't available
#   FAULT=scale_o examples/run_attn_llama_worker.sh    # inject a bad-GPU fault
#   PIPELINE=1 examples/run_attn_llama_worker.sh       # overlap send with compute
#   LOOP=1 examples/run_attn_llama_worker.sh           # keep serving after each session
#
# Env knobs (all optional):
#   BIND_HOST  interface to listen on                              (default: 0.0.0.0)
#   PORT       TCP port                                            (default: 9101)
#   DEVICE     torch device for the forward pass                   (default: cuda:0)
#   FAULT      none|flip_v|scale_o|drop_softmax|drop_rope          (default: none)
#   PIPELINE   1 to enable --pipeline                              (default: 0)
#   LOOP       1 to restart the worker forever                     (default: 0)
#   PYTHON     python interpreter                                  (default: python)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BIND_HOST="${BIND_HOST:-0.0.0.0}"
PORT="${PORT:-9101}"
DEVICE="${DEVICE:-cuda:0}"
FAULT="${FAULT:-none}"
PIPELINE="${PIPELINE:-0}"
LOOP="${LOOP:-0}"
PYTHON="${PYTHON:-python}"

ARGS=(examples/multi_machine_attn_llama.py
      --role worker
      --bind "${BIND_HOST}:${PORT}"
      --device "${DEVICE}"
      --inject-fault "${FAULT}")
[ "${PIPELINE}" = "1" ] && ARGS+=(--pipeline)

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "[worker] repo:    ${REPO_ROOT}"
echo "[worker] listen:  ${BIND_HOST}:${PORT}   (coordinator should use --worker-host ${IP:-<this-host-ip>} --worker-port ${PORT})"
echo "[worker] device:  ${DEVICE}   fault: ${FAULT}   pipeline: ${PIPELINE}"
echo "[worker] cmd:     ${PYTHON} ${ARGS[*]}"
echo

if [ "${LOOP}" = "1" ]; then
  echo "[worker] LOOP=1 — restarting after every session (Ctrl-C to stop)"
  while true; do "${PYTHON}" "${ARGS[@]}"; echo "[worker] session ended; restarting…"; sleep 0.5; done
else
  exec "${PYTHON}" "${ARGS[@]}"
fi
