#!/usr/bin/env bash
# Run the multi-machine FFN example as the GPU *worker* (untrusted side).
#
# Loads the SwiGLU MLP weights on the GPU, then waits for one coordinator to
# connect over TCP, runs the forward pass per request, and ships only the
# linear outputs (y1/y3/y2) back. Holds no SLALOM keys; never sees x.
#
# Pair this with examples/run_cpu_coordinator.sh on the other machine.
#
# Usage:
#   examples/run_gpu_worker.sh                      # bind 0.0.0.0:9100, cuda:0
#   PORT=9100 DEVICE=cuda:0 examples/run_gpu_worker.sh
#   DEVICE=cpu examples/run_gpu_worker.sh           # if CUDA isn't available
#   FAULT=scale_y2 examples/run_gpu_worker.sh       # inject a bad-GPU fault
#   PIPELINE=1 examples/run_gpu_worker.sh           # overlap send with compute
#   LOOP=1 examples/run_gpu_worker.sh               # keep serving after each session
#
# Env knobs (all optional):
#   BIND_HOST  interface to listen on            (default: 0.0.0.0)
#   PORT       TCP port                          (default: 9100)
#   DEVICE     torch device for the forward pass (default: cuda:0)
#   FAULT      none|flip_y1|scale_y2|drop_silu   (default: none)
#   PIPELINE   1 to enable --pipeline            (default: 0)
#   LOOP       1 to restart the worker forever   (default: 0)
#   PYTHON     python interpreter                (default: python)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BIND_HOST="${BIND_HOST:-0.0.0.0}"
PORT="${PORT:-9100}"
DEVICE="${DEVICE:-cuda:0}"
FAULT="${FAULT:-none}"
PIPELINE="${PIPELINE:-0}"
LOOP="${LOOP:-0}"
PYTHON="${PYTHON:-python}"

ARGS=(examples/multi_machine_ffn.py
      --role worker
      --bind "${BIND_HOST}:${PORT}"
      --device "${DEVICE}"
      --inject-fault "${FAULT}")
[ "${PIPELINE}" = "1" ] && ARGS+=(--pipeline)

# Print a reachable address so the coordinator side knows what to point at.
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
