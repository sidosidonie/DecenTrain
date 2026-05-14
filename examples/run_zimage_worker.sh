#!/usr/bin/env bash
# Run the multi-machine mini-Z-Image example as the GPU *worker*.
#
# Loads N blocks of (RMSNorm + zimage-attn + RMSNorm + SwiGLU MLP) on the GPU.
# Waits for one coordinator to connect, runs the full N-block forward per
# request, and streams 7 tensors per block back (q,k,v,o + w1,w3,w2) tagged
# with (block_idx<<4 | op_kind). A background sender thread overlaps wire
# transfer of block b with GPU compute of block b+1.
#
# Pair with examples/run_zimage_coordinator.sh on the other machine.
#
# Usage:
#   examples/run_zimage_worker.sh                                # bind 0.0.0.0:9103, cuda:0
#   FAULT=scale_o FAULT_BLOCK=3 examples/run_zimage_worker.sh    # fault at a specific block
#   NO_STREAM=1 examples/run_zimage_worker.sh                    # send tensors only after compute completes
#   LOOP=1 examples/run_zimage_worker.sh
#
# Env knobs:
#   BIND_HOST    interface to listen on                                    (default: 0.0.0.0)
#   PORT         TCP port                                                  (default: 9103)
#   DEVICE       torch device                                              (default: cuda:0)
#   FAULT        none|flip_v|scale_o|scale_w2|flip_w1|drop_silu            (default: none)
#   FAULT_BLOCK  block index to inject the fault into (0..N_LAYERS-1)      (default: 0)
#   NO_STREAM    1 to disable the streaming sender (sequential send)       (default: 0)
#   LOOP         1 to restart the worker forever                           (default: 0)
#   PYTHON       python interpreter                                        (default: python)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BIND_HOST="${BIND_HOST:-0.0.0.0}"
PORT="${PORT:-9103}"
DEVICE="${DEVICE:-cuda:0}"
FAULT="${FAULT:-none}"
FAULT_BLOCK="${FAULT_BLOCK:-0}"
NO_STREAM="${NO_STREAM:-0}"
LOOP="${LOOP:-0}"
PYTHON="${PYTHON:-python}"

ARGS=(examples/multi_machine_zimage.py
      --role worker
      --bind "${BIND_HOST}:${PORT}"
      --device "${DEVICE}"
      --inject-fault "${FAULT}"
      --fault-block "${FAULT_BLOCK}")
[ "${NO_STREAM}" = "1" ] && ARGS+=(--no-stream)

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "[worker] repo:    ${REPO_ROOT}"
echo "[worker] listen:  ${BIND_HOST}:${PORT}   (coordinator should use --worker-host ${IP:-<this-host-ip>} --worker-port ${PORT})"
echo "[worker] device:  ${DEVICE}   fault: ${FAULT}@b${FAULT_BLOCK}   stream: $([ "${NO_STREAM}" = "1" ] && echo off || echo on)"
echo "[worker] cmd:     ${PYTHON} ${ARGS[*]}"
echo

if [ "${LOOP}" = "1" ]; then
  echo "[worker] LOOP=1 — restarting after every session (Ctrl-C to stop)"
  while true; do "${PYTHON}" "${ARGS[@]}"; echo "[worker] session ended; restarting…"; sleep 0.5; done
else
  exec "${PYTHON}" "${ARGS[@]}"
fi
