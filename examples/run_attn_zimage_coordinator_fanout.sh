#!/usr/bin/env bash
# Run N CPU coordinators in parallel against ONE worker host running N
# attn-zimage worker processes on consecutive ports (PORT_BASE..PORT_BASE+N-1).
#
# Worker side first: examples/run_attn_zimage_workers_fanout.sh
#
# Usage:
#   WORKER_HOST=10.128.0.3 examples/run_attn_zimage_coordinator_fanout.sh
#
# Env knobs:
#   WORKER_HOST  worker address                         (REQUIRED)
#   N            number of parallel coordinators        (default: 5)
#   PORT_BASE    first worker port                      (default: 9102)
#   ROUNDS       forward rounds per stream              (default: 100)
#   WARMUP       rounds excluded from each summary      (default: 10)
#   DIM          model dim                              (default: 1536)
#   HEADS        attention heads                        (default: 12)
#   HEAD_DIM     per-head dimension                     (default: 128)
#   BATCH        batch size                             (default: 2)
#   SEQ          sequence length                        (default: 1024)
#   QK_NORM      rms|none                               (default: rms)
#   ROPE_THETA   complex-cis RoPE base                  (default: 10000.0)
#   WIRE_DTYPE   fp16|fp32 on-wire activation dtype     (default: fp16)
#   THRESHOLD    SLALOM MSE threshold                   (default: auto)
#   LINK_GBPS    nominal link bandwidth (Gbit/s)        (default: unset)
#   PIPELINE     1 to enable --pipeline (must match workers) (default: 0)
#   THREADS      OMP/torch threads per coordinator      (default: 8)
#   OUTDIR       per-stream logs + JSON reports dir     (default: fanout_attn_zimage_<ts>)
#   PYTHON       python interpreter                     (default: python)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${WORKER_HOST:-}" ]; then
  echo "error: set WORKER_HOST to the worker's address, e.g.:" >&2
  echo "       WORKER_HOST=10.128.0.3 $0" >&2
  exit 2
fi

N="${N:-5}"
PORT_BASE="${PORT_BASE:-9102}"
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
THREADS="${THREADS:-8}"
PYTHON="${PYTHON:-python}"
OUTDIR="${OUTDIR:-fanout_attn_zimage_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

echo "[fanout] worker:   ${WORKER_HOST}  ports ${PORT_BASE}..$((PORT_BASE+N-1))"
echo "[fanout] streams:  ${N}   attn-zimage dim=${DIM} heads=${HEADS} head_dim=${HEAD_DIM} qk_norm=${QK_NORM} batch=${BATCH} seq=${SEQ} wire=${WIRE_DTYPE}"
echo "[fanout] rounds:   ${ROUNDS} (warmup ${WARMUP})   threads/stream: ${THREADS}   outdir: ${OUTDIR}"
echo

pids=()
for i in $(seq 0 $((N-1))); do
  port=$((PORT_BASE + i))
  ARGS=(examples/multi_machine_attn_zimage.py
        --role coordinator
        --worker-host "${WORKER_HOST}" --worker-port "${port}"
        --rounds "${ROUNDS}" --warmup "${WARMUP}"
        --dim "${DIM}" --heads "${HEADS}" --head-dim "${HEAD_DIM}"
        --batch "${BATCH}" --seq "${SEQ}"
        --qk-norm "${QK_NORM}" --rope-theta "${ROPE_THETA}"
        --wire-dtype "${WIRE_DTYPE}"
        --json-report "${OUTDIR}/coord_${i}.json")
  [ -n "${THRESHOLD:-}" ] && ARGS+=(--threshold "${THRESHOLD}")
  [ -n "${LINK_GBPS:-}" ] && ARGS+=(--link-gbps "${LINK_GBPS}")
  [ "${PIPELINE}" = "1" ] && ARGS+=(--pipeline)
  OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
    "${PYTHON}" "${ARGS[@]}" > "${OUTDIR}/coord_${i}.log" 2>&1 &
  pids+=($!)
  echo "[fanout] stream ${i} -> ${WORKER_HOST}:${port}   pid ${pids[-1]}   log ${OUTDIR}/coord_${i}.log"
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo
echo "[fanout] all ${N} streams finished (rc=${rc}); per-stream logs + reports in ${OUTDIR}/"
echo

"${PYTHON}" - "${OUTDIR}" "${WARMUP}" <<'PY'
import glob, json, os, sys
d, warmup = sys.argv[1], int(sys.argv[2])
files = sorted(glob.glob(os.path.join(d, "coord_*.json")))
if not files:
    print("(no JSON reports found)"); sys.exit(0)

def _stats(rounds, warmup):
    measured = rounds[warmup:] if warmup else rounds
    if not measured: return None
    e2e = [r["end_to_end_t"] for r in measured]
    gpu = [r["gpu_forward_t"] for r in measured]
    wire = [r["wire_recv_t"] for r in measured]
    verify = [r["cpu_verify_t"] for r in measured]
    passed = sum(1 for r in measured if r.get("ok"))
    e2e_s = sorted(e2e)
    p50 = e2e_s[len(e2e_s)//2]
    p95 = e2e_s[min(len(e2e_s)-1, int(0.95*len(e2e_s)))]
    mean = sum(e2e)/len(e2e)
    return {"n": len(measured), "passed": passed, "p50": p50, "p95": p95,
            "mean": mean, "round_per_s": 1000.0/mean if mean else 0.0,
            "gpu_ms": sum(gpu)/len(gpu), "wire_ms": sum(wire)/len(wire),
            "verify_ms": sum(verify)/len(verify)}

print(f"=== fan-out summary: {len(files)} parallel streams ===\n")
print(f"{'stream':<14}{'round/s':>9}{'p50 ms':>9}{'p95 ms':>9}{'wire ms':>9}{'verify ms':>11}{'passed':>11}")
agg_rps = agg_pass = agg_total = 0.0
for f in files:
    name = os.path.basename(f)
    s = _stats(json.load(open(f))["per_round"], warmup)
    if s is None: print(f"{name:<14}(no measured rounds)"); continue
    agg_rps += s["round_per_s"]
    agg_pass += s["passed"]; agg_total += s["n"]
    print(f"{name:<14}{s['round_per_s']:>9.2f}{s['p50']:>9.1f}{s['p95']:>9.1f}"
          f"{s['wire_ms']:>9.1f}{s['verify_ms']:>11.1f}"
          f"{s['passed']:>5}/{s['n']:<5}")
print(f"\naggregate throughput  ≈ {agg_rps:.2f} round/s  (sum over {len(files)} streams)")
print(f"verification          {int(agg_pass)}/{int(agg_total)} rounds passed across all streams")
PY
