#!/usr/bin/env python3
"""
Distributed Verification Performance Analyzer

Reads a YAML system description and outputs detailed performance breakdown
with bottleneck analysis for the DecenTrain distributed verification architecture.

Hardware specs (GPUs, CPUs, networks, models) are defined in hardware/*.yaml
and referenced by name in system configs.

Usage:
  python tools/perf_analyze.py configs/my_system.yaml
  python tools/perf_analyze.py configs/my_system.yaml --json
  python tools/perf_analyze.py --presets

Calibrated against real benchmarks:
  Qwen3.5-9B / RTX 5090 / 18C: verified forward 1950ms (14.3x), decode 0.4 tok/s
  Z-Image / A100 / fp32: verified 1-step 16115ms (2.3x)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Hardware Catalog — loaded from hardware/*.yaml
# ═══════════════════════════════════════════════════════════════════════════════

HARDWARE_DIR = Path(__file__).parent.parent / "hardware"

def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}

def load_hardware_catalog(extra_dirs: Optional[List[str]] = None) -> Dict[str, Dict[str, dict]]:
    """Load all hardware definitions from hardware/ folder and optional extra dirs."""
    catalog: Dict[str, Dict[str, dict]] = {
        "gpus": {}, "cpus": {}, "networks": {}, "models": {}, "quantization": {},
    }
    dirs = [HARDWARE_DIR]
    if extra_dirs:
        dirs.extend(Path(d) for d in extra_dirs)

    for d in dirs:
        for key, fname in [("gpus", "gpus.yaml"), ("cpus", "cpus.yaml"),
                           ("networks", "networks.yaml"), ("models", "models.yaml"),
                           ("quantization", "quantization.yaml")]:
            data = _load_yaml(d / fname)
            catalog[key].update(data)

    return catalog


TEE_OVERHEAD: Dict[str, float] = {
    "none": 0.0, "tdx": 3.0, "sgx": 10.0, "sev-snp": 7.0,
}

# Distribution strategies
STRATEGIES = {"standalone", "pipeline_parallel", "tensor_parallel", "data_parallel"}


# ═══════════════════════════════════════════════════════════════════════════════
# Parsed Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GPUConfig:
    name: str = ""
    bf16_tflops: float = 0
    memory_gb: float = 0
    memory_bw_gbps: float = 0
    nvlink: bool = False
    nvlink_bw_gbps: float = 0

@dataclass
class CPUConfig:
    name: str = ""
    cores: int = 1
    fp32_gflops: float = 0
    memory_bw_gbps: float = 0
    tee: str = "none"

@dataclass
class NetworkConfig:
    name: str = ""
    bandwidth_gbps: float = 0
    latency_us: float = 0

@dataclass
class ModelConfig:
    name: str = ""
    type: str = "llm"
    hidden_size: int = 0
    intermediate_size: int = 0
    num_layers: int = 0
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    num_verified_attn_layers: int = 0
    num_verified_mlp_layers: int = 0
    param_bytes: float = 0
    num_refiner_layers: int = 0
    default_steps: int = 4
    default_image_size: int = 512
    latent_seq_len: int = 1024

    @property
    def total_transformer_layers(self) -> int:
        return self.num_layers + self.num_refiner_layers

    @property
    def linear_checks_per_fwd(self) -> int:
        return self.num_verified_attn_layers * 4 + self.num_verified_mlp_layers * 3

    @property
    def matmul_checks_per_fwd(self) -> int:
        return self.num_verified_attn_layers * 2

    @property
    def total_checks_per_fwd(self) -> int:
        return self.linear_checks_per_fwd + self.matmul_checks_per_fwd + self.num_verified_attn_layers


@dataclass
class QuantConfig:
    """Quantization method and its performance effects."""
    method: str = "none"
    weight_bits: int = 16
    activation_bits: int = 16
    memory_factor: float = 1.0      # VRAM multiplier vs bf16
    compute_factor: float = 1.0     # GPU throughput multiplier (>1 = faster)
    threshold_factor: float = 1.0   # MSE threshold multiplier
    kv_cache_bits: int = 16
    verify_compatible: str = "full"  # full | partial | none
    notes: str = ""

    @property
    def label(self) -> str:
        if self.method == "none":
            return "bf16"
        return self.method


@dataclass
class ClusterConfig:
    """A group of GPUs running one model with a distribution strategy."""
    name: str = ""
    gpu_type: str = ""
    count: int = 1
    strategy: str = "standalone"  # standalone | pipeline_parallel | tensor_parallel | data_parallel
    model: str = ""
    quantization: QuantConfig = field(default_factory=QuantConfig)
    # resolved at parse time
    gpu: GPUConfig = field(default_factory=GPUConfig)
    model_cfg: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class VerifyConfig:
    use_slalom: bool = True
    freivalds_k: int = 10
    verify_every_n: int = 1
    max_workers: int = 0
    compression: str = "fp16"

@dataclass
class InferenceConfig:
    batch_size: int = 1
    seq_len: int = 32
    gen_tokens: int = 50
    num_steps: int = 0
    image_size: int = 0

@dataclass
class CalibrationData:
    """Real measured numbers from calibrate.py, used to adjust analytical model."""
    source_file: str = ""
    source_gpu: str = ""
    source_model: str = ""
    # Measured per-op times
    gpu_decode_ms: float = 0       # real GPU decode per token (LLM)
    gpu_step_ms: float = 0         # real GPU per step (diffusion)
    verified_decode_ms: float = 0
    verified_step_ms: float = 0
    linear_verify_ms: float = 0    # real avg linear verify time
    matmul_verify_ms: float = 0
    transfer_linear_ms: float = 0
    transfer_matmul_ms: float = 0

    @property
    def has_data(self) -> bool:
        return self.gpu_decode_ms > 0 or self.gpu_step_ms > 0


@dataclass
class SystemConfig:
    cpu: CPUConfig = field(default_factory=CPUConfig)
    clusters: List[ClusterConfig] = field(default_factory=list)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    calibrations: List[CalibrationData] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Config Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve(raw: dict, catalog_section: dict) -> dict:
    """Merge catalog definition with inline overrides."""
    ref = raw.get("type", raw.get("preset", raw.get("name", "")))
    base = dict(catalog_section.get(ref, {}))
    for k, v in raw.items():
        if k not in ("type", "preset") and v is not None:
            base[k] = v
    if "name" not in base:
        base["name"] = ref
    return base


def _parse_gpu(d: dict, catalog: Dict[str, dict]) -> GPUConfig:
    r = _resolve(d, catalog.get("gpus", {}))
    return GPUConfig(
        name=r.get("name", ""), bf16_tflops=float(r.get("bf16_tflops", 0)),
        memory_gb=float(r.get("memory_gb", 0)), memory_bw_gbps=float(r.get("memory_bw_gbps", 0)),
        nvlink=bool(r.get("nvlink", False)), nvlink_bw_gbps=float(r.get("nvlink_bw_gbps", 0)),
    )


def _parse_model(d: dict, catalog: Dict[str, dict]) -> ModelConfig:
    ref = d if isinstance(d, str) else d.get("type", d.get("preset", d.get("name", "")))
    if isinstance(ref, str) and ref in catalog.get("models", {}):
        base = dict(catalog["models"][ref])
        if isinstance(d, dict):
            for k, v in d.items():
                if k not in ("type", "preset") and v is not None:
                    base[k] = v
        base.setdefault("name", ref)
        d = base
    elif isinstance(d, str):
        d = {"name": d}

    return ModelConfig(
        name=d.get("name", ""), type=d.get("type", "llm"),
        hidden_size=int(d.get("hidden_size", 0)),
        intermediate_size=int(d.get("intermediate_size", 0)),
        num_layers=int(d.get("num_layers", 0)),
        num_attention_heads=int(d.get("num_attention_heads", 0)),
        num_kv_heads=int(d.get("num_kv_heads", 0)),
        head_dim=int(d.get("head_dim", 0)),
        num_verified_attn_layers=int(d.get("num_verified_attn_layers", d.get("num_layers", 0))),
        num_verified_mlp_layers=int(d.get("num_verified_mlp_layers", d.get("num_layers", 0))),
        param_bytes=float(d.get("param_bytes", 0)),
        num_refiner_layers=int(d.get("num_refiner_layers", 0)),
        default_steps=int(d.get("default_steps", 4)),
        default_image_size=int(d.get("default_image_size", 512)),
        latent_seq_len=int(d.get("latent_seq_len", 1024)),
    )


def _parse_quant(raw, catalog: Dict[str, dict]) -> QuantConfig:
    """Parse quantization config — string name or inline dict."""
    if raw is None or raw == "none":
        return QuantConfig()
    if isinstance(raw, str):
        qd = catalog.get("quantization", {}).get(raw, {})
        if not qd:
            print(f"  Warning: unknown quantization method '{raw}'", file=sys.stderr)
            return QuantConfig(method=raw)
        qd["method"] = raw
    else:
        ref = raw.get("method", raw.get("type", "none"))
        qd = dict(catalog.get("quantization", {}).get(ref, {}))
        qd.update({k: v for k, v in raw.items() if k not in ("type",) and v is not None})
        qd.setdefault("method", ref)

    return QuantConfig(
        method=qd.get("method", "none"),
        weight_bits=int(qd.get("weight_bits", 16)),
        activation_bits=int(qd.get("activation_bits", 16)),
        memory_factor=float(qd.get("memory_factor", 1.0)),
        compute_factor=float(qd.get("compute_factor", 1.0)),
        threshold_factor=float(qd.get("threshold_factor", 1.0)),
        kv_cache_bits=int(qd.get("kv_cache_bits", 16)),
        verify_compatible=qd.get("verify_compatible", "full"),
        notes=qd.get("notes", ""),
    )


def parse_config(path: str) -> SystemConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    # Load hardware catalog from default dir + any extra hardware_dirs
    extra = raw.get("hardware_dirs", [])
    if isinstance(extra, str):
        extra = [extra]
    catalog = load_hardware_catalog(extra)

    cfg = SystemConfig()

    # CPU
    cpu_raw = raw.get("cpu", {})
    cr = _resolve(cpu_raw, catalog.get("cpus", {}))
    cfg.cpu = CPUConfig(
        name=cr.get("name", ""), cores=int(cr.get("cores", 1)),
        fp32_gflops=float(cr.get("fp32_gflops", 0)),
        memory_bw_gbps=float(cr.get("memory_bw_gbps", 0)),
        tee=str(cpu_raw.get("tee", cr.get("tee", "none"))),
    )

    # Network
    net_raw = raw.get("network", {})
    nr = _resolve(net_raw, catalog.get("networks", {}))
    cfg.network = NetworkConfig(
        name=nr.get("name", ""), bandwidth_gbps=float(nr.get("bandwidth_gbps", 0)),
        latency_us=float(nr.get("latency_us", 0)),
    )

    # Clusters — new format
    if "clusters" in raw:
        for cl in raw["clusters"]:
            gpu_ref = cl.get("gpu_type", cl.get("gpu", ""))
            gpu_d = {"type": gpu_ref} if isinstance(gpu_ref, str) else gpu_ref
            gpu = _parse_gpu(gpu_d, catalog)

            model_ref = cl.get("model", "")
            model_d = {"type": model_ref} if isinstance(model_ref, str) else model_ref
            model = _parse_model(model_d, catalog)

            strategy = cl.get("strategy", "standalone")
            count = int(cl.get("count", 1))
            if count > 1 and strategy == "standalone":
                strategy = "pipeline_parallel"

            quant = _parse_quant(cl.get("quantization", "none"), catalog)

            cfg.clusters.append(ClusterConfig(
                name=cl.get("name", f"{count}x{gpu.name}:{model.name}"),
                gpu_type=gpu.name, count=count, strategy=strategy,
                model=model.name, quantization=quant, gpu=gpu, model_cfg=model,
            ))

    # Legacy format: gpus + models lists -> auto-generate clusters
    elif "gpus" in raw:
        gpus_raw = raw.get("gpus", [])
        models_raw = raw.get("models", [])
        for i, g in enumerate(gpus_raw):
            gpu = _parse_gpu(g, catalog)
            count = int(g.get("count", 1))
            model_ref = models_raw[i] if i < len(models_raw) else {}
            model = _parse_model(model_ref, catalog)
            strategy = "pipeline_parallel" if count > 1 else "standalone"
            cfg.clusters.append(ClusterConfig(
                name=f"{count}x{gpu.name}:{model.name}",
                gpu_type=gpu.name, count=count, strategy=strategy,
                model=model.name, gpu=gpu, model_cfg=model,
            ))

    # Verify
    vr = raw.get("verify", {})
    cfg.verify = VerifyConfig(
        use_slalom=vr.get("use_slalom", True), freivalds_k=int(vr.get("freivalds_k", 10)),
        verify_every_n=int(vr.get("verify_every_n", 1)),
        max_workers=int(vr.get("max_workers", 0)), compression=vr.get("compression", "fp16"),
    )

    # Inference
    ir = raw.get("inference", {})
    cfg.inference = InferenceConfig(
        batch_size=int(ir.get("batch_size", 1)), seq_len=int(ir.get("seq_len", 32)),
        gen_tokens=int(ir.get("gen_tokens", 50)),
        num_steps=int(ir.get("num_steps", 0)), image_size=int(ir.get("image_size", 0)),
    )

    # Calibration files
    cal_refs = raw.get("calibration", [])
    if isinstance(cal_refs, str):
        cal_refs = [cal_refs]
    for cal_path in cal_refs:
        cal = _load_calibration(cal_path, Path(path).parent)
        if cal:
            cfg.calibrations.append(cal)

    return cfg


def _load_calibration(cal_path: str, config_dir: Path) -> Optional[CalibrationData]:
    """Load a calibration YAML file."""
    p = Path(cal_path)
    if not p.is_absolute():
        # Try relative to config dir, then project root
        for base in [config_dir, Path.cwd()]:
            candidate = base / p
            if candidate.exists():
                p = candidate
                break
    if not p.exists():
        print(f"  Warning: calibration file not found: {cal_path}", file=sys.stderr)
        return None

    with open(p) as f:
        raw = yaml.safe_load(f)

    cal_data = raw.get("calibration", {})
    factors = cal_data.get("factors", {})
    machine = cal_data.get("machine", {})
    bench = cal_data.get("benchmark", {})

    return CalibrationData(
        source_file=str(p),
        source_gpu=machine.get("gpu", ""),
        source_model=bench.get("model", ""),
        gpu_decode_ms=float(factors.get("gpu_decode_ms", 0)),
        gpu_step_ms=float(factors.get("gpu_step_ms", 0)),
        verified_decode_ms=float(factors.get("verified_decode_ms", 0)),
        verified_step_ms=float(factors.get("verified_step_ms", 0)),
        linear_verify_ms=float(factors.get("linear_verify_ms", 0)),
        matmul_verify_ms=float(factors.get("matmul_verify_ms", 0)),
        transfer_linear_ms=float(factors.get("transfer_linear_ms", 0)),
        transfer_matmul_ms=float(factors.get("transfer_matmul_ms", 0)),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Analytical Cost Model (calibrated)
# ═══════════════════════════════════════════════════════════════════════════════

FREIVALDS_EFFECTIVE_BW = 3.4e9
SLALOM_EFFECTIVE_BW = 40e9
GPU_UTIL_DECODE = 0.35
GPU_UTIL_PREFILL = 0.55


def _cr(compression: str) -> float:
    return 0.5 if compression == "fp16" else 1.0


def _gpu_forward_ms(model: ModelConfig, gpu: GPUConfig, batch: int, seq_len: int,
                    num_gpus: int, strategy: str) -> float:
    H, I = model.hidden_size, model.intermediate_size
    nh, nkv, hd = model.num_attention_heads, model.num_kv_heads, model.head_dim
    S = seq_len

    attn_flops = 2 * batch * S * H * (H + 2 * nkv * hd + H) + 4 * batch * nh * S * S * hd
    mlp_flops = 6 * batch * S * H * I
    total_flops = model.num_layers * (attn_flops + mlp_flops)

    weight_bytes = model.param_bytes / num_gpus
    mem_ms = weight_bytes / (gpu.memory_bw_gbps * 1e9) * 1000
    util = GPU_UTIL_DECODE if S == 1 else GPU_UTIL_PREFILL
    compute_ms = (total_flops / num_gpus) / (gpu.bf16_tflops * 1e12 * util) * 1000

    base = max(compute_ms, mem_ms)

    if num_gpus > 1:
        if strategy == "pipeline_parallel":
            base *= 1.15  # pipeline bubble ~15%
        elif strategy == "tensor_parallel":
            # all-reduce per layer: 2 * (num_gpus-1)/num_gpus * hidden * 2bytes
            # use NVLink if available, else network/PCIe
            ar_bytes = 2 * (num_gpus - 1) / num_gpus * H * 2  # ring all-reduce
            layers = model.num_layers
            if gpu.nvlink and gpu.nvlink_bw_gbps > 0:
                ar_ms = layers * ar_bytes / (gpu.nvlink_bw_gbps * 1e9) * 1000
            else:
                ar_ms = layers * ar_bytes / (gpu.memory_bw_gbps * 0.3 * 1e9) * 1000  # PCIe fallback
            base += ar_ms
        elif strategy == "data_parallel":
            base *= 1.0  # each GPU runs full model independently

    return base


def _transfer_bytes(model: ModelConfig, batch: int, seq_len: int, compression: str) -> float:
    H, I = model.hidden_size, model.intermediate_size
    nh, nkv, hd = model.num_attention_heads, model.num_kv_heads, model.head_dim
    S = seq_len
    cr = _cr(compression)
    attn_numel = batch * S * H * 3 + batch * S * nkv * hd * 2 + batch * nh * S * S + batch * nh * S * hd + batch * S * H
    mlp_numel = batch * S * (H + I + I + H)
    return (model.num_verified_attn_layers * attn_numel + model.num_verified_mlp_layers * mlp_numel) * 4 * cr


def _linear_verify_ms(h_in: int, h_out: int, batch: int, seq: int, k: int,
                      use_slalom: bool, cpu: CPUConfig) -> float:
    bw_scale = min(cpu.memory_bw_gbps / 90, 3.0)
    if use_slalom:
        data = (batch * seq * h_out + h_out * k + batch * seq * h_in + h_in * k) * 4
        return (data / (SLALOM_EFFECTIVE_BW * bw_scale)) * 1000
    else:
        data = h_in * h_out * 4 + (batch * seq * (h_in + h_out) * k * 4)
        return (data / (FREIVALDS_EFFECTIVE_BW * bw_scale)) * 1000


def _matmul_verify_ms(m: int, n: int, p: int, batch_heads: int, k: int, cpu: CPUConfig) -> float:
    bw_scale = min(cpu.memory_bw_gbps / 90, 3.0)
    data = (batch_heads * (m * n + n * p + m * p) + p * k) * 4
    return (data / (SLALOM_EFFECTIVE_BW * bw_scale)) * 1000


@dataclass
class VerifyCost:
    linear_seq_ms: float = 0
    matmul_seq_ms: float = 0
    elementwise_seq_ms: float = 0
    total_seq_ms: float = 0
    total_parallel_ms: float = 0
    transfer_bytes: float = 0
    n_linear: int = 0
    n_matmul: int = 0
    n_elementwise: int = 0


def _verify_cost(model: ModelConfig, cpu: CPUConfig, batch: int, seq: int,
                 vp: VerifyConfig, tee_pct: float) -> VerifyCost:
    vc = VerifyCost()
    H, I = model.hidden_size, model.intermediate_size
    nh, nkv, hd = model.num_attention_heads, model.num_kv_heads, model.head_dim
    k = vp.freivalds_k

    attn_linear = (
        _linear_verify_ms(H, H, batch, seq, k, vp.use_slalom, cpu)
        + _linear_verify_ms(H, nkv * hd, batch, seq, k, vp.use_slalom, cpu) * 2
        + _linear_verify_ms(H, H, batch, seq, k, vp.use_slalom, cpu)
    )
    mlp_linear = (
        _linear_verify_ms(H, I, batch, seq, k, vp.use_slalom, cpu) * 2
        + _linear_verify_ms(I, H, batch, seq, k, vp.use_slalom, cpu)
    )
    vc.linear_seq_ms = (
        model.num_verified_attn_layers * attn_linear
        + model.num_verified_mlp_layers * mlp_linear
    )
    vc.n_linear = model.linear_checks_per_fwd

    qk_ms = _matmul_verify_ms(seq, hd, seq, batch * nh, k, cpu)
    pv_ms = _matmul_verify_ms(seq, seq, hd, batch * nh, k, cpu)
    vc.matmul_seq_ms = model.num_verified_attn_layers * (qk_ms + pv_ms)
    vc.n_matmul = model.matmul_checks_per_fwd

    soft_numel = batch * nh * seq * seq
    bw_scale = min(cpu.memory_bw_gbps / 90, 3.0)
    vc.elementwise_seq_ms = model.num_verified_attn_layers * (soft_numel * 8 / (SLALOM_EFFECTIVE_BW * bw_scale)) * 1000
    vc.n_elementwise = model.num_verified_attn_layers

    vc.linear_seq_ms /= vp.verify_every_n
    vc.matmul_seq_ms /= vp.verify_every_n
    vc.elementwise_seq_ms /= vp.verify_every_n

    vc.total_seq_ms = (vc.linear_seq_ms + vc.matmul_seq_ms + vc.elementwise_seq_ms) * (1 + tee_pct / 100)

    workers = vp.max_workers if vp.max_workers > 0 else cpu.cores
    max_par = min(workers, cpu.cores)
    if vp.use_slalom:
        eff = min(max_par, cpu.cores * 0.7)
    else:
        eff = min(max_par, max(1, math.sqrt(cpu.memory_bw_gbps / 10)))
    vc.total_parallel_ms = vc.total_seq_ms / max(1, eff)
    vc.transfer_bytes = _transfer_bytes(model, batch, seq, vp.compression) / vp.verify_every_n
    return vc


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhaseResult:
    name: str = ""
    gpu_ms: float = 0
    transfer_ms: float = 0
    verify_ms: float = 0
    total_ms: float = 0
    bottleneck: str = ""
    tok_per_sec: float = 0
    iterations: int = 0

@dataclass
class ClusterResult:
    cluster_name: str = ""
    model_name: str = ""
    model_type: str = ""
    gpu_desc: str = ""
    strategy: str = ""
    num_gpus: int = 0
    oom: bool = False
    oom_detail: str = ""
    phases: List[PhaseResult] = field(default_factory=list)
    total_latency_ms: float = 0
    throughput_label: str = ""
    throughput_value: float = 0
    transfer_mb_per_fwd: float = 0
    overhead_vs_origin: float = 0
    verify_seq_ms: float = 0
    verify_parallel_ms: float = 0
    n_checks_per_fwd: int = 0
    calibrated: bool = False
    calibration_source: str = ""

@dataclass
class AnalysisResult:
    system_summary: str = ""
    cluster_results: List[ClusterResult] = field(default_factory=list)
    aggregate_cpu_load_ms: float = 0
    aggregate_network_mbps: float = 0
    warnings: List[str] = field(default_factory=list)
    # aliases for draw_topology compat
    @property
    def model_results(self):
        return self.cluster_results


def _bottleneck(gpu_ms: float, xfer_ms: float, verify_ms: float) -> str:
    stages = {"GPU compute": gpu_ms, "Network": xfer_ms, "CPU verify": verify_ms}
    return max(stages, key=stages.get)


def _effective_gpus_for_verify(cluster: ClusterConfig) -> int:
    """How many 'equivalent GPUs' feed verification data."""
    if cluster.strategy == "data_parallel":
        return 1  # each GPU runs independently; verify one at a time
    return cluster.count


def _find_calibration(calibrations: List[CalibrationData], gpu_name: str, model_name: str) -> Optional[CalibrationData]:
    """Find best matching calibration for this GPU+model combo."""
    for cal in calibrations:
        if cal.source_gpu and gpu_name and cal.source_gpu.lower() in gpu_name.lower():
            return cal
    for cal in calibrations:
        if cal.has_data:
            return cal
    return None


def analyze_cluster(cluster: ClusterConfig, cpu: CPUConfig, net: NetworkConfig,
                    vp: VerifyConfig, inf: InferenceConfig,
                    calibrations: Optional[List[CalibrationData]] = None) -> ClusterResult:
    model = cluster.model_cfg
    gpu = cluster.gpu
    ng = cluster.count
    strategy = cluster.strategy

    cr = ClusterResult(
        cluster_name=cluster.name, model_name=model.name, model_type=model.type,
        gpu_desc=f"{ng}x {gpu.name}", strategy=strategy, num_gpus=ng,
    )

    # Quantization effects
    quant = cluster.quantization
    quant_mem = quant.memory_factor  # VRAM reduction
    quant_compute = quant.compute_factor  # throughput boost

    if quant.verify_compatible == "none":
        cr.oom = True
        cr.oom_detail = f"Quantization '{quant.method}' is not compatible with Freivalds verification"
        return cr

    # Data parallel: each GPU has full model
    mem_per_gpu = model.param_bytes * quant_mem / (ng if strategy != "data_parallel" else 1)
    if mem_per_gpu > gpu.memory_gb * 1e9 * 0.85:
        cr.oom = True
        cr.oom_detail = f"{mem_per_gpu/1e9:.1f}GB > {gpu.memory_gb*0.85:.1f}GB per GPU"
        return cr

    tee_pct = TEE_OVERHEAD.get(cpu.tee, 0)
    net_bps = net.bandwidth_gbps * 1e9 / 8
    total_checks = model.total_checks_per_fwd

    # Calibration: find matching real measurements to scale analytical estimates
    cal = _find_calibration(calibrations or [], gpu.name, model.name)
    gpu_cal_factor = 1.0  # multiplier for GPU forward time
    verify_cal_factor = 1.0  # multiplier for CPU verify time
    if cal and cal.has_data:
        cr.calibrated = True
        cr.calibration_source = cal.source_file
        # Compute correction factors by comparing analytical vs measured
        if model.type == "llm" and cal.gpu_decode_ms > 0:
            analytical_decode = _gpu_forward_ms(model, gpu, 1, 1, 1, "standalone")
            if analytical_decode > 0:
                gpu_cal_factor = cal.gpu_decode_ms / analytical_decode
        elif model.type == "diffusion" and cal.gpu_step_ms > 0:
            analytical_step = _gpu_forward_ms(model, gpu, 1, model.latent_seq_len, 1, "standalone")
            if analytical_step > 0:
                gpu_cal_factor = cal.gpu_step_ms / analytical_step
        if cal.linear_verify_ms > 0:
            H, I = model.hidden_size, model.intermediate_size
            analytical_linear = _linear_verify_ms(H, I, 1, 1, vp.freivalds_k, vp.use_slalom, cpu)
            if analytical_linear > 0:
                verify_cal_factor = cal.linear_verify_ms / analytical_linear

    # Quantization affects GPU forward: less memory to read (memory_factor) + faster kernels (compute_factor)
    # For memory-bound decode: time scales with memory_factor / compute_factor
    # For compute-bound prefill: time scales with 1 / compute_factor
    quant_gpu_factor = quant_mem / quant_compute  # combined effect on GPU time

    # For data_parallel, GPU count accelerates batch throughput not single-request latency
    compute_gpus = ng if strategy != "data_parallel" else 1
    # For tensor_parallel, all GPUs send verify data (each sends its shard)
    verify_gpus = _effective_gpus_for_verify(cluster)

    if model.type == "llm":
        batch, seq = inf.batch_size, inf.seq_len

        # Prefill
        gpu_ms = _gpu_forward_ms(model, gpu, batch, seq, compute_gpus, strategy) * gpu_cal_factor * quant_gpu_factor
        vc = _verify_cost(model, cpu, batch, seq, vp, tee_pct)
        xfer_per_gpu = vc.transfer_bytes / max(1, verify_gpus)
        xfer_ms = (xfer_per_gpu / net_bps) * 1000 + net.latency_us / 1000 * total_checks / vp.verify_every_n
        total = max(gpu_ms, xfer_ms, vc.total_parallel_ms)
        cr.phases.append(PhaseResult(
            name="prefill", gpu_ms=gpu_ms, transfer_ms=xfer_ms,
            verify_ms=vc.total_parallel_ms, total_ms=total,
            bottleneck=_bottleneck(gpu_ms, xfer_ms, vc.total_parallel_ms),
            tok_per_sec=seq / (total / 1000) if total > 0 else 0,
        ))
        cr.transfer_mb_per_fwd = vc.transfer_bytes / 1e6
        cr.verify_seq_ms = vc.total_seq_ms
        cr.verify_parallel_ms = vc.total_parallel_ms
        cr.n_checks_per_fwd = total_checks

        # Decode
        gpu_ms_d = _gpu_forward_ms(model, gpu, batch, 1, compute_gpus, strategy) * gpu_cal_factor * quant_gpu_factor
        vc_d = _verify_cost(model, cpu, batch, 1, vp, tee_pct)
        xfer_d = vc_d.transfer_bytes / max(1, verify_gpus)
        xfer_ms_d = (xfer_d / net_bps) * 1000 + net.latency_us / 1000 * total_checks / vp.verify_every_n
        total_d = max(gpu_ms_d, xfer_ms_d, vc_d.total_parallel_ms)
        cr.phases.append(PhaseResult(
            name="decode (per token)", gpu_ms=gpu_ms_d, transfer_ms=xfer_ms_d,
            verify_ms=vc_d.total_parallel_ms, total_ms=total_d,
            bottleneck=_bottleneck(gpu_ms_d, xfer_ms_d, vc_d.total_parallel_ms),
            tok_per_sec=1000 / total_d if total_d > 0 else 0,
            iterations=inf.gen_tokens,
        ))

        cr.total_latency_ms = cr.phases[0].total_ms + inf.gen_tokens * total_d
        cr.throughput_label = "tok/s (decode)"
        cr.throughput_value = cr.phases[1].tok_per_sec
        if strategy == "data_parallel":
            cr.throughput_label = f"tok/s (decode, {ng}x DP)"
            cr.throughput_value *= ng  # N independent streams

        origin = _gpu_forward_ms(model, gpu, batch, 1, 1, "standalone") * gpu_cal_factor * quant_gpu_factor
        cr.overhead_vs_origin = total_d / origin if origin > 0 else 0

    elif model.type == "diffusion":
        steps = inf.num_steps if inf.num_steps > 0 else model.default_steps
        seq = model.latent_seq_len
        batch = inf.batch_size

        gpu_ms = _gpu_forward_ms(model, gpu, batch, seq, compute_gpus, strategy) * gpu_cal_factor * quant_gpu_factor
        vc = _verify_cost(model, cpu, batch, seq, vp, tee_pct)
        xfer_per_gpu = vc.transfer_bytes / max(1, verify_gpus)
        xfer_ms = (xfer_per_gpu / net_bps) * 1000 + net.latency_us / 1000 * total_checks / vp.verify_every_n
        total = max(gpu_ms, xfer_ms, vc.total_parallel_ms)

        cr.phases.append(PhaseResult(
            name="per denoising step", gpu_ms=gpu_ms, transfer_ms=xfer_ms,
            verify_ms=vc.total_parallel_ms, total_ms=total,
            bottleneck=_bottleneck(gpu_ms, xfer_ms, vc.total_parallel_ms),
            iterations=steps,
        ))
        cr.transfer_mb_per_fwd = vc.transfer_bytes / 1e6
        cr.verify_seq_ms = vc.total_seq_ms
        cr.verify_parallel_ms = vc.total_parallel_ms
        cr.n_checks_per_fwd = total_checks

        cr.total_latency_ms = steps * total
        cr.throughput_label = f"img/s ({steps} steps)"
        cr.throughput_value = 1000 / cr.total_latency_ms if cr.total_latency_ms > 0 else 0
        if strategy == "data_parallel":
            cr.throughput_label = f"img/s ({steps}st, {ng}x DP)"
            cr.throughput_value *= ng

        origin = _gpu_forward_ms(model, gpu, batch, seq, 1, "standalone") * gpu_cal_factor * quant_gpu_factor
        cr.overhead_vs_origin = total / origin if origin > 0 else 0

    return cr


def run_analysis(cfg: SystemConfig) -> AnalysisResult:
    result = AnalysisResult()

    gpu_desc = ", ".join(f"{c.count}x{c.gpu.name}" for c in cfg.clusters)
    result.system_summary = (
        f"CPU: {cfg.cpu.name} ({cfg.cpu.cores}C, TEE={cfg.cpu.tee}) | "
        f"GPU: {gpu_desc} | "
        f"Network: {cfg.network.name} ({cfg.network.bandwidth_gbps} Gbps)"
    )

    for cluster in cfg.clusters:
        vp = cfg.verify
        if vp.max_workers == 0:
            vp = VerifyConfig(
                use_slalom=vp.use_slalom, freivalds_k=vp.freivalds_k,
                verify_every_n=vp.verify_every_n, max_workers=cfg.cpu.cores,
                compression=vp.compression,
            )
        cr = analyze_cluster(cluster, cfg.cpu, cfg.network, vp, cfg.inference, cfg.calibrations)
        result.cluster_results.append(cr)

    result.aggregate_cpu_load_ms = sum(cr.verify_seq_ms for cr in result.cluster_results if not cr.oom)
    result.aggregate_network_mbps = sum(cr.transfer_mb_per_fwd for cr in result.cluster_results if not cr.oom)

    for cr in result.cluster_results:
        if cr.oom:
            result.warnings.append(f"{cr.cluster_name}: OOM — {cr.oom_detail}")
    total_gpus = sum(c.count for c in cfg.clusters)
    if total_gpus > 1 and cfg.network.bandwidth_gbps < 10:
        result.warnings.append("Low bandwidth (<10 Gbps) with multiple GPUs — network will likely bottleneck")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════════

def _ms(v: float) -> str:
    if v >= 1000: return f"{v/1000:.2f}s"
    if v >= 1: return f"{v:.1f}ms"
    if v >= 0.001: return f"{v*1000:.1f}us"
    return f"{v*1e6:.1f}ns"

STRATEGY_LABELS = {
    "standalone": "Standalone",
    "pipeline_parallel": "Pipeline Parallel",
    "tensor_parallel": "Tensor Parallel",
    "data_parallel": "Data Parallel",
}


def print_text(result: AnalysisResult, cfg: SystemConfig) -> None:
    W = 120
    print(f"\n{'='*W}")
    print(f"  DecenTrain Distributed Verification — Performance Analysis")
    print(f"{'='*W}")
    print(f"  {result.system_summary}")
    vm = "SLALOM (preprocessed)" if cfg.verify.use_slalom else "Standard Freivalds"
    print(f"  Verify: {vm} k={cfg.verify.freivalds_k} every_n={cfg.verify.verify_every_n} compression={cfg.verify.compression}")

    if result.warnings:
        print(f"\n  WARNINGS:")
        for w in result.warnings:
            print(f"    ! {w}")

    for cr in result.cluster_results:
        model = cr.model_name
        strat = STRATEGY_LABELS.get(cr.strategy, cr.strategy)

        cluster_cfg = next((c for c in cfg.clusters if c.name == cr.cluster_name), None)
        quant = cluster_cfg.quantization if cluster_cfg else QuantConfig()
        quant_label = f" | Quant: {quant.label}" if quant.method != "none" else ""

        print(f"\n{'─'*W}")
        print(f"  CLUSTER: {cr.cluster_name}")
        print(f"  Model: {model} ({cr.model_type}) | GPU: {cr.gpu_desc} | Strategy: {strat}{quant_label}")
        print(f"{'─'*W}")

        if cr.calibrated:
            print(f"  [CALIBRATED from: {Path(cr.calibration_source).name}]")

        if quant.method != "none":
            compat_icon = {"full": "OK", "partial": "PARTIAL", "none": "INCOMPATIBLE"}[quant.verify_compatible]
            print(f"  Quantization: {quant.method} (W{quant.weight_bits}A{quant.activation_bits}) "
                  f"VRAM {quant.memory_factor:.0%} | Compute {quant.compute_factor:.1f}x | "
                  f"Verify: {compat_icon}")
            if quant.verify_compatible == "partial":
                print(f"    Note: {quant.notes}")

        if cr.oom:
            print(f"  *** OUT OF MEMORY: {cr.oom_detail} ***")
            continue

        mc = next((c.model_cfg for c in cfg.clusters if c.name == cr.cluster_name), None)
        if mc:
            param_effective = mc.param_bytes * quant.memory_factor / 1e9
            print(f"  Arch: H={mc.hidden_size} I={mc.intermediate_size} L={mc.num_layers} "
                  f"heads={mc.num_attention_heads} kv={mc.num_kv_heads} hd={mc.head_dim}")
            print(f"  Params: {mc.param_bytes/1e9:.1f}GB -> {param_effective:.1f}GB (quantized) | "
                  f"Verified: {mc.num_verified_attn_layers} attn + {mc.num_verified_mlp_layers} MLP")
        print(f"  Checks/fwd: {cr.n_checks_per_fwd}")

        print(f"\n  CPU Verification (TEE={cfg.cpu.tee}):")
        print(f"    Sequential: {_ms(cr.verify_seq_ms)} | Parallel: {_ms(cr.verify_parallel_ms)}")

        print(f"\n  Phase Breakdown:")
        print(f"  {'Phase':<25} {'GPU':>10} {'Network':>10} {'Verify':>10} {'Total':>10} {'Bottleneck':<16} {'Throughput':<20}")
        print(f"  {'-'*105}")

        for p in cr.phases:
            tp = ""
            if p.tok_per_sec > 0:
                tp = f"{p.tok_per_sec:.1f} tok/s"
            elif p.iterations > 0 and cr.model_type == "diffusion":
                tp = f"{p.iterations} steps"
            print(f"  {p.name:<25} {_ms(p.gpu_ms):>10} {_ms(p.transfer_ms):>10} "
                  f"{_ms(p.verify_ms):>10} {_ms(p.total_ms):>10} {p.bottleneck:<16} {tp:<20}")

        print(f"\n  Summary:")
        print(f"    Total latency: {_ms(cr.total_latency_ms)}")
        print(f"    Throughput: {cr.throughput_value:.1f} {cr.throughput_label}")
        print(f"    Transfer/fwd: {cr.transfer_mb_per_fwd:.1f} MB | Overhead vs origin: {cr.overhead_vs_origin:.1f}x")

        print(f"\n  Bottleneck Analysis:")
        primary = max(
            {"GPU compute": cr.phases[-1].gpu_ms, "Network": cr.phases[-1].transfer_ms,
             "CPU verify": cr.phases[-1].verify_ms},
            key=lambda k: {"GPU compute": cr.phases[-1].gpu_ms, "Network": cr.phases[-1].transfer_ms,
                           "CPU verify": cr.phases[-1].verify_ms}[k],
        )
        print(f"    Primary: {primary}")
        if primary == "Network":
            dp = cr.phases[-1]
            needed = cr.transfer_mb_per_fwd * 1e6 / (dp.gpu_ms / 1000) if dp.gpu_ms > 0 else 0
            print(f"    Need ~{needed * 8 / 1e9:.0f} Gbps to become GPU-bound")
        elif primary == "GPU compute":
            print(f"    GPU-bound — verification adds minimal overhead")
        elif primary == "CPU verify" and not cfg.verify.use_slalom:
            print(f"    Switch to SLALOM for ~100-200x speedup")

    # Aggregate
    active = [cr for cr in result.cluster_results if not cr.oom]
    if len(active) > 1:
        print(f"\n{'─'*W}")
        print(f"  AGGREGATE: {len(active)} clusters -> 1 CPU TEE")
        print(f"{'─'*W}")
        workers = cfg.verify.max_workers or cfg.cpu.cores
        eff = min(workers, cfg.cpu.cores * 0.7)
        print(f"  CPU verify (seq): {_ms(result.aggregate_cpu_load_ms)} | "
              f"(parallel, {eff:.0f}w): {_ms(result.aggregate_cpu_load_ms / max(1, eff))}")
        print(f"  Total network: {result.aggregate_network_mbps:.1f} MB/fwd")

    print(f"\n{'='*W}")


def print_json(result: AnalysisResult) -> None:
    out: Dict[str, Any] = {"system": result.system_summary, "warnings": result.warnings, "clusters": []}
    for cr in result.cluster_results:
        cd: Dict[str, Any] = {
            "name": cr.cluster_name, "model": cr.model_name, "type": cr.model_type,
            "gpu": cr.gpu_desc, "strategy": cr.strategy, "oom": cr.oom,
        }
        if cr.oom:
            cd["oom_detail"] = cr.oom_detail
        else:
            cd["phases"] = [{
                "name": p.name, "gpu_ms": round(p.gpu_ms, 3), "transfer_ms": round(p.transfer_ms, 3),
                "verify_ms": round(p.verify_ms, 3), "total_ms": round(p.total_ms, 3),
                "bottleneck": p.bottleneck, "tok_per_sec": round(p.tok_per_sec, 2),
            } for p in cr.phases]
            cd["total_latency_ms"] = round(cr.total_latency_ms, 2)
            cd["throughput"] = f"{cr.throughput_value:.1f} {cr.throughput_label}"
            cd["transfer_mb_per_fwd"] = round(cr.transfer_mb_per_fwd, 2)
            cd["overhead_vs_origin"] = round(cr.overhead_vs_origin, 2)
        out["clusters"].append(cd)
    print(json.dumps(out, indent=2))


def print_presets() -> None:
    catalog = load_hardware_catalog()
    print("\n=== Hardware Catalog (from hardware/*.yaml) ===\n")
    print("GPU types:")
    for name, spec in sorted(catalog["gpus"].items()):
        nv = " NVLink" if spec.get("nvlink") else ""
        print(f"  {name:<16} {spec.get('bf16_tflops',0):>6.0f} TFLOPS  "
              f"{spec.get('memory_gb',0):>5.0f}GB  {spec.get('memory_bw_gbps',0):>5.0f} GB/s{nv}")
    print("\nCPU types:")
    for name, spec in sorted(catalog["cpus"].items()):
        tee = ",".join(spec.get("tee_support", [])) or "none"
        print(f"  {name:<18} {spec.get('cores',0):>4}C  {spec.get('fp32_gflops',0):>5.0f} GFLOPS  "
              f"{spec.get('memory_bw_gbps',0):>5.0f} GB/s  TEE:[{tee}]")
    print("\nNetwork types:")
    for name, spec in sorted(catalog["networks"].items()):
        print(f"  {name:<14} {spec.get('bandwidth_gbps',0):>6.1f} Gbps  {spec.get('latency_us',0):>6.0f} us")
    print("\nModel types:")
    for name, spec in sorted(catalog["models"].items()):
        t = spec.get("type", "llm")
        p = float(spec.get("param_bytes", 0)) / 1e9
        print(f"  {name:<16} {t:<10} {p:>5.1f}GB  H={spec.get('hidden_size',0)} L={spec.get('num_layers',0)}")
    print("\nQuantization methods:")
    for name, spec in sorted(catalog["quantization"].items()):
        w = spec.get("weight_bits", 16)
        a = spec.get("activation_bits", 16)
        mem = spec.get("memory_factor", 1.0)
        comp = spec.get("compute_factor", 1.0)
        vc = spec.get("verify_compatible", "full")
        print(f"  {name:<20} W{w}A{a}  VRAM {mem:.0%}  Compute {comp:.1f}x  Verify:{vc}")
    print(f"\nStrategies: {', '.join(sorted(STRATEGIES))}")
    print("TEE types: none, tdx, sgx, sev-snp")


def main():
    parser = argparse.ArgumentParser(description="DecenTrain Performance Analyzer")
    parser.add_argument("config", nargs="?", help="YAML config file")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--presets", action="store_true", help="List hardware catalog")
    args = parser.parse_args()

    if args.presets:
        print_presets()
        return
    if not args.config:
        parser.print_help()
        sys.exit(1)

    cfg = parse_config(args.config)
    result = run_analysis(cfg)
    if args.json:
        print_json(result)
    else:
        print_text(result, cfg)


if __name__ == "__main__":
    main()
