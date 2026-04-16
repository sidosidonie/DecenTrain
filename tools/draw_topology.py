#!/usr/bin/env python3
"""
Draw system topology diagram from a YAML config.

Generates an architecture diagram showing CPU TEE, GPU workers,
network connections, model assignments, and key parameters.

Usage:
  python tools/draw_topology.py configs/my_system.yaml
  python tools/draw_topology.py configs/my_system.yaml -o output/topology.png
  python tools/draw_topology.py configs/my_system.yaml --with-perf   # include perf numbers
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent))
from perf_analyze import (
    parse_config, run_analysis, SystemConfig, AnalysisResult, TEE_OVERHEAD,
    STRATEGY_LABELS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Colors & Style
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "cpu_face": "#D4E6F1",
    "cpu_edge": "#2471A3",
    "cpu_tee_face": "#A9DFBF",
    "cpu_tee_edge": "#1E8449",
    "gpu_face": "#FADBD8",
    "gpu_edge": "#C0392B",
    "model_llm": "#F9E79F",
    "model_diffusion": "#D2B4DE",
    "network_line": "#566573",
    "network_label_bg": "#FDFEFE",
    "verify_face": "#E8F8F5",
    "verify_edge": "#17A589",
    "title_color": "#2C3E50",
    "text_dark": "#2C3E50",
    "text_mid": "#566573",
    "text_light": "#ABB2B9",
    "arrow_color": "#2471A3",
    "bandwidth_good": "#27AE60",
    "bandwidth_mid": "#F39C12",
    "bandwidth_bad": "#E74C3C",
}

def _bw_color(gbps: float) -> str:
    if gbps >= 25:
        return COLORS["bandwidth_good"]
    if gbps >= 5:
        return COLORS["bandwidth_mid"]
    return COLORS["bandwidth_bad"]


def _fmt_bytes(b: float) -> str:
    if b >= 1e9:
        return f"{b/1e9:.1f}GB"
    if b >= 1e6:
        return f"{b/1e6:.1f}MB"
    if b >= 1e3:
        return f"{b/1e3:.1f}KB"
    return f"{b:.0f}B"


def _fmt_ms(v: float) -> str:
    if v >= 1000:
        return f"{v/1000:.2f}s"
    if v >= 1:
        return f"{v:.1f}ms"
    return f"{v*1000:.0f}us"


# ═══════════════════════════════════════════════════════════════════════════════
# Drawing Primitives
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_box(ax, x, y, w, h, face, edge, text_lines, fontsize=8,
              title=None, title_size=9, lw=2, rounded=True, icon=None):
    """Draw a labeled rounded box."""
    style = "round,pad=0.02" if rounded else "square,pad=0"
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=style, facecolor=face, edgecolor=edge, linewidth=lw,
        zorder=2,
    )
    ax.add_patch(box)

    text_y = y + h * 0.15
    if title:
        prefix = f"{icon} " if icon else ""
        ax.text(x, y + h/2 - h * 0.18, f"{prefix}{title}",
                ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=COLORS["text_dark"], zorder=3)
        text_y = y - h * 0.05

    for i, line in enumerate(text_lines):
        ax.text(x, text_y - i * fontsize * 0.015, line,
                ha="center", va="center", fontsize=fontsize,
                color=COLORS["text_mid"], zorder=3, family="monospace")

    return box


def _draw_connection(ax, x1, y1, x2, y2, label="", color=COLORS["network_line"],
                     lw=2, style="-", label_bg=None):
    """Draw a connection line with optional label."""
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw,
            linestyle=style, zorder=1, alpha=0.8)

    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        bg = label_bg or COLORS["network_label_bg"]
        ax.text(mx, my, label, ha="center", va="center", fontsize=7,
                color=color, fontweight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=bg,
                          edgecolor=color, alpha=0.95, linewidth=0.8))


def _draw_arrow_label(ax, x, y, text, color=COLORS["arrow_color"], fontsize=6.5):
    """Small annotation arrow/label."""
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=color, style="italic", zorder=4)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Topology Drawing
# ═══════════════════════════════════════════════════════════════════════════════

def draw_topology(cfg: SystemConfig, result: AnalysisResult | None = None,
                  output_path: str | None = None, show_perf: bool = False):
    n_clusters = len(cfg.clusters)
    total_gpus = sum(c.count for c in cfg.clusters)

    fig_w = max(10, 3.5 * n_clusters + 2)
    fig_h = max(7, 5 + (1 if show_perf and result else 0))
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.set_xlim(-0.5, fig_w - 0.5)
    ax.set_ylim(-0.5, fig_h - 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Title ──
    tee_label = cfg.cpu.tee.upper() if cfg.cpu.tee != "none" else "No TEE"
    title = f"DecenTrain: 1 CPU ({tee_label}) + {total_gpus} GPU Workers ({n_clusters} clusters)"
    ax.text(fig_w / 2, fig_h - 0.3, title,
            ha="center", va="top", fontsize=13, fontweight="bold",
            color=COLORS["title_color"])

    # ── CPU TEE Box ──
    cpu_x = fig_w / 2
    cpu_y = fig_h - 1.7
    cpu_w = max(3.0, fig_w * 0.35)
    cpu_h = 1.6

    tee_overhead = TEE_OVERHEAD.get(cfg.cpu.tee, 0)
    is_tee = cfg.cpu.tee != "none"

    cpu_lines = [
        f"{cfg.cpu.cores} cores | {cfg.cpu.fp32_gflops} GFLOPS | {cfg.cpu.memory_bw_gbps} GB/s DRAM",
    ]
    if is_tee:
        cpu_lines.append(f"TEE overhead: ~{tee_overhead:.0f}%")

    verify_mode = "SLALOM" if cfg.verify.use_slalom else "Freivalds"
    cpu_lines.append(f"Verify: {verify_mode} k={cfg.verify.freivalds_k} every_n={cfg.verify.verify_every_n}")

    if show_perf and result:
        total_verify = sum(cr.verify_parallel_ms for cr in result.cluster_results if not cr.oom)
        cpu_lines.append(f"Aggregate verify: {_fmt_ms(total_verify)}/fwd")

    face = COLORS["cpu_tee_face"] if is_tee else COLORS["cpu_face"]
    edge = COLORS["cpu_tee_edge"] if is_tee else COLORS["cpu_edge"]
    cpu_title = f"CPU TEE: {cfg.cpu.name}" if is_tee else f"CPU: {cfg.cpu.name}"

    _draw_box(ax, cpu_x, cpu_y, cpu_w, cpu_h, face, edge,
              cpu_lines, fontsize=7, title=cpu_title, title_size=9.5, lw=2.5)

    ve_y = cpu_y - cpu_h / 2 + 0.22
    _draw_box(ax, cpu_x, ve_y, cpu_w * 0.55, 0.28,
              COLORS["verify_face"], COLORS["verify_edge"],
              [f"ThreadPool verify ({verify_mode})"],
              fontsize=6.5, lw=1, rounded=True)

    # ── Network label ──
    net_y = cpu_y - cpu_h / 2 - 0.6
    bw_color = _bw_color(cfg.network.bandwidth_gbps)
    net_label = f"{cfg.network.name}  {cfg.network.bandwidth_gbps} Gbps  {cfg.network.latency_us:.0f}us latency"
    ax.text(fig_w / 2, net_y + 0.15, "\u2500" * int(fig_w * 5), ha="center", va="center",
            fontsize=6, color=bw_color, alpha=0.5)
    ax.text(fig_w / 2, net_y, net_label,
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=bw_color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=bw_color, linewidth=1.5, alpha=0.95), zorder=5)

    # ── Cluster boxes (bottom row) ──
    gpu_row_y = net_y - 1.8
    gpu_w = min(2.8, (fig_w - 1) / max(n_clusters, 1) - 0.3)
    gpu_h = 2.4 if show_perf and result else 2.0
    total_w = n_clusters * (gpu_w + 0.3) - 0.3
    start_x = (fig_w - total_w) / 2 + gpu_w / 2

    strat_short = {
        "standalone": "", "pipeline_parallel": "PP",
        "tensor_parallel": "TP", "data_parallel": "DP",
    }

    for i, cluster in enumerate(cfg.clusters):
        gx = start_x + i * (gpu_w + 0.3)
        gy = gpu_row_y
        gpu = cluster.gpu
        model = cluster.model_cfg
        model_face = COLORS["model_llm"] if model.type == "llm" else COLORS["model_diffusion"]

        strategy_tag = strat_short.get(cluster.strategy, "")
        gpu_title = f"{cluster.count}x {gpu.name}" if cluster.count > 1 else gpu.name
        if strategy_tag:
            gpu_title += f" [{strategy_tag}]"

        gpu_lines = [
            f"{gpu.bf16_tflops} TFLOPS | {gpu.memory_gb}GB",
            f"Mem BW: {gpu.memory_bw_gbps} GB/s",
        ]
        if cluster.count > 1:
            strat_label = STRATEGY_LABELS.get(cluster.strategy, cluster.strategy)
            gpu_lines.append(f"{strat_label}: {cluster.count} ranks")
            if gpu.nvlink and cluster.strategy == "tensor_parallel":
                gpu_lines[-1] += f" (NVLink {gpu.nvlink_bw_gbps}GB/s)"

        _draw_box(ax, gx, gy, gpu_w, gpu_h,
                  COLORS["gpu_face"], COLORS["gpu_edge"],
                  gpu_lines, fontsize=7, title=f"GPU: {gpu_title}", title_size=8, lw=2)

        # Model sub-box
        if model.name:
            m_y = gy - gpu_h / 2 + 0.55
            m_w = gpu_w * 0.85
            m_h = 0.55
            m_lines = [f"{model.name}"]
            if model.type == "llm":
                m_lines.append(f"H={model.hidden_size} L={model.num_layers} {model.param_bytes/1e9:.1f}GB")
            else:
                steps = cfg.inference.num_steps or model.default_steps
                m_lines.append(f"H={model.hidden_size} L={model.num_layers} {steps} steps")
            _draw_box(ax, gx, m_y, m_w, m_h,
                      model_face, COLORS["gpu_edge"],
                      m_lines, fontsize=6.5, lw=1, rounded=True)

        # Perf annotation
        if show_perf and result and i < len(result.cluster_results):
            cr = result.cluster_results[i]
            if not cr.oom:
                perf_y = gy - gpu_h / 2 + 0.12
                if model.type == "llm":
                    perf_text = f"{cr.throughput_value:.1f} tok/s | {cr.transfer_mb_per_fwd:.0f}MB/fwd"
                else:
                    perf_text = f"{cr.throughput_value:.2f} img/s | {cr.transfer_mb_per_fwd:.0f}MB/fwd"
                ax.text(gx, perf_y, perf_text, ha="center", va="center",
                        fontsize=6, color=COLORS["text_mid"], style="italic", zorder=3)

        # Connection lines
        line_bot = gy + gpu_h / 2
        line_top = cpu_y - cpu_h / 2

        _draw_connection(ax, gx, line_bot, gx, net_y - 0.15,
                         color=bw_color, lw=1.5, style="-")
        _draw_connection(ax, gx, net_y + 0.15, cpu_x, line_top,
                         color=bw_color, lw=1.5, style="-")

        if result and i < len(result.cluster_results) and not result.cluster_results[i].oom:
            cr = result.cluster_results[i]
            xfer_label = f"{cr.transfer_mb_per_fwd:.0f} MB"
            mid_y = (line_bot + net_y) / 2
            ax.text(gx + gpu_w * 0.35, mid_y, xfer_label,
                    ha="left", va="center", fontsize=6, color=bw_color,
                    bbox=dict(boxstyle="round,pad=0.08", facecolor="white",
                              edgecolor="none", alpha=0.8), zorder=4)

        ax.annotate("", xy=(gx, net_y - 0.12), xytext=(gx, net_y - 0.25),
                     arrowprops=dict(arrowstyle="->", color=bw_color, lw=1.5), zorder=4)

    # ── Legend ──
    legend_y = gpu_row_y - gpu_h / 2 - 0.5
    legend_items = [
        ("[G]", "GPU compute", COLORS["gpu_edge"]),
        ("[N]", "Network transfer (verification data)", bw_color),
        ("[V]", "CPU verify (SLALOM/Freivalds)", COLORS["verify_edge"]),
    ]
    if show_perf and result:
        active = [mr for mr in result.model_results if not mr.oom]
        if active:
            bottlenecks = set(p.bottleneck for mr in active for p in mr.phases)
            bn_str = " + ".join(sorted(bottlenecks))
            legend_items.append(("[!]", f"Bottleneck: {bn_str}", COLORS["bandwidth_bad"]))

    for j, (icon, text, lcolor) in enumerate(legend_items):
        ax.text(0.3, legend_y - j * 0.25, f"{icon} {text}",
                ha="left", va="center", fontsize=7, color=lcolor)

    # ── Compression label ──
    if cfg.verify.compression == "fp16":
        ax.text(fig_w - 0.3, legend_y, "compression: fp16 (2x reduction)",
                ha="right", va="center", fontsize=6.5, color=COLORS["text_light"])

    plt.tight_layout(pad=0.5)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"Topology diagram saved to: {output_path}")
    else:
        default_path = "output/topology.png"
        Path(default_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(default_path, dpi=180, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"Topology diagram saved to: {default_path}")

    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Draw system topology diagram from YAML config",
    )
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("-o", "--output", help="Output image path (default: output/topology.png)")
    parser.add_argument("--with-perf", action="store_true",
                        help="Include performance numbers from analysis")
    args = parser.parse_args()

    cfg = parse_config(args.config)

    result = None
    if args.with_perf:
        result = run_analysis(cfg)

    draw_topology(cfg, result=result, output_path=args.output, show_perf=args.with_perf)


if __name__ == "__main__":
    main()
