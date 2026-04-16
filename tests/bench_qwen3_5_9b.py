"""
Qwen3.5-9B end-to-end benchmark with verified inference profiling.

Measures prefill and decode phases separately:
  - Prefill: single forward pass over the full prompt (KV cache fill)
  - Decode: autoregressive token-by-token generation
  - Verification overhead breakdown (D2H transfer, Freivalds, elementwise)
  - Profiling export (CSV, JSON, plot)

Usage:
  python tests/bench_qwen3_5_9b.py
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from verified_diffusers.zimage.config import VerifyConfig
from verified_llm.llm_model import create_llm_model

MODEL_ID = "Qwen/Qwen3.5-9B"
DTYPE = torch.bfloat16
DEVICE = "cuda"
PROMPT = "Explain the concept of verified computation in three sentences."
WARMUP_RUNS = 1
TIMED_RUNS = 3
GEN_MAX_TOKENS = 50
PROFILE_DIR = "output/qwen3_5_9b_profile"


def _get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


def _tokenize(tokenizer, text: str, device: str = DEVICE):
    return tokenizer(text, return_tensors="pt").to(device)


def _free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _gpu_mem_mb():
    return torch.cuda.memory_allocated() / 1024**2


def _gpu_peak_mb():
    return torch.cuda.max_memory_allocated() / 1024**2


# ── Step 1: Architecture probe ──────────────────────────────────────────────

def probe_architecture():
    print("=" * 70)
    print("STEP 1: Architecture Probe")
    print("=" * 70)

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    # Qwen3.5 is multimodal; text config is nested under text_config
    text_cfg = getattr(config, "text_config", config)

    print(f"  Model:              {MODEL_ID}")
    print(f"  Config class:       {type(config).__name__}")
    print(f"  hidden_size:        {text_cfg.hidden_size}")
    print(f"  num_hidden_layers:  {text_cfg.num_hidden_layers}")
    print(f"  num_attention_heads:{text_cfg.num_attention_heads}")
    print(f"  num_key_value_heads:{getattr(text_cfg, 'num_key_value_heads', 'N/A')}")
    print(f"  head_dim:           {getattr(text_cfg, 'head_dim', text_cfg.hidden_size // text_cfg.num_attention_heads)}")
    print(f"  intermediate_size:  {getattr(text_cfg, 'intermediate_size', 'N/A')}")
    print(f"  vocab_size:         {text_cfg.vocab_size}")
    print(f"  attn_output_gate:   {getattr(text_cfg, 'attn_output_gate', False)}")

    # Show layer types (hybrid architecture)
    layer_types = getattr(text_cfg, "layer_types", None)
    if layer_types:
        from collections import Counter
        counts = Counter(layer_types)
        print(f"  layer_types:        {dict(counts)}")
        print(f"    full_attention:   {counts.get('full_attention', 0)} layers (verified)")
        print(f"    linear_attention: {counts.get('linear_attention', 0)} layers (pass-through)")

    print("  [OK] Architecture probe complete.\n")
    return text_cfg


# ── Greedy generation with prefill/decode timing ──────────────────────────

def _greedy_generate(model, input_ids, max_new_tokens, runtime=None):
    """Manual greedy generation loop that separately times prefill and decode.

    Returns (output_ids, prefill_ms, decode_ms, per_token_ms_list).
    """
    generated = input_ids.clone()

    # ── Prefill ──
    torch.cuda.synchronize()
    t_prefill_start = time.perf_counter()

    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    if runtime:
        runtime.flush()
    torch.cuda.synchronize()
    t_prefill_end = time.perf_counter()

    prefill_ms = (t_prefill_end - t_prefill_start) * 1000
    next_token = out.logits[:, -1:].argmax(dim=-1)
    generated = torch.cat([generated, next_token], dim=-1)

    # ── Decode ──
    per_token_ms = []
    eos_id = getattr(model.config, "eos_token_id", None)
    if isinstance(eos_id, list):
        eos_id = set(eos_id)
    elif eos_id is not None:
        eos_id = {eos_id}
    else:
        eos_id = set()

    torch.cuda.synchronize()
    t_decode_start = time.perf_counter()

    for step in range(max_new_tokens - 1):
        t_tok_start = time.perf_counter()
        out = model(
            input_ids=next_token,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        if runtime:
            runtime.flush()
        torch.cuda.synchronize()
        t_tok_end = time.perf_counter()

        per_token_ms.append((t_tok_end - t_tok_start) * 1000)
        next_token = out.logits[:, -1:].argmax(dim=-1)
        generated = torch.cat([generated, next_token], dim=-1)

        if next_token.item() in eos_id:
            break

    torch.cuda.synchronize()
    t_decode_end = time.perf_counter()
    decode_ms = (t_decode_end - t_decode_start) * 1000

    return generated, prefill_ms, decode_ms, per_token_ms


# ── Step 2: Origin baseline ─────────────────────────────────────────────────

def benchmark_origin(tokenizer):
    print("=" * 70)
    print("STEP 2: Origin Baseline (no verification)")
    print("=" * 70)

    torch.cuda.reset_peak_memory_stats()
    mem_before = _gpu_mem_mb()

    model = create_llm_model(
        MODEL_ID, verify=False, dtype=DTYPE, device=DEVICE,
        trust_remote_code=True,
    )
    model.eval()

    mem_loaded = _gpu_mem_mb()
    print(f"  VRAM after load: {mem_loaded:.0f} MB (delta: {mem_loaded - mem_before:.0f} MB)")

    inputs = _tokenize(tokenizer, PROMPT)

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(**inputs)
    torch.cuda.synchronize()

    # Timed forward passes (prefill only, no KV cache)
    fwd_times = []
    with torch.no_grad():
        for i in range(TIMED_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(**inputs)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            fwd_times.append((t1 - t0) * 1000)

    avg_fwd = sum(fwd_times) / len(fwd_times)
    print(f"  Forward pass (avg {TIMED_RUNS} runs): {avg_fwd:.1f} ms")
    print(f"  Forward pass times: {[f'{t:.1f}' for t in fwd_times]}")

    # Timed generation with prefill/decode split
    gen_inputs = _tokenize(tokenizer, PROMPT)
    n_input = gen_inputs["input_ids"].shape[1]

    with torch.no_grad():
        out_ids, prefill_ms, decode_ms, per_tok = _greedy_generate(
            model, gen_inputs["input_ids"], GEN_MAX_TOKENS,
        )

    n_output = out_ids.shape[1] - n_input
    gen_time = prefill_ms + decode_ms
    decode_tok_per_sec = len(per_tok) / (decode_ms / 1000) if decode_ms > 0 else 0
    avg_tok_ms = sum(per_tok) / len(per_tok) if per_tok else 0

    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    peak_mem = _gpu_peak_mb()

    print(f"  Prefill:  {n_input} tokens in {prefill_ms:.1f} ms "
          f"({n_input / (prefill_ms / 1000):.0f} tok/s)")
    print(f"  Decode:   {len(per_tok)} tokens in {decode_ms:.1f} ms "
          f"({decode_tok_per_sec:.1f} tok/s, avg {avg_tok_ms:.1f} ms/tok)")
    print(f"  Total:    {n_output} tokens in {gen_time:.1f} ms")
    print(f"  Peak VRAM: {peak_mem:.0f} MB")
    print(f"  Output: {text[:200]!r}...")

    results = {
        "fwd_avg_ms": avg_fwd,
        "fwd_times": fwd_times,
        "prefill_ms": prefill_ms,
        "prefill_tokens": n_input,
        "decode_ms": decode_ms,
        "decode_tokens": len(per_tok),
        "decode_tok_per_sec": decode_tok_per_sec,
        "decode_avg_tok_ms": avg_tok_ms,
        "per_token_ms": per_tok,
        "gen_time_ms": gen_time,
        "gen_tokens": n_output,
        "peak_vram_mb": peak_mem,
        "text": text,
        "output_ids": out_ids.cpu(),
    }

    _free_model(model)
    print()
    return results


# ── Step 3: Verified inference ──────────────────────────────────────────────

def benchmark_verified(tokenizer):
    print("=" * 70)
    print("STEP 3: Verified Inference")
    print("=" * 70)

    cfg = VerifyConfig(
        enabled=True,
        freivalds_k=4,
        mse_threshold=5e-2,               # Relaxed for bf16 (3 decimal digits precision)
        elementwise_mse_threshold=1e-2,    # Relaxed for bf16
        max_workers=4,
        max_verify_tensor_numel=4_000_000,
        fail_on_error=False,
        profile_enabled=True,
        profile_dir=PROFILE_DIR,
        profile_prefix="qwen3_5_9b",
    )

    torch.cuda.reset_peak_memory_stats()
    mem_before = _gpu_mem_mb()

    model = create_llm_model(
        MODEL_ID, verify=True, config=cfg, dtype=DTYPE, device=DEVICE,
        trust_remote_code=True,
    )
    model.eval()

    mem_loaded = _gpu_mem_mb()
    print(f"  VRAM after load: {mem_loaded:.0f} MB (delta: {mem_loaded - mem_before:.0f} MB)")

    runtime = model._verify_runtime
    inputs = _tokenize(tokenizer, PROMPT)

    # Warmup
    with torch.no_grad():
        _ = model(**inputs)
    runtime.flush()
    runtime.clear_errors()
    runtime.profiler._records.clear()

    # Timed forward passes (prefill only, no KV cache)
    fwd_times = []
    with torch.no_grad():
        for i in range(TIMED_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(**inputs)
            runtime.flush()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            fwd_times.append((t1 - t0) * 1000)

    avg_fwd = sum(fwd_times) / len(fwd_times)
    records_fwd = list(runtime.profiler.records)
    n_fail_fwd = sum(1 for r in records_fwd if not r.ok)

    print(f"  Forward pass (avg {TIMED_RUNS} runs): {avg_fwd:.1f} ms")
    print(f"  Forward pass times: {[f'{t:.1f}' for t in fwd_times]}")
    print(f"  Verify checks ({TIMED_RUNS} passes): {len(records_fwd)} total, {n_fail_fwd} failures")

    # Clear profiler for generation benchmark
    runtime.profiler._records.clear()
    runtime._errors.clear()

    # Timed generation with prefill/decode split
    gen_inputs = _tokenize(tokenizer, PROMPT)
    n_input = gen_inputs["input_ids"].shape[1]

    with torch.no_grad():
        out_ids, prefill_ms, decode_ms, per_tok = _greedy_generate(
            model, gen_inputs["input_ids"], GEN_MAX_TOKENS, runtime=runtime,
        )

    n_output = out_ids.shape[1] - n_input
    gen_time = prefill_ms + decode_ms
    decode_tok_per_sec = len(per_tok) / (decode_ms / 1000) if decode_ms > 0 else 0
    avg_tok_ms = sum(per_tok) / len(per_tok) if per_tok else 0

    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    peak_mem = _gpu_peak_mb()

    records_gen = list(runtime.profiler.records)
    n_fail_gen = sum(1 for r in records_gen if not r.ok)

    print(f"  Prefill:  {n_input} tokens in {prefill_ms:.1f} ms "
          f"({n_input / (prefill_ms / 1000):.0f} tok/s)")
    print(f"  Decode:   {len(per_tok)} tokens in {decode_ms:.1f} ms "
          f"({decode_tok_per_sec:.1f} tok/s, avg {avg_tok_ms:.1f} ms/tok)")
    print(f"  Total:    {n_output} tokens in {gen_time:.1f} ms")
    print(f"  Peak VRAM: {peak_mem:.0f} MB")
    print(f"  Verify checks (generation): {len(records_gen)} total, {n_fail_gen} failures")
    print(f"  Output: {text[:200]!r}...")

    # Export profiling
    profiler = runtime.profiler
    Path(PROFILE_DIR).mkdir(parents=True, exist_ok=True)
    profiler.export_csv(
        f"{PROFILE_DIR}/qwen3_5_9b_detail.csv",
        f"{PROFILE_DIR}/qwen3_5_9b_summary.csv",
    )
    profiler.export_json(
        f"{PROFILE_DIR}/qwen3_5_9b_detail.json",
        f"{PROFILE_DIR}/qwen3_5_9b_summary.json",
    )
    profiler.plot(f"{PROFILE_DIR}/qwen3_5_9b_breakdown.png")
    print(f"  Profiling exported to {PROFILE_DIR}/")

    results = {
        "fwd_avg_ms": avg_fwd,
        "fwd_times": fwd_times,
        "prefill_ms": prefill_ms,
        "prefill_tokens": n_input,
        "decode_ms": decode_ms,
        "decode_tokens": len(per_tok),
        "decode_tok_per_sec": decode_tok_per_sec,
        "decode_avg_tok_ms": avg_tok_ms,
        "per_token_ms": per_tok,
        "gen_time_ms": gen_time,
        "gen_tokens": n_output,
        "peak_vram_mb": peak_mem,
        "text": text,
        "output_ids": out_ids.cpu(),
        "n_checks_fwd": len(records_fwd),
        "n_fail_fwd": n_fail_fwd,
        "n_checks_gen": len(records_gen),
        "n_fail_gen": n_fail_gen,
        "profiler": profiler,
    }

    runtime.shutdown()
    _free_model(model)
    print()
    return results


# ── Step 4: Performance report ──────────────────────────────────────────────

def print_report(origin, verified):
    print("=" * 70)
    print("PERFORMANCE REPORT: Qwen3.5-9B Verified Inference")
    print("=" * 70)

    print(f"\n  Model:  {MODEL_ID}")
    print(f"  dtype:  {DTYPE}")
    print(f"  Prompt: {PROMPT!r}")
    print(f"  GPU:    {torch.cuda.get_device_name(0)}")
    print(f"  Architecture: Hybrid (8 full_attention verified + 24 linear_attention pass-through)")
    print(f"  Verified: 8 attention + 32 MLP layers")

    def _ratio(a, b):
        return a / b if b > 0 else float("inf")

    fwd_overhead = _ratio(verified["fwd_avg_ms"], origin["fwd_avg_ms"])
    prefill_overhead = _ratio(verified["prefill_ms"], origin["prefill_ms"])
    decode_overhead = _ratio(verified["decode_ms"], origin["decode_ms"])
    gen_overhead = _ratio(verified["gen_time_ms"], origin["gen_time_ms"])

    print(f"\n  {'Metric':<30} {'Origin':>12} {'Verified':>12} {'Overhead':>10}")
    print(f"  {'-'*64}")
    print(f"  {'Forward pass (ms)':<30} {origin['fwd_avg_ms']:>12.1f} {verified['fwd_avg_ms']:>12.1f} {fwd_overhead:>9.2f}x")
    print(f"  {'Prefill (ms)':<30} {origin['prefill_ms']:>12.1f} {verified['prefill_ms']:>12.1f} {prefill_overhead:>9.2f}x")
    print(f"  {'  input tokens':<30} {origin['prefill_tokens']:>12} {verified['prefill_tokens']:>12} {'':>10}")
    print(f"  {'  prefill tok/s':<30} {origin['prefill_tokens']/(origin['prefill_ms']/1000):>12.0f} {verified['prefill_tokens']/(verified['prefill_ms']/1000):>12.0f} {'':>10}")
    print(f"  {'Decode (ms)':<30} {origin['decode_ms']:>12.1f} {verified['decode_ms']:>12.1f} {decode_overhead:>9.2f}x")
    print(f"  {'  output tokens':<30} {origin['decode_tokens']:>12} {verified['decode_tokens']:>12} {'':>10}")
    print(f"  {'  decode tok/s':<30} {origin['decode_tok_per_sec']:>12.1f} {verified['decode_tok_per_sec']:>12.1f} {'':>10}")
    print(f"  {'  avg ms/tok':<30} {origin['decode_avg_tok_ms']:>12.1f} {verified['decode_avg_tok_ms']:>12.1f} {'':>10}")
    print(f"  {'Total generation (ms)':<30} {origin['gen_time_ms']:>12.1f} {verified['gen_time_ms']:>12.1f} {gen_overhead:>9.2f}x")
    print(f"  {'Peak VRAM (MB)':<30} {origin['peak_vram_mb']:>12.0f} {verified['peak_vram_mb']:>12.0f} {'':>10}")

    # Token equivalence
    tokens_match = torch.equal(origin["output_ids"], verified["output_ids"])
    print(f"\n  Token equivalence: {'PASS' if tokens_match else 'FAIL'}")
    if not tokens_match:
        print(f"    Origin:   {origin['text'][:100]!r}")
        print(f"    Verified: {verified['text'][:100]!r}")

    # Verification stats
    print(f"\n  Verification Statistics (generation pass):")
    print(f"    Total checks:  {verified['n_checks_gen']}")
    print(f"    Failures:      {verified['n_fail_gen']}")
    pass_rate = ((verified['n_checks_gen'] - verified['n_fail_gen']) / verified['n_checks_gen'] * 100) if verified['n_checks_gen'] > 0 else 0
    print(f"    Pass rate:     {pass_rate:.1f}%")

    # Profiler breakdown
    profiler = verified.get("profiler")
    if profiler and profiler.records:
        df = profiler.to_frame()
        print(f"\n  Profiling Breakdown (generation pass, {len(df)} records):")
        print(f"  {'Category':<20} {'Op':<25} {'Count':>6} {'Total ms':>10} {'Avg ms':>10}")
        print(f"  {'-'*71}")

        summary = df.groupby(["category", "op_name"]).agg(
            count=("duration_ms", "count"),
            total=("duration_ms", "sum"),
            avg=("duration_ms", "mean"),
        ).reset_index()

        for _, row in summary.iterrows():
            print(f"  {row['category']:<20} {row['op_name']:<25} {row['count']:>6} {row['total']:>10.1f} {row['avg']:>10.2f}")

        # Category totals
        cat_totals = df.groupby("category")["duration_ms"].sum()
        print(f"\n  Category Totals:")
        for cat, total in cat_totals.items():
            print(f"    {cat:<20} {total:>10.1f} ms")

        # Skipped ops
        n_skipped = len(df[df["category"] == "verify_skip"])
        if n_skipped > 0:
            print(f"\n  Skipped verifications (tensor too large): {n_skipped}")

    print(f"\n{'=' * 70}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"Model: {MODEL_ID}")
    print(f"dtype: {DTYPE}\n")

    model_config = probe_architecture()
    tokenizer = _get_tokenizer()

    origin_results = benchmark_origin(tokenizer)
    verified_results = benchmark_verified(tokenizer)
    print_report(origin_results, verified_results)


if __name__ == "__main__":
    main()
