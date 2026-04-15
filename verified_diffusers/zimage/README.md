# Verified ZImage

这个目录提供了 Z-Image 的可验证推理实现，核心目标是：

- 在 GPU 上执行原始推理计算；
- 在 CPU 上用 Freivalds 算法异步验证关键矩阵算子；
- 使用 CUDA stream/event 让 D2H 传输与 CPU 校验尽量与后续 GPU 计算并行。

## 主要入口

- `create_verified_zimage_pipeline(...)`
- `patch_zimage_pipeline(pipe, config)`

## 快速使用

```python
import torch
from verified_diffusers.zimage import create_verified_zimage_pipeline, VerifyConfig

cfg = VerifyConfig(
    enabled=True,
    freivalds_k=8,
    mse_threshold=1e-5,
    profile_enabled=True,
    profile_dir="output/zimage_verify_profile",
)

pipe = create_verified_zimage_pipeline(
    model_id="Tongyi-MAI/Z-Image",
    dtype=torch.bfloat16,
    device_map="cuda",
    verify_config=cfg,
)

out = pipe(
    prompt="Astronaut in a jungle, cold color palette, muted colors, detailed",
    num_inference_steps=4,
    max_sequence_length=128,
)

paths = pipe.export_profile("run1")
print(paths)
pipe.shutdown()
```

## 结构说明

- `runtime.py`: stream/event、异步任务队列、flush 与错误聚合。
- `layers.py`: `VerifyLinearModule` 与 `VerifyMatmul`。
- `attention.py`: 重写 ZImage attention，显式验证 QK/KV/to_out。
- `mlp.py`: 重写 FFN（w1/w3/w2）。
- `transformer_block.py`: 组合 attention + mlp，保持原 block 语义。
- `transformer.py`: 将原 transformer 的 block 与关键 linear 模块替换为验证版。
- `profiler.py`: 导出细粒度 CSV 和汇总图。

## Profiling

可统计并导出以下指标：

- `compute`: GPU 上的 matmul/linear 计算时间（event timing）
- `transfer`: GPU->CPU 的 D2H 传输时间
- `verify`: CPU Freivalds 校验耗时

导出内容：

- `*_detail.csv`
- `*_summary.csv`
- `*_plot.png`

## 测试

- `tests/test_zimage_verify_ops.py`
- `tests/test_zimage_pipeline_fast.py`
- `tests/test_zimage_pipeline_slow.py`（需要设置 `RUN_SLOW_ZIMAGE=1`）

运行 fast 测试：

```bash
pytest tests/test_zimage_verify_ops.py tests/test_zimage_pipeline_fast.py -q
```

运行 slow 测试：

```bash
RUN_SLOW_ZIMAGE=1 pytest tests/test_zimage_pipeline_slow.py -m slow -q
```
