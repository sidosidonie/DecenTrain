from typing import  OrderedDict, List

import pandas as pd
from verified_llm.utils.log_utils import g_logger
import torch
import time
from torch import nn
from pprint import pprint
from dataclasses import dataclass, field

VERIFY_INSTANCE = "verify"
ORIGIN_INSTANCE = "origin"

@dataclass
class TimeRecords:

    times : List[float] = field(default_factory=list)

    def avg_time(self, skip=3):
        valid = self.times[skip:]
        return sum(valid) / len(valid) if valid else 0.0

    def add_time(self, t : float):
        self.times.append(t)

class ModelInfo:

    def __init__(self, module : nn.Module):
        self.module = module
        ## Some layer infos
        self.input_shapes = {}
        self.output_shapes = {}
        self.times = {k: TimeRecords() for k in self.names}

    @property
    def module_to_name(self):
        return OrderedDict({m: n for n, m in self.module.named_modules()})

    @property
    def class_names(self):
        return OrderedDict({n : m.__class__.__name__ for m,n in self.module_to_name.items()})

    @property
    def names(self):
        return list(self.module_to_name.values())

    def add_elapsed(self, layer_name, t):
        self.times.get(layer_name).add_time(t)

    def get_elapsed(self, layer_name):
        if layer_name in self.times.keys():
            return self.times.get(layer_name).avg_time()
        else:
            g_logger.error(f"Can not find elapsed for {layer_name}")
            return 0.


    def add_input_shape(self, layer_name, s):
        if layer_name not in self.input_shapes.keys():
            self.input_shapes[layer_name] = s

    def add_output_shape(self, layer_name, s):
        if layer_name not in self.output_shapes.keys():
            self.output_shapes[layer_name] = s

def to_df(ori : ModelInfo, ver : ModelInfo):
    data = {
        "layer_name" : ori.names,
        "class" : list(ori.class_names.values()),
        "in_shapes" : [ori.input_shapes.get(k) for k in ori.names],
        "out_shapes" : [ori.output_shapes.get(k) for k in ori.names],
        ORIGIN_INSTANCE: [ori.get_elapsed(l) for l in ori.names],
        VERIFY_INSTANCE: [ver.get_elapsed(l) for l in ori.names],
    }


    return pd.DataFrame(data)

class ModelInfoHook:
    def __init__(self, model, streams = [torch.cuda.default_stream()]):
        # map module -> name
        self.module_to_name = {m: n for n, m in model.named_modules()}
        self.start_time = {}
        self.model_info= ModelInfo(model)

        self.add_hook(model)
        self.streams = streams

    def add_hook(self, model : nn.Module):
        for m in model.modules():
            m.register_forward_pre_hook(self.pre_hook)
            m.register_forward_hook(self.post_hook)

    def _sync_all(self):
        for s in self.streams:
            s.synchronize()

    def pre_hook(self, module, inputs):
        self._sync_all()
        self.start_time[module] = time.perf_counter()

    def post_hook(self, module, inputs, output):
        self._sync_all()

        elapsed = time.perf_counter() - self.start_time[module]
        name = self.model_info.module_to_name[module]
        self.model_info.add_elapsed(name, elapsed)
        self.model_info.add_input_shape(name, [tuple(x.shape) for x in inputs])
        self.model_info.add_output_shape(name, tuple(output.shape))

def test_hook():
    from diffusers.models.attention import FeedForward
    from verified_llm.verify_linear import SyncVerifyLinear, replace_linear

    ffd = FeedForward(100, 150)
    ffd = ffd.to("cuda")
    ffd.eval()

    g_logger.info("========== Run Original model ==========")
    hook_ori = ModelInfoHook(ffd)
    input_tensor = torch.randn(20, 100, device="cuda")
    for i in range(6):
        ffd(input_tensor)

    g_logger.info("========== Run Original model ==========")
    gpu_stream = torch.cuda.default_stream()
    cpu_stream = torch.cuda.Stream()
    replace_linear(ffd, gpu_stream, cpu_stream)
    hook_ver = ModelInfoHook(ffd, [gpu_stream, cpu_stream])
    for i in range(6):
        ffd(input_tensor)

    df = to_df(hook_ori.model_info, hook_ver.model_info)
    print(df.to_string())

def test_model_info():
    from diffusers.models.attention import FeedForward
    ffd = FeedForward(100, 150)
    info = ModelInfo(ffd)
    print(info.names())
    info.add_instance(ORIGIN_INSTANCE)
    info.add_instance(VERIFY_INSTANCE)
    info.add_elapsed(ORIGIN_INSTANCE, "net", 1.0)
    info.add_elapsed(ORIGIN_INSTANCE, "net", 1.3)
    info.add_elapsed(ORIGIN_INSTANCE, "net", 1.4)
    info.add_elapsed(ORIGIN_INSTANCE, "net", 1.5)
    print(info.get_elapsed(ORIGIN_INSTANCE, "net").avg_time(skip=3))
