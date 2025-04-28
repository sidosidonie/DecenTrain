import torch
from torch.optim import AdamW
from transformers.modeling_outputs import CausalLMOutputWithPast
from verified_training.utils.log_utils import g_logger
from verified_training.utils.profiler import *
from verified_training.llm_model import *
import deepspeed

class Train:

    def __init__(self, model_path, dataloader, verify,
                 num_epochs, bs, 
                 ds_config, use_half, cpu_only, log_file = "logs/perf.json"):
        self._model_path = model_path
        self._verify = verify

        self._model = create_llm_model(model_path, verify)

        self._dataloader = dataloader
        self._num_epochs = num_epochs
        self._batch_size = bs
        self._ds_config = ds_config
        self._use_half = use_half
        self._cpu_only = cpu_only
        self._prof = Profiler("./logs/prof.json")
        self._log_file = log_file

    def dump_result(self):
        dump = {
            "cpu_only" : self._cpu_only,
            "use_half": self._use_half,
            "bs": self._batch_size,
            "ds_config": self.get_ds_config(),
            "num_epochs" : self._num_epochs,
            "model_path": self._model_path,
            "verify": self._verify,
            "perf": self._prof.dur_dict()
        }
        with open(self._log_file, "w+") as fp:
            import json
            json.dump(dump, fp, indent=4)

    def set_device(self):
        if self._cpu_only:
            device = torch.device("cpu")
        else:
            assert torch.cuda.is_available()
            device = torch.device("cuda")
        return device

    def get_ds_config(self):
        if isinstance(self._ds_config, str):
            return self._ds_config
        elif isinstance(self._ds_config, bool):
            if self._ds_config:
                if self._cpu_only:
                    return "ds_config/ds_config_cpu.json"
                else:
                    if self._use_half:
                        return "ds_config/ds_config_fp16.json"
                    else:
                        return "ds_config/ds_config.json"

        return None

    def do(self):
        if self._cpu_only:
            device = torch.device("cpu")
        else:
            assert torch.cuda.is_available()
            device = torch.device("cuda")

        g_logger.info(f"Setting device to {device}")
        self._model.to(device)
        #torch.autograd.set_detect_anomaly(True)

        ds_config = self.get_ds_config()
        if ds_config is not None:
            g_logger.info("Use deepspeed to train")
            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=self._model,
                model_parameters=self._model.parameters(),
                config=ds_config
            )
        else:
            g_logger.info("Use original to train")
            model_engine = self._model
            optimizer = AdamW(self._model.parameters(), lr=1e-5, weight_decay=0.01)

        g_logger.info("Create detailed profiler...")
        breakdown = self._prof.add_time_span("breakdown")
        total = self._prof.add_time_span("total")

        self._model.train()
        criterion = torch.nn.CrossEntropyLoss()
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=5, active=10, repeat=0),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for epoch in range(self._num_epochs):
                for inputs, targets in self._dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()               # Clear previous gradients

                    breakdown.new_iter()
                    total.new_iter()

                    total.record("st")
                    breakdown.record("forward-st")
                    outputs = model_engine(inputs)             # Forward pass

                    if isinstance(outputs, CausalLMOutputWithPast):
                        outputs = outputs.logits

                    # print("####### checking nan")
                    # print(outputs.isnan().any())
                    # print(outputs)
                    loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
                    breakdown.record("forward-ed")

                    if ds_config is not None:
                        model_engine.backward(loss)                     # Backward pass
                    else:
                        loss.backward()
                    breakdown.record("backward-ed")

                    if ds_config is not None:
                        model_engine.step()                    # Update model parameters
                    else:
                        optimizer.step()

                    breakdown.record("update-ed")
                    total.record("ed")

                    # prof.step()
                    g_logger.info("Break, only test for one pass")
                    break

            #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
            #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
            self.dump_result()