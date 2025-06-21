import torch
from torch import nn
from torch_mlir import fx, compiler_utils

class MySubModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    @torch.export
    def forward(self, x):
        return self.linear(x)

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub = MySubModule()

    def forward(self, x):
        return self.sub(x) + 1

from torch_mlir import enable_private_function_export

enable_private_function_export(True)

module = MyModule()
example_input = torch.randn(2, 10)

# Export to MLIR
mlir_module = fx.export_and_import(module, example_input)

# Print MLIR
print(mlir_module.operation)

