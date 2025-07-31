
echo "Install torch_mlir"
pip install --pre torch-mlir torchvision   --extra-index-url https://download.pytorch.org/whl/nightly/cpu   -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels

pip install transformers flash-attn datasets
pip install -r requirements.txt
