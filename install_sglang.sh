#!/bin/bash
# install_sglang.sh
# SGLang 0.5.1.post1 + CUDA 12.8 完整安装脚本
# 使用 --no-deps 绕过依赖冲突，手动控制所有包版本

set -e

echo "=========================================="
echo " SGLang Installation (CUDA 12.8 compat)"
echo "=========================================="
echo "  sglang:     0.5.1.post1"
echo "  sgl-kernel: 0.3.6.post1"
echo "  flashinfer: 0.2.5"
echo "  torch:      2.7.1 (cu124)"
echo "=========================================="

# 1. PyTorch (cu124 wheel 兼容 CUDA 12.8)
echo ""
echo "[1/6] Installing PyTorch 2.7.1 (cu124)..."
pip install torch==2.7.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# 2. SGLang (跳过依赖解析避免冲突)
echo ""
echo "[2/6] Installing SGLang 0.5.1.post1..."
pip install sglang==0.5.1.post1 --no-deps

# 3. sgl-kernel (从 SGLang 官方 cu124 wheel 源)
echo ""
echo "[3/6] Installing sgl-kernel 0.3.6.post1..."
pip install sgl-kernel==0.3.6.post1 --no-deps \
    --index-url https://docs.sglang.ai/whl/cu124/

# 4. flashinfer (锁定与 torch 2.7 兼容的旧版)
echo ""
echo "[4/6] Installing flashinfer-python 0.2.5..."
pip install flashinfer-python==0.2.5 --no-deps \
    --index-url https://docs.sglang.ai/whl/cu124/

# 5. 手动补装运行时依赖
echo ""
echo "[5/6] Installing runtime dependencies..."
pip install aiohttp numpy requests tqdm transformers tokenizers \
    fastapi uvicorn pydantic pillow psutil packaging \
    protobuf grpcio interegular outlines triton

# 6. Router 和 NIXL
echo ""
echo "[6/6] Installing sglang-router and NIXL..."
pip install sglang-router
pip install "nixl[cu12]"

# 验证
echo ""
echo "=========================================="
echo " Verifying installation..."
echo "=========================================="
python -c "import torch; print(f'  torch:      {torch.__version__} (CUDA {torch.version.cuda})')"
python -c "import sglang; print(f'  sglang:     {sglang.__version__}')"
python -c "import sgl_kernel; print('  sgl_kernel: OK')"
python -c "import flashinfer; print('  flashinfer: OK')"
echo "=========================================="
echo " Installation complete!"
echo "=========================================="
