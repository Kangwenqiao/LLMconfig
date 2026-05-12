#!/bin/bash
set -e

echo "=== 部署 AIGC Rewriter 服务 ==="

# 安装 uv
if ! command -v uv &> /dev/null; then
    echo "安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 克隆或更新项目
PROJECT_DIR="$HOME/LLMconfig"
if [ -d "$PROJECT_DIR" ]; then
    echo "更新项目..."
    cd "$PROJECT_DIR"
    git pull
else
    echo "克隆项目..."
    git clone https://github.com/Kangwenqiao/LLMconfig.git "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# 创建模型目录
mkdir -p models

# 下载模型（如果不存在）
MODEL_FILE="models/qwen3-merged-aigc_zhv3-Q4_K_M.gguf"
if [ ! -f "$MODEL_FILE" ]; then
    echo "下载模型（约 1.1GB）..."
    wget -O "$MODEL_FILE" \
        https://huggingface.co/skskk/aigc-rewriter/resolve/main/qwen3-merged-aigc_zhv3-Q4_K_M.gguf
else
    echo "模型已存在，跳过下载"
fi

# 安装依赖
echo "安装依赖..."
uv sync

# 创建 systemd 服务
echo "创建 systemd 服务..."
sudo tee /etc/systemd/system/aigc-rewriter.service > /dev/null <<'EOF'
[Unit]
Description=AIGC Rewriter API Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/LLMconfig
Environment="PATH=/home/ubuntu/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="LLM_HOST=0.0.0.0"
Environment="LLM_PORT=1002"
Environment="LLM_WORKERS=1"
Environment="LLM_GPU_LAYERS=-1"
Environment="LLM_N_CTX=4096"
Environment="LLM_IDLE_TIMEOUT=300"
ExecStart=/home/ubuntu/.local/bin/uv run aigc_rewriter_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 重载 systemd
sudo systemctl daemon-reload

# 启动服务
echo "启动服务..."
sudo systemctl enable aigc-rewriter
sudo systemctl restart aigc-rewriter

# 等待服务启动
sleep 5

# 检查服务状态
sudo systemctl status aigc-rewriter --no-pager

echo ""
echo "=== 部署完成 ==="
echo "服务地址: http://117.50.218.77:1002"
echo "健康检查: curl http://117.50.218.77:1002/"
echo "API 文档: http://117.50.218.77:1002/docs"