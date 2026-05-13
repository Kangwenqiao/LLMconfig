#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/LLMconfig}"
MODEL_URL="https://huggingface.co/skskk/aigc-rewriter/resolve/main/qwen3-merged-aigc_zhv3-Q4_K_M.gguf"
MODEL_FILE="$PROJECT_DIR/models/qwen3-merged-aigc_zhv3-Q4_K_M.gguf"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3-aigc-chat:latest}"
SERVER_PORT="${SERVER_PORT:-8000}"
OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"

echo "=== Deploy AIGC Rewriter Ollama API ==="

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v ollama >/dev/null 2>&1; then
    echo "ollama is required. Install and start ollama first."
    exit 1
fi

if [ -d "$PROJECT_DIR/.git" ]; then
    cd "$PROJECT_DIR"
    git pull
else
    git clone https://github.com/Kangwenqiao/LLMconfig.git "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

mkdir -p models

if [ ! -f "$MODEL_FILE" ]; then
    curl -L -C - --fail -o "$MODEL_FILE" "$MODEL_URL"
fi

cat > Modelfile.qwen3-aigc-chat <<'EOF'
FROM ./models/qwen3-merged-aigc_zhv3-Q4_K_M.gguf
TEMPLATE """{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{- end }}<|im_start|>assistant
<think>

</think>

"""
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
PARAMETER stop <|im_end|>
PARAMETER stop <|endoftext|>
EOF

ollama create "$OLLAMA_MODEL" -f Modelfile.qwen3-aigc-chat

for model in qwen2.5:1.5b deepseek-r1:1.5b qwen3-aigc:latest; do
    ollama rm "$model" >/dev/null 2>&1 || true
done

uv sync

chmod +x scripts/watchdog.sh
PROJECT_DIR="$PROJECT_DIR" OLLAMA_MODEL="$OLLAMA_MODEL" \
    OLLAMA_KEEP_ALIVE="$OLLAMA_KEEP_ALIVE" SERVER_PORT="$SERVER_PORT" \
    scripts/watchdog.sh install

sleep 2
curl -s "http://127.0.0.1:${SERVER_PORT}/"
echo
