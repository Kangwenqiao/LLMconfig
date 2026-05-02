# AIGC Rewriter

基于 [skskk/aigc-rewriter](https://huggingface.co/skskk/aigc-rewriter) 模型的文本重写服务，将 AI 生成的内容改写为更自然、更人类化的表达。

## 功能特点

- 文本重写：将 AI 生成的文本改写为更自然的表达
- API 服务：提供 REST API 接口，方便集成
- JSON Schema 约束：确保输出格式规范
- GPU 加速：支持 GPU 推理，速度快

## 项目结构

```
.
├── aigc_rewriter_server.py  # API 服务
├── aigc_rewriter.py         # 命令行工具
├── models/                  # 模型文件目录
│   └── qwen3-merged-aigc_zhv3-Q4_K_M.gguf
├── API_DOC.md              # API 文档
├── pyproject.toml          # 项目配置
└── README.md
```

## 安装

### 1. 安装依赖

```bash
# 使用 uv 安装依赖
uv sync
```

### 2. 下载模型

```bash
# 创建模型目录
mkdir -p models

# 下载模型（约 1.1GB）
proxychains4 wget -O models/qwen3-merged-aigc_zhv3-Q4_K_M.gguf \
  https://huggingface.co/skskk/aigc-rewriter/resolve/main/qwen3-merged-aigc_zhv3-Q4_K_M.gguf
```

## 使用方法

### 启动 API 服务

```bash
uv run aigc_rewriter_server.py
```

服务将在 `http://localhost:1001` 启动。

### API 接口

#### 健康检查

```bash
curl http://localhost:1001/
```

#### 文本重写

```bash
curl -X POST "http://localhost:1001/rewrite" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "人工智能技术的发展给人类社会带来了深远的影响。",
    "temperature": 0.7
  }'
```

**响应示例：**

```json
{
  "rewritten_text": "人工智能技术的发展给人类社会带来深远的影响。",
  "original_length": 23,
  "rewritten_length": 22
}
```

#### 简单接口

```bash
curl "http://localhost:1001/rewrite/simple?text=你的文本"
```

### 命令行工具

```bash
# 交互模式
uv run aigc_rewriter.py -i

# 单次重写
uv run aigc_rewriter.py --text "你的文本"

# 从文件读取
uv run aigc_rewriter.py --file input.txt --output output.txt
```

## API 文档

详细 API 文档请参考 [API_DOC.md](./API_DOC.md)

## 技术栈

- **模型**: Qwen3 (GGUF 格式, Q4_K_M 量化)
- **推理引擎**: llama-cpp-python
- **Web 框架**: FastAPI
- **包管理**: uv

## 模型信息

| 属性 | 值 |
|------|-----|
| 模型名称 | skskk/aigc-rewriter |
| 架构 | Qwen3 |
| 量化 | Q4_K_M |
| 模型大小 | ~1.1 GB |
| 上下文长度 | 4096 |

## License

MIT
