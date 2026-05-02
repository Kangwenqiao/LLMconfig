#!/usr/bin/env python3
"""
AIGC Rewriter API Service
基于 skskk/aigc-rewriter GGUF 模型的文本重写服务
"""

import os
import gc
import re
import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn

# 模型配置 - 下载到项目路径的models目录
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "qwen3-merged-aigc_zhv3-Q4_K_M.gguf"

# 全局模型实例
llm: Optional[Llama] = None

# JSON Schema for structured output
REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "rewritten_text": {
            "type": "string",
            "description": "重写后的文本内容"
        }
    },
    "required": ["rewritten_text"]
}

SYSTEM_PROMPT = (
    "你是一个专业的文本重写助手。请将用户提供的文本重写为更自然、更人类化的表达，保持原意不变。"
    "输出必须是一个JSON对象，只包含rewritten_text字段。rewritten_text的值只能是改写后的纯正文，"
    "不要包含JSON字符串、工具标签、解释说明、反引号、Markdown代码块、.toJSONString、JSON.stringify或重复文本。"
)


class RewriteRequest(BaseModel):
    """重写请求"""
    text: str
    temperature: float = 0.7
    max_tokens: int = 2048


class RewriteResponse(BaseModel):
    """重写响应"""
    rewritten_text: str
    original_length: int
    rewritten_length: int


def _remove_model_markup(text: str) -> str:
    """移除模型可能输出的思考、代码块和工具标签。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|[^>]+?\|>", "", text)
    text = re.sub(
        r"</?(?:tool_call|tool_response|tool|function_call|function_response)[^>]*>",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def _truncate_artifact_suffix(text: str) -> str:
    """截断正文后面混入的代码、工具或序列化残留。"""
    artifact_patterns = [
        r"<(?:tool_call|tool_response|tool|function_call|function_response)\b",
        r"\.toJSONString\b",
        r"\btoJSONString\s*\(",
        r"\bJSON\.stringify\b",
        r"\bjson\.dumps\b",
        r"```",
    ]
    for pattern in artifact_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            text = text[:match.start()]
    return text.strip()


def _extract_json_rewritten_text(text: str) -> Optional[str]:
    """从完整或嵌入式 JSON 中提取 rewritten_text。"""
    decoder = json.JSONDecoder()
    candidates = [text.strip()]

    for match in re.finditer(r"\{", text):
        candidates.append(text[match.start():].strip())

    for candidate in candidates:
        try:
            data, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("rewritten_text"), str):
            return data["rewritten_text"]

    match = re.search(r'"rewritten_text"\s*:\s*("(?:[^"\\]|\\.)*")', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return match.group(1).strip('"')

    return None


def _clean_json_residue(text: str) -> str:
    """移除夹在正文标点后的 JSON 闭合残片。"""
    text = re.sub(r"([。！？!?\.]['\"’”]?)\s*[}\]]+(?=\s|$)", r"\1", text)
    text = re.sub(r"([。！？!?\.]['\"’”]?)\s*[}\]]+(?=[^\w\u4e00-\u9fff]|$)", r"\1", text)
    text = re.sub(r"([。！？!?\.]['\"’”]?)\s*[}\]]+", r"\1", text)
    text = re.sub(r"\s*[,，]\s*[}\]]+\s*$", "", text)
    return text.strip()


def _dedupe_repeated_text(text: str) -> str:
    """清理模型偶发的整段重复或连续重复句。"""
    normalized = lambda value: re.sub(r"\s+", "", value)

    if len(text) >= 20:
        for unit_len in range(8, len(text) // 2 + 1):
            unit = text[:unit_len].strip()
            if not unit:
                continue
            repeated = unit
            count = 1
            while len(repeated) < len(text) and text.startswith(repeated + unit):
                repeated += unit
                count += 1
            if count >= 2 and normalized(text[len(repeated):]) in {"", normalized(unit[: len(text) - len(repeated)])}:
                text = unit
                break

        middle = len(text) // 2
        window = min(80, middle)
        for split_at in range(max(1, middle - window), min(len(text), middle + window) + 1):
            left = text[:split_at].strip()
            right = text[split_at:].strip()
            if left and normalized(left) == normalized(right):
                text = left
                break

    parts = re.findall(r".+?[。！？!?](?:[’”])?|.+$", text, flags=re.DOTALL)
    cleaned_parts: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if cleaned_parts and normalized(cleaned_parts[-1]) == normalized(part):
            continue
        cleaned_parts.append(part)

    return "".join(cleaned_parts).strip()


def extract_pure_text(content: str, original_text: str) -> str:
    """
    从模型输出中提取纯净的重写文本。

    API 的 rewritten_text 必须只包含正文；这里兜底清理完整 JSON、半截 JSON、
    工具标签、代码块、解释说明和偶发重复输出。
    """
    text = _remove_model_markup(content or "")

    extracted = _extract_json_rewritten_text(text)
    if extracted is not None:
        text = extracted
        nested = _extract_json_rewritten_text(_remove_model_markup(text))
        if nested is not None:
            text = nested

    text = _truncate_artifact_suffix(text)
    text = _remove_model_markup(text)
    text = re.split(
        r"<(?:tool_call|tool_response|tool|function_call|function_response)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    text = re.sub(
        r"^\s*\{?\s*['\"]?rewritten_text['\"]?\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.strip().strip("` \t\r\n")

    if text.startswith('"') and text.endswith('"'):
        try:
            text = json.loads(text)
        except json.JSONDecodeError:
            text = text[1:-1]

    text = _truncate_artifact_suffix(text)
    text = _clean_json_residue(text)
    text = re.sub(r"([。！？!?\.])['\"’”]?\s*[}\]\{]+[\s}\]\{'\"`]*$", r"\1", text)
    text = re.sub(r"^[\s{\[}\]'\"`]+", "", text)
    text = re.sub(r"[\s{\[}\]'\"`]+$", "", text)

    text = _dedupe_repeated_text(_clean_json_residue(text.strip()))
    return text.strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global llm

    # 启动时加载模型
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"模型文件不存在: {MODEL_PATH}\n"
            f"请先下载模型: proxychains4 wget -O {MODEL_PATH} "
            "https://huggingface.co/skskk/aigc-rewriter/resolve/main/qwen3-merged-aigc_zhv3-Q4_K_M.gguf"
        )

    print("正在加载模型...")
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_gpu_layers=-1,  # 全部加载到GPU
        n_ctx=4096,
        verbose=False
    )
    print("模型加载完成，服务已启动")

    yield

    # 关闭时释放资源
    print("正在释放GPU资源...")
    if llm is not None:
        del llm
        llm = None
    gc.collect()
    print("GPU资源已释放")


app = FastAPI(
    title="AIGC Rewriter API",
    description="降AIGC重写服务 - 将AI生成的内容改写为更加自然、人类化的表达",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """健康检查"""
    return {"status": "ok", "model": "aigc-rewriter", "message": "服务运行中"}


@app.post("/rewrite", response_model=RewriteResponse)
async def rewrite_text(request: RewriteRequest):
    """
    重写文本接口

    将输入的文本进行重写，使表达更加自然流畅
    """
    if llm is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.text}
        ]

        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=["<|im_end|>"],
            response_format={
                "type": "json_object",
                "schema": REWRITE_SCHEMA
            }
        )

        content = response["choices"][0]["message"]["content"]
        rewritten = extract_pure_text(content, request.text)

        return RewriteResponse(
            rewritten_text=rewritten,
            original_length=len(request.text),
            rewritten_length=len(rewritten)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重写失败: {str(e)}")


@app.api_route("/rewrite/simple", methods=["GET", "POST"])
async def rewrite_simple(text: str, temperature: float = 0.7):
    """
    简单重写接口 (GET参数方式)
    """
    if llm is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    if not text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]

        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=2048,
            temperature=temperature,
            stop=["<|im_end|>"],
            response_format={
                "type": "json_object",
                "schema": REWRITE_SCHEMA
            }
        )

        content = response["choices"][0]["message"]["content"]
        rewritten = extract_pure_text(content, text)

        return {"result": rewritten}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重写失败: {str(e)}")


def main():
    """启动服务"""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=1001,
        log_level="info"
    )


if __name__ == "__main__":
    main()
