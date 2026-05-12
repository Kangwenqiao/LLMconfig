#!/usr/bin/env python3
"""
OpenAI-compatible API Server
基于 Ollama 的模型推理服务，提供 OpenAI 兼容接口
"""

import os
import time
import httpx
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Ollama 配置
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-aigc-chat:latest")
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8000"))
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "24h")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "local"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False
    strength: str | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


def clean_model_output(content: str) -> str:
    """Remove model reasoning markers from user-facing responses."""
    if not content:
        return ""
    text = content.strip()
    if text.startswith("<think>"):
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        else:
            text = text.removeprefix("<think>")
    text = text.strip()
    if text.startswith("assistant"):
        text = text.removeprefix("assistant").strip()
    if text.startswith("<think>"):
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        else:
            text = text.removeprefix("<think>")
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    text = text.lstrip(" -—,，、。；;：:")
    prefixes = ("改写后的正文：", "改写后的文本：", "输出：", "正文：")
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text.removeprefix(prefix)
            break
    text = remove_repeated_sentences(text)
    return text.strip()


def remove_repeated_sentences(text: str) -> str:
    """Trim obvious sentence-level loops from small GGUF completion models."""
    sentences = [s for s in re.split(r"(?<=[。！？!?])", text) if s.strip()]
    if len(sentences) < 2:
        return text

    result: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        normalized = re.sub(r"\s+", "", sentence)
        previous = re.sub(r"\s+", "", result[-1]) if result else ""
        if normalized in seen:
            break
        if previous and previous.endswith(normalized) and len(previous) - len(normalized) <= 2:
            result[-1] = sentence
            break
        if previous and normalized.endswith(previous) and len(normalized) - len(previous) <= 2:
            break
        seen.add(normalized)
        result.append(sentence)
    return "".join(result) if result else text


async def run_completion(messages: list[dict], temperature: float, max_tokens: int) -> str:
    """Call Ollama through the chat endpoint to preserve the GGUF chat template behavior."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
                "stream": False,
            },
        )

        if resp.status_code != 200:
            return await run_completion_generate(client, messages, temperature, max_tokens)

        return clean_model_output(resp.json().get("message", {}).get("content", ""))


async def run_completion_generate(
    client: httpx.AsyncClient,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    """Fallback for Ollama builds that cannot chat with this GGUF."""
    prompt = render_qwen_chat(messages)
    resp = await client.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "stop": ["<|im_end|>", "<|endoftext|>", "\n\n"],
            },
            "raw": True,
            "stream": False,
        }
    )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Ollama error: {resp.text}"
        )

    return clean_model_output(resp.json().get("response", ""))


def render_qwen_chat(messages: list[dict]) -> str:
    """Render the Qwen chat format used by the original llama-cpp chat completion."""
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "\n".join(parts)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 检查 Ollama 是否可用
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{OLLAMA_HOST}/api/tags")
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                print(f"Ollama connected, available models: {model_names}")
        except Exception as e:
            print(f"Warning: Cannot connect to Ollama at {OLLAMA_HOST}: {e}")

    yield


app = FastAPI(
    title="OpenAI-Compatible API (Ollama)",
    description="Ollama-based LLM inference server with OpenAI-compatible interface",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """健康检查"""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{OLLAMA_HOST}/api/tags")
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                return {
                    "status": "ok",
                    "ollama_host": OLLAMA_HOST,
                    "ollama_model": OLLAMA_MODEL,
                    "available_models": [m.get("name") for m in models],
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return {"status": "ok", "ollama_host": OLLAMA_HOST, "ollama_model": OLLAMA_MODEL}


@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{OLLAMA_HOST}/api/tags")
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                return {
                    "object": "list",
                    "data": [
                        {
                            "id": m.get("name", "unknown"),
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "ollama",
                        }
                        for m in models
                    ]
                }
        except Exception:
            pass

    return {
        "object": "list",
        "data": [
            {
                "id": OLLAMA_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollama",
            }
        ]
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI兼容的聊天补全接口"""
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        content = await run_completion(
            messages,
            request.temperature,
            request.max_tokens,
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                )
            ],
        )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Ollama request timeout")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Completion failed: {str(e)}")


def main():
    """启动服务"""
    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info"
    )


if __name__ == "__main__":
    main()
