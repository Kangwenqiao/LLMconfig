#!/usr/bin/env python3
"""
OpenAI-compatible API Server
基于 Ollama 的模型推理服务，提供 OpenAI 兼容接口
"""

import os
import time
import httpx
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Ollama 配置
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-aigc")
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8000"))


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "local"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False


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
    text = content.strip()
    if text.startswith(""):
        if "" in text:
            text = text.split("", 1)[1]
        else:
            text = text.removeprefix("")
    return text.strip()


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

    # 获取最后一条用户消息作为输入（降AIGC场景）
    user_message = request.messages[-1].content if request.messages else ""

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # 调用 Ollama API (使用 generate 接口更简单)
            resp = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": user_message,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens,
                    },
                    "stream": False,
                }
            )

            if resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Ollama error: {resp.text}"
                )

            result = resp.json()
            content = result.get("response", "")
            content = clean_model_output(content)

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