"""
视觉大模型的几家 API，统一成一个「几张图 + 一段话 → 一段文字」的调用。

四家：
  anthropic   Claude          官方 SDK
  openai      GPT             REST（chat/completions）
  doubao      火山引擎 豆包    REST，跟 OpenAI 同一套协议，只是 base_url 不同
  gemini      Google Gemini   REST（generateContent）
  local       本地起的服务     REST，OpenAI 兼容口（vLLM / SGLang / Ollama），key 可不填

**key 不在这里存。** 平台那边有个「大模型 API」页面，key 存在平台数据库里，
每次请求随 llm 字段一起带过来（局域网内）。没带的话退回环境变量里的
ANTHROPIC_API_KEY（老部署方式，保留）。

为什么 OpenAI/豆包/Gemini 走 REST 不装各家 SDK：三个 SDK 三套依赖、三套版本坑，
而这里用到的只有一个接口；REST 用 httpx 几十行，而且能拿假传输层把整条路径
测到，不用真花钱。
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass

from . import config

_logger = logging.getLogger("vision_service.llm")

PROVIDERS = ("anthropic", "openai", "doubao", "gemini", "local")

DEFAULT_BASE_URL = {
    "openai": "https://api.openai.com/v1",
    "doubao": "https://ark.cn-beijing.volces.com/api/v3",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    # 本地起的服务（vLLM / SGLang / Ollama 都能开 OpenAI 兼容口），key 可以不填
    "local": "http://127.0.0.1:8000/v1",
}
# 不需要 key 的提供方：本地服务
_KEY_OPTIONAL = {"local"}

# 请求超时：几张图几百 token，正常几秒；给到两分钟是为了排队时不误判
_TIMEOUT = 120.0


@dataclass
class LLM:
    provider: str
    model: str
    api_key: str
    base_url: str | None = None
    # 估算花费用，$/百万 token。不填就估不出来（显示 0），账以各家后台为准
    price_in: float = 0.0
    price_out: float = 0.0

    def label(self) -> str:
        return f"{self.provider}:{self.model}"


def from_env() -> LLM | None:
    """老部署方式：只配了 ANTHROPIC_API_KEY 环境变量。"""
    if not config.ANTHROPIC_API_KEY:
        return None
    pin, pout = config.SEEK_PRICE_PER_M.get(config.SEEK_MODEL, (0.0, 0.0))
    return LLM("anthropic", config.SEEK_MODEL, config.ANTHROPIC_API_KEY, price_in=pin, price_out=pout)


def from_dict(d: dict | None) -> LLM | None:
    if not d or not d.get("provider") or not d.get("model"):
        return None
    p = str(d["provider"]).lower()
    if p not in PROVIDERS:
        raise ValueError(f"不认识的模型提供方：{p}（可选 {', '.join(PROVIDERS)}）")
    return LLM(provider=p, model=str(d["model"]), api_key=str(d.get("api_key") or ""),
               base_url=(str(d["base_url"]).rstrip("/") if d.get("base_url") else None),
               price_in=float(d.get("price_in") or 0.0), price_out=float(d.get("price_out") or 0.0))


def estimate_usd(llm: LLM, input_tokens: int, output_tokens: int) -> float:
    return round(input_tokens / 1e6 * llm.price_in + output_tokens / 1e6 * llm.price_out, 4)


# ── 各家 ──────────────────────────────────────────────────────────────

def _b64(b: bytes) -> str:
    return base64.standard_b64encode(b).decode("ascii")


def _anthropic(llm: LLM, system: str, user: str, jpegs: list[bytes], max_tokens: int, client=None) -> tuple[str, dict]:
    if client is None:
        import anthropic

        client = anthropic.Anthropic(api_key=llm.api_key, max_retries=3, timeout=_TIMEOUT)
    content: list[dict] = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(b)}}
                           for b in jpegs]
    content.append({"type": "text", "text": user})
    resp = client.messages.create(
        model=llm.model, max_tokens=max_tokens, system=system,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": content}],
    )
    text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
    u = getattr(resp, "usage", None)
    return text, {"input": int(getattr(u, "input_tokens", 0) or 0), "output": int(getattr(u, "output_tokens", 0) or 0)}


def _openai_compatible(llm: LLM, system: str, user: str, jpegs: list[bytes], max_tokens: int, http=None) -> tuple[str, dict]:
    """OpenAI 和豆包（火山引擎 Ark）都是这套：/chat/completions + image_url(data:)。"""
    import httpx

    base = llm.base_url or DEFAULT_BASE_URL[llm.provider]
    content: list[dict] = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(b)}"}} for b in jpegs]
    content.append({"type": "text", "text": user})
    body = {"model": llm.model, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
    http = http or httpx.Client(timeout=_TIMEOUT)
    headers = {"Content-Type": "application/json"}
    if llm.api_key:
        headers["Authorization"] = f"Bearer {llm.api_key}"
    resp = http.post(f"{base}/chat/completions", json=body, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"{llm.provider} 返回 {resp.status_code}：{resp.text[:300]}")
    d = resp.json()
    text = ""
    try:
        msg = d["choices"][0]["message"]["content"]
        # 有的实现把 content 给成 [{type:text,text}] 列表
        text = msg if isinstance(msg, str) else "".join(p.get("text", "") for p in msg if isinstance(p, dict))
    except (KeyError, IndexError, TypeError):
        pass
    u = d.get("usage") or {}
    return text, {"input": int(u.get("prompt_tokens") or 0), "output": int(u.get("completion_tokens") or 0)}


def _gemini(llm: LLM, system: str, user: str, jpegs: list[bytes], max_tokens: int, http=None) -> tuple[str, dict]:
    import httpx

    base = llm.base_url or DEFAULT_BASE_URL["gemini"]
    parts: list[dict] = [{"inline_data": {"mime_type": "image/jpeg", "data": _b64(b)}} for b in jpegs]
    parts.append({"text": user})
    body = {"system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"maxOutputTokens": max_tokens}}
    http = http or httpx.Client(timeout=_TIMEOUT)
    resp = http.post(f"{base}/models/{llm.model}:generateContent", json=body,
                     headers={"x-goog-api-key": llm.api_key, "Content-Type": "application/json"})
    if resp.status_code != 200:
        raise RuntimeError(f"gemini 返回 {resp.status_code}：{resp.text[:300]}")
    d = resp.json()
    text = ""
    try:
        text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
    except (KeyError, IndexError, TypeError):
        pass
    u = d.get("usageMetadata") or {}
    return text, {"input": int(u.get("promptTokenCount") or 0), "output": int(u.get("candidatesTokenCount") or 0)}


def chat_vision(llm: LLM, system: str, user: str, jpegs: list[bytes], max_tokens: int = 300,
                client=None, http=None) -> tuple[str, dict]:
    """几张 JPEG + 文字 → (回答文字, {input, output} token 数)。client/http 是给测试塞桩的。"""
    if not llm.api_key and llm.provider not in _KEY_OPTIONAL:
        raise RuntimeError(f"{llm.provider} 没配 API key")
    if llm.provider == "anthropic":
        return _anthropic(llm, system, user, jpegs, max_tokens, client=client)
    if llm.provider in ("openai", "doubao", "local"):
        return _openai_compatible(llm, system, user, jpegs, max_tokens, http=http)
    if llm.provider == "gemini":
        return _gemini(llm, system, user, jpegs, max_tokens, http=http)
    raise ValueError(f"不认识的模型提供方：{llm.provider}")


def ping(llm: LLM, client=None, http=None) -> dict:
    """key 对不对、模型名对不对：发一句最短的话，看回不回。不带图，几乎不花钱。"""
    t0 = time.monotonic()
    try:
        text, usage = chat_vision(llm, "只回一个词。", "回复 OK", [], max_tokens=10, client=client, http=http)
        return {"ok": True, "latency_ms": int((time.monotonic() - t0) * 1000), "reply": (text or "")[:40],
                "usage": usage, "error": None}
    except Exception as e:  # noqa: BLE001 测试连通性就是要把错误原样带回给人看
        return {"ok": False, "latency_ms": int((time.monotonic() - t0) * 1000), "reply": None,
                "usage": None, "error": f"{type(e).__name__}: {str(e)[:300]}"}


def describe(llm: LLM | None) -> dict:
    """给 status 用：不带 key。"""
    if llm is None:
        return {"provider": None, "model": None}
    return {"provider": llm.provider, "model": llm.model}


__all__ = ["LLM", "PROVIDERS", "DEFAULT_BASE_URL", "from_env", "from_dict", "chat_vision", "ping",
           "estimate_usd", "describe"]
