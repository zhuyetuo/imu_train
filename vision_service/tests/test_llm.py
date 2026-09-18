"""四家大模型 API 的统一调用。

REST 那三家（OpenAI / 豆包 / Gemini）用 httpx 的假传输层：请求真的被编出来、
经过真的 httpx 客户端、由我们自己的处理函数回应。这样测的是**发出去的东西对不对**
（地址、鉴权头、图怎么放、模型名在哪），而不是"函数被调了一次"。
"""

from __future__ import annotations

import base64
import json
import types

import httpx
import pytest

from vision_service import llm as L


def _http(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


JPEGS = [b"\xff\xd8\xff\xe0aaa", b"\xff\xd8\xff\xe0bbb"]


def test_from_dict_和默认地址():
    assert L.from_dict(None) is None and L.from_dict({"provider": "openai"}) is None
    x = L.from_dict({"provider": "OpenAI", "model": "gpt-5", "api_key": "k", "base_url": "https://x/v1/", "price_in": "1.5"})
    assert x.provider == "openai" and x.base_url == "https://x/v1" and x.price_in == 1.5 and x.label() == "openai:gpt-5"
    with pytest.raises(ValueError, match="不认识"):
        L.from_dict({"provider": "baidu", "model": "m"})
    assert L.estimate_usd(L.LLM("openai", "m", "k", price_in=2.0, price_out=10.0), 1_000_000, 100_000) == 3.0


@pytest.mark.parametrize("provider,default_base", [("openai", "https://api.openai.com/v1"),
                                                   ("doubao", "https://ark.cn-beijing.volces.com/api/v3"),
                                                   ("zhipu", "https://open.bigmodel.cn/api/paas/v4")])
def test_openai_协议_请求形状和用量(provider, default_base):
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["auth"] = req.headers.get("authorization")
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": ' {"label":"舔身体"} '}}],
                                         "usage": {"prompt_tokens": 1234, "completion_tokens": 9}})

    llm = L.LLM(provider, "some-model", "sk-test")
    text, usage = L.chat_vision(llm, "SYS", "USER", JPEGS, http=_http(handler))
    assert text.strip() == '{"label":"舔身体"}' and usage == {"input": 1234, "output": 9}
    assert seen["url"] == f"{default_base}/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    b = seen["body"]
    assert b["model"] == "some-model" and b["messages"][0] == {"role": "system", "content": "SYS"}
    parts = b["messages"][1]["content"]
    assert [p["type"] for p in parts] == ["image_url", "image_url", "text"]
    assert parts[0]["image_url"]["url"] == "data:image/jpeg;base64," + base64.standard_b64encode(JPEGS[0]).decode()
    assert parts[-1]["text"] == "USER"


def test_openai_自定义_base_url_和列表型_content():
    def handler(req):
        assert str(req.url).startswith("https://my-proxy/v1/chat/completions")
        return httpx.Response(200, json={"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]})

    text, usage = L.chat_vision(L.LLM("openai", "m", "k", base_url="https://my-proxy/v1"), "s", "u", [], http=_http(handler))
    assert text == "ab" and usage == {"input": 0, "output": 0}


def test_gemini_请求形状和用量():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["key"] = req.headers.get("x-goog-api-key")
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"label":"none"}'}]}}],
                                         "usageMetadata": {"promptTokenCount": 500, "candidatesTokenCount": 7}})

    text, usage = L.chat_vision(L.LLM("gemini", "gemini-2.5-flash", "gk"), "SYS", "USER", JPEGS, http=_http(handler))
    assert text == '{"label":"none"}' and usage == {"input": 500, "output": 7}
    assert seen["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    assert seen["key"] == "gk"
    b = seen["body"]
    assert b["system_instruction"]["parts"][0]["text"] == "SYS"
    parts = b["contents"][0]["parts"]
    assert parts[0]["inline_data"]["mime_type"] == "image/jpeg" and parts[-1]["text"] == "USER"
    assert b["generationConfig"]["maxOutputTokens"] == 300


def test_http_非200_报错带原文():
    def handler(req):
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with pytest.raises(RuntimeError, match="401.*bad key"):
        L.chat_vision(L.LLM("doubao", "m", "k"), "s", "u", [], http=_http(handler))


def test_没_key_直接报错_不发请求():
    def handler(req):
        raise AssertionError("不该发出去")

    with pytest.raises(RuntimeError, match="没配 API key"):
        L.chat_vision(L.LLM("openai", "m", ""), "s", "u", [], http=_http(handler))


def test_anthropic_走_sdk_形状():
    calls = []

    class C:
        class messages:  # noqa: N801
            @staticmethod
            def create(**kw):
                calls.append(kw)
                return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="ok")],
                                             usage=types.SimpleNamespace(input_tokens=3, output_tokens=1))

    text, usage = L.chat_vision(L.LLM("anthropic", "claude-opus-5", "ak"), "SYS", "USER", JPEGS, client=C())
    assert text == "ok" and usage == {"input": 3, "output": 1}
    kw = calls[0]
    assert kw["model"] == "claude-opus-5" and kw["system"] == "SYS"
    assert [b["type"] for b in kw["messages"][0]["content"]] == ["image", "image", "text"]


def test_ping_成功和失败都不抛():
    ok = L.ping(L.LLM("openai", "m", "k"), http=_http(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})))
    assert ok["ok"] is True and ok["reply"] == "OK" and ok["latency_ms"] >= 0
    bad = L.ping(L.LLM("openai", "m", "k"), http=_http(lambda r: httpx.Response(403, text="nope")))
    assert bad["ok"] is False and "403" in bad["error"]
    assert L.ping(L.LLM("gemini", "m", ""))["ok"] is False


def test_from_env(monkeypatch):
    monkeypatch.setattr(L.config, "ANTHROPIC_API_KEY", "")
    assert L.from_env() is None
    monkeypatch.setattr(L.config, "ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(L.config, "SEEK_MODEL", "claude-opus-5")
    e = L.from_env()
    assert e.provider == "anthropic" and e.price_in == 5.0


def test_接口_llm_test(monkeypatch):
    from fastapi.testclient import TestClient

    from vision_service import app as appmod

    monkeypatch.setattr(L, "ping", lambda llm, **kw: {"ok": True, "latency_ms": 1, "reply": llm.model, "error": None, "usage": None})
    with TestClient(appmod.app) as tc:
        r = tc.post("/api/v1/llm/test", json={"llm": {"provider": "doubao", "model": "doubao-seed", "api_key": "k"}})
        assert r.status_code == 200 and r.json()["reply"] == "doubao-seed"
        assert tc.post("/api/v1/llm/test", json={"llm": {"provider": "nope", "model": "m"}}).status_code == 422


def test_local_不要_key_也不发鉴权头():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["auth"] = req.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    text, _ = L.chat_vision(L.LLM("local", "Qwen/Qwen2.5-VL-7B-Instruct", ""), "s", "u", JPEGS, http=_http(handler))
    assert text == "ok" and seen["url"] == "http://127.0.0.1:8386/v1/chat/completions" and seen["auth"] is None
    # 给了 key（vLLM 开了 --api-key）就带上
    L.chat_vision(L.LLM("local", "m", "tok", base_url="http://gpu:8000/v1"), "s", "u", [], http=_http(handler))
    assert seen["auth"] == "Bearer tok" and seen["url"].startswith("http://gpu:8000/v1/")
