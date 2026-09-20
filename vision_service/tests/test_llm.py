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


def test_看不清和不是这些行为要分开(monkeypatch):
    """label=None 有两种：模型正常工作、正确拒绝（不是这些行为），和根本看不清
    （送进来的候选本身是垃圾）。混成一个 none 的话，一批 none 回来完全不知道
    该调提示词还是回头修出候选的那一层。"""
    from vision_service import seek

    labels = [seek.Label(name="舔", parts=["后爪", "前爪"])]

    clear_no = seek.parse_answer('{"see":"clear","desc":"侧卧不动，口鼻朝前","label":"none",'
                                 '"confidence":0,"note":"没有理毛动作"}', labels)
    assert clear_no["label"] is None and clear_no["see"] == "clear"
    assert "侧卧" in clear_no["desc"]

    # 看不清时即使给了标签也不采信——提示词里写明了看不清一律 none
    unclear = seek.parse_answer('{"see":"unclear","desc":"画面太暗，只看到一团",'
                                '"label":"舔","body_part":"后爪","confidence":0.8,"note":"猜的"}', labels)
    assert unclear["label"] is None and unclear["body_part"] is None
    assert unclear["confidence"] == 0.0 and unclear["see"] == "unclear"

    hit = seek.parse_answer('{"see":"clear","desc":"侧卧，头转向身后，口鼻接触左后肢",'
                            '"label":"舔","body_part":"后爪","confidence":0.7,"note":"口鼻贴着后爪"}', labels)
    assert hit["label"] == "舔" and hit["body_part"] == "后爪" and hit["confidence"] == 0.7
    assert hit["see"] == "clear" and "口鼻接触左后肢" in hit["desc"]

    # 老模型不给 see 字段：不能当成看不清而把结果丢掉
    old = seek.parse_answer('{"label":"舔","body_part":"后爪","confidence":0.6,"note":"x"}', labels)
    assert old["label"] == "舔" and old["see"] == "unknown"
    assert seek.parse_answer("不是 JSON", labels)["see"] == "unknown"


def test_多帧拼成一张带序号的图(monkeypatch):
    """分开发几张时模型容易当成几只不同的狗（俯拍裁出来的狗本来就难认），
    而舔和啃在单帧上几乎一样、差别全在几帧之间的变化。"""
    import cv2
    import numpy as np

    from vision_service import seek

    def jpg(color):
        return bytes(cv2.imencode(".jpg", np.full((80, 120, 3), color, np.uint8))[1].tobytes())

    frames = [jpg(c) for c in (60, 120, 180, 200)]
    sheet = seek.tile_frames(frames, cols=2, cell=120)
    img = cv2.imdecode(np.frombuffer(sheet, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[0] == 160 and img.shape[1] == 240          # 2x2 格
    assert seek.tile_frames([frames[0]]) == frames[0]           # 一帧不用拼
    assert seek.tile_frames([]) == b""
    assert seek.tile_frames(["不是图片".encode(), frames[0]]) == frames[0]   # 坏的跳过


def test_ask_默认拼图_提示词里说清楚序号是时间顺序(monkeypatch):
    from vision_service import llm as llmmod
    from vision_service import seek

    import cv2
    import numpy as np

    frames = [bytes(cv2.imencode(".jpg", np.full((60, 60, 3), c, np.uint8))[1].tobytes())
              for c in (50, 150, 250)]
    seen = {}

    def fake(llm, system, user, jpegs, max_tokens=300, client=None, http=None):
        seen.update(system=system, user=user, n=len(jpegs))
        return '{"see":"clear","desc":"d","label":"none","confidence":0,"note":"n"}', {"input": 1, "output": 1}
    monkeypatch.setattr(llmmod, "chat_vision", fake)

    seek.ask(frames, [seek.Label(name="舔")], 6.0, llmmod.LLM(provider="anthropic", api_key="k", model="m"))
    assert seen["n"] == 1                                        # 三帧拼成一张发
    assert "序号" in seen["user"] and "时间顺序" in seen["user"]
    assert "同一只狗的连续几秒" in seen["system"] and "see=unclear" in seen["system"]
    assert "desc" in seen["user"] and "不要写结论" in seen["user"]

    seek.ask(frames, [seek.Label(name="舔")], 6.0,
             llmmod.LLM(provider="anthropic", api_key="k", model="m"), tile=False)
    assert seen["n"] == 3                                        # 关掉拼图就分开发


def test_命令行也能用豆包等非Claude的一家(monkeypatch):
    """平台那边的 key 存在平台数据库里、随请求带过来；命令行工具不经过平台只能读环境。
    原来 from_env 只认 ANTHROPIC_API_KEY，等于"平台上配好了豆包，命令行却用不了"
    ——同一台机器两套配置。"""
    import pytest

    from vision_service import config
    from vision_service import llm as llmmod

    monkeypatch.setattr(config, "SEEK_PROVIDER", "doubao")
    monkeypatch.setattr(config, "SEEK_API_KEY", "d3e2")
    monkeypatch.setattr(config, "SEEK_MODEL", "doubao-seed-1-6-vision-250815")
    monkeypatch.setattr(config, "SEEK_BASE_URL", "")
    monkeypatch.setattr(config, "SEEK_PRICE_IN", 0.0)
    monkeypatch.setattr(config, "SEEK_PRICE_OUT", 0.0)
    l = llmmod.from_env()
    assert l.provider == "doubao" and l.api_key == "d3e2"
    assert l.model == "doubao-seed-1-6-vision-250815" and l.base_url is None
    # base_url 不填时走这家的默认地址（跟界面上那个一样）
    assert llmmod.DEFAULT_BASE_URL["doubao"] == "https://ark.cn-beijing.volces.com/api/v3"

    monkeypatch.setattr(config, "SEEK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    assert llmmod.from_env().base_url == "https://ark.cn-beijing.volces.com/api/v3"

    # 没填 key：返回 None 而不是拿空 key 去撞 401
    monkeypatch.setattr(config, "SEEK_API_KEY", "")
    assert llmmod.from_env() is None
    # 本地服务不需要 key
    monkeypatch.setattr(config, "SEEK_PROVIDER", "local")
    assert llmmod.from_env().provider == "local"
    # 写错提供方：当场报，别等到发请求
    monkeypatch.setattr(config, "SEEK_PROVIDER", "doubaoo")
    with pytest.raises(ValueError, match="不认识的模型提供方"):
        llmmod.from_env()

    # 不填 SEEK_PROVIDER 时还是老路（ANTHROPIC_API_KEY）
    monkeypatch.setattr(config, "SEEK_PROVIDER", "")
    monkeypatch.setattr(config, "SEEK_MODEL", "claude-opus-5")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-ant-x")
    a = llmmod.from_env()
    assert a.provider == "anthropic" and a.price_in == 5.0
