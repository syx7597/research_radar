"""
v3 统一 LLM 客户端 —— 收敛全仓 13 份 call_llm 副本。

- key 读取顺序: env DEEPSEEK_API_KEY > apikey.txt 的 `apikey=`/`api_key=` 行
  （代码里永远不写死 key）
- 磁盘缓存（按 messages 哈希），重跑免费
- 重试 x3 指数退避；usage 计数（calls / cache_hits / tokens in+out）
"""

import hashlib
import json
import os
import re
import threading
import time
import atexit
from pathlib import Path

import requests

_LOCK = threading.Lock()

ROOT       = Path(__file__).resolve().parent.parent.parent
CACHE_PATH = ROOT / "pipeline" / "v3" / "cache" / "llm_cache.json"
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

STATS = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0,
         "errors": 0}


def _load_endpoint() -> tuple[str, str]:
    """返回 (chat_completions_url, api_key)。

    apikey.txt 里有多组服务配置（base_url= / api_key= 就近成组），
    只取 base_url 含 deepseek 的那组；env 变量优先。
    """
    env_key = os.getenv("DEEPSEEK_API_KEY", "")
    env_url = os.getenv("DEEPSEEK_BASE_URL", "")
    if env_key:
        return _to_chat_url(env_url or "https://api.deepseek.com"), env_key

    groups, cur_url = [], ""
    f = ROOT / "apikey.txt"
    if f.exists():
        for ln in f.read_text(encoding="utf-8").splitlines():
            mu = re.match(r"\s*\"?base_url\"?\s*[=:]\s*\"?([^\s\",]+)", ln)
            if mu:
                cur_url = mu.group(1)
                continue
            mk = re.match(r"\s*\"?api_?key\"?\s*[=:]\s*\"?(sk-[A-Za-z0-9]+)", ln)
            if mk:
                groups.append((cur_url, mk.group(1)))
    for url, key in groups:
        if "deepseek" in url.lower():
            return _to_chat_url(url), key
    if groups:
        return _to_chat_url(groups[-1][0] or "https://api.deepseek.com"), groups[-1][1]
    raise RuntimeError("未找到 DeepSeek key：请设 DEEPSEEK_API_KEY 或在 apikey.txt 配 base_url/api_key")


def _to_chat_url(base: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


API_URL, _KEY = _load_endpoint()

_cache: dict = {}
if CACHE_PATH.exists():
    try:
        _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        _cache = {}
_dirty = 0


def _save_cache():
    CACHE_PATH.write_text(json.dumps(_cache, ensure_ascii=False), encoding="utf-8")


atexit.register(_save_cache)


def chat(messages: list[dict], max_tokens: int = 900, temperature: float = 0.0) -> str:
    """返回 assistant 文本；失败返回空串（调用方自行决定跳过/重试）。"""
    global _dirty
    key = hashlib.sha256(json.dumps(
        [MODEL, messages, max_tokens, temperature], ensure_ascii=False
    ).encode("utf-8")).hexdigest()[:24]
    with _LOCK:
        if key in _cache:
            STATS["cache_hits"] += 1
            return _cache[key]

    payload = {"model": MODEL, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    headers = {"Authorization": f"Bearer {_KEY}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=90)
            r.raise_for_status()
            data = r.json()
            txt = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage", {})
            with _LOCK:
                STATS["calls"] += 1
                STATS["prompt_tokens"]     += usage.get("prompt_tokens", 0)
                STATS["completion_tokens"] += usage.get("completion_tokens", 0)
                _cache[key] = txt
                _dirty += 1
                if _dirty % 20 == 0:
                    _save_cache()
            return txt
        except Exception:
            if attempt == 2:
                STATS["errors"] += 1
                return ""
            time.sleep(2.0 * (attempt + 1))
    return ""


def parse_json(text: str, default=None):
    """稳健 JSON 解析：剥 ```json 围栏，退化用首个 [...] / {...}。"""
    if not text:
        return default
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        for pat in (r"\[.*\]", r"\{.*\}"):
            m = re.search(pat, t, re.S)
            if m:
                try:
                    return json.loads(m.group())
                except Exception:
                    pass
    return default
