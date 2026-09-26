# -*- coding: utf-8 -*-
"""LLM 润色 dev/test 的问题句(模板→自然中文),金标保护 + Qwen 模型链 + 缓存。
只改 question 文本,gold_answer/gold_support 不变;改写后若丢关键 token 则回退模板句。

  python ca_agraphrag/polish_llm.py --n 5      # 小批测试
  python ca_agraphrag/polish_llm.py            # 全量 dev+test
"""
import json
import re
import sys
import time
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ca_agraphrag" / "data"
CACHE = DATA / "polish_cache.json"
PUBLIC = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL_CHAIN = ["qwen3.7-max-2026-05-17", "qwen3.7-max-2026-05-20", "qwen3.7-plus-2026-05-26"]
_cur = [0]

SYS = ("你是中文改写助手。把给定的雷达知识库问题从模板句改写成自然、口语化的中文提问。"
       "严格要求:(1)所有型号名/国家/波段/模式/平台/数值/专有名词与原文完全一致,一字不改、不翻译;"
       "(2)不得增删任何限定条件;(3)不要回答问题;(4)只输出改写后的问题一句话,不要引号、不要解释。")


def load_key():
    grp, cur = [], {}
    for ln in (ROOT / "apikey.txt").read_text(encoding="utf-8").splitlines():
        mu = re.match(r'\s*"?base_url"?\s*[=:]\s*"?([^\s",]+)', ln)
        if mu:
            cur = {"base_url": mu.group(1)}
        mk = re.match(r'\s*"?api_?key"?\s*[=:]\s*"?(sk-[A-Za-z0-9._\-]+)', ln)
        if mk and cur is not None:
            cur["key"] = mk.group(1); grp.append(cur); cur = {}
    for g in grp:
        if "dashscope" in g.get("base_url", "") or g.get("key", "").startswith("sk-ws"):
            return g["key"]
    raise RuntimeError("apikey.txt 未找到 dashscope / sk-ws key")


KEY = load_key()
HEADERS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}


def call(prompt):
    while _cur[0] < len(MODEL_CHAIN):
        model = MODEL_CHAIN[_cur[0]]
        payload = {"model": model, "temperature": 0.4,
                   "messages": [{"role": "system", "content": SYS},
                                {"role": "user", "content": prompt}]}
        try:
            r = requests.post(PUBLIC, headers=HEADERS, json=payload, timeout=120,
                              proxies={"http": None, "https": None})
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            body = r.text.lower()
            if any(x in body for x in ("arrear", "quota", "insufficient", "balance")):
                print(f"[额度耗尽] {model} -> 切下一个模型", flush=True)
                _cur[0] += 1; continue
            print(f"[HTTP {r.status_code}] {r.text[:120]}", flush=True)
            return None
        except Exception as e:
            print(f"[异常] {e}", flush=True); time.sleep(2); return None
    print("!!! 所有 Qwen 模型额度耗尽,请补充或换号", flush=True)
    return "__EXHAUSTED__"


def required_tokens(q):
    """金标保护:改写后必须仍包含这些关键 token(型号/值)。"""
    toks, s = set(), q.get("gold_support", [])
    t = q["type"]
    def addname(x):
        if x and isinstance(x, str) and len(x) >= 2:
            toks.add(x)
    if t in ("single_hop", "two_hop", "negation"):
        addname(s[0].get("head"))
        if t == "negation":
            addname(s[0].get("candidate"))
    elif t == "comparison":
        addname(s[0].get("a")); addname(s[1].get("b"))
    elif t in ("count", "enumerate"):
        addname(s[0].get("tail"))
    elif t == "multi_constraint":
        addname(s[0]["c1"][1]); addname(s[0]["c2"][1])
    return toks


def polish_file(name, limit=None, no_api=False):
    items = [json.loads(l) for l in (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        items = items[:limit]
    out, reverted, exhausted = [], 0, False
    for i, q in enumerate(items):
        raw = q["question"]
        if raw in cache:
            new = cache[raw]
        elif no_api:                    # 定稿模式:未润色的保留模板句,不再调 API
            new = raw
        else:
            new = call(raw)
            if new == "__EXHAUSTED__":
                exhausted = True; break
            if new:
                cache[raw] = new
                if i % 20 == 0:
                    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        # 金标保护:关键 token 必须全在;否则回退模板
        if not new or not all(tok in new for tok in required_tokens(q)):
            new = raw; reverted += 1
        q["question_template"] = raw
        q["question"] = new
        out.append(q)
        if (i + 1) % 50 == 0:
            print(f"  [{name}] {i+1}/{len(items)}  (回退 {reverted})", flush=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    if not exhausted:
        (DATA / f"{name}_polished.jsonl").write_text(
            "\n".join(json.dumps(o, ensure_ascii=False) for o in out), encoding="utf-8")
    print(f"[{name}] 润色 {len(out)} 条,金标保护回退 {reverted} 条"
          f"{' (额度中断,未写盘)' if exhausted else f' -> {name}_polished.jsonl'}", flush=True)
    return exhausted


def main():
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else None
    if n:
        print(f"=== 小批测试 {n} 条(dev)===")
        polish_file("dev", limit=n)
        # 展示对比
        pol = [json.loads(l) for l in (DATA / "dev_polished.jsonl").read_text(encoding="utf-8").splitlines()]
        for q in pol[:n]:
            print(f"  模板: {q['question_template']}")
            print(f"  润色: {q['question']}\n")
    elif "--finalize" in sys.argv:
        for name in ("dev", "test"):
            polish_file(name, no_api=True)
    else:
        for name in ("dev", "test"):
            if polish_file(name):
                break


if __name__ == "__main__":
    main()
