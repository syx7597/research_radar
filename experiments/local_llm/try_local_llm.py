"""
Demonstrate the RUNTIME on a LOCAL Ollama model (no external API).
=================================================================
Proves the "保密边界内可部署" claim: the dispatcher/parser/answerer can run on a
local open-weight model served by Ollama; the KG, retrieval and operators are
already offline; and because the LLM does NOT enter the trust path, the facts come
from deterministic traces — so even a small local model keeps outputs auditable.

The LLM backend is selected by env vars (read by qa_strategy_pipeline / qa_router):
  LLM_URL    e.g. http://localhost:11434/v1/chat/completions   (Ollama OpenAI-compat)
  LLM_MODEL  e.g. qwen2.5:7b   (exactly as shown by `ollama list`)
  LLM_KEY    ollama            (any non-empty string)
  LLM_TIMEOUT 120              (local models can be slower)

Run (after `export`-ing the above):
  python experiments/local_llm/try_local_llm.py
See README_local.md for running on the server vs via an Xshell SSH tunnel.
"""
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if not os.getenv("LLM_URL"):
    print("请先设置本地后端环境变量，例如：")
    print("  export LLM_URL=http://localhost:11434/v1/chat/completions")
    print("  export LLM_MODEL=<你的ollama模型名，如 qwen2.5:7b>")
    print("  export LLM_KEY=ollama")
    print("  export LLM_TIMEOUT=120")
    sys.exit(1)

import requests
from qa_strategy_pipeline import llm_call, LLM_URL, LLM_MODEL

print(f"本地后端: {LLM_MODEL} @ {LLM_URL}\n")

# 0) connectivity check
print("① 连通性测试 ...", flush=True)
ans = llm_call([{"role": "user", "content": "只回复两个字：可用"}], max_tokens=8)
if ans.startswith("[LLM_ERROR"):
    print(f"   连接失败: {ans}")
    print("   检查: ollama 是否在运行? 端口/隧道是否通? 模型名是否正确(ollama list)?")
    sys.exit(1)
print(f"   OK，模型回复: {ans}\n")

# 1) end-to-end on questions that need NO retriever (route to exhaustive)
from qa_strategy_pipeline import StrategyPipeline
print("② 端到端(用本地模型做 路由+解析+复述，算子确定性求值)...\n", flush=True)
pipe = StrategyPipeline()
demo = [
    "Marconi 一共研制了多少款雷达？",
    "列出 Westinghouse Electric Corporation 研制的雷达",
    "工作在 S 波段的雷达有哪些？",
]
for q in demo:
    try:
        r = pipe.run(q)
        ev = r.get("evidence", {})
        n = ev.get("n", len(ev.get("heads", [])) if isinstance(ev, dict) else "?")
        print(f"Q: {q}")
        print(f"   题型={r['qtype']}  策略={r['strategy']}  确定性结果数={n}")
        print(f"   回答(本地模型复述): {str(r['answer'])[:180]}\n", flush=True)
    except Exception as e:
        print(f"Q: {q}\n   出错: {e}\n")

print("=> 全程未调用任何外部 API：路由/解析/复述均由本地 Ollama 模型完成，")
print("   事实数(确定性结果数)由算子在离线图谱上算出，不依赖模型——即可审计、可本地部署。")
