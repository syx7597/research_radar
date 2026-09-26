"""
Question-type router for Strategy-Routed GraphRAG.

Zero-shot classifier (DeepSeek) that maps a natural-language question
to one of 11 types defined in lexicon/question_types.json. The output
type is then used to dispatch the appropriate retrieval strategy.

Usage:
  from qa_router import QuestionRouter
  router = QuestionRouter()
  qtype, strategy = router.route("AN/TPY-2 由哪家公司研制？")
  # -> ('single_hop', 'lookup')
"""

import os
import json
import re
import time
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TYPES_PATH = ROOT / "lexicon" / "question_types.json"

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
# swappable LLM backend (deployment portability): default DeepSeek; override via env
# (LLM_URL / LLM_MODEL / LLM_KEY) to a LOCAL Ollama OpenAI-compatible endpoint.
LLM_URL = os.getenv("LLM_URL", DEEPSEEK_URL)
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_KEY = os.getenv("LLM_KEY", DEEPSEEK_KEY)
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))


SYSTEM_PROMPT = """你是雷达知识图谱问答系统的查询分类器。给定一个中文或英文问题，输出最匹配的题型ID。

题型定义：
- single_hop: 直接事实查询（X 的属性是什么；X 由谁研制；X 的频段）
- two_hop_bridge: 需要桥接中间实体的两跳查询（X 的研制方所在国；升级前型号的研制公司）
- three_hop_chain: 三跳及以上链式查询
- relation_inverse: 反向查询，已知客体找主体（"由 X 研制了哪些雷达"；"工作在 S 波段的雷达有哪些"）
- agg_count: 计数题，问"多少款/几种/数量"
- agg_enum: 列举题，问"列出所有/都有哪些/请列举"
- set_compare: 比较两个实体（X 和 Y 的共同点/差异/是否相同）
- negation: 否定/排除题（不部署在/没有/是否被 X 使用，含"是否"且预期否定答案）
- attr_filter: 多约束筛选题（既...又...；且；并且；同时满足两个条件）
- unanswerable: KG 中明显无法回答的问题
- distractor: 易混淆实体题（型号变体如 AN/SPY-1 vs AN/SPY-1A）

注意：
- relation_inverse 与 agg_enum 的区别：前者强调"由客体反查主体"，后者强调"列举/全部"。当两者都适用时优先 agg_enum。
- single_hop 与 negation 的区别：question 含"是否/没有/不"且只问一个实体 → negation。
- agg_count 与 agg_enum 的区别：问"多少/几"是 count；问"列出/都有哪些"是 enum。
- **set_compare 优先级最高**：只要问题同时出现两个并列实体（"A 和 B"/"A 与 B"/"A、B"）并要求比较或判断异同，无论是否带"是否/相同/不同"等词，一律归 set_compare，不要归 negation。
- three_hop_chain 与 two_hop_bridge 区别：问题中如果出现两次桥接（"X 的 Y 的 Z"或"X 的 Y 由谁研制？属于哪国？"两段连续追问）→ three_hop_chain；只有一次桥接 → two_hop_bridge。
- 反向枚举 vs 桥接：如果问题的关键是"已知 A 找还有哪些 B（其中 A 和 B 共享一个属性）"，且回答需要先桥接再枚举（如"A 的研制方还研制了哪些 X"）→ two_hop_bridge（包含枚举末段，但依然算 bridge）。
- **single_hop 优先于 enum/inverse**：如果问题的主语是一个**具体实体**（特定雷达型号 / 公司 / 武器型号），即使疑问词是"哪些 NN"，也归 single_hop。
  例：「AN/APG-63 装备在哪些国家？」主语是 AN/APG-63 这一具体雷达，问该雷达的国家——是 single_hop（关系=operatedBy，从 AN/APG-63 出发查 tail），不是 agg_enum。
  反例（仍是 agg_enum/relation_inverse）：「美国装备了哪些雷达？」「Raytheon 研制了哪些雷达？」——主语是国家或公司这种"汇总者"，要列举其下属的雷达。

示例：
- "AN/SPY-1 和 AN/TPY-2 的研制公司是否相同？" → set_compare（不是 negation）
- "X 和 Y 的共同部署平台" → set_compare
- "AN/TPS-70 的前代型号由哪家公司研制？属于哪个国家？" → three_hop_chain
- "AN/MPQ-65 的研制公司还研制了哪些雷达？" → two_hop_bridge
- "AN/APG-63 radar family 装备在哪些国家？" → single_hop（具体雷达 + 单值/少值答案）

仅输出题型 ID，不要其他内容。"""


def _call_llm(prompt: str, max_tokens: int = 32, retries: int = 2) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {LLM_KEY}",
               "Content-Type": "application/json"}
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(LLM_URL, headers=headers, json=payload, timeout=LLM_TIMEOUT)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    return f"[LLM_ERROR: {last_err}]"


VALID_TYPES = {
    "single_hop", "two_hop_bridge", "three_hop_chain", "relation_inverse",
    "agg_count", "agg_enum", "set_compare", "negation",
    "attr_filter", "unanswerable", "distractor",
}


def _parse_type(text: str) -> str:
    """Extract the type id from the LLM's free-form output."""
    text = text.strip().lower()
    for typ in VALID_TYPES:
        if typ in text:
            return typ
    # fallback: closest match by token
    tokens = re.findall(r"[a-z_]+", text)
    for tok in tokens:
        if tok in VALID_TYPES:
            return tok
    return "single_hop"  # default


class QuestionRouter:
    def __init__(self):
        with open(TYPES_PATH, encoding="utf-8") as f:
            self.spec = json.load(f)
        self.type_to_strategy = self.spec["type_to_strategy"]

    def route(self, question: str) -> tuple[str, str, str]:
        """Returns (type_id, strategy_id, raw_llm_output)."""
        raw = _call_llm(question)
        if raw.startswith("[LLM_ERROR"):
            return "single_hop", "lookup", raw
        qtype = _parse_type(raw)
        strategy = self.type_to_strategy.get(qtype, "lookup")
        return qtype, strategy, raw


if __name__ == "__main__":
    # quick sanity test
    router = QuestionRouter()
    samples = [
        "AN/TPY-2 是由哪家公司研制的？",
        "美国一共使用了多少款雷达？",
        "请列出所有部署在战斗机上的雷达。",
        "AN/TPY-2 是否被中国使用？",
        "AN/TPY-2 的研制公司属于哪个国家？",
        "美国研制且工作在 S 波段的雷达？",
    ]
    for q in samples:
        qtype, strategy, raw = router.route(q)
        print(f"  [{qtype:20s} → {strategy:18s}] (raw={raw!r:30s}) {q}")
