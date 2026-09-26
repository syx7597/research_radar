"""
RoG-style uniform baseline (Reasoning on Graphs, Luo et al., ICLR 2024).

Implements the planning-retrieval-reasoning paradigm but uses DeepSeek instead of
the LoRA-tuned LLaMA2 (since their checkpoint targets Freebase relations, not our
28-relation RadarKG). The framework is identical:

  1. Planning  : LLM emits k candidate relation paths in the form
                 <PATH>r1<SEP>r2<SEP>...</PATH>  (with optional r^-1 inverse markers
                 to handle questions where the start entity is on the tail side).
  2. Retrieval : walk each path from start_entity through the KG (forward when r,
                 inverse when r^-1).
  3. Reasoning : LLM produces final answer from the walked path traces.

Crucially this uses *one* uniform strategy for all question types — no router,
no per-type dispatch. Comparing this to our Strategy-Routed pipeline isolates
the contribution of routing.
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

import json
import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lexicon import load_relations
from qa_strategy_pipeline import KGIndex, llm_call, ANSWER_SYSTEM, get_relation_specs


PLANNER_SYSTEM = """你是雷达知识图谱推理路径规划器（RoG-style）。给定一个问题，输出 k=3 条候选关系路径。每条路径必须能从问题中提到的实体出发到达答案。

可用关系列表（schema id (head_type -> tail_type)）：
{relation_specs}

输出格式（**严格**遵守，不要其他内容，不要 markdown）：

START: <主实体名>
<PATH>r1<SEP>r2<SEP>...</PATH>
<PATH>r1<SEP>r2<SEP>...</PATH>
<PATH>r1<SEP>r2<SEP>...</PATH>

规则：
- START 是问题中明确提到的、可作为起点遍历图的实体。包括雷达型号 / 厂商 / 国家 / 频段 / 模式等。
- 每条 <PATH> 可包含 1-3 个关系。
- 关系名加 ^-1 表示反向遍历（即"从 tail 回到 head"），例如：
  * "美国共有多少款雷达" → START=美国, <PATH>operatedBy^-1</PATH>（从美国反向找所有装备它的雷达）
  * "AN/TPY-2 由谁研制" → START=AN/TPY-2, <PATH>developedBy</PATH>（正向）
  * "AN/TPY-2 的研制公司在哪个国家" → START=AN/TPY-2, <PATH>developedBy<SEP>affiliatedTo</PATH>
  * "Lockheed Martin 研制了哪些雷达" → START=Lockheed Martin, <PATH>developedBy^-1</PATH>
- 给出 3 条路径以增加召回率（可有重复，可选不同关系）。
- 中文同义词请映射到英文 schema id。
- 频段单字母（S/X/L），不带"波段"。
"""

PLANNER_USER_TPL = "问题：{question}\n\n请按格式输出 START 和 3 条 <PATH>。"


def plan_paths(question: str) -> dict:
    """Call LLM to generate (start_entity, [path1, path2, path3]). Each path is
    a list of (relation_id, is_inverse) tuples."""
    sys_p = PLANNER_SYSTEM.format(relation_specs=get_relation_specs())
    raw = llm_call([{"role": "system", "content": sys_p},
                    {"role": "user",   "content": PLANNER_USER_TPL.format(question=question)}],
                   max_tokens=400)
    start_m = re.search(r"START\s*[:：]\s*(.+?)(?:\n|$)", raw)
    start = start_m.group(1).strip() if start_m else ""
    paths = []
    for pm in re.finditer(r"<PATH>(.*?)</PATH>", raw, flags=re.DOTALL):
        body = pm.group(1).strip()
        rels = []
        for tok in body.split("<SEP>"):
            tok = tok.strip()
            if not tok:
                continue
            if tok.endswith("^-1"):
                rels.append((tok[:-3].strip(), True))
            else:
                rels.append((tok, False))
        if rels:
            paths.append(rels)
    return {"start": start, "paths": paths, "raw": raw}


def walk_path(start: str, path: list, kg: KGIndex) -> dict:
    """Walk a single relation path from start entity. Path is [(rel, is_inverse), ...]."""
    current = {start}
    trace = []
    for rel, inverse in path:
        next_set = set()
        edges = []
        for v in current:
            if inverse:
                # u such that (u, rel, v) → use heads_with(rel, v)
                ws = kg.heads_with(rel, v)
                for u in ws:
                    edges.append((u, rel + "^-1", v))
                    next_set.add(u)
            else:
                ws = kg.tails_of(v, rel)
                for w in ws:
                    edges.append((v, rel, w))
                    next_set.add(w)
        trace.append({"step_relation": rel + ("^-1" if inverse else ""),
                      "in": sorted(current), "out": sorted(next_set), "edges_n": len(edges),
                      "edges_sample": edges[:8]})
        if not next_set:
            return {"trace": trace, "final": [], "broken_at": rel + ("^-1" if inverse else "")}
        current = next_set
    return {"trace": trace, "final": sorted(current)}


def render_rog_context(plan: dict, walks: list) -> str:
    lines = [f"【RoG Planner 输出】",
             f"START: {plan['start']}"]
    for i, p in enumerate(plan["paths"]):
        path_str = "<SEP>".join(r + ("^-1" if inv else "") for r, inv in p)
        lines.append(f"  Path{i+1}: {path_str}")
    lines.append("\n【KG 路径遍历结果】")
    union_final = set()
    for i, w in enumerate(walks):
        final = w.get("final", [])
        broken = w.get("broken_at")
        union_final |= set(final)
        if broken:
            lines.append(f"  Path{i+1}: 在关系 {broken} 处中断（KG中无匹配）")
        else:
            lines.append(f"  Path{i+1}: 终态实体 {len(final)} 个")
            for e in final[:8]:
                lines.append(f"    - {e}")
            if len(final) > 8:
                lines.append(f"    ... 共 {len(final)} 个")
    lines.append(f"\n【三条路径并集】共 {len(union_final)} 个实体")
    for e in sorted(union_final)[:60]:
        lines.append(f"  - {e}")
    if len(union_final) > 60:
        lines.append(f"  ... 共 {len(union_final)} 个")
    return "\n".join(lines)


ROG_ANSWER_SYSTEM = """你是基于 KG 推理路径回答问题的 reasoner（RoG-style）。基于路径遍历得到的实体集合，用中文回答问题。

规则：
1. 答案必须基于路径遍历结果。集合为空时回答"未知"。
2. 计数题（多少款/几种）：直接输出整数 + 简短说明。
3. 列举题（列出/有哪些）：基于"三条路径并集"枚举。
4. 是否题：检查目标实体是否在并集中。
5. 桥接题：给出最终的目标实体（路径终态）。
6. 简洁，1-3 句。
"""


def run_rog_baseline(question: str, kg: KGIndex) -> dict:
    plan = plan_paths(question)
    walks = []
    if plan["start"] and plan["paths"]:
        for p in plan["paths"]:
            w = walk_path(plan["start"], p, kg)
            walks.append(w)
    ctx = render_rog_context(plan, walks)
    user = (f"问题：{question}\n\n"
            f"路径遍历证据：\n{ctx}\n\n"
            f"请给出答案：")
    answer = llm_call([{"role": "system", "content": ROG_ANSWER_SYSTEM},
                       {"role": "user",   "content": user}], max_tokens=1200)
    return {"plan": plan, "walks": walks, "context": ctx, "answer": answer}


if __name__ == "__main__":
    with open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8") as f:
        triples = json.load(f)
    kg = KGIndex(triples)

    samples = [
        "美国一共使用了多少款雷达？",
        "AN/TPY-2 是由哪家公司研制的？",
        "AN/TPY-2 的研制公司属于哪个国家？",
        "AN/TPY-2 是否被中国使用？",
        "Lockheed Martin 研制的雷达有哪些？",
    ]
    for q in samples:
        out = run_rog_baseline(q, kg)
        print(f"\n──── {q}")
        print(f"  start={out['plan']['start']!r}")
        print(f"  paths=", out['plan']['paths'])
        union = set()
        for w in out['walks']:
            union |= set(w.get('final', []))
        print(f"  walked union: {len(union)} entities")
        print(f"  ANSWER: {out['answer'][:200]}")
