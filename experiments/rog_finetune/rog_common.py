# -*- coding: utf-8 -*-
"""Shared planner prompt + ChatML formatting for RoG-FT (Qwen-7B-Chat planner).
Used by both train_lora.py and infer_plans.py so train/inference prompts match.
The PLANNER_SYSTEM text is identical to qa_rog_baseline.py (the zero-shot RoG
planner) so the only change is zero-shot -> fine-tuned, not the prompt.
"""
import json
import os

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


def build_relation_specs(lexicon_dir):
    rels = json.load(open(os.path.join(lexicon_dir, "relations.json"), encoding="utf-8"))["relations"]
    lines = []
    for rid, rec in rels.items():
        if not isinstance(rec, dict):
            continue
        syns = "/".join([rec.get("zh_label", "")] + rec.get("zh_synonyms", []))
        head_t = "/".join(rec.get("head_types", []))
        tail_t = "/".join(rec.get("tail_types", []))
        lines.append(f"  {rid:18s} ({head_t} -> {tail_t}): {syns}")
    return "\n".join(lines)


def get_planner_system(lexicon_dir):
    return PLANNER_SYSTEM.format(relation_specs=build_relation_specs(lexicon_dir))


def build_chatml(system, user, assistant=None):
    """Qwen1 ChatML. If assistant is None, returns the prompt prefix ending at
    the assistant turn opener (for inference / loss masking split)."""
    prefix = (f"<|im_start|>system\n{system}<|im_end|>\n"
              f"<|im_start|>user\n{user}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    if assistant is None:
        return prefix
    return prefix + f"{assistant}<|im_end|>\n"


def build_llama2(system, user, assistant=None):
    """LLaMA-2-chat prompt format. If assistant is None, returns the prompt prefix."""
    prefix = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] "
    if assistant is None:
        return prefix
    return prefix + f"{assistant} </s>"


def build_prompt(fmt, system, user, assistant=None):
    """Dispatch on prompt format: 'qwen' (ChatML) or 'llama2' ([INST] ...)."""
    if fmt == "llama2":
        return build_llama2(system, user, assistant)
    return build_chatml(system, user, assistant)
