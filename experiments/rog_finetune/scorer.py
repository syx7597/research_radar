# -*- coding: utf-8 -*-
# Auto-extracted scorer functions from experiments/run_e2e_30.py (verbatim, UTF-8 preserved).
import re
from qa_strategy_pipeline import KGIndex



def _extract_int(text: str) -> int | None:
    """Extract first non-trivial integer from answer text. Skip digits embedded
    in model-name patterns (e.g., AN/APS-124, RDR-1400C) to reduce false matches."""
    if not text:
        return None
    # mask common model-name patterns first
    masked = re.sub(r"[A-Z][A-Z0-9]*[/\-][A-Z]?\d+[A-Za-z\d/\-()]*", " ", text)
    # \d+ without \b — in Python 3 unicode mode, CJK chars count as \w so \b would fail
    m = re.search(r"\d+", masked)
    if m:
        return int(m.group(0))
    # fallback: any digit at all
    m2 = re.search(r"\d+", text)
    return int(m2.group(0)) if m2 else None



def _yes_or_no(text: str) -> str | None:
    """Parse 是/否. 'yes' if affirmative, 'no' if negative, None if unclear."""
    t = text.strip()
    head = t[:80]
    no_kw  = ["否", "不是", "没有", "不被", "不属于", "不部署", "未被", "未发现", "未知", "无相关",
              "no", "No", "NO"]
    yes_kw = ["是", "有", "属于", "已被", "yes", "Yes", "YES", "确实"]
    has_no  = any(k in head for k in no_kw)
    has_yes = any(k in head for k in yes_kw)
    if has_no and not has_yes:
        return "no"
    if has_yes and not has_no:
        return "yes"
    if has_no:    return "no"   # negation usually appears first
    if has_yes:   return "yes"
    return None



def score_question(q: dict, predicted_answer: str, kg: KGIndex) -> dict:
    qtype = q["type"]
    text = predicted_answer or ""

    if qtype == "agg_count":
        gold_count = q.get("gold_answer")
        if not isinstance(gold_count, int):
            cons = q.get("gold_constraint") or {}
            gold_count = len(kg.heads_with(cons.get("relation",""), cons.get("tail","")))
        pred_count = _extract_int(text)
        correct = (pred_count == gold_count)
        # also report relative error for partial credit logging
        rel_err = abs((pred_count or 0) - gold_count) / max(1, gold_count)
        return {"metric":"exact_count", "gold":gold_count, "pred":pred_count,
                "correct": correct, "rel_err": rel_err}

    if qtype in ("agg_enum", "relation_inverse", "attr_filter"):
        if qtype == "attr_filter":
            cons_list = q.get("gold_constraints") or []
            sets = [kg.heads_with(c["relation"], c["tail"]) for c in cons_list]
            gold_set = set.intersection(*sets) if sets else set()
        else:
            cons = q.get("gold_constraint") or {}
            gold_set = kg.heads_with(cons.get("relation",""), cons.get("tail","")) if cons else set()
            if isinstance(q.get("gold_answer"), list):
                # if curated list given, prefer it as the gold
                gold_set = set(q["gold_answer"])
        if not gold_set:
            return {"metric":"recall", "skip":"no_gold_set"}
        hit = sum(1 for g in gold_set if g in text)
        recall = hit / len(gold_set)
        return {"metric":"recall", "gold_size":len(gold_set), "hit":hit, "recall":recall,
                "correct": recall >= 0.6}  # >=60% recall counted as correct

    if qtype == "negation":
        gold_answer = str(q.get("gold_answer", "")).lower()
        gold_polarity = "no" if any(k in gold_answer for k in ["否", "no", "未"]) else "yes"
        pred_polarity = _yes_or_no(text)
        correct = (pred_polarity == gold_polarity)
        return {"metric":"polarity", "gold_polarity":gold_polarity,
                "pred_polarity":pred_polarity, "correct": correct}

    if qtype == "set_compare":
        gold_answer = str(q.get("gold_answer", "")).lower()
        gold_polarity = "yes" if "是" in gold_answer or "yes" in gold_answer else "no"
        pred_polarity = _yes_or_no(text)
        correct = (pred_polarity == gold_polarity)
        return {"metric":"polarity", "gold_polarity":gold_polarity,
                "pred_polarity":pred_polarity, "correct": correct}

    if qtype in ("two_hop_bridge", "three_hop_chain"):
        gold_answer = q.get("gold_answer", "")
        if isinstance(gold_answer, list):
            gold_strs = gold_answer
        else:
            # split on ; or , or 、 or 和
            gold_strs = re.split(r"[；;,，、和]", str(gold_answer))
            gold_strs = [g.strip() for g in gold_strs if g.strip()]
        if not gold_strs:
            return {"metric":"substring", "skip":"no_gold"}
        hits = sum(1 for g in gold_strs if g in text)
        # accept if at least 1 key entity is mentioned
        return {"metric":"substring", "gold":gold_strs, "hits":hits,
                "correct": hits >= 1}

    if qtype in ("single_hop", "distractor"):
        gold = str(q.get("gold_answer", "")).strip()
        if not gold:
            return {"metric":"substring", "skip":"no_gold"}
        # exact substring match (case-sensitive for radar names; case-insensitive otherwise)
        match = gold in text or gold.lower() in text.lower()
        return {"metric":"substring", "gold":gold, "correct": match}

    if qtype == "unanswerable":
        # correct if the answer expresses uncertainty / absence
        no_keywords = ["未知", "无相关", "没有", "无法", "无此", "no record",
                       "not found", "unknown", "no triple", "未提供", "无", "n/a"]
        match = any(k in text for k in no_keywords)
        return {"metric":"polarity", "expected":"unknown_or_no", "correct": match}

    return {"metric":"none", "skip":f"unknown_type_{qtype}"}


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────
