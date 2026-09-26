# -*- coding: utf-8 -*-
"""Provenance-cited intelligence reports — the "analysis assistant" deliverable.

Every fact in a report is rendered deterministically from the KG with its
source + confidence + evidence, so the report is auditable: each claim traces
back to a specific triple. An optional LLM executive summary is constrained to
the cited facts only (and we verify it does not invent entities/numbers).
"""
import os, re, json, sys
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qa_strategy_pipeline import llm_call, _alias_resolve_tail
from lexicon import build_alias_index, load_relations

# relation -> Chinese label for report rendering
_RELS = load_relations()["relations"]
def _zh(rel):
    rec = _RELS.get(rel)
    return (rec.get("zh_label") if isinstance(rec, dict) and rec.get("zh_label") else rel)

# which relations to surface in a dossier, in display order
DOSSIER_RELS = [
    "countryOfOrigin", "developedBy", "operatedBy", "exportedTo",
    "hasFrequencyBand", "hasMode", "hasFunction", "hasTechType",
    "deployedOn", "range_km", "peak_power_kW", "weight_kg", "year",
    "upgradeOf", "similarTo", "compatibleWith", "meetsStandard",
]
COMPARE_DIMS = ["countryOfOrigin", "developedBy", "hasFrequencyBand",
                "hasMode", "hasFunction", "deployedOn", "range_km"]


class ProvenanceKG:
    """Indexes triples with their provenance for auditable reporting."""
    def __init__(self, triples):
        self.by_h = defaultdict(list)              # head -> [triple]
        self.prov = {}                             # (h,r,t) -> {source,confidence,evidence}
        self.heads = set()
        for t in triples:
            h, r, ta = t["head"], t["relation"], t["tail"]
            self.by_h[h].append(t)
            self.prov[(h, r, ta)] = {"source": t.get("source", "?"),
                                     "confidence": t.get("confidence"),
                                     "evidence": t.get("evidence", "")}
            self.heads.add(h)
        self.aliases = build_alias_index()
        self._norm = {}                            # normalized head -> canonical head
        for h in self.heads:
            self._norm.setdefault(self._normkey(h), h)

    @staticmethod
    def _normkey(s):
        return re.sub(r"\s+", "", str(s).lower()).replace("雷达", "").replace("radar", "")

    def resolve(self, entity):
        """Return (canonical_head, found). found=False means the entity is not in
        the KG (a coverage gap → caller may invoke web_search)."""
        if entity in self.heads:
            return entity, True
        for c in _alias_resolve_tail(entity, self.aliases):
            if c in self.heads:
                return c, True
        nk = self._normkey(entity)
        if nk in self._norm:
            return self._norm[nk], True
        # fuzzy: KG head whose normalized form starts with / contains the input
        cands = [h for k, h in self._norm.items() if k.startswith(nk) or nk in k]
        if cands:
            return min(cands, key=len), True
        return entity, False

    def facts(self, entity, rels=None):
        e, found = self.resolve(entity)
        out = defaultdict(list)
        for t in self.by_h.get(e, []):
            r = t["relation"]
            if rels and r not in rels:
                continue
            out[r].append({"value": t["tail"], "source": t.get("source", "?"),
                           "confidence": t.get("confidence"), "evidence": t.get("evidence", "")})
        return e, out, found


CONF_THRESHOLD = 0.75


def _is_low(f, thr=CONF_THRESHOLD):
    c = f.get("confidence")
    return isinstance(c, (int, float)) and c < thr


def _cite(f, thr=CONF_THRESHOLD):
    conf = f.get("confidence")
    conf_s = f"，置信:{conf:.2f}" if isinstance(conf, (int, float)) else ""
    warn = " ⚠低置信" if _is_low(f, thr) else ""
    return f"〔来源:{f.get('source','?')}{conf_s}{warn}〕"


class RadarReporter:
    def __init__(self, triples, conf_threshold=CONF_THRESHOLD):
        self.kg = ProvenanceKG(triples)
        self.thr = conf_threshold

    # ---------- 装备档案 ----------
    def dossier(self, entity, summary=True):
        e, facts, found = self.kg.facts(entity, set(DOSSIER_RELS))
        if not found:
            return {"entity": entity, "found": False, "markdown": "",
                    "n_claims": 0, "facts": [],
                    "note": f"知识图谱未收录「{entity}」，建议联网检索补全（web_search）。"}
        lines = [f"# 装备档案：{e}", ""]
        claim_count = 0
        cited_facts = []          # all facts (label, value, source, confidence)
        reliable_facts = []       # conf >= threshold — feed the auto-summary
        n_low = 0
        for r in DOSSIER_RELS:
            if r not in facts:
                continue
            label = _zh(r)
            rendered = []
            for f in facts[r][:8]:
                rendered.append(f"{f['value']} {_cite(f, self.thr)}")
                claim_count += 1
                cited_facts.append((label, f["value"], f.get("source"), f.get("confidence")))
                if _is_low(f, self.thr):
                    n_low += 1
                else:
                    reliable_facts.append((label, f["value"], f.get("source"), f.get("confidence")))
            lines.append(f"- **{label}**：" + "；".join(rendered))
        if not facts:
            lines.append("_知识图谱中未收录该实体的结构化事实。_")
        if n_low:
            lines += ["", f"> ⚠ 含 {n_low} 条低置信线索（置信<{self.thr}），已标注；自动概述仅采用高置信事实，建议人工核验。"]
        md = "\n".join(lines)
        report = {"entity": e, "found": True, "markdown": md,
                  "n_claims": claim_count, "n_low_conf": n_low, "facts": cited_facts}
        if summary and reliable_facts:
            report["summary"] = self._summary(e, reliable_facts)
            report["faithful"] = self._faithfulness(report["summary"], reliable_facts, e)
        return report

    # ---------- 对比报告 ----------
    def compare(self, a, b, dims=None):
        dims = dims or COMPARE_DIMS
        ea, fa, fda = self.kg.facts(a, set(dims))
        eb, fb, fdb = self.kg.facts(b, set(dims))
        missing = [x for x, ok in ((a, fda), (b, fdb)) if not ok]
        lines = [f"# 对比分析：{ea}  vs  {eb}", "",
                 "| 维度 | " + ea + " | " + eb + " |", "|---|---|---|"]
        for r in dims:
            la = "；".join(f"{x['value']} {_cite(x, self.thr)}" for x in fa.get(r, [])[:4]) or "—"
            lb = "；".join(f"{x['value']} {_cite(x, self.thr)}" for x in fb.get(r, [])[:4]) or "—"
            lines.append(f"| {_zh(r)} | {la} | {lb} |")
        # explicit same/diff on developer & country
        diffs = []
        for r in ("developedBy", "countryOfOrigin"):
            sa = {x["value"] for x in fa.get(r, [])}
            sb = {x["value"] for x in fb.get(r, [])}
            if sa and sb:
                diffs.append(f"- {_zh(r)}：{'相同' if sa==sb else '不同'}"
                             f"（{ea}={ '、'.join(sorted(sa)) }；{eb}={ '、'.join(sorted(sb)) }）")
        if diffs:
            lines += ["", "## 关键异同", *diffs]
        if missing:
            lines += ["", f"> 注：{ '、'.join(missing) } 未被知识图谱收录，建议联网检索补全。"]
        return {"a": ea, "b": eb, "markdown": "\n".join(lines), "missing": missing}

    # ---------- LLM 概述（受约束，不改事实） ----------
    def _summary(self, entity, cited_facts):
        facts_str = "\n".join(f"- {lab}：{val}" for lab, val, _s, _c in cited_facts)
        sys_p = ("你是雷达情报分析师。仅依据下面给出的已核验事实，写一段 2-3 句的中文概述。"
                 "不得引入事实之外的任何实体、型号、数字或结论。")
        user = f"目标：{entity}\n已核验事实：\n{facts_str}\n\n请写概述："
        out = llm_call([{"role": "system", "content": sys_p},
                        {"role": "user", "content": user}], max_tokens=300)
        return out if out and not out.startswith("[LLM_ERROR") else ""

    def _faithfulness(self, summary, cited_facts, entity):
        """Lightweight check: do the entity/number tokens mentioned in the summary
        all appear in the cited facts (or the subject entity itself)?"""
        vocab = {str(val) for _lab, val, _s, _c in cited_facts}
        vocab.add(str(entity))                      # the subject is allowed
        toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9/\-().]{2,}", summary)
        checked = [t for t in toks if any(ch.isdigit() for ch in t) or "/" in t or "-" in t]
        unsupported = [t for t in checked if not any(t in v or v in t for v in vocab)]
        ratio = 1 - len(unsupported) / max(len(checked), 1)
        return {"checked_tokens": len(checked), "unsupported": unsupported, "support_ratio": round(ratio, 3)}


def load_reporter():
    triples = json.load(open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8"))
    return RadarReporter(triples)
