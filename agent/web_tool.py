# -*- coding: utf-8 -*-
"""web_search + online extraction + entity linking.

When the KG does not cover an entity (e.g. a model not yet ingested), search
Wikipedia, extract schema-constrained triples with the LLM, link tail values to
the KG's canonical surface forms via the bilingual alias layer, and return them
tagged source=wikipedia + confidence (clearly separated from the curated KG).
"""
import os, re, json, sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests
from qa_strategy_pipeline import llm_call, _alias_resolve_tail
from lexicon import build_alias_index

WIKI = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "RadarIntelAgent/0.1 (research)"}

# relations we try to extract online (schema id -> hint)
EXTRACT_RELS = {
    "developedBy": "研制厂商/公司", "countryOfOrigin": "原产国", "operatedBy": "使用国",
    "hasFrequencyBand": "工作频段(单字母 S/X/L/C/Ku...)", "deployedOn": "部署平台/载机/舰艇",
    "hasFunction": "用途/功能", "hasTechType": "体制(如AESA/PESA/脉冲多普勒)",
    "exportedTo": "出口国", "range_km": "探测距离(km, 纯数字)", "year": "服役/研制年份",
}


def wiki_search(query, lang="en"):
    base = f"https://{lang}.wikipedia.org/w/api.php"
    try:
        r = requests.get(base, params={"action": "opensearch", "search": query,
                                       "limit": 1, "namespace": 0, "format": "json"},
                         headers=HEADERS, timeout=12)
        data = r.json()
        if not data[1]:
            return None, None, None
        title = data[1][0]; url = data[3][0]
        r2 = requests.get(base, params={"action": "query", "prop": "extracts", "exintro": 1,
                                        "explaintext": 1, "titles": title, "format": "json",
                                        "redirects": 1}, headers=HEADERS, timeout=12)
        pages = r2.json()["query"]["pages"]
        extract = next(iter(pages.values())).get("extract", "")
        return title, extract, url
    except Exception as e:
        return None, f"[wiki error: {type(e).__name__}]", None


EXTRACT_SYS = """你从给定文本中抽取关于某型号雷达的结构化三元组。
只抽取下列关系，关系名用英文 id：
{rels}
输出 JSON 数组，每项 {{"relation":"<id>","value":"<取值>"}}；
国家用中文（美国/中国/俄罗斯…），公司名保留原文，频段用单字母。
文本没提到的关系不要编。只输出 JSON 数组。"""


def extract_triples(entity, text, aliases):
    if not text or len(text) < 20:
        return []
    rels = "\n".join(f"  {k}: {v}" for k, v in EXTRACT_RELS.items())
    raw = llm_call([{"role": "system", "content": EXTRACT_SYS.format(rels=rels)},
                    {"role": "user", "content": f"型号：{entity}\n文本：{text[:3000]}\n\n抽取："}],
                   max_tokens=500)
    m = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for it in arr:
        rel, val = it.get("relation"), str(it.get("value", "")).strip()
        if rel not in EXTRACT_RELS or not val:
            continue
        # entity-link the tail to KG canonical form via alias layer
        linked = _alias_resolve_tail(val, aliases)
        canon = linked[1] if len(linked) > 1 else val
        out.append({"head": entity, "relation": rel, "tail": canon,
                    "raw_tail": val, "source": "wikipedia", "confidence": 0.6})
    return out


class WebTool:
    def __init__(self):
        self.aliases = build_alias_index()

    @staticmethod
    def _query_variants(query):
        q = re.sub(r"(装备档案|档案|参数|介绍|概览|简介|equipment|archive|dossier|parameters?|specs?)", " ", query, flags=re.I)
        q = re.sub(r"\s+", " ", q).strip()
        variants = [q]
        # model-designation token (e.g. AN/SPY-1, S-300, AN/TPY-2)
        m = re.search(r"[A-Za-z]{1,5}[/\-][A-Za-z0-9/().\-]*\d[A-Za-z0-9/().\-]*", query)
        if m:
            tok = m.group(0)
            variants += [f"{tok} radar", tok]
        seen, out = set(), []
        for v in variants:
            if v and v.lower() not in seen:
                seen.add(v.lower()); out.append(v)
        return out

    def search_extract(self, query):
        title = text = url = None
        for v in self._query_variants(query):
            for lang in ("en", "zh"):
                title, text, url = wiki_search(v, lang)
                if title:
                    break
            if title:
                break
        if not title:
            return {"found": False, "text": f"联网未找到「{query}」的条目。"}
        triples = extract_triples(query, text, self.aliases)
        lines = [f"【联网检索·Wikipedia】{title}（{url}）",
                 "（以下为联网抽取，未经图谱核验，置信度较低）"]
        for t in triples:
            lines.append(f"  - {t['relation']}: {t['tail']}〔source=wikipedia, conf={t['confidence']}〕")
        if not triples:
            lines.append("  （未抽到结构化字段，摘要：" + (text or "")[:200] + "）")
        return {"found": True, "title": title, "url": url, "triples": triples,
                "text": "\n".join(lines)}
