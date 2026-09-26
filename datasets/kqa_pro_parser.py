"""
KQA Pro-aware LLM parser for cross-domain validation of the main innovation.

Replaces the radar-specific parser in qa_strategy_pipeline.parse_args.
Outputs the same args schema (primary_entity, secondary_entity, relation_chain,
constraints, forbidden_tail, answer_target) so the existing operators can run.

Key design:
  - The system prompt enumerates the top N KQA Pro relations + attributes
    (with brief head→tail type hints derived from the KB).
  - No radar-specific disambiguation rules (no `developedBy vs countryOfOrigin`,
    no bilingual considerations).
  - The parser is asked to use Wikidata-style relation names verbatim.

This module provides:
  - build_relation_specs(kb_path) → str  (one-time build, cached to disk)
  - parse_args(question, qtype, strategy) → dict
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qa_strategy_pipeline import llm_call


SPEC_CACHE = ROOT / "datasets" / "kqa_pro" / "_parser_specs.txt"
TOP_REL = 60
TOP_ATTR = 30


def build_relation_specs(kb_path):
    """Profile KB to produce a relation+attribute spec string for the prompt."""
    with open(kb_path, encoding="utf-8") as f:
        kb = json.load(f)
    ents = kb["entities"]
    id_to_name = {**{k: v["name"] for k, v in ents.items()},
                  **{k: v["name"] for k, v in kb["concepts"].items()}}
    rel_freq = Counter()
    rel_sig = {}
    for eid, e in ents.items():
        for r in e.get("relations", []):
            if r.get("direction") != "forward":
                continue
            pred = r.get("predicate", "?")
            rel_freq[pred] += 1
            if pred not in rel_sig:
                ht = e.get("instanceOf", [""])
                tt = ents.get(r.get("object"), {}).get("instanceOf", [""])
                rel_sig[pred] = (
                    id_to_name.get(ht[0], "?") if ht else "?",
                    id_to_name.get(tt[0], "?") if tt else "?",
                )
    attr_freq = Counter()
    attr_type = {}
    for e in ents.values():
        for a in e.get("attributes", []):
            k = a.get("key", "?")
            attr_freq[k] += 1
            if k not in attr_type:
                attr_type[k] = a.get("value", {}).get("type", "?")

    lines = ["RELATIONS (top {} by frequency):".format(TOP_REL)]
    for r, n in rel_freq.most_common(TOP_REL):
        h, t = rel_sig.get(r, ("?", "?"))
        lines.append(f"  - {r}  ({h} → {t})")
    lines.append("")
    lines.append("ATTRIBUTES (top {} by frequency):".format(TOP_ATTR))
    for a, n in attr_freq.most_common(TOP_ATTR):
        lines.append(f"  - {a}  ({attr_type.get(a, '?')})")
    spec = "\n".join(lines)
    SPEC_CACHE.write_text(spec, encoding="utf-8")
    return spec


def get_relation_specs():
    if SPEC_CACHE.exists():
        return SPEC_CACHE.read_text(encoding="utf-8")
    return build_relation_specs(ROOT / "datasets" / "kqa_pro" / "kb.json")


SYSTEM_TEMPLATE = """You are an information extractor for a Wikidata-style KGQA system.
The knowledge graph contains 17K entities and 360 relations + 629 attributes drawn from Wikidata
(films, people, countries, awards, sports teams, etc.). Each question asks about facts in this KG.

Below is the most frequent relations and attributes you should prefer; use the exact lowercase
relation name verbatim. If a question implies a relation NOT in this list, you may invent a sensible
Wikidata-style relation name (e.g., "spouse", "child", "father", "mother", "headquarters location").

{relation_specs}

Output a JSON object with EXACTLY these fields (no markdown, no commentary):
{{
  "primary_entity":   "the main entity in the question, verbatim if possible",
  "secondary_entity": "second entity for comparison questions (else empty string)",
  "relation_chain":   ["list of relations / attributes to traverse from primary_entity"],
  "constraints":      [{{"relation": "rel_name", "tail": "value"}}],
  "forbidden_tail":   "entity to test for absence (for verify questions, else empty)",
  "answer_target":    "one of: count, enum, entity, yesno, literal"
}}

Rules:
  - Use lowercase Wikidata relation names from the list above when applicable.
  - For SelectBetween (binary comparison) questions, fill primary_entity and secondary_entity with
    the two compared entities and put the comparison attribute (e.g., "duration", "population")
    in relation_chain.
  - For Count questions, set answer_target=count. If FindAll-class questions reference a concept
    (e.g., "Pennsylvania counties"), put the concept name in constraints with
    relation="instance of".
  - For Verify questions, set answer_target=yesno and put the value being verified in forbidden_tail.
  - For QueryAttrQualifier / QueryRelationQualifier (asking about a qualifier of a statement),
    put the base attribute first in relation_chain, then the qualifier key.
  - For multi-hop questions, list the relations in traversal order.
  - For superlative ("largest", "smallest", "first", "last") questions, put the ranking attribute
    in relation_chain.

Output JSON only.
"""


def make_system_prompt():
    return SYSTEM_TEMPLATE.format(relation_specs=get_relation_specs())


USER_TEMPLATE = """Question: {question}
Given question type: {qtype}  (operator: {strategy})

Output JSON."""


def parse_args(question: str, qtype: str, strategy: str) -> dict:
    """Run the LLM parser; return args dict in the standard schema."""
    sys_p = make_system_prompt()
    user = USER_TEMPLATE.format(question=question, qtype=qtype, strategy=strategy)
    raw = llm_call(
        [{"role": "system", "content": sys_p},
         {"role": "user",   "content": user}],
        max_tokens=600,
    )
    # Strip code fences if any
    txt = raw.strip()
    if txt.startswith("```"):
        txt = txt.split("```", 2)[1] if txt.count("```") >= 2 else txt
        if txt.startswith("json"):
            txt = txt[4:]
        txt = txt.strip("` \n")
    try:
        args = json.loads(txt)
    except json.JSONDecodeError:
        # Try a relaxed extraction
        import re
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            try:
                args = json.loads(m.group(0))
            except Exception:
                args = {}
        else:
            args = {}
    # Normalize
    out = {
        "primary_entity":   args.get("primary_entity", "") or "",
        "secondary_entity": args.get("secondary_entity", "") or "",
        "relation_chain":   args.get("relation_chain") or [],
        "constraints":      args.get("constraints") or [],
        "forbidden_tail":   args.get("forbidden_tail", "") or "",
        "answer_target":    args.get("answer_target", "entity") or "entity",
    }
    if not isinstance(out["relation_chain"], list):
        out["relation_chain"] = [str(out["relation_chain"])]
    if not isinstance(out["constraints"], list):
        out["constraints"] = []
    return out


if __name__ == "__main__":
    # Build spec cache on first run
    spec = build_relation_specs(ROOT / "datasets" / "kqa_pro" / "kb.json")
    print(spec[:1200])
    print()
    print("--- testing parser on sample questions ---")
    samples = [
        ("Who was the prize winner when Mrs. Miniver got the Academy Award for Best Writing, Adapted Screenplay?", "two_hop_bridge", "path_plan"),
        ("Does My Neighbor Totoro or Hannah Arendt, originally in German, possess the longer run-time?", "set_compare", "dual_subgraph"),
        ("How many Pennsylvania counties have a population greater than 7800 or a population less than 40000000?", "agg_count", "exhaustive"),
        ("How old is Sacha Baron Cohen?", "single_hop", "lookup"),
    ]
    for q, qt, st in samples:
        args = parse_args(q, qt, st)
        print(f"\nQ: {q}")
        print(f"  → {json.dumps(args, ensure_ascii=False)}")
