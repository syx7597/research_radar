"""
KQA Pro adapter — convert KQA Pro KB (entities/concepts/relations/attributes
JSON) to our flat-triple format, and extract per-question topic entities.

KQA Pro types → our 11-way typology mapping is in `qtype_mapping` below.
Both forward and backward relations are emitted; attributes are emitted as
literal-tail triples with the relation = attribute key.
"""
import json
from pathlib import Path
from collections import defaultdict


KQAPRO_TO_OURS = {
    # KQA Pro program-output function  →  (our_qtype, our_strategy)
    "QueryName":              ("single_hop",    "lookup"),
    "What":                   ("single_hop",    "lookup"),
    "QueryAttr":              ("single_hop",    "lookup"),
    "QueryRelation":          ("single_hop",    "lookup"),
    "QueryAttrQualifier":     ("attr_filter",   "constrained_join"),
    "QueryRelationQualifier": ("attr_filter",   "constrained_join"),
    "Count":                  ("agg_count",     "exhaustive"),
    "SelectBetween":          ("set_compare",   "dual_subgraph"),
    "SelectAmong":            ("agg_enum",      "exhaustive"),   # argmax-among → enumerate then pick
    "VerifyStr":              ("negation",      "complement"),
    "VerifyYear":             ("negation",      "complement"),
    "VerifyNum":              ("negation",      "complement"),
    "VerifyDate":             ("negation",      "complement"),
}


def value_to_str(v):
    """Convert KQA Pro attribute value record to string."""
    if isinstance(v, dict):
        t = v.get("type", "")
        val = v.get("value", "")
        unit = v.get("unit", "")
        if unit and unit != "1":
            return f"{val} {unit}"
        return str(val)
    return str(v)


def kb_to_flat_triples(kb_path):
    """Walk every entity → emit relation triples (both directions de-duped)
    and attribute triples. Returns (triples, entity_name_to_id, id_to_name)."""
    with open(kb_path, encoding="utf-8") as f:
        kb = json.load(f)

    ents = kb["entities"]
    concepts = kb["concepts"]

    id_to_name = {**{k: v["name"] for k, v in ents.items()},
                  **{k: v["name"] for k, v in concepts.items()}}
    # name → list of ids (collisions exist)
    name_to_id = defaultdict(list)
    for k, v in id_to_name.items():
        name_to_id[v].append(k)

    seen = set()
    triples = []

    for eid, e in ents.items():
        ename = e["name"]
        # instanceOf concept edges (entity → concept name)
        for cid in e.get("instanceOf", []):
            if cid in id_to_name:
                key = (ename, "instance of", id_to_name[cid])
                if key not in seen:
                    seen.add(key)
                    triples.append({
                        "head": ename, "head_type": "Entity",
                        "relation": "instance of",
                        "tail": id_to_name[cid], "tail_type": "Concept",
                        "source": "kqa_pro_kb",
                    })

        # forward relations only (to avoid double-counting; KQA Pro records
        # each edge twice with direction tag)
        for r in e.get("relations", []):
            if r.get("direction") != "forward":
                continue
            obj_id = r.get("object")
            obj_name = id_to_name.get(obj_id, obj_id)
            pred = r.get("predicate", "?")
            key = (ename, pred, obj_name)
            if key not in seen:
                seen.add(key)
                triples.append({
                    "head": ename, "head_type": "Entity",
                    "relation": pred,
                    "tail": obj_name, "tail_type": "Entity",
                    "source": "kqa_pro_kb",
                })

        # attributes → literal-tail triples
        for a in e.get("attributes", []):
            key_pred = a.get("key", "?")
            val_str = value_to_str(a.get("value", ""))
            key = (ename, key_pred, val_str)
            if key not in seen:
                seen.add(key)
                triples.append({
                    "head": ename, "head_type": "Entity",
                    "relation": key_pred,
                    "tail": val_str, "tail_type": "Literal",
                    "source": "kqa_pro_kb",
                })

    return triples, name_to_id, id_to_name


def topic_entities_of(question_record):
    """Extract topic entities mentioned in the question's program — anything
    passed to a `Find` op."""
    out = []
    for step in question_record.get("program", []):
        if step.get("function") == "Find":
            for arg in step.get("inputs", []):
                if arg:
                    out.append(arg)
    return out


def gold_qtype_strategy(question_record):
    """Use the LAST program function to derive (our_qtype, our_strategy)."""
    prog = question_record.get("program", [])
    if not prog:
        return ("unknown", "lookup")
    final_fn = prog[-1]["function"]
    return KQAPRO_TO_OURS.get(final_fn, ("single_hop", "lookup"))


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    triples, n2i, i2n = kb_to_flat_triples(ROOT / "datasets" / "kqa_pro" / "kb.json")
    print(f"flat triples: {len(triples)}")
    print(f"distinct entities: {len(set(t['head'] for t in triples) | set(t['tail'] for t in triples))}")
    print(f"distinct relations: {len(set(t['relation'] for t in triples))}")
    # sample
    for t in triples[:5]:
        print(f"  {t['head']} --[{t['relation']}]--> {t['tail']}")
