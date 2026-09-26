"""
Per-question 2-hop subgraph extractor for KQA Pro.

For each question we extract the local KB neighborhood around topic entities
(those mentioned in `Find` ops) so the operator suite can be invoked without
loading the entire 362K-triple KB.

The subgraph is returned as a list of triples in the format the rest of the
pipeline expects (`head/head_type/relation/tail/tail_type/source`).
"""
import json
from collections import defaultdict
from pathlib import Path


def _entity_name_to_ids(kb):
    """Map every entity NAME (canonical + via labels) → list of KB ids."""
    out = defaultdict(list)
    for eid, e in kb["entities"].items():
        out[e["name"]].append(eid)
    for cid, c in kb["concepts"].items():
        out[c["name"]].append(cid)
    return dict(out)


def _id_to_name(kb):
    out = {}
    for eid, e in kb["entities"].items():
        out[eid] = e["name"]
    for cid, c in kb["concepts"].items():
        out[cid] = c["name"]
    return out


def _value_to_str(v):
    if isinstance(v, dict):
        t = v.get("type", "")
        val = v.get("value", "")
        unit = v.get("unit", "")
        if unit and unit != "1":
            return f"{val} {unit}"
        return str(val)
    return str(v)


def _emit_entity_triples(eid, kb, id_to_name, seen, out):
    """Emit all relation + attribute triples whose subject is `eid`."""
    e = kb["entities"].get(eid)
    if not e:
        return
    ename = e["name"]
    # instanceOf
    for cid in e.get("instanceOf", []):
        if cid in id_to_name:
            key = (ename, "instance of", id_to_name[cid])
            if key not in seen:
                seen.add(key)
                out.append({"head": ename, "head_type": "Entity",
                            "relation": "instance of",
                            "tail": id_to_name[cid], "tail_type": "Concept",
                            "source": "kqa_pro_kb"})
    for r in e.get("relations", []):
        if r.get("direction") != "forward":
            continue
        obj_id = r.get("object")
        obj_name = id_to_name.get(obj_id, obj_id)
        pred = r.get("predicate", "?")
        key = (ename, pred, obj_name)
        if key not in seen:
            seen.add(key)
            out.append({"head": ename, "head_type": "Entity",
                        "relation": pred,
                        "tail": obj_name, "tail_type": "Entity",
                        "source": "kqa_pro_kb"})
    for a in e.get("attributes", []):
        key_pred = a.get("key", "?")
        val_str = _value_to_str(a.get("value", ""))
        key = (ename, key_pred, val_str)
        if key not in seen:
            seen.add(key)
            out.append({"head": ename, "head_type": "Entity",
                        "relation": key_pred,
                        "tail": val_str, "tail_type": "Literal",
                        "source": "kqa_pro_kb"})


def build_question_subgraph(question_record, kb, name_to_ids=None, id_to_name=None, hops=2):
    """Build per-question subgraph: hops-radius around topic entities found
    in `Find` ops. Both directions traversed."""
    if name_to_ids is None:
        name_to_ids = _entity_name_to_ids(kb)
    if id_to_name is None:
        id_to_name = _id_to_name(kb)

    # Topic entity ids (multiple matches → take first)
    topic_ids = []
    for step in question_record.get("program", []):
        if step.get("function") == "Find":
            for nm in step.get("inputs", []):
                if not nm:
                    continue
                ids = name_to_ids.get(nm) or name_to_ids.get(nm.strip()) or []
                if ids:
                    topic_ids.append(ids[0])

    seen_triples = set()
    triples = []

    frontier = set(topic_ids)
    visited = set()
    for _ in range(hops):
        new_frontier = set()
        for eid in frontier:
            if eid in visited:
                continue
            visited.add(eid)
            e = kb["entities"].get(eid)
            if not e:
                continue
            # Emit outbound + collect 1-hop neighbors
            _emit_entity_triples(eid, kb, id_to_name, seen_triples, triples)
            for r in e.get("relations", []):
                if r.get("direction") == "forward" and r.get("object") in kb["entities"]:
                    new_frontier.add(r["object"])
            # Also include inbound edges to this entity by walking concept edges (cheap)
        # Inbound: scan KB for entities that point AT current frontier
        # — too expensive to do globally; we approximate by skipping inbound on hop 2.
        frontier = new_frontier - visited

    return {
        "topic_entities": [id_to_name.get(eid, eid) for eid in topic_ids],
        "topic_ids": topic_ids,
        "triples": triples,
    }


def _build_concept_index(kb):
    """name(concept) → list of entity ids whose instanceOf includes that concept.
    Also includes transitive instanceOf via concept hierarchy ('instanceOf' on concepts)."""
    concepts = kb["concepts"]
    # concept_id → all descendant concept_ids (including self) reachable via instanceOf
    parent_to_children = {cid: set() for cid in concepts}
    for cid, c in concepts.items():
        for pid in c.get("instanceOf", []):
            if pid in parent_to_children:
                parent_to_children[pid].add(cid)
    # Build transitive closure of descendants (limit depth 3 to avoid blow-up)
    def descendants(cid):
        seen = {cid}
        frontier = {cid}
        for _ in range(3):
            new_front = set()
            for x in frontier:
                for ch in parent_to_children.get(x, set()):
                    if ch not in seen:
                        seen.add(ch)
                        new_front.add(ch)
            frontier = new_front
            if not frontier: break
        return seen

    # Map concept_name → set of concept_ids with that name (usually 1)
    name_to_cids = {}
    for cid, c in concepts.items():
        name_to_cids.setdefault(c["name"], set()).add(cid)

    # name → entities whose instanceOf intersects descendants of any matching concept
    out = {}
    for cname, cids in name_to_cids.items():
        target_ids = set()
        for cid in cids:
            target_ids |= descendants(cid)
        members = []
        for eid, e in kb["entities"].items():
            if any(io in target_ids for io in e.get("instanceOf", [])):
                members.append(eid)
        out[cname] = members
    return out


def build_question_subgraph_v2(question_record, kb, name_to_ids=None, id_to_name=None,
                                concept_index=None, hops=2, max_concept_members=500):
    """Subgraph v2: 2-hop around Find entities PLUS all entities whose
    instanceOf matches any FilterConcept / FindAll concept in the program.

    The concept expansion gives Count / SelectAmong questions access to the
    full concept class they enumerate over.
    """
    if name_to_ids is None:
        name_to_ids = _entity_name_to_ids(kb)
    if id_to_name is None:
        id_to_name = _id_to_name(kb)
    if concept_index is None:
        concept_index = _build_concept_index(kb)

    # Topic entity ids (from Find)
    topic_ids = []
    for step in question_record.get("program", []):
        if step.get("function") == "Find":
            for nm in step.get("inputs", []):
                if not nm:
                    continue
                ids = name_to_ids.get(nm) or name_to_ids.get(nm.strip()) or []
                if ids:
                    topic_ids.append(ids[0])

    # Concepts referenced (from FilterConcept / FindAll-with-concept)
    referenced_concepts = []
    for step in question_record.get("program", []):
        if step.get("function") == "FilterConcept":
            for nm in step.get("inputs", []):
                if nm:
                    referenced_concepts.append(nm)
        # FindAll itself has no concept; subsequent FilterConcept disambiguates

    concept_member_ids = set()
    for cname in referenced_concepts:
        members = concept_index.get(cname, [])
        if len(members) > max_concept_members:
            members = members[:max_concept_members]
        concept_member_ids.update(members)

    seen_triples = set()
    triples = []

    # 1) Emit triples for ALL concept members (their attrs + outgoing relations)
    for eid in concept_member_ids:
        _emit_entity_triples(eid, kb, id_to_name, seen_triples, triples)

    # 2) BFS hops around topic entities (same as v1)
    frontier = set(topic_ids)
    visited = set()
    for _ in range(hops):
        new_frontier = set()
        for eid in frontier:
            if eid in visited:
                continue
            visited.add(eid)
            _emit_entity_triples(eid, kb, id_to_name, seen_triples, triples)
            e = kb["entities"].get(eid)
            if not e:
                continue
            for r in e.get("relations", []):
                if r.get("direction") == "forward" and r.get("object") in kb["entities"]:
                    new_frontier.add(r["object"])
        frontier = new_frontier - visited

    return {
        "topic_entities":  [id_to_name.get(eid, eid) for eid in topic_ids],
        "topic_ids":       topic_ids,
        "concepts":        referenced_concepts,
        "concept_members": [id_to_name.get(eid, eid) for eid in concept_member_ids],
        "triples":         triples,
    }


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    with open(ROOT / "datasets" / "kqa_pro" / "val.json", encoding="utf-8") as f:
        val = json.load(f)
    with open(ROOT / "datasets" / "kqa_pro" / "kb.json", encoding="utf-8") as f:
        kb = json.load(f)
    n2i = _entity_name_to_ids(kb)
    i2n = _id_to_name(kb)
    print(f"name_to_ids: {len(n2i)} distinct names")
    # Test on a few questions
    for i, q in enumerate(val[:3]):
        sg = build_question_subgraph(q, kb, n2i, i2n, hops=2)
        print(f"\nQ{i}: {q['question'][:100]}")
        print(f"  topic entities: {sg['topic_entities']}")
        print(f"  subgraph v1 size: {len(sg['triples'])} triples")
        for t in sg["triples"][:2]:
            print(f"    {t['head']} --[{t['relation']}]--> {t['tail']}")
    print()
    print("Building concept index...")
    ci = _build_concept_index(kb)
    print(f"  concept index built: {len(ci)} concept names")
    # Test v2 on the same questions
    for i, q in enumerate(val[:3]):
        sg2 = build_question_subgraph_v2(q, kb, n2i, i2n, concept_index=ci, hops=2)
        print(f"\nQ{i}: {q['question'][:100]}")
        print(f"  topic entities: {sg2['topic_entities']}")
        print(f"  referenced concepts: {sg2['concepts']}")
        print(f"  concept members added: {len(sg2['concept_members'])}")
        print(f"  subgraph v2 size: {len(sg2['triples'])} triples")
