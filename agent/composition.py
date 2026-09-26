# -*- coding: utf-8 -*-
"""Typed operator-composition executor — the analytical core of the radar agent.

The LLM (planner.py) emits a JSON *plan tree* of typed operator nodes; this module
evaluates it *deterministically* by recursing over the tree, reusing the paper's
six retrieval operators (via KGIndex + the bilingual alias layer + equivalence-class
fallback). Entities flow inside the tree as Python sets — never re-transcribed by an
LLM between steps — so there is no entity-name drift and every step is auditable.

Value types flowing through the tree:
  EntitySet  : set[str]              (operator outputs over the KG)
  Scalar     : int | float           (count / aggregate)
  Bool       : bool                  (membership / comparison)
  Compare    : dict                  (dual-subgraph: same / only_a / only_b)

Node schemas (JSON):
  {"op":"constraint", "relation": "<rel>", "value": "<tail>"}            -> EntitySet
  {"op":"intersect",  "args": [node, node, ...]}                          -> EntitySet
  {"op":"union",      "args": [node, ...]}                                -> EntitySet
  {"op":"difference", "args": [node_a, node_b]}                           -> EntitySet  (a - b)
  {"op":"path",       "start": "<entity>", "chain": ["r1", "r2^-1", ...]} -> EntitySet
  {"op":"count",      "arg": node}                                        -> Scalar
  {"op":"enumerate",  "arg": node}                                        -> EntitySet  (passthrough, for listing)
  {"op":"aggregate",  "func":"avg|sum|max|min|count", "attribute":"<rel>", "over": node} -> Scalar
  {"op":"compare",    "a":"<e>", "b":"<e>", "relation":"<rel>"}           -> Compare
  {"op":"contains",   "set": node, "member":"<entity>"}                   -> Bool
"""
import os, re, sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qa_strategy_pipeline import KGIndex, _heads_for_constraint, _alias_resolve_tail
from lexicon import build_alias_index


class PlanError(Exception):
    pass


class CompositionExecutor:
    def __init__(self, kg: KGIndex, aliases: dict | None = None):
        self.kg = kg
        self.aliases = aliases if aliases is not None else build_alias_index()

    # ---- leaf helpers (reuse the paper's operators) ----
    def _constraint_heads(self, relation: str, value: str) -> set:
        """Heads satisfying (?, relation, value), with alias + equivalence-class
        fallback — exactly the paper's constrained-join leaf resolution."""
        heads, info = _heads_for_constraint({"relation": relation, "tail": value},
                                            self.kg, self.aliases)
        return set(heads), info

    def _resolve_entity(self, value: str) -> str:
        """Pick the KG-canonical surface form for a start entity."""
        for cand in _alias_resolve_tail(value, self.aliases):
            # an entity is usable as a path start if it appears as a head or tail
            if any(k[0] == cand for k in self.kg.by_head_relation) or \
               any(k[1] == cand for k in self.kg.by_relation_tail):
                return cand
        return value

    def _walk(self, start: str, chain: list) -> tuple[set, list]:
        current = {self._resolve_entity(start)}
        steps = []
        for rel in chain:
            inverse = rel.endswith("^-1")
            r = rel[:-3].strip() if inverse else rel
            nxt = set()
            for v in current:
                nxt |= (self.kg.heads_with(r, v) if inverse else self.kg.tails_of(v, r))
            steps.append({"relation": rel, "in": len(current), "out": len(nxt)})
            current = nxt
            if not current:
                break
        return current, steps

    def distinct_tails(self, relation: str, limit: int = 25) -> list:
        """Actual values a relation takes in the KG — used by the reflection node
        to repair grounding errors (e.g. value '在役' → KG uses '服役中')."""
        vals = []
        for (r, t), heads in self.kg.by_relation_tail.items():
            if r == relation and heads:
                vals.append((len(heads), t))
        vals.sort(reverse=True)
        return [t for _, t in vals[:limit]]

    def _numeric(self, entity: str, attribute: str) -> list:
        out = []
        for t in self.kg.tails_of(entity, attribute):
            m = re.search(r"-?\d+(?:\.\d+)?", str(t))
            if m:
                out.append(float(m.group(0)))
        return out

    # ---- recursive evaluator ----
    def evaluate(self, node: dict) -> tuple:
        """Return (value, trace). trace is a nested dict recording each step."""
        if not isinstance(node, dict) or "op" not in node:
            raise PlanError(f"bad node: {node!r}")
        op = node["op"]
        m = getattr(self, f"_op_{op}", None)
        if m is None:
            raise PlanError(f"unknown op: {op}")
        return m(node)

    def _eval_set(self, node) -> tuple[set, dict]:
        v, tr = self.evaluate(node)
        if not isinstance(v, set):
            raise PlanError(f"expected EntitySet from {node.get('op')}, got {type(v).__name__}")
        return v, tr

    # --- operators ---
    def _op_constraint(self, n):
        heads, info = self._constraint_heads(n["relation"], n["value"])
        return heads, {"op": "constraint", "relation": info["relation"],
                       "value": n["value"], "n": len(heads), "sample": sorted(heads)[:5]}

    def _op_intersect(self, n):
        sets, trs = [], []
        for a in n["args"]:
            s, t = self._eval_set(a); sets.append(s); trs.append(t)
        res = set.intersection(*sets) if sets else set()
        return res, {"op": "intersect", "n": len(res), "args": trs, "sample": sorted(res)[:8]}

    def _op_union(self, n):
        res, trs = set(), []
        for a in n["args"]:
            s, t = self._eval_set(a); res |= s; trs.append(t)
        return res, {"op": "union", "n": len(res), "args": trs, "sample": sorted(res)[:8]}

    def _op_difference(self, n):
        a, ta = self._eval_set(n["args"][0]); b, tb = self._eval_set(n["args"][1])
        res = a - b
        return res, {"op": "difference", "n": len(res), "args": [ta, tb], "sample": sorted(res)[:8]}

    def _op_path(self, n):
        res, steps = self._walk(n["start"], n.get("chain", []))
        return res, {"op": "path", "start": n["start"], "chain": n.get("chain", []),
                     "n": len(res), "steps": steps, "sample": sorted(res)[:8]}

    def _op_count(self, n):
        s, t = self._eval_set(n["arg"])
        return len(s), {"op": "count", "value": len(s), "arg": t}

    def _op_enumerate(self, n):
        s, t = self._eval_set(n["arg"])
        return s, {"op": "enumerate", "n": len(s), "items": sorted(s), "arg": t}

    def _op_aggregate(self, n):
        s, t = self._eval_set(n["over"])
        attr, func = n["attribute"], n["func"].lower()
        vals, covered = [], 0
        for e in s:
            ev = self._numeric(e, attr)
            if ev:
                covered += 1
                vals.append(sum(ev) / len(ev))   # avg if an entity has multiple values
        if func == "count":
            res = len(vals)
        elif not vals:
            res = None
        elif func == "avg":
            res = sum(vals) / len(vals)
        elif func == "sum":
            res = sum(vals)
        elif func == "max":
            res = max(vals)
        elif func == "min":
            res = min(vals)
        else:
            raise PlanError(f"bad aggregate func: {func}")
        return res, {"op": "aggregate", "func": func, "attribute": attr,
                     "value": res, "set_n": len(s), "covered": covered,
                     "coverage": f"{covered}/{len(s)} 个实体有该属性", "over": t}

    def _op_compare(self, n):
        a, b, r = n["a"], n["b"], n["relation"]
        ca, cb = self._resolve_entity(a), self._resolve_entity(b)
        sa = self.kg.tails_of(ca, r); sb = self.kg.tails_of(cb, r)
        res = {"a": a, "b": b, "relation": r,
               "a_values": sorted(sa), "b_values": sorted(sb),
               "same": sorted(sa & sb), "only_a": sorted(sa - sb), "only_b": sorted(sb - sa),
               "identical": sa == sb and len(sa) > 0}
        return res, {"op": "compare", **{k: res[k] for k in ("a", "b", "relation", "same", "only_a", "only_b", "identical")}}

    def _op_contains(self, n):
        s, t = self._eval_set(n["set"])
        cands = set(_alias_resolve_tail(n["member"], self.aliases))
        hit = bool(s & cands) or n["member"] in s
        return hit, {"op": "contains", "member": n["member"], "in_set": hit, "set_n": len(s), "arg": t}


def is_degenerate(value, trace) -> bool:
    """A result is 'degenerate' (likely a grounding/plan error, not a true empty
    answer) when the operators found no input grounding."""
    op = trace.get("op")
    if value is None:
        return True
    if isinstance(value, set):
        return len(value) == 0
    if op == "count":
        return value == 0
    if op == "contains":
        return trace.get("set_n", 0) == 0          # member-check over an empty set = no evidence
    if op == "compare":
        return not trace.get("same") and not trace.get("only_a") and not trace.get("only_b")
    if op == "aggregate":
        return value is None or trace.get("covered", 0) == 0
    return False


def empty_leaves(trace) -> list:
    """Collect (relation, value) constraint leaves that returned 0 — the grounding
    failures the reflection node should repair. Also flags empty path starts."""
    out = []
    def walk(t):
        if not isinstance(t, dict):
            return
        if t.get("op") == "constraint" and t.get("n", 1) == 0:
            out.append({"kind": "constraint", "relation": t.get("relation"), "value": t.get("value")})
        if t.get("op") == "path" and t.get("n", 1) == 0:
            out.append({"kind": "path", "start": t.get("start"), "chain": t.get("chain")})
        for k in ("arg", "over"):
            if k in t:
                walk(t[k])
        for k in ("args",):
            for c in t.get(k, []):
                walk(c)
    walk(trace)
    return out


def load_executor(triples_path=None):
    import json
    triples_path = triples_path or (ROOT / "graphrag_index" / "merged_triples.json")
    triples = json.load(open(triples_path, encoding="utf-8"))
    return CompositionExecutor(KGIndex(triples))
