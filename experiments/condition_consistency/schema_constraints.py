"""KB role-field token tries for the existing flat KoPL serialization.

This is a conventional constrained-decoding baseline, not a new KGQA method.
No model, gold program, answer, execution result or question enters the rule.
Only attribute/relation/qualifier arguments are constrained. Other arguments,
function names and program grammar still need the frozen executor's validation.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy


FIELD_ROLES = {
    **{f"Filter{x}": {0: "attribute"} for x in ("Str", "Num", "Year", "Date")},
    **{f"QFilter{x}": {0: "qualifier"} for x in ("Str", "Num", "Year", "Date")},
    "Relate": {0: "relation"},
    "SelectBetween": {0: "attribute"}, "SelectAmong": {0: "attribute"},
    "QueryAttr": {0: "attribute"},
    "QueryAttrUnderCondition": {0: "attribute", 1: "qualifier"},
    "QueryAttrQualifier": {0: "attribute", 2: "qualifier"},
    "QueryRelationQualifier": {0: "relation", 1: "qualifier"},
}
ROLES = ("attribute", "relation", "qualifier")
KNOWN_FUNCTIONS = frozenset(FIELD_ROLES) | {
    "FindAll", "Find", "FilterConcept", "And", "Or", "What", "Count",
    "VerifyStr", "VerifyNum", "VerifyYear", "VerifyDate", "QueryRelation",
}
_END = -1


class ConstraintPrefixError(ValueError):
    """A recognized field has no legal continuation; never silently fail open."""


class SchemaFieldConstraint:
    """Stateless per-beam Hugging Face ``prefix_allowed_tokens_fn`` adapter.

    Tries cover the checkpoint tokenizer's canonical encoding of each KB field,
    with zero or one ASCII space on either side. They do not enumerate every
    possible BPE segmentation or arbitrary whitespace. Prefix inspection uses
    the complete beam prefix; the bounded cache is only an optimization.
    """

    def __init__(self, tokenizer, names_by_role, decoder_start_token_id=2):
        self.tokenizer = tokenizer
        self.decoder_start_token_id = decoder_start_token_id
        self.all_tokens = sorted(set(tokenizer.get_vocab().values()))
        self.eos = tokenizer.eos_token_id
        self.bos = tokenizer.bos_token_id
        self.pad = tokenizer.pad_token_id
        self.markers = {}
        for marker in ("<func>", "<arg>"):
            ids = tokenizer.encode(marker, add_special_tokens=False)
            if len(ids) != 1 or marker in tokenizer.all_special_tokens:
                raise ValueError(f"Expected a single ordinary added token: {marker}")
            self.markers[marker] = ids[0]
        if self.eos is None or self.markers["<func>"] == self.markers["<arg>"]:
            raise ValueError("Distinct KoPL markers and an EOS token are required")
        self.closers = {self.eos, *self.markers.values()}
        self.tries = {}
        self.field_counts = {}
        self.node_counts = {}
        self._function_cache = OrderedDict()
        for role in ROLES:
            names = sorted(set(names_by_role[role]))
            if not names:
                raise ValueError(f"Empty KB vocabulary for {role}")
            root = {}
            nodes = 1
            for name in names:
                if not name or name != name.strip() or any(m in name for m in self.markers):
                    raise ValueError(f"Field cannot be represented unambiguously: {name!r}")
                for before in ("", " "):
                    for after in ("", " "):
                        ids = tokenizer.encode(before + name + after, add_special_tokens=False)
                        if not ids or any(i in self.closers or i in tokenizer.all_special_ids for i in ids):
                            raise ValueError(f"Special/delimiter token inside KB field: {name!r}")
                        if tokenizer.decode(ids, skip_special_tokens=False,
                                            clean_up_tokenization_spaces=False) != before + name + after:
                            raise ValueError(f"Tokenizer does not preserve KB field: {name!r}")
                        node = root
                        for token in ids:
                            if token not in node:
                                node[token] = {}
                                nodes += 1
                            node = node[token]
                        node[_END] = True
            self.tries[role] = root
            self.field_counts[role] = len(names)
            self.node_counts[role] = nodes

    def _function(self, tokens):
        key = tuple(tokens)
        if key not in self._function_cache:
            self._function_cache[key] = self.tokenizer.decode(
                tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            if len(self._function_cache) > 4096:
                self._function_cache.popitem(last=False)
        return self._function_cache[key]

    def inspect(self, prefix_ids):
        """Describe the next-token restriction without using inference labels.

        ``allowed=None`` means unconstrained; an empty list means a hard failure.
        An EOS is a terminator except at the initial decoder-start position.
        Inspection is intended for prefixes reached by the callback; offline
        traces must be checked at *every* position to catch earlier violations.
        """
        ids = list(prefix_ids)
        state = dict(mode="free", function=None, argument=None, role=None,
                     allowed=None, reason="before_argument")
        if ids and ids[0] == self.decoder_start_token_id:
            ids = ids[1:]
        if ids and ids[0] == self.bos:
            ids = ids[1:]
        # Final baseline decoding drops specials. Use exactly those semantics
        # for function recognition and reject delimiter text assembled from
        # ordinary BPE pieces. Otherwise a lexical delimiter could be invisible
        # to the token-ID slot parser. This is a hard unsupported-path failure,
        # not an unrestricted fallback (normal added-token paths are unchanged).
        text = self.tokenizer.decode(ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
        if any(text.count(marker) != ids.count(token) for marker, token in self.markers.items()):
            return {**state, "allowed": [], "reason": "non_atomic_delimiter"}
        if self.eos in ids:
            tail = ids[ids.index(self.eos) + 1:]
            valid_tail = all(token == self.pad for token in tail)
            return {**state, "mode": "ended", "allowed": [self.pad] if valid_tail else [],
                    "reason": "after_eos" if valid_tail else "tokens_after_eos"}
        func_id, arg_id = self.markers["<func>"], self.markers["<arg>"]
        last_func = max((i for i, token in enumerate(ids) if token == func_id), default=-1)
        step = ids[last_func + 1:]
        args = [i for i, token in enumerate(step) if token == arg_id]
        if not args:
            return state
        function = self._function(step[:args[0]])
        argument = len(args) - 1
        role = FIELD_ROLES.get(function, {}).get(argument)
        state.update(function=function, argument=argument, role=role)
        if role is None:
            state["reason"] = "non_field_argument" if function in KNOWN_FUNCTIONS else "unknown_function"
            return state
        state["mode"] = "field"
        node = self.tries[role]
        for token in step[args[-1] + 1:]:
            if token not in node:
                return {**state, "allowed": [], "reason": "unmatched_field_prefix"}
            node = node[token]
        allowed = set(node) - {_END}
        if _END in node:
            allowed.update(self.closers)
        return {**state, "allowed": sorted(allowed), "reason": "kb_field_trie"}

    def __call__(self, batch_id, input_ids):
        # Do not key state by batch_id: beam search reorders and duplicates beams.
        ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        state = self.inspect(ids)
        if state["allowed"] == []:
            raise ConstraintPrefixError(
                f"{state['reason']}: function={state['function']}, argument={state['argument']}")
        return self.all_tokens if state["allowed"] is None else state["allowed"]

    def generation_kwargs(self, generation_config):
        """Return a ready-to-pass generate adapter, rejecting forced-EOS conflict.

        The caller must explicitly disable forced EOS for BOTH experimental arms
        and record this protocol change. A maximum-length unfinished program must
        remain a truncation failure, not a silently accepted completed field.
        This method neither calls a model nor changes the frozen baseline config.
        """
        if getattr(generation_config, "forced_eos_token_id", None) is not None:
            raise ValueError("Disable forced_eos_token_id explicitly in both comparison arms")
        return {"generation_config": deepcopy(generation_config),
                "prefix_allowed_tokens_fn": self}
