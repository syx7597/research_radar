"""Source-only fixed retrieval and shared evidence/answer-contract tests."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.radar_domain import source_rag as rag


def chunk(chunk_id, text, title="Product source", source_id="source-a", entities=None):
    return {"source_id": source_id, "chunk_id": chunk_id, "title": title, "text": text,
            "entities": entities or [], "page_start": 1, "page_end": 1,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest()}


def claim(**changes):
    row = {"subject": "Example", "attribute": "frequency", "value_kind": "range", "value": None,
           "min_value": "2.5", "max_value": "3.5", "unit": "GHz", "condition": None,
           "event": None, "unknown_scope": None, "bound_inclusive": None,
           "citations": [{"source_id": "source-a", "chunk_id": "chunk-1"}]}
    row.update(changes)
    return row


class RadarSourceRagTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [chunk("chunk-1", "The frequency range is 2.5 to 3.5 GHz.")]
        self.evidence = rag.evidence_for_retrieval(self.chunks)

    def parsed(self, claims, status="supported", text="依据原文给出范围及单位。"):
        return rag.parse_answer(json.dumps({"answer_text": text, "evidence_status": status,
                                           "claims": claims}, ensure_ascii=False), self.evidence)

    def test_mixed_tokenization_preserves_decimal_alias_and_microsecond_terms(self):
        tokens = rag.tokenize("脉冲宽度 AN/APY-9：0.2 μs；9.0–9.16 GHz")
        for term in ("脉", "冲", "脉冲", "冲宽", "宽度", "an/apy-9", "0.2", "us", "9.0", "9.16", "ghz"):
            self.assertIn(term, tokens)
        terms = rag.query_terms("某型号的脉冲重复频率和波束宽度是什么？")
        for term in ("pulse", "repetition", "frequency", "prf", "beam", "width", "beamwidth"):
            self.assertIn(term, terms)
        self.assertEqual(len(terms), len(set(terms)))

    def test_bm25_is_fixed_global_top_four_with_deterministic_ties(self):
        documents = [chunk(f"chunk-{number}", "frequency") for number in reversed(range(6))]
        index = rag.BM25Index(documents)
        rows = index.retrieve("frequency")
        self.assertEqual([row["chunk_id"] for row in rows], ["chunk-0", "chunk-1", "chunk-2", "chunk-3"])
        expected = math.log(1 + 0.5 / 6.5)
        # Each identical document contains exactly one matching token; BM25 tf=1.
        self.assertAlmostEqual(rows[0]["bm25_score"], expected)
        self.assertEqual(index.retrieve("zzznomatch"), [])
        english = rag.BM25Index([chunk("power", "Peak transmitter power is specified."),
                                chunk("frequency", "Frequency and band are specified.")])
        self.assertEqual(english.retrieve("功率是多少？")[0]["chunk_id"], "power")

    def test_index_reads_only_explicit_chunk_file_and_ignores_entity_metadata(self):
        documents = [chunk("chunk-1", "plain source text", entities=["SECRETALIAS"])]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "chunks.jsonl"
            source.write_text(json.dumps(documents[0]) + "\n")
            real_read = Path.read_text
            opened = []
            def guarded(path, *args, **kwargs):
                self.assertEqual(path, source)
                opened.append(path)
                return real_read(path, *args, **kwargs)
            with patch.object(Path, "read_text", guarded):
                index = rag.BM25Index.from_path(source)
                self.assertEqual(index.retrieve("SECRETALIAS"), [])
            self.assertEqual(opened, [source])
        with self.assertRaisesRegex(ValueError, "fixed source-text schema"):
            rag.BM25Index([{**documents[0], "expected_fact_ids": ["gold"]}])
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            rag.BM25Index([{**documents[0], "text": "tampered"}])

    def test_record_mapping_strips_answers_and_has_same_prompt_as_rag(self):
        records = [{"fact_id": "do-not-expose", "value": "secret-value", "scope_notes_zh": "answer-hint",
                    "citation": {"source_id": "source-a", "chunk_id": "chunk-1"}}] * 2
        mapped = rag.evidence_for_records(records, self.chunks)
        self.assertEqual(mapped["mapping"], {"record_count": 2, "unique_chunk_count": 1, "truncated_chunk_count": 0})
        by_rag = rag.evidence_for_retrieval([{**self.chunks[0], "bm25_score": 123.0}])
        first = rag.answer_messages("频率？", mapped)
        self.assertEqual(first, rag.answer_messages("频率？", by_rag))
        for forbidden in ("do-not-expose", "secret-value", "answer-hint", "bm25_score", "record_count"):
            self.assertNotIn(forbidden, json.dumps(first, ensure_ascii=False))
        with self.assertRaisesRegex(ValueError, "no bound"):
            rag.evidence_for_records([{"citation": {"source_id": "other", "chunk_id": "chunk-1"}}], self.chunks)

    def test_record_mapping_uses_fixed_chunk_sort_and_reports_truncation(self):
        chunks = [chunk(f"chunk-{i}", "source") for i in reversed(range(6))]
        records = [{"citation": {"source_id": row["source_id"], "chunk_id": row["chunk_id"]}} for row in chunks]
        evidence = rag.evidence_for_records(records, chunks)
        self.assertEqual([row["chunk_id"] for row in evidence["chunks"]], ["chunk-0", "chunk-1", "chunk-2", "chunk-3"])
        self.assertEqual(evidence["mapping"]["truncated_chunk_count"], 2)
        self.assertEqual(rag.evidence_for_records([], chunks)["chunks"], [])

    def test_claim_shapes_preserve_ranges_discrete_options_and_unknown_scope(self):
        self.assertEqual(self.parsed([claim()])["claims"][0]["max_value"], "3.5")
        options = claim(value_kind="options", value=["0.2", "0.8"], min_value=None, max_value=None, unit="us")
        self.assertEqual(self.parsed([options])["claims"][0]["value"], ["0.2", "0.8"])
        unknown = claim(value_kind="unknown", min_value=None, max_value=None,
                        unknown_scope="cited_source_chunk", unit=None)
        self.assertEqual(self.parsed([unknown])["claims"][0]["unknown_scope"], "cited_source_chunk")
        for invalid in (claim(min_value="4", max_value="3"), claim(value_kind="scalar", value="2", min_value="1"),
                        {**unknown, "unknown_scope": "entire_database"}, {**unknown, "value": 0},
                        {**options, "value": ["0.2"]}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.parsed([invalid])

    def test_one_sided_strict_bounds_and_approximation_remain_explicit(self):
        lower = claim(value_kind="lower_bound", min_value="60", max_value=None, bound_inclusive=False)
        upper = claim(value_kind="upper_bound", min_value=None, max_value="5", bound_inclusive=True)
        approximate = claim(value_kind="approximate", value="3.0", min_value=None, max_value=None)
        for row in (lower, upper, approximate):
            self.assertEqual(self.parsed([row])["claims"][0], row)
        for wrong in ({**lower, "bound_inclusive": None}, {**upper, "min_value": "0"},
                      {**approximate, "bound_inclusive": True}, {**lower, "value": "60"}):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                self.parsed([wrong])

    def test_claim_citations_and_finite_values_are_checked_but_truth_is_not_claimed(self):
        for invalid in (claim(citations=[]), claim(citations=[{"source_id": "absent", "chunk_id": "chunk-1"}]),
                        claim(value_kind="scalar", value="NaN", min_value=None, max_value=None),
                        claim(value_kind="scalar", value=True, min_value=None, max_value=None),
                        {**claim(), "fact_id": "forbidden"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.parsed([invalid])
        # Membership alone cannot detect a false read of a real citation: this is
        # intentionally accepted mechanically and must receive separate review.
        changed = claim(min_value="999", max_value="1000")
        self.assertEqual(self.parsed([changed])["claims"][0]["min_value"], "999")

    def test_empty_evidence_allows_only_explicit_insufficiency_and_no_claims(self):
        evidence = {"chunks": []}
        base = {"answer_text": "所给证据不足，无法确定。", "evidence_status": "insufficient_evidence", "claims": []}
        self.assertEqual(rag.parse_answer(json.dumps(base), evidence), base)
        for wrong in ({**base, "claims": [claim()]}, {**base, "evidence_status": "supported"}):
            with self.assertRaisesRegex(ValueError, "Empty evidence"):
                rag.parse_answer(json.dumps(wrong), evidence)
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            rag.parse_answer('{"answer_text":"a","answer_text":"b"}', evidence)
        with self.assertRaises(ValueError):
            rag.parse_answer('```json\n{}\n```', evidence)
        with self.assertRaisesRegex(ValueError, "Natural-language"):
            self.parsed([claim()], text="")


if __name__ == "__main__":
    unittest.main()
