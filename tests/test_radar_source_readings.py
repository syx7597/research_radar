"""Provenance failure cases; these checks do not establish semantic gold."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.radar_domain import source_readings as sr


class ReadingChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.local = self.root / "local"
        self.local.mkdir()
        self.source = self.root / "original.pdf"
        self.source.write_bytes(b"%PDF-fixture")
        self.derived = self.root / "page.txt"
        self.derived.write_text("Model A and B\nPower 10 W; standby 2 W")
        text = sr.compact(self.derived.read_text())
        self.chunk = {"chunk_id": "s:p001", "family_id": "f", "source_id": "s",
                      "page_one_based": 1, "source_path": "original.pdf",
                      "source_sha256": sr.sha(self.source), "source_uri": "https://example.test/fixture",
                      "derived_path": "page.txt", "derived_sha256": sr.sha(self.derived),
                      "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text}
        self.manifest = {"schema": "source_reading_chunks_v1", "inputs_sha256": {},
                         "normalization": "collapse_whitespace_only", "chunks": [self.chunk]}
        sr.write_new(self.local / "chunks.json", self.manifest)
        self.patchers = [patch.object(sr, "ROOT", self.root), patch.object(sr, "LOCAL", self.local),
                         patch.object(sr, "build_chunks", return_value=copy.deepcopy(self.manifest))]
        for p in self.patchers:
            p.start()

    def tearDown(self):
        for p in reversed(self.patchers):
            p.stop()
        self.tmp.cleanup()

    def packet(self):
        p = sr.Packet("fixture AI", "T")
        p.add("s", 1, ["A", "B"], "radar", "power", "scalar", "2 W", "W",
              [["mode", "standby"]], ["standby 2 W"], "power table")
        p.save("draft.json", [{"chunk_id": "s:p001", "scope_review": "fixture table"}])
        return json.loads((self.local / "draft.json").read_text())

    def test_repeated_quote_requires_explicit_occurrence(self):
        with self.assertRaises(ValueError):
            sr.spans("mode 2 W; another 2 W", ["2 W"])
        self.assertEqual(sr.spans("mode 2 W; another 2 W", [("2 W", 1)])[0]["start"], 18)

    def test_removed_page_not_hidden_by_recomputed_text_hash(self):
        d = copy.deepcopy(self.manifest)
        d["chunks"] = []
        (self.local / "chunks.json").write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError, "declared source/page scope"):
            sr.load_chunks()

    def test_changed_original_rejected_even_when_text_unchanged(self):
        self.source.write_bytes(b"%PDF-different")
        with self.assertRaisesRegex(ValueError, "archived bytes"):
            sr.load_chunks()

    def test_shared_subjects_do_not_multiply_evidence(self):
        self.packet()
        a = sr.audit([self.local / "draft.json"])
        self.assertEqual(a["counts"]["readings"], 1)
        self.assertFalse(a["model_run_ready"])

    def test_changed_span_and_wrong_family_rejected(self):
        d = self.packet()
        for change in ("span", "family"):
            modified = copy.deepcopy(d)
            if change == "span":
                modified["records"][0]["citation"]["spans"][0]["start"] += 1
            else:
                modified["records"][0]["family_id"] = "unrelated"
            with self.assertRaises(ValueError):
                sr.validate(modified, {"s:p001": self.chunk})

    def test_coverage_cannot_claim_unannotated_page(self):
        d = self.packet()
        d["coverage"].append({"chunk_id": "s:p002", "scope_review": "claimed"})
        with self.assertRaisesRegex(ValueError, "Coverage"):
            sr.validate(d, {"s:p001": self.chunk})

    def test_preserve_author_draft_and_reject_path_escape(self):
        self.packet()
        p = sr.Packet("fixture AI", "T")
        with self.assertRaisesRegex(ValueError, "basename"):
            p.save("../outside.json", [])
        with self.assertRaises(FileExistsError):
            sr.write_new(self.local / "draft.json", {})

    def test_same_packet_cannot_be_counted_twice(self):
        self.packet()
        with self.assertRaisesRegex(ValueError, "Duplicate IDs or page ownership"):
            sr.audit([self.local / "draft.json", self.local / "draft.json"])


if __name__ == "__main__":
    unittest.main()
