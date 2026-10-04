#!/usr/bin/env python3
"""Archive the three already reviewed institutional URLs for human inspection.

Full HTML/text stay in ignored data/. Public metadata contains only the existing
short quotes and locators. This creates no accepted facts or evaluation answers.
Network is opt-in via --fetch; default verifies the saved capture, without IO writes.
"""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "artifacts/thesis_direction_review/radar_primary_screening_001_003/ai_screening.json"
DEST = ROOT / "data/radar_review/primary_capture_001_003_v1"
MANIFEST = ROOT / "artifacts/thesis_direction_review/radar_primary_screening_001_003/capture_manifest_v1.json"
ALLOWED = {
    "primary-dsca-18-51": "https://samm.dsca.mil/policy-memoranda/dsca-18-51",
    "primary-redstone-1981": "https://history.redstone.army.mil/ihist-1981.html",
    "primary-redstone-1984": "https://history.redstone.army.mil/ihist-1984.html",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class PageText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)

    def text(self):
        return " ".join(" ".join(self.parts).split()) + "\n"


def local(relative):
    candidate = (ROOT / relative).resolve()
    if not candidate.is_relative_to(DEST.resolve()):
        raise ValueError("Capture path outside fixed data directory")
    return candidate


def verify():
    manifest = json.loads(MANIFEST.read_text())
    if manifest["screening_sha256"] != sha(SCREEN.read_bytes()):
        raise ValueError("Screening version mismatch")
    screen = {s["source_id"]: s for s in json.loads(SCREEN.read_text())["sources"]}
    if {s["source_id"] for s in manifest["sources"]} != set(ALLOWED):
        raise ValueError("Capture source set mismatch")
    for source in manifest["sources"]:
        source_id = source["source_id"]
        if source["requested_url"] != ALLOWED[source_id]:
            raise ValueError("Source URL mismatch")
        if source["capture_status"] != "captured":
            if source["capture_status"] != "unavailable" or source["accepted_as_fact"]:
                raise ValueError("Unexpected failed-capture status")
            continue
        for key in ("html", "text"):
            data = local(source[key]["file"]).read_bytes()
            if sha(data) != source[key]["sha256"] or len(data) != source[key]["bytes"]:
                raise ValueError("Source bytes changed")
        text = local(source["text"]["file"]).read_text()
        quote = source["evidence_text"]
        if quote != screen[source_id]["short_quote"]:
            raise ValueError("Quote changed from screened source")
        if source["quote_binding_verified"] != bool(source["quote_occurrences"]):
            raise ValueError("Quote binding flag mismatch")
        for loc in source["quote_occurrences"]:
            if text[loc["char_start"]:loc["char_end"]] != quote:
                raise ValueError("Quote locator mismatch")
        parser = PageText()
        parser.feed(local(source["html"]["file"]).read_bytes().decode(source["encoding"]))
        if parser.text() != text:
            raise ValueError("Text is not a reproducible HTML rendition")
    return {"verified_source_files": sum(s["capture_status"] == "captured" for s in manifest["sources"]),
            "located_screened_quotes": sum(s.get("quote_binding_verified", False) for s in manifest["sources"]),
            "unavailable_sources": sum(s["capture_status"] == "unavailable" for s in manifest["sources"]), "human_accepted": 0,
            "fact_values_changed": False, "final_qa_answers_generated": False}


def fetch():
    if MANIFEST.exists() or DEST.exists():
        raise FileExistsError("Capture already exists; verify it or create an explicit new version")
    screening = json.loads(SCREEN.read_text())
    staged, entries = {}, []
    for source in screening["sources"]:
        source_id = source["source_id"]
        url = ALLOWED[source_id]
        if source["url"] != url:
            raise ValueError("Screened URL changed")
        # System curl uses the operating system trust store (the local Python
        # installation has no default CA bundle). TLS verification stays on.
        result = subprocess.run([
            "/usr/bin/curl", "--fail", "--location", "--silent", "--show-error",
            "--max-time", "35", "--max-filesize", "2000000", "--proto", "=https",
            "--proto-redir", "=https", "--user-agent", "Mozilla/5.0 (academic source verification)",
            "--write-out", "\n%{content_type}\n%{url_effective}", url,
        ], check=False, capture_output=True)
        if result.returncode:
            entries.append({"source_id": source_id, "title": source["title"], "requested_url": url,
                            "capture_status": "unavailable", "curl_exit_code": result.returncode,
                            "error": result.stderr.decode("utf-8", errors="replace").strip()[:500],
                            "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
                            "human_review_status": "pending", "accepted_as_fact": False})
            continue
        data, content_type, final_url_bytes = result.stdout.rsplit(b"\n", 2)
        content_type = content_type.decode("ascii")
        if len(data) > 2_000_000 or not content_type.startswith("text/html"):
            raise ValueError("Unexpected document size or type")
        charset = re.search(r"charset=([^; ]+)", content_type)
        encoding = charset.group(1) if charset else "utf-8"
        final_url = final_url_bytes.decode("utf-8")
        parser = PageText()
        parser.feed(data.decode(encoding))
        text = parser.text()
        quote = source["short_quote"]
        matches = list(re.finditer(re.escape(quote), text))
        html_rel = str((DEST / f"{source_id}.html").relative_to(ROOT))
        text_rel = str((DEST / f"{source_id}.txt").relative_to(ROOT))
        text_bytes = text.encode("utf-8")
        staged[html_rel], staged[text_rel] = data, text_bytes
        locators = [{"char_start": m.start(), "char_end": m.end(),
                     "offset_unit": "unicode_codepoint", "end_exclusive": True} for m in matches]
        entries.append({"source_id": source_id, "title": source["title"], "capture_status": "captured",
                        "requested_url": url, "resolved_url": final_url, "encoding": encoding,
                        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                        "html": {"file": html_rel, "sha256": sha(data), "bytes": len(data)},
                        "text": {"file": text_rel, "sha256": sha(text_bytes), "bytes": len(text_bytes)},
                        "evidence_text": quote, "quote_occurrences": locators,
                        "quote_binding_verified": bool(matches),
                        "capture_note": "Exact screened quote located" if matches else "Downloaded HTML does not contain the screened quote; may be an access page or changed content. Do not use for fact acceptance.",
                        "locator_note": source["locator"],
                        "supports": source["supports"], "does_not_establish": source["does_not_establish"],
                        "human_review_status": "pending", "accepted_as_fact": False})
    DEST.mkdir(parents=True)
    for relative, data in staged.items():
        (ROOT / relative).write_bytes(data)
    report = {"version": "radar_primary_capture_001_003_v1", "screening_sha256": sha(SCREEN.read_bytes()),
              "capture_script_sha256": sha(Path(__file__).read_bytes()), "sources": entries,
              "source_classification": "AI-proposed institutional primary sources, human verification pending",
              "rendition": "HTMLParser text nodes excluding script/style; whitespace normalized; HTML retained",
              "scope": "Source and event attribution review only; not replacement radar facts or final QA gold",
              "public_fulltext_included": False, "accepted_records": 0,
              "final_qa_answers_generated": False, "original_packet_modified": False}
    MANIFEST.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        fetch()
    print(json.dumps(verify(), ensure_ascii=False))
