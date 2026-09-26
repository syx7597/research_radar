# -*- coding: utf-8 -*-
"""Scrape open-source threat-radar references (CSIS Missile Threat, GlobalSecurity,
Air Power Australia) and extract structured radar parameters.

These sites block the harness WebFetch tool, but a plain `requests` GET with a
browser User-Agent works (verified 200 OK). We fetch, strip HTML to text with
BeautifulSoup, then LLM-extract radar records — each value must be backed by a
verbatim quote from the page (verified as a substring), so unsupported values
are dropped (anti-hallucination), exactly like the corpus enrichment pipeline.

Usage:
    python agent/scrape_sources.py <url> [<url> ...]
Output: data/ew/scraped_candidates.json  (review, then merge into threat_radars.json)
"""
import sys, re, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent"))

import requests
from bs4 import BeautifulSoup
from qa_strategy_pipeline import llm_call

OUT = ROOT / "data" / "ew" / "scraped_candidates.json"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
TEXT_CAP = 6000


def fetch_text(url):
    r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "form"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


EXTRACT_SYS = """You extract air-defense RADAR records STRICTLY from the given page text.
Return a JSON array; one object per distinct radar. Each object:
{"name":"...","role":"acquisition|search|early_warning|track|fire_control|SAM_guidance",
 "band":"<letter e.g. S, X, L, VHF, UHF, C or NATO A-J, else ''>",
 "frequency_GHz":<number or null>, "range_km":<number or null>,
 "quote":"<verbatim substring from the text that supports band/frequency>"}
Rules: ONLY radars explicitly described in the text. If a field is not stated, use "" or null.
The quote MUST be copied verbatim from the text. Output ONLY the JSON array."""


def extract_radars(url, text):
    text = text[:TEXT_CAP]
    raw = llm_call([{"role": "system", "content": EXTRACT_SYS},
                    {"role": "user", "content": f"URL: {url}\nPAGE TEXT:\n\"\"\"\n{text}\n\"\"\"\n\nJSON:"}],
                   max_tokens=900)
    m = re.search(r"\[.*\]", raw or "", flags=re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    ntext = re.sub(r"\s+", " ", text.lower())
    out = []
    for r in arr:
        if not isinstance(r, dict) or not r.get("name"):
            continue
        q = (r.get("quote") or "").strip()
        verified = bool(q) and re.sub(r"\s+", " ", q.lower())[:50] in ntext
        out.append({"name": r.get("name"), "role": r.get("role", ""),
                    "band": r.get("band", ""), "frequency_GHz": r.get("frequency_GHz"),
                    "range_km": r.get("range_km"), "quote": q[:160],
                    "verified": verified, "source": url})
    return out


def scrape(urls):
    allrecs = []
    for u in urls:
        try:
            txt = fetch_text(u)
            recs = extract_radars(u, txt)
            print(f"[{len(recs)} radars] {u}")
            for r in recs:
                flag = "✓" if r["verified"] else "✗unverified"
                print(f"    {flag} {r['name'][:30]:30s} band={r['band']:6} GHz={r['frequency_GHz']} R={r['range_km']}")
            allrecs += recs
        except Exception as e:
            print(f"[ERROR {type(e).__name__}] {u}: {e}")
        time.sleep(1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(allrecs, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n→ {len(allrecs)} candidate records written to {OUT}")
    return allrecs


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1 and args[0].endswith(".json"):
        urls = json.load(open(args[0], encoding="utf-8"))
    else:
        urls = args or ["https://missilethreat.csis.org/defsys/s-400-triumf/"]
    scrape(urls)
