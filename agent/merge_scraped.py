# -*- coding: utf-8 -*-
"""Merge scraped radar candidates into the threat-radar library.

Takes data/ew/scraped_candidates.json (from scrape_sources.py), keeps only
quote-verified records with usable data, maps role->purpose / band->bands /
role->tracking_method, dedups against existing threat_radars.json by normalized
name+aliases, and:
  - appends genuinely-new radars (tier reference, carrying the source URL)
  - fills missing frequency_GHz / range_km on existing matches
Prints a report; never overwrites curated fields that already exist.
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAND = ROOT / "data" / "ew" / "scraped_candidates.json"
TR = ROOT / "data" / "ew" / "threat_radars.json"

VALID_BANDS = {"VHF", "UHF", "L", "S", "C", "X", "Ku", "Ka", "K",
               "A", "B", "D", "E", "F", "G", "H", "I", "J"}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def clean_bands(s):
    if not s:
        return []
    up = s.upper()
    toks = re.findall(r"\b(VHF|UHF|KU|KA|[A-L]|S|X)\b", up)
    out = []
    for t in toks:
        t = {"KU": "Ku", "KA": "Ka"}.get(t, t)
        if t in VALID_BANDS and t not in out:
            out.append(t)
    return out[:3]


def role_to_purpose(role):
    r = (role or "").lower()
    p = []
    if any(k in r for k in ["fire control", "fire-control", "guidance", "engagement", "illuminat"]):
        p += ["fire_control", "SAM_guidance", "track"]
    if any(k in r for k in ["early warning", "early-warning", "warning"]):
        p += ["early_warning", "search"]
    if any(k in r for k in ["acquisition", "surveillance", "search", "detect"]):
        p += ["acquisition", "search"]
    if "height" in r:
        p += ["acquisition", "track"]
    if "gci" in r or "ground control" in r:
        p += ["GCI", "search"]
    # dedup preserve order
    seen = set(); out = []
    for x in p:
        if x not in seen:
            seen.add(x); out.append(x)
    return out or ["search"]


def role_to_tracking(role, band_str):
    r = (role or "").lower(); b = (band_str or "").lower()
    if any(k in r for k in ["fire control", "fire-control", "guidance", "engagement", "illuminat"]):
        if any(k in (r + b) for k in ["illuminat", "continuous wave", "cw"]):
            return "cw_illuminator"
        return "monopulse"
    return "none"


def main():
    cands = json.load(open(CAND, encoding="utf-8")) if CAND.exists() else []
    db = json.load(open(TR, encoding="utf-8"))
    radars = db["radars"]
    alias_index = {}
    for r in radars:
        for a in [r["name"]] + r.get("aliases", []):
            alias_index[norm(a)] = r

    def find(name):
        nk = norm(name)
        if nk in alias_index:
            return alias_index[nk]
        for k, r in alias_index.items():
            if nk and len(nk) >= 4 and (nk in k or k in nk) and abs(len(nk) - len(k)) <= 3:
                return r
        return None

    added, enriched, skipped = [], [], 0
    seen_new = set()
    for c in cands:
        if not c.get("verified") or not c.get("name"):
            skipped += 1; continue
        bands = clean_bands(c.get("band"))
        ghz = c.get("frequency_GHz")
        rng = c.get("range_km")
        if not (bands or ghz or rng):
            skipped += 1; continue
        match = find(c["name"])
        if match:
            ev = match.setdefault("_enrich_note", [])
            if ghz and not match.get("frequency_GHz"):
                match["frequency_GHz"] = float(ghz); enriched.append((match["name"], "GHz", ghz))
            if rng and not match.get("range_km"):
                try:
                    match["range_km"] = float(re.search(r"[\d.]+", str(rng)).group()); enriched.append((match["name"], "range", rng))
                except Exception:
                    pass
            continue
        nk = norm(c["name"])
        if nk in seen_new:
            continue
        seen_new.add(nk)
        rec = {"name": c["name"].strip(),
               "aliases": [c["name"].strip()],
               "system": "IADS (open-source)", "country": "未知",
               "role": c.get("role", ""),
               "purpose": role_to_purpose(c.get("role")),
               "tracking_method": role_to_tracking(c.get("role"), c.get("band")),
               "sources": [c.get("source", "")]}
        if bands:
            rec["bands"] = bands
        if ghz:
            try: rec["frequency_GHz"] = float(ghz)
            except Exception: pass
        if rng:
            try: rec["range_km"] = float(re.search(r"[\d.]+", str(rng)).group())
            except Exception: pass
        radars.append(rec)
        added.append(rec["name"])

    # strip helper keys
    for r in radars:
        r.pop("_enrich_note", None)
    json.dump(db, open(TR, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"candidates: {len(cands)} | added: {len(added)} | enriched existing: {len(enriched)} | skipped: {skipped}")
    print("ADDED:", ", ".join(added) if added else "(none)")
    print("ENRICHED:", "; ".join(f"{n}+{w}={v}" for n, w, v in enriched) if enriched else "(none)")


if __name__ == "__main__":
    main()
