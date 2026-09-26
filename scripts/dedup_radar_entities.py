"""
Deduplicate Radar / RadarSystem entities that differ only by superficial naming
(prefix, suffix, family notation) — NOT by genuine variant marker.

SAFE to merge:
  X            ↔ AN/X
  X            ↔ X radar / X (radar) / X 雷达
  X            ↔ X family / X 系列
  X            ↔ X/X(V)   (e.g. APG-70 ↔ AN/APG-70/70(V))
  X            ↔ X/Y(V)   when Y is X with suffix

DO NOT merge:
  X            ↔ X(V)1 / X(V)2 / X(V)4   (real generation variants)
  X            ↔ X/Y    when X and Y are different model numbers (e.g., AN/APG-66/68)

Process per merge group:
  1. Pick canonical (prefer AN/-prefixed, then longest meaningful id)
  2. Merge attribute values into canonical (don't overwrite existing non-empty)
  3. Rewrite all edges (head & tail) to point to canonical id
  4. Add duplicates to canonical's aliases
  5. Delete duplicates

Conservative safety guard: if two entities in a group have CONFLICTING country_of_origin
(both set, different non-empty values) → DO NOT merge that pair.

Idempotent: safe to re-run.
"""
import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
ENT_PATH = ROOT / "data" / "v2" / "radarkg_v2_entities.json"
EDGE_PATH = ROOT / "data" / "v2" / "radarkg_v2_edges.json"
REPORT_PATH = ROOT / "data" / "v2" / "dedup_radar_report.md"


# ────────────────────────────────────────────────────────────
#  Strict canonical key:
#  Encode all "safe-to-merge" variations into the same key.
# ────────────────────────────────────────────────────────────

def has_explicit_variant(rid: str) -> bool:
    """True if id contains a (V)N variant marker we must NOT merge across."""
    return bool(re.search(r"\([Vv]\)\d+", rid))


def safe_canonical_key(rid: str) -> str | None:
    """Return a canonical key if this id is one of the 'family base' surface forms.
    Returns None if id has an explicit (V)N marker (cannot safely merge into base).
    """
    s = rid.strip()
    if has_explicit_variant(s):
        return None
    # Strip leading AN/
    s = re.sub(r"^AN/", "", s, flags=re.I)
    # Strip Chinese 雷达 / English radar suffix (with optional parens)
    s = re.sub(r"\s*\(?\s*[Rr]adar\s*\)?\s*$", "", s)
    s = re.sub(r"\s*雷达\s*$", "", s)
    # Strip family / 系列
    s = re.sub(r"\s+family\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*系列\s*$", "", s)
    s = re.sub(r"\s*改进型\s*$", "", s)
    # Strip /X(V) family notation: e.g. "APG-70/70(V)" → "APG-70";  "APG-65/65(V)" → "APG-65"
    # Pattern: base + '/' + same-or-related-number + optional (V) (with no digit after V)
    m = re.match(r"^(.+?)/(\d+)(\([Vv]\))?\s*$", s)
    if m:
        base, num2 = m.group(1), m.group(2)
        # If base ends with same number suffix → same radar family
        if base.endswith(num2):
            s = base
        # If base is just a prefix + number, treat as family if num2 matches that number
        else:
            mb = re.match(r"^(.+?)(\d+)$", base)
            if mb and mb.group(2) == num2:
                s = base
            # else: e.g. "AN/APG-66/68" — different model, don't merge
            else:
                return None
    s = s.strip().lower()
    return s


def build_groups(radars: list[dict]) -> dict[str, list[dict]]:
    """Return {canonical_key: [radar_entities…]} for groups with ≥2 entries."""
    groups = defaultdict(list)
    for r in radars:
        key = safe_canonical_key(r["id"])
        if key is None: continue
        groups[key].append(r)
    return {k: v for k, v in groups.items() if len(v) >= 2}


# ────────────────────────────────────────────────────────────
#  Per-group merge logic
# ────────────────────────────────────────────────────────────

def pick_canonical(group: list[dict]) -> dict:
    """Pick the best canonical entity from a group:
      1) prefer id starting with 'AN/'
      2) prefer id without slash (cleaner base)
      3) prefer the longest among ties (carries more info)
    """
    def score(e):
        rid = e["id"]
        an = 1 if rid.upper().startswith("AN/") else 0
        no_slash = 1 if "/" not in rid else 0
        return (an, no_slash, len(rid))
    return max(group, key=score)


def merge_attrs(canonical: dict, dup: dict):
    """Merge dup's attributes into canonical (don't overwrite non-empty existing values)."""
    for k, v in dup.items():
        if k in ("id", "type"): continue
        if v in (None, "", [], 0): continue
        cur = canonical.get(k)
        if cur in (None, "", [], 0):
            canonical[k] = v
        elif k == "aliases" and isinstance(cur, list) and isinstance(v, list):
            canonical[k] = sorted(set(cur) | set(v))


def check_safe_to_merge(group: list[dict]) -> tuple[bool, str]:
    """Verify there's no hard conflict (e.g., different country_of_origin)."""
    countries = {e.get("country_of_origin") for e in group if e.get("country_of_origin")}
    if len(countries) > 1:
        return False, f"country conflict: {countries}"
    return True, ""


# ────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────

def main():
    print("Loading v2 KG…")
    with open(ENT_PATH, encoding="utf-8") as f:
        ent_data = json.load(f)
    with open(EDGE_PATH, encoding="utf-8") as f:
        edge_data = json.load(f)
    entities = ent_data["entities"]
    edges = edge_data["edges"]

    radars = [e for e in entities if e["type"] in ("Radar", "RadarSystem")]
    print(f"  Total entities: {len(entities)}, radars: {len(radars)}, edges: {len(edges)}")

    groups = build_groups(radars)
    print(f"  Groups eligible for safe merging: {len(groups)}")

    # Build canonical mapping: old_id → canonical_id
    id_map: dict[str, str] = {}
    merge_log = []
    skipped_log = []

    for key, grp in groups.items():
        safe, reason = check_safe_to_merge(grp)
        if not safe:
            skipped_log.append((key, [e["id"] for e in grp], reason))
            continue
        canonical = pick_canonical(grp)
        for e in grp:
            if e["id"] != canonical["id"]:
                id_map[e["id"]] = canonical["id"]
        merge_log.append((canonical["id"], [e["id"] for e in grp if e["id"] != canonical["id"]]))

    print(f"  Safe merges: {len(merge_log)} canonical entities, "
          f"{len(id_map)} duplicates to absorb")
    print(f"  Skipped (conflict): {len(skipped_log)}")

    # Apply merges
    ent_by_id = {e["id"]: e for e in entities}

    # 1. Merge attrs into canonical
    for canonical_id, dup_ids in merge_log:
        canonical_ent = ent_by_id[canonical_id]
        aliases = list(canonical_ent.get("aliases", []) or [])
        for dup_id in dup_ids:
            dup_ent = ent_by_id.get(dup_id)
            if not dup_ent: continue
            merge_attrs(canonical_ent, dup_ent)
            # Record dup id as alias
            if dup_id not in aliases:
                aliases.append(dup_id)
            if dup_ent.get("name_en") and dup_ent["name_en"] not in aliases:
                aliases.append(dup_ent["name_en"])
        canonical_ent["aliases"] = sorted(set(a for a in aliases if a))

    # 2. Rewrite edges
    rewritten_h = 0
    rewritten_t = 0
    dropped = 0
    seen_keys = set()
    new_edges = []
    for ed in edges:
        h, t = ed["head"], ed["tail"]
        new_h = id_map.get(h, h)
        new_t = id_map.get(t, t)
        if new_h != h: rewritten_h += 1
        if new_t != t: rewritten_t += 1
        ed["head"], ed["tail"] = new_h, new_t
        key = (new_h, ed["relation"], new_t)
        if key in seen_keys:
            dropped += 1
            continue
        if new_h == new_t:  # self-loop guard
            dropped += 1
            continue
        seen_keys.add(key)
        new_edges.append(ed)

    # 3. Delete merged duplicate entities
    new_entities = [e for e in entities if e["id"] not in id_map]

    # Save
    edge_data["edges"] = new_edges
    edge_data["count"] = len(new_edges)
    ent_data["entities"] = new_entities
    ent_data["count"] = len(new_entities)
    with open(EDGE_PATH, "w", encoding="utf-8") as f:
        json.dump(edge_data, f, ensure_ascii=False, indent=2)
    with open(ENT_PATH, "w", encoding="utf-8") as f:
        json.dump(ent_data, f, ensure_ascii=False, indent=2)

    # Report
    print(f"\nMerged {len(id_map)} dup entities into {len(merge_log)} canonicals")
    print(f"  Edges: rewrote head {rewritten_h}, tail {rewritten_t}, deduped {dropped}")
    print(f"  Final: {len(new_entities)} entities, {len(new_edges)} edges")
    print()
    print("Sample merges:")
    for canon, dups in merge_log[:15]:
        print(f"  {canon}  ←  {dups}")
    if skipped_log:
        print()
        print("Skipped merges (country/dev conflict):")
        for key, ids, reason in skipped_log:
            print(f"  {key}: {ids}  ({reason})")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"# Radar Entity Deduplication Report\n\n")
        f.write(f"- Groups examined: {len(groups)}\n")
        f.write(f"- Safe merges: {len(merge_log)}\n")
        f.write(f"- Duplicate entities absorbed: {len(id_map)}\n")
        f.write(f"- Edge head rewrites: {rewritten_h}\n")
        f.write(f"- Edge tail rewrites: {rewritten_t}\n")
        f.write(f"- Duplicate edges dropped: {dropped}\n\n")
        f.write("## All merges\n\n")
        for canon, dups in merge_log:
            f.write(f"- `{canon}` ← {', '.join('`'+d+'`' for d in dups)}\n")
        if skipped_log:
            f.write("\n## Skipped (conflict)\n\n")
            for key, ids, reason in skipped_log:
                f.write(f"- `{key}`: {ids}  — {reason}\n")


if __name__ == "__main__":
    main()
