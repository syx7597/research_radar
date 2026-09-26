"""
Bilingual lexicon for RadarKG.

Loaders:
  load_relations()  → relation lexicon (28 relations, zh/en labels, synonyms, templates)
  load_entity_aliases() → entity alias table (countries, modes, tech_types, ...)
  build_alias_index() → flat surface_form → canonical mapping (for retrieval-time lookup)
  build_relation_lookup() → zh keyword → relation_id mapping (for question routing)
"""

import json
from pathlib import Path

LEXICON_DIR = Path(__file__).parent
RELATIONS_PATH      = LEXICON_DIR / "relations.json"
ENTITY_ALIASES_PATH = LEXICON_DIR / "entity_aliases.json"


def load_relations() -> dict:
    with open(RELATIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_entity_aliases() -> dict:
    with open(ENTITY_ALIASES_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_alias_index() -> dict:
    """Flatten alias table into surface_form → {canonical, type, zh, en} mapping.

    Used at query time for cross-lingual entity linking. The same surface form
    may map to multiple canonicals across types (rare); we keep the first match.
    """
    aliases = load_entity_aliases()
    index: dict[str, dict] = {}

    def _add(form: str, canonical: str, ent_type: str, zh: str = "", en: str = ""):
        if not form:
            return
        index.setdefault(form, {"canonical": canonical, "type": ent_type, "zh": zh, "en": en})
        index.setdefault(form.lower(), {"canonical": canonical, "type": ent_type, "zh": zh, "en": en})

    for canonical, rec in aliases.get("countries", {}).items():
        _add(canonical, canonical, "Country", rec.get("zh", ""), rec.get("en", ""))
        _add(rec.get("en", ""), canonical, "Country", rec.get("zh", ""), rec.get("en", ""))
        for a in rec.get("aliases", []):
            _add(a, canonical, "Country", rec.get("zh", ""), rec.get("en", ""))

    for canonical, rec in aliases.get("radar_modes", {}).items():
        _add(canonical, canonical, "RadarMode", rec.get("zh", ""), rec.get("en", ""))
        _add(rec.get("en", ""), canonical, "RadarMode", rec.get("zh", ""), rec.get("en", ""))
        for a in rec.get("aliases", []):
            _add(a, canonical, "RadarMode", rec.get("zh", ""), rec.get("en", ""))

    for canonical, rec in aliases.get("tech_types", {}).items():
        _add(canonical, canonical, "TechType", rec.get("zh", ""), rec.get("en", ""))
        _add(rec.get("en", ""), canonical, "TechType", rec.get("zh", ""), rec.get("en", ""))
        for a in rec.get("aliases", []):
            _add(a, canonical, "TechType", rec.get("zh", ""), rec.get("en", ""))

    for canonical, rec in aliases.get("functions", {}).items():
        _add(canonical, canonical, "Function", rec.get("zh", ""), rec.get("en", ""))
        _add(rec.get("zh", ""), canonical, "Function", rec.get("zh", ""), rec.get("en", ""))
        for a in rec.get("aliases", []):
            _add(a, canonical, "Function", rec.get("zh", ""), rec.get("en", ""))

    for canonical, rec in aliases.get("naval_vessel_classes", {}).items():
        _add(canonical, canonical, "NavalVessel", rec.get("zh", ""), "")
        _add(rec.get("zh", ""), canonical, "NavalVessel", rec.get("zh", ""), "")
        for a in rec.get("aliases", []):
            _add(a, canonical, "NavalVessel", rec.get("zh", ""), "")

    return index


def build_relation_lookup() -> dict:
    """Map Chinese keyword → list of candidate relation_ids.

    Used by the question-type router and entity-relation linking. Keywords with
    higher specificity (longer strings) should be tried first by the caller.
    Non-dict entries (e.g., schema-internal markers like _v2_new_relations_note)
    are skipped.
    """
    relations = load_relations()["relations"]
    lookup: dict[str, list[str]] = {}
    for rel_id, rec in relations.items():
        if not isinstance(rec, dict):
            continue  # skip notes / markers
        terms = [rec.get("zh_label", "")] + rec.get("zh_synonyms", [])
        for term in terms:
            if not term:
                continue
            lookup.setdefault(term, [])
            if rel_id not in lookup[term]:
                lookup[term].append(rel_id)
    return lookup


def render_triple_text(triple: dict, lang: str = "zh", relations: dict | None = None) -> str:
    """Render a triple to natural text in the chosen language for indexing/display."""
    if relations is None:
        relations = load_relations()["relations"]
    rel = triple.get("relation", "")
    head = triple.get("head", "")
    tail = triple.get("tail", "")
    spec = relations.get(rel, {})
    template_key = "triple_text_zh" if lang == "zh" else "triple_text_en"
    template = spec.get(template_key, f"{{head}} {rel} {{tail}}")
    return template.format(head=head, tail=tail)
