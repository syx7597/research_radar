#!/usr/bin/env python3
"""Build a small, deterministic AI-prepared packet for human source review.

Only selected local KG attributes and existing source snapshots are read.
No network, API, model, GPU, original-data mutation, or accepted gold creation.
Run normally to create the packet; --check-only rebuilds and verifies its bytes.
Existing outputs are refused unless --overwrite is explicitly provided.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/thesis_direction_review/radar_review_batch"
ENTITY_PATH = "kg_v3/entities.json"
CHUNK_PATH = "pipeline/v3/work/chunks.jsonl"
TEMPLATE_PATH = "templates/radar_review/facts.csv"
REVIEWER = "AI_initial_screening_not_human_verified"
SOURCE_DOC_IDS = {"AN/MPQ-65": "an_mpq-65", "AN/APY-9": "an_apy-9",
                  "AN/SPG-51": "an_spg-51", "AN/SPN-35": "an_spn-35",
                  "AN/SPG-59": "an_spg-59", "EL/M-2080": "el_m-2080"}
HEADERS = "fact_id,entity_id,entity_name,variant,attribute,event_type,value_raw,value_kind,value_status,value,min_value,max_value,unit_raw,unit_std,condition_status,condition_raw,conditions_json,doc_id,source_uri,source_file,source_sha256,locator,evidence_text,review_status,reviewer,review_notes".split(",")


def spec(entity, attribute, source_key, quote, category, note, **fields):
    return dict(entity=entity, attribute=attribute, source_key=source_key, quote=quote,
                category=category, note=note, fields=fields)


# Purposive, bounded examples chosen for human inspection, not a random sample.
# Values below are AI-prepared readings of the quoted local source, never gold.
SPECS = [
    spec("AN/MPQ-65", "type_description", "Type",
         "Mobile; surface-to-air missile; /; anti-ballistic missile; system",
         "model_system_attribution", "来源页为MIM-104 Patriot系统，不能把系统类型直接当雷达型号类型；主体归属待核验。",
         value_kind="text", value_status="ambiguous",
         value="Mobile surface-to-air missile / anti-ballistic missile system"),
    spec("AN/MPQ-65", "service_entry", "In\u00a0service", "Since 1981",
         "event_attribution", "保留来源第一个时间事件；来源主体是Patriot系统，1981不能据此确认为AN/MPQ-65服役年份。",
         event_type="in_service", value_kind="event", value_status="ambiguous",
         value="1981", unit_raw="", unit_std="year"),
    spec("AN/MPQ-65", "service_entry", "In\u00a0service", "initial operational capacity 1984",
         "event_attribution", "第二个时间事件单列；旧属性只存1981。1984的主体和事件定义仍待人工核验，不是替代金标。",
         event_type="initial_operational_capacity", value_kind="event", value_status="ambiguous",
         value="1984", unit_raw="", unit_std="year"),
    spec("AN/APY-9", "frequency", "Frequency", "UHF-Band 0.3–3.0\u00a0GHz (99.93–9.99\u00a0cm)",
         "numeric_range", "按快照预填区间两端；旧值只存3.0 GHz。需核实这是型号实际工作频段还是一般UHF频段说明；不把波长另造事实。",
         value_kind="range", value_status="ambiguous", min_value="0.3", max_value="3.0",
         unit_raw="GHz", unit_std="GHz"),
    spec("AN/SPG-51", "frequency", "Frequency", "Illuminator: 10.25-10.5 GHz",
         "component_scoped_range", "原文照射部件范围单列；旧值只存10.5 GHz。部件名称仅忠实保留原文，不宣称型号版本已厘清。",
         value_kind="range", value_status="stated", min_value="10.25", max_value="10.5",
         unit_raw="GHz", unit_std="GHz", condition_status="explicit", condition_raw="Illuminator",
         conditions_json='[{"kind":"verbatim_component_label","text":"Illuminator"}]'),
    spec("AN/SPG-51", "frequency", "Frequency", "Tracking Radar: 5.45 GHz - 5.825 GHz",
         "component_scoped_range", "原文跟踪部件范围单列；不与照射部件范围合并。出处引用及具体配置需人工核验。",
         value_kind="range", value_status="stated", min_value="5.45", max_value="5.825",
         unit_raw="GHz", unit_std="GHz", condition_status="explicit", condition_raw="Tracking Radar",
         conditions_json='[{"kind":"verbatim_component_label","text":"Tracking Radar"}]'),
    spec("AN/SPG-51", "prf", "PRF", "4100 pps (surface)",
         "context_scoped_numeric", "surface是原文标签，仅记录，不擅自扩展其模式/目标含义。当前条目未与air数值混合。",
         value_kind="scalar", value_status="stated", value="4100", unit_raw="pps", unit_std="pps",
         condition_status="explicit", condition_raw="surface",
         conditions_json='[{"kind":"verbatim_context_label","text":"surface"}]'),
    spec("AN/SPG-51", "prf", "PRF", "9600-16700 pps (air)",
         "context_scoped_numeric", "air是原文标签，范围单列；旧属性只存surface的4100。标签实际语义及条件完整性待核验。",
         value_kind="range", value_status="stated", min_value="9600", max_value="16700",
         unit_raw="pps", unit_std="pps", condition_status="explicit", condition_raw="air",
         conditions_json='[{"kind":"verbatim_context_label","text":"air"}]'),
    spec("AN/SPN-35", "frequency", "Frequency", "9.0 to 9.2\u00a0GHz band",
         "unit_dimension_and_range", "快照为GHz频率范围，旧图谱却存9000.0 kg；仅预填原文范围，不保留明显量纲冲突为金标。",
         value_kind="range", value_status="stated", min_value="9.0", max_value="9.2",
         unit_raw="GHz", unit_std="GHz"),
    spec("AN/SPN-35", "pulse_width", "Pulsewidth", "0.2 microseconds",
         "unit_dimension", "快照为时间量，旧图谱却存0.3218 km。预填数值不变，仅将microseconds标作us；仍需人工核验。",
         value_kind="scalar", value_status="stated", value="0.2", unit_raw="microseconds", unit_std="us"),
    spec("AN/SPG-59", "service_entry", "Introduced", "Canceled 1963",
         "event_type", "原文说Canceled，旧字段名却是service_entry；预标cancellation以提醒复核，不把取消年份解释为服役年份。",
         event_type="cancellation", value_kind="event", value_status="stated", value="1963",
         unit_raw="", unit_std="year"),
    spec("EL/M-2080", "power", "Power", "Classified",
         "unknown_numeric_text", "快照没有给出功率数值，数值和单位留空；不把用途猜测变成数值或已证实能力。not_recorded仅描述本快照。",
         value_kind="unknown", value_status="not_recorded"),
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_info(relative: str) -> dict:
    data = (ROOT / relative).read_bytes()
    return {"path": relative, "bytes": len(data), "sha256": digest(data)}


def pointer_part(text: str) -> str:
    return text.replace("~", "~0").replace("/", "~1")


def resolve_pointer(value, pointer: str):
    if not pointer.startswith("/"):
        raise ValueError("Expected an absolute JSON pointer")
    for raw in pointer[1:].split("/"):
        part = raw.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def build_packet() -> dict[str, bytes]:
    template = list(csv.reader(io.StringIO((ROOT / TEMPLATE_PATH).read_text(encoding="utf-8"))))
    if template != [HEADERS]:
        raise ValueError("Facts template changed; explicitly review the packet schema before generating")
    entities = json.loads((ROOT / ENTITY_PATH).read_text(encoding="utf-8"))
    wanted = {(entry["entity"], entry["attribute"]) for entry in SPECS}
    selected = {}
    for entity_index, entity in enumerate(entities):
        if entity["name"] not in {name for name, _ in wanted}:
            continue
        for attribute_index, attribute in enumerate(entity.get("attributes", [])):
            key = (entity["name"], attribute["attr"])
            if (key in wanted and attribute.get("tier") == "v3_struct"
                    and attribute.get("doc_id") == SOURCE_DOC_IDS[entity["name"]]):
                if key in selected:
                    raise ValueError(f"Ambiguous selection for {key}")
                selected[key] = (entity_index, attribute_index, attribute)
    if set(selected) != wanted:
        raise ValueError("One or more explicitly selected KG attributes are missing")
    doc_ids = {item[2]["doc_id"] for item in selected.values()}
    chunks = {}
    with (ROOT / CHUNK_PATH).open(encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            item = json.loads(line)
            if item.get("doc_id") in doc_ids and item.get("chunk_id") == item["doc_id"] + "#ibx":
                if item["chunk_id"] in chunks:
                    raise ValueError("Duplicate selected chunk ID")
                chunks[item["chunk_id"]] = (line_number, item, digest(line.encode("utf-8")))
    rows, source_records, source_files = [], [], {}
    for number, entry in enumerate(SPECS, 1):
        entity_index, attribute_index, attribute = selected[(entry["entity"], entry["attribute"])]
        doc_id = attribute["doc_id"]
        relative = f"radar_corpus/raw/{doc_id}.json"
        source = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        source_files[relative] = file_info(relative)
        source_pointer = "/infobox/" + pointer_part(entry["source_key"])
        raw_source_value = resolve_pointer(source, source_pointer)
        if not isinstance(raw_source_value, str) or raw_source_value.count(entry["quote"]) != 1:
            raise ValueError(f"Expected one exact source excerpt for {entry['entity']} {entry['attribute']}")
        start = raw_source_value.index(entry["quote"])
        end = start + len(entry["quote"])
        chunk_id = doc_id + "#ibx"
        chunk_info = None
        if chunk_id in chunks:
            line_number, chunk, line_hash = chunks[chunk_id]
            chunk_pointer = "/struct/" + pointer_part(entry["source_key"])
            # The existing chunk is cleaned relative to the raw snapshot. Do
            # not force text equality or replace the archived source text.
            chunk_value = resolve_pointer(chunk, chunk_pointer)
            chunk_info = {"file": CHUNK_PATH, "line": line_number, "chunk_id": chunk_id,
                          "line_sha256": line_hash, "json_pointer": chunk_pointer,
                          "value_excerpt": str(chunk_value)[:180],
                          "value_excerpt_truncated": len(str(chunk_value)) > 180}
        locator = {"json_pointer": source_pointer, "char_start": start, "char_end": end,
                   "offset_unit": "unicode_codepoint", "end_exclusive": True}
        if chunk_info:
            locator.update(chunk_file=CHUNK_PATH, chunk_line=chunk_info["line"], chunk_id=chunk_id)
        row = {field: "" for field in HEADERS}
        row.update(fact_id=f"radar-review-001-{number:02d}", entity_id="kg_v3:" + entry["entity"],
                   entity_name=entry["entity"], attribute=entry["attribute"], value_raw=attribute["value_raw"],
                   condition_status="not_stated", conditions_json="[]", doc_id=doc_id,
                   source_uri=source["source_en"], source_file=relative,
                   source_sha256=source_files[relative]["sha256"],
                   locator=json.dumps(locator, ensure_ascii=False, sort_keys=True),
                   evidence_text=entry["quote"], review_status="needs_review", reviewer=REVIEWER,
                   review_notes="AI预填，未人工核验；不是金标或已接受替代事实。型号版本未明确，variant留空。" + entry["note"])
        row.update(entry["fields"])
        rows.append(row)
        source_records.append({
            "fact_id": row["fact_id"], "category": entry["category"],
            "entity_json_pointer": f"/{entity_index}",
            "attribute_json_pointer": f"/{entity_index}/attributes/{attribute_index}",
            "extracted_entity_name": entry["entity"], "original_attribute": attribute,
            "source_file": relative, "source_sha256": row["source_sha256"],
            "source_uri": row["source_uri"], "source_locator": locator,
            "source_quote": entry["quote"], "chunk_locator": chunk_info,
            "source_excerpt_only": True, "human_verified": False,
            "annotation_status": "AI proposed reading for review; not an accepted replacement or gold"})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=HEADERS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_data = stream.getvalue().encode("utf-8")
    categories = sorted({entry["category"] for entry in SPECS})
    readme = f"""# 首批雷达事实核查包：AI预填，待人工核验

本包含{len(rows)}条待核验记录，来自{len(selected)}条旧属性、{len(doc_ids)}个实体标签和{len(source_files)}份本地网页快照。所有记录均为`needs_review`；没有人工金标，没有修改旧知识图谱，没有独立问答标注。

这是有目的选取的问题样例，不是随机抽样，也不是错误频率估计。覆盖型号/系统归属、两个事件的拆分、频率区间、部件及原文情境标签、量纲错映射、取消/服役事件混淆以及未记录数值。具体类别：{', '.join(categories)}。未额外编造关系案例来凑类别。

- `facts.csv`严格使用既有事实模板。`value_raw`保留旧属性原值；value/min/max、事件和条件栏只是根据短摘录预填的待核验解释，不能直接导入accepted知识库。
- `source_manifest.json`保存原属性、知识图谱JSON位置、网页URL、本地快照哈希、短摘录位置和可用chunk位置。CSV的`locator`为JSON；字符偏移按Unicode码点计数，右端不含。chunk文本经过历史清洗，不要求与快照逐字相同。
- `value_status=stated`只表示数字或文本在这份快照中出现，不表示其真实性已经核实。`not_recorded`仅指本快照没有数值，不能推断真实参数不存在。`condition_status=not_stated`也不能解释为无条件成立。
- 部件或surface/air等标签只保留原文，不扩展为未说明的配置、目标或测试条件。年份的year来自日历事件语境，仅预填unit_std；原文未写单位，unit_raw留空。
- 人工首先检查主体是雷达、整套系统还是平台，再查事件/配置及网页所引的一手出处；本轮未联网追踪文献、未核实真实性。AN/MPQ-65的来源指向Patriot系统页，不能直接接受两个年份为该雷达事实。
- 型号版本没有确认，`variant`留空并写入备注。source_uri与source_file分列，哈希属于本地文件。用于审阅的独立问答应在事实核验和版本冻结后另行编写。

生成：`python3 scripts/prepare_radar_review_packet.py`。默认拒绝覆盖任何已有输出。
只读自校验与确定性重建：`python3 scripts/prepare_radar_review_packet.py --check-only`。

自校验仅保证字段、来源定位、字面证据、数量、哈希和输出复现，不替代人工事实核验。输入只读取本地文件；无网络、模型/API或GPU调用。
""".encode("utf-8")
    input_paths = [ENTITY_PATH, CHUNK_PATH, TEMPLATE_PATH,
                   "docs/research/RADAR_DATA_ANNOTATION.md", "scripts/prepare_radar_review_packet.py"]
    manifest = {
        "packet_version": "radar_review_batch_001", "status": "needs_review",
        "prepared_by": REVIEWER, "human_verified": False, "accepted_gold_records": 0,
        "selection": "Fixed purposive sample; not random; not a prevalence estimate",
        "facts": len(rows), "selected_existing_attributes": len(selected),
        "entity_labels": sorted({entry["entity"] for entry in SPECS}), "categories": categories,
        "inputs": [file_info(path) for path in input_paths] + [source_files[path] for path in sorted(source_files)],
        "outputs": {"facts.csv": {"bytes": len(csv_data), "sha256": digest(csv_data)},
                    "README.md": {"bytes": len(readme), "sha256": digest(readme)}},
        "records": source_records,
        "validation": {"template_columns_match": True, "unique_fact_ids": True,
                       "all_source_pointers_resolve": True, "all_excerpts_exact": True,
                       "source_hashes_checked": True, "chunk_links_available": sum(bool(r["chunk_locator"]) for r in source_records),
                       "no_accepted_records": True, "numeric_truth_verified": False}}
    packet = {"facts.csv": csv_data, "README.md": readme, "source_manifest.json": json_bytes(manifest)}
    validate_packet(packet)
    return packet


def validate_packet(packet: dict[str, bytes]) -> None:
    manifest = json.loads(packet["source_manifest.json"])
    reader = csv.DictReader(io.StringIO(packet["facts.csv"].decode("utf-8")))
    rows = list(reader)
    if reader.fieldnames != HEADERS or len(rows) != 12 or len({row["fact_id"] for row in rows}) != 12:
        raise ValueError("Invalid header, fact count, or duplicate fact IDs")
    records = {record["fact_id"]: record for record in manifest["records"]}
    for info in manifest["inputs"]:
        if file_info(info["path"]) != info:
            raise ValueError(f"Source changed during preparation: {info['path']}")
    for name, info in manifest["outputs"].items():
        if digest(packet[name]) != info["sha256"] or len(packet[name]) != info["bytes"]:
            raise ValueError("Output content/hash mismatch")
    entities = json.loads((ROOT / ENTITY_PATH).read_text(encoding="utf-8"))
    chunk_lines = (ROOT / CHUNK_PATH).read_bytes().splitlines(keepends=True)
    for row in rows:
        record = records[row["fact_id"]]
        if row["review_status"] != "needs_review" or row["reviewer"] != REVIEWER or row["variant"]:
            raise ValueError("A record was incorrectly marked accepted/human-reviewed or given an unverified variant")
        if row["value_kind"] not in {"text", "scalar", "range", "lower_bound", "upper_bound", "approximate", "date", "event", "unknown"}:
            raise ValueError("Unknown value_kind")
        if row["value_status"] not in {"stated", "not_recorded", "not_applicable", "conflicting", "ambiguous"}:
            raise ValueError("Unknown value_status")
        if row["value_kind"] == "range" and (not row["min_value"] or not row["max_value"] or row["value"]
                                                or float(row["min_value"]) > float(row["max_value"])):
            raise ValueError("A range must retain ordered endpoints without a collapsed scalar")
        if row["value_kind"] == "unknown" and any(row[key] for key in ("value", "min_value", "max_value", "unit_raw", "unit_std")):
            raise ValueError("Unknown numeric facts must not receive fabricated numbers or units")
        if resolve_pointer(entities, record["attribute_json_pointer"]) != record["original_attribute"]:
            raise ValueError("Original KG attribute no longer matches its recorded pointer")
        source = json.loads((ROOT / row["source_file"]).read_text(encoding="utf-8"))
        locator = json.loads(row["locator"])
        text = resolve_pointer(source, locator["json_pointer"])
        if text[locator["char_start"]:locator["char_end"]] != row["evidence_text"] or len(row["evidence_text"]) > 180:
            raise ValueError("Source quote is not an exact short located excerpt")
        if row["value_kind"] in {"scalar", "range", "event"}:
            for key in ("value", "min_value", "max_value"):
                if row[key] and row[key] not in row["evidence_text"]:
                    raise ValueError("A proposed numeral is absent from the exact source excerpt")
        if row["unit_raw"] and row["unit_raw"] not in row["evidence_text"]:
            raise ValueError("A raw unit must occur literally in the source excerpt")
        conditions = json.loads(row["conditions_json"])
        if row["condition_status"] == "explicit":
            if not row["condition_raw"] or row["condition_raw"] not in row["evidence_text"] or not conditions:
                raise ValueError("Explicit condition lacks literal source support")
        elif row["condition_status"] == "not_stated":
            if row["condition_raw"] or conditions:
                raise ValueError("Unstated conditions must not be filled in")
        else:
            raise ValueError("Unexpected condition status in this fixed packet")
        chunk = record["chunk_locator"]
        if chunk:
            line = chunk_lines[chunk["line"] - 1]
            item = json.loads(line)
            if digest(line) != chunk["line_sha256"] or item["chunk_id"] != chunk["chunk_id"]:
                raise ValueError("Chunk identity or line hash mismatch")
            if str(resolve_pointer(item, chunk["json_pointer"]))[:180] != chunk["value_excerpt"]:
                raise ValueError("Chunk pointer/excerpt mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(DEFAULT_OUTPUT.resolve()):
        raise ValueError("Output must stay within the authorized radar_review_batch directory")
    if args.check_only and args.overwrite:
        parser.error("--check-only and --overwrite are mutually exclusive")
    names = ("facts.csv", "README.md", "source_manifest.json")
    if not args.check_only and not args.overwrite and any((args.output_dir / name).exists() for name in names):
        raise FileExistsError("Packet already exists; use --check-only or explicitly --overwrite")
    packet = build_packet()
    if args.check_only:
        actual = {name: (args.output_dir / name).read_bytes() for name in names}
        validate_packet(actual)
        if actual != packet:
            raise ValueError("Stored packet differs from deterministic rebuild")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, data in packet.items():
            (args.output_dir / name).write_bytes(data)
        validate_packet({name: (args.output_dir / name).read_bytes() for name in names})
    manifest = json.loads(packet["source_manifest.json"])
    print(json.dumps({"mode": "verified_without_writing" if args.check_only else "created_and_verified",
                      "facts": manifest["facts"], "selected_existing_attributes": manifest["selected_existing_attributes"],
                      "entity_labels": len(manifest["entity_labels"]), "categories": manifest["categories"],
                      "all_records_status": "needs_review", "human_verified": False,
                      "chunk_links": manifest["validation"]["chunk_links_available"],
                      "manifest_sha256": digest(packet["source_manifest.json"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
