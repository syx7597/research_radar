"""Build a private source/QA audit packet; never read model outputs or score files.

The fixed risk-stratified sample is not an unbiased accuracy estimate. Generating
this packet does not perform human review. References are a separate AI sidecar.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = "results/radar_domain/coverage_v2_citation_sensitivity_v1/external_review_protocol.json"
FAMILIES = ("eec_ranger_x_band", "furuno_drs_nxt", "gamic_gmwr", "garmin_gmr_fantom",
            "jrc_jma5200mk2", "metek_mrr", "raymarine_cyclone", "vaisala_wrs300")
STRATA = ("quantity_form", "qualified_attribute", "table_binding")
METADATA = ("question_id", "family_id", "primary_type", "question_sha256")
EMPTY_FIELDS = ("reviewer_name", "reviewer_role", "review_started_at", "review_completed_at",
                "prior_reference_exposure", "independent_answer", "independent_locators",
                "independent_uncertainties", "independent_saved_at", "reference_opened_at",
                "reference_judgment", "revised_answer", "revision_reason", "human_verified")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def local(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Path must be relative and remain inside the repository")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Path escapes repository through a symlink")
    return resolved


def select_metadata(rows, protocol):
    """The selector deliberately accepts metadata only, never references/scores."""
    if any(set(row) != set(METADATA) for row in rows):
        raise ValueError("Selection accepts only the four registered metadata fields")
    if len({row["question_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate question_id")
    selected = []
    for family in protocol["families"]:
        for stratum in protocol["strata"]:
            candidates = []
            for row in rows:
                if row["family_id"] == family and row["primary_type"] == stratum:
                    material = "|".join((str(protocol["seed"]), family, stratum,
                                         row["question_id"], row["question_sha256"]))
                    candidates.append(dict(row, rank_sha256=digest(material.encode("utf-8"))))
            if not candidates:
                raise ValueError(f"Missing registered stratum; no fallback: {family}/{stratum}")
            selected.append(min(candidates, key=lambda row: (row["rank_sha256"], row["question_id"])))
    return selected


def validate_protocol(protocol):
    required = {"schema": "radar_coverage_external_review_protocol_v1", "status": "registered",
                "seed": 20261008, "selection_rule": "sha256_utf8_pipe_rank_v1",
                "per_family_per_stratum": 1, "human_gold": False,
                "purpose": "source_and_QA_reference_fact_verification",
                "citation_sensitivity_results_seen_at_registration": False,
                "previous_original_coverage_results_known_to_project": True}
    if any(protocol.get(key) != value for key, value in required.items()):
        raise ValueError("Protocol does not match the registered external review design")
    if protocol.get("families") != list(FAMILIES) or protocol.get("strata") != list(STRATA):
        raise ValueError("Registered families/strata changed")
    paths = protocol["paths"]
    inputs = [paths[key] for key in ("selection", "snapshot", "chunks", "scoring_adjudication", "author_rubric_response")]
    inputs += paths["author_packets"]
    if len(inputs) != 7 or set(inputs) != set(protocol["input_sha256"]):
        raise ValueError("Protocol input roles/hash inventory mismatch")
    if any(not (name.startswith("data/radar_sources_v2/") or
                name.startswith("artifacts/thesis_direction_review/radar_questions_v1/")) for name in inputs):
        raise ValueError("Input inventory must not contain model outputs or scores")
    if paths["private_dir"] != "data/radar_sources_v2/external_review_v1" or paths["public_manifest"] != "artifacts/thesis_direction_review/radar_external_review_v1/manifest.json":
        raise ValueError("Unexpected output destinations")


def escaped_section(chunk, archive_name, derived_text):
    e = html.escape
    return ("<!doctype html><html lang='zh'><meta charset='utf-8'><title>归档原文整节</title>"
            "<style>body{max-width:1000px;margin:2em auto}pre{white-space:pre-wrap}</style>"
            "<h1>归档 HTML 整节文本（不是整站）</h1><p>页面仅显示机器抽取的完整归档节文本；"
            "原始 HTML 的脚本不会在此执行。空白和表格布局可能有抽取瑕疵。</p><p>来源："
            + e(chunk["source_id"]) + "；归档地址：" + e(chunk["source_uri"]) + "</p><p>原始字节 SHA256："
            + e(chunk["source_sha256"]) + "；<a download href='" + e(archive_name, quote=True)
            + "'>下载原始归档字节（不自动执行）</a></p><pre>" + e(derived_text) + "</pre></html>").encode("utf-8")


def review_html(questions, reference_sha, protocol_sha):
    # Only question/source data are embedded; no reference text or support spans.
    payload = json.dumps({"questions": questions, "reference_sha256": reference_sha,
                          "protocol_sha256": protocol_sha}, ensure_ascii=False).replace("<", "\\u003c")
    return (HTML_TEMPLATE.replace("__PAYLOAD__", payload)).encode("utf-8")


def build(protocol_path=DEFAULT_PROTOCOL, root=ROOT):
    root = Path(root).resolve()
    protocol_bytes = local(root, protocol_path).read_bytes()
    protocol = json.loads(protocol_bytes)
    validate_protocol(protocol)
    inputs = {}
    for name, expected in protocol["input_sha256"].items():
        value = local(root, name).read_bytes()
        if digest(value) != expected:
            raise ValueError(f"Input hash mismatch: {name}")
        inputs[name] = json.loads(value)
    paths = protocol["paths"]
    authors = [row for name in paths["author_packets"] for row in inputs[name]["questions"]]
    if len(authors) != 96 or Counter(row["family_id"] for row in authors) != Counter({family: 12 for family in FAMILIES}):
        raise ValueError("Expected frozen roster of 96 questions / 12 per family")
    for row in authors:
        if digest(row["question_zh"].encode("utf-8")) != row["question_sha256"]:
            raise ValueError("Question text fingerprint mismatch")
    selected = select_metadata([{key: row[key] for key in METADATA} for row in authors], protocol)
    author_by_id = {row["question_id"]: row for row in authors}
    if inputs[paths["chunks"]].get("normalization") != "collapse_whitespace_only":
        raise ValueError("Unexpected archived text normalization")
    raw_chunks = inputs[paths["chunks"]]["chunks"]
    chunks = {row["chunk_id"]: row for row in raw_chunks}
    if len(chunks) != len(raw_chunks):
        raise ValueError("Duplicate source chunk IDs")
    private = Path(paths["private_dir"])
    outputs, source_hashes, source_index = {}, {}, {}
    questions, references = [], []
    interpretations = inputs[paths["author_rubric_response"]].get("required_facts_scoring_interpretation", [])
    for picked in selected:
        row = author_by_id[picked["question_id"]]
        locators = []
        if not row["support"]:
            raise ValueError("Selected question has no archived source support")
        for support in row["support"]:
            chunk = chunks[support["chunk_id"]]
            if chunk["family_id"] != row["family_id"] or digest(chunk["text"].encode("utf-8")) != support["text_sha256"] or support["text_sha256"] != chunk["text_sha256"]:
                raise ValueError("Question source family/text binding mismatch")
            for span in support["spans"]:
                if not (0 <= span["start"] < span["end"] <= len(chunk["text"])) or chunk["text"][span["start"]:span["end"]] != span["quote"]:
                    raise ValueError("Reference source span mismatch")
            source_id = chunk["source_id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", source_id):
                raise ValueError("Unsafe source_id")
            source_path = Path(chunk["source_path"])
            derived_path = Path(chunk["derived_path"])
            if any(not str(path).startswith("data/radar_sources_v2/") for path in (source_path, derived_path)):
                raise ValueError("Source archive outside the bound source collection")
            source = local(root, source_path).read_bytes()
            derived = local(root, derived_path).read_bytes()
            if digest(source) != chunk["source_sha256"] or digest(derived) != chunk["derived_sha256"]:
                raise ValueError("Archived source/derived text hash mismatch")
            if re.sub(r"\s+", " ", derived.decode("utf-8")).strip() != chunk["text"]:
                raise ValueError("Chunk is not the complete archived page/section text")
            for name, value in ((source_path, source), (derived_path, derived)):
                source_hashes[str(name)] = digest(value)
            page = chunk["page_one_based"]
            if page is not None and (type(page) is not int or page < 1 or not source.startswith(b"%PDF-")):
                raise ValueError("PDF source/page mismatch")
            archive_name = source_id + (".pdf" if page is not None else ".html.bin")
            archive_path = private / "sources" / archive_name
            if archive_path in outputs and outputs[archive_path] != source:
                raise ValueError("Source ID used for different archive bytes")
            outputs[archive_path] = source
            if page is None:
                display_name = source_id + ".section.html"
                display = escaped_section(chunk, archive_name, derived.decode("utf-8"))
                display_path = private / "sources" / display_name
                if display_path in outputs and outputs[display_path] != display:
                    raise ValueError("HTML source ID has multiple section texts")
                outputs[display_path] = display
                href = "sources/" + display_name
            else:
                href = "sources/" + archive_name + f"#page={page}"
            locators.append({"source_id": source_id, "chunk_id": chunk["chunk_id"],
                             "source_uri": chunk["source_uri"], "page_one_based": page,
                             "scope": "complete_archived_PDF" if page else "complete_archived_HTML_section_not_whole_site",
                             "href": href, "source_sha256": chunk["source_sha256"]})
            source_index[chunk["chunk_id"]] = locators[-1]
        questions.append({**picked, "question": row["question_zh"], "source_version": row["source_version"], "sources": locators})
        references.append({"question_id": row["question_id"], "reference_grade": "AI_reference_not_human_gold",
                           "reference": row["reference"], "support": row["support"],
                           "scoring_interpretation": [item for item in interpretations if item["question_id"] == row["question_id"]]})
    reference_doc = {"schema": "radar_external_reference_sidecar_v1", "human_gold": False,
                     "protocol_sha256": digest(protocol_bytes), "rows": references,
                     "scoring_adjudication": inputs[paths["scoring_adjudication"]]}
    reference_bytes = encoded(reference_doc)
    outputs[private / "references.json"] = reference_bytes
    outputs[private / "questions.json"] = encoded({"schema": "radar_external_questions_v1", "rows": questions})
    outputs[private / "selection.json"] = encoded({"protocol_sha256": digest(protocol_bytes), "rows": selected})
    outputs[private / "source_index.json"] = encoded({"chunks": list(source_index.values())})
    empty_rows = [{"question_id": row["question_id"], **dict.fromkeys(EMPTY_FIELDS)} for row in selected]
    outputs[private / "blank_responses.json"] = encoded({"schema": "radar_external_human_responses_v1", "human_gold": False,
        "protocol_sha256": digest(protocol_bytes), "reviewer_identity": None, "reference_file_loaded_at": None, "rows": empty_rows})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["question_id", *EMPTY_FIELDS])
    writer.writeheader()
    writer.writerows(empty_rows)
    outputs[private / "blank_responses.csv"] = stream.getvalue().encode("utf-8-sig")
    outputs[private / "index.html"] = review_html(questions, digest(reference_bytes), digest(protocol_bytes))
    manifest = {"schema": "radar_external_review_manifest_v1", "status": "packet_prepared_human_review_not_performed",
        "purpose": protocol["purpose"], "human_gold": False, "human_reviews_completed": 0,
        "reviewer_identity": None, "human_review_started_at": None, "human_review_completed_at": None,
        "selection_seed": protocol["seed"], "question_count": len(selected), "family_count": len(FAMILIES),
        "counts_by_family": dict(Counter(row["family_id"] for row in selected)),
        "counts_by_primary_type": dict(Counter(row["primary_type"] for row in selected)),
        "model_outputs_read_for_selection": False, "model_inference_runs": 0,
        "citation_sensitivity_results_seen_at_registration": False,
        "previous_original_coverage_results_known_to_project": True,
        "protocol_sha256": digest(protocol_bytes), "generator_sha256": digest(Path(__file__).read_bytes()),
        "input_sha256": protocol["input_sha256"], "source_files_sha256": source_hashes,
        "private_outputs_sha256": {str(path): digest(value) for path, value in sorted(outputs.items())},
        "limits": protocol["limits"]}
    outputs[Path(paths["public_manifest"])] = encoded(manifest)
    return outputs


def publish(outputs, root=ROOT, check=False):
    destinations = {local(Path(root), str(path)): value for path, value in outputs.items()}
    if check:
        for path, value in destinations.items():
            if not path.is_file() or path.read_bytes() != value:
                raise ValueError(f"Reproduction mismatch: {path}")
        return
    if any(path.exists() for path in destinations):
        raise FileExistsError("Packet already exists; use --check, never overwrite human responses")
    for path, value in destinations.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(value)


HTML_TEMPLATE = r'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>来源与参考事实核验（待真人填写）</title>
<style>body{font:16px/1.6 sans-serif;max-width:1050px;margin:2em auto;padding:1em}article{border:1px solid #bbb;padding:1em;margin:1em 0}textarea{display:block;width:97%;min-height:70px}input{margin:.3em}pre{white-space:pre-wrap}button{padding:.5em;margin:.4em}summary{cursor:pointer}.notice{background:#fff2d1;padding:1em}.reference{background:#eef5ff;padding:1em}</style>
<h1>来源与 QA 参考事实核验</h1><p class="notice">这是 8 组各 3 题的固定风险分层样本，不是全部 96 题的无偏抽样；本轮不含 simple attribute 和 listing。参考由 AI 起草，尚非人工金标。此页不展示任何模型回答。生成核验包不等于已完成人审。</p>
<p>建议先对照原 PDF 页或归档 HTML 整节完成独立作答，再查看 AI 参考。文件或浏览器状态不能证明完全盲审，请如实记录此前是否看过参考。HTML 展示为完整归档节的转义文本，不执行归档脚本；不是整个网站。</p>
<label>核验者姓名/代号 <input id="reviewer_name"></label><label>身份/专业背景 <input id="reviewer_role"></label>
<label>此前看过本题参考答案？ <select id="prior_reference_exposure"><option value=""></option><option>未看过</option><option>看过部分或全部</option><option>不确定</option></select></label>
<p><button id="choose">主动加载 AI 参考 sidecar</button><input type="file" id="refs" accept="application/json,.json" hidden><button id="complete">本人标记本次填写完成</button><button id="download">下载并实际保存填写结果 JSON</button><span id="status"></span></p>
<p>请手动选择同目录 references.json。加载时间与每题首次展开时间会记录；展开前必须保存该题原始独立回答/不确定性。浏览器本地保存不代表已提交，完成后请点击下载并确认文件已保存。也可复制 blank_responses.csv / blank_responses.json 填写并另存，请保留原空白模板。完成按钮仅记录填写者的操作，不自动把参考升级为人工金标。</p><main id="items"></main>
<script id="packet" type="application/json">__PAYLOAD__</script><script>
const p=JSON.parse(document.getElementById('packet').textContent), key='radar-external-'+p.protocol_sha256;
const blank=()=>({schema:'radar_external_human_responses_v1',protocol_sha256:p.protocol_sha256,human_gold:false,reviewer_identity:null,reference_file_loaded_at:null,review_started_at:null,review_completed_at:null,rows:p.questions.map(q=>({question_id:q.question_id,independent_answer:'',independent_locators:'',independent_uncertainties:'',independent_saved_at:null,reference_opened_at:null,independent_at_first_reference:null,reference_judgment:'',revised_answer:'',revision_reason:'',human_verified:null}))});
let state=blank(), refs=null;try{const old=JSON.parse(localStorage.getItem(key));if(old&&old.protocol_sha256===p.protocol_sha256)state=old}catch(e){}
const $=id=>document.getElementById(id), stamp=()=>new Date().toISOString();
const save=()=>{try{localStorage.setItem(key,JSON.stringify(state));$('status').textContent='本机已保存；仍需下载文件'}catch(e){$('status').textContent='本机保存失败，请下载文件'}};
for(const id of ['reviewer_name','reviewer_role','prior_reference_exposure']){$(id).value=state[id]||'';$(id).oninput=()=>{state[id]=$(id).value;state.reviewer_identity={name:$('reviewer_name').value,role:$('reviewer_role').value};state.review_completed_at=null;save()}}
function field(parent,label,name,row){const l=document.createElement('label');l.textContent=label;const t=document.createElement('textarea');t.value=row[name]||'';t.oninput=()=>{row[name]=t.value;save()};l.append(t);parent.append(l);return t}
function showReference(box,ref){box.replaceChildren();const heading=document.createElement('h3');heading.textContent='AI 参考（请对照原文核验，不是人工标答）';box.append(heading);const answer=document.createElement('p');answer.textContent=ref.reference.answer_zh||'';box.append(answer);for(const item of ref.scoring_interpretation||[]){const note=document.createElement('p');note.textContent='冻结的参考措辞解释：'+(item.interpretation_zh||[]).join(' ')+' '+(item.source_conditions_interpretation_zh||'');box.append(note)}const details=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre');summary.textContent='展开原参考要点、原文定位及完整解释';pre.textContent=JSON.stringify(ref,null,2);details.append(summary,pre);box.append(details);box.hidden=false}
for(const q of p.questions){const row=state.rows.find(r=>r.question_id===q.question_id), card=document.createElement('article'), h=document.createElement('h2');h.textContent=q.question_id+' · '+q.primary_type;card.append(h);const question=document.createElement('p');question.textContent=q.question;card.append(question);const version=document.createElement('p');version.textContent=q.source_version;card.append(version);
for(const src of q.sources){const a=document.createElement('a');a.href=src.href;a.target='_blank';a.rel='noopener';a.textContent=src.source_id+(src.page_one_based?'：PDF 第 '+src.page_one_based+' 页':'：归档整节文本');card.append(a,document.createElement('br'))}
const before=[field(card,'独立回答（不知道可留空，在不确定性中说明）','independent_answer',row),field(card,'自己定位的页码/行/表格依据','independent_locators',row),field(card,'不确定性或来源/题目问题','independent_uncertainties',row)];
const lock=document.createElement('button');lock.textContent=row.independent_saved_at?'原始独立记录已保存':'保存此题原始独立判断';lock.onclick=()=>{if(!row.independent_answer.trim()&&!row.independent_uncertainties.trim()){alert('请先记录回答或不确定性。');return}if(!row.independent_saved_at){row.independent_saved_at=stamp();state.review_started_at=state.review_started_at||row.independent_saved_at;row.independent_snapshot={answer:row.independent_answer,locators:row.independent_locators,uncertainties:row.independent_uncertainties,saved_at:row.independent_saved_at,prior_reference_exposure:$('prior_reference_exposure').value,reference_file_loaded_at:state.reference_file_loaded_at};save()}before.forEach(t=>t.disabled=true);lock.disabled=true;lock.textContent='原始独立记录已保存'};card.append(lock);if(row.independent_saved_at){before.forEach(t=>t.disabled=true);lock.disabled=true}
const reveal=document.createElement('button');reveal.textContent='展开此题 AI 参考（记录时间）';const box=document.createElement('div');box.className='reference';box.hidden=true;
reveal.onclick=()=>{if(!row.independent_saved_at){alert('请先保存该题独立判断。');return}if(!refs){alert('请先用上方按钮选择 references.json。');return}if(!row.reference_opened_at){row.reference_opened_at=stamp();row.independent_at_first_reference=JSON.parse(JSON.stringify(row.independent_snapshot));save()}showReference(box,refs.rows.find(r=>r.question_id===q.question_id))};card.append(reveal,box);
field(card,'查看原文/参考后，对参考是否受支持、需更正或无法判断的意见','reference_judgment',row);field(card,'修订后的回答（如有）','revised_answer',row);field(card,'修订原因（区分原文判断与参考影响）','revision_reason',row);$('items').append(card)}
$('choose').onclick=()=>$('refs').click();$('refs').onchange=async()=>{try{const file=$('refs').files[0];if(!file)return;const bytes=await file.arrayBuffer();const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))).map(x=>x.toString(16).padStart(2,'0')).join('');if(hash!==p.reference_sha256)throw Error('参考 sidecar SHA256 不匹配');const doc=JSON.parse(new TextDecoder().decode(bytes));if(doc.protocol_sha256!==p.protocol_sha256)throw Error('协议不匹配');refs=doc;state.reference_file_loaded_at=state.reference_file_loaded_at||stamp();save();alert('AI 参考已加载。每题仍需先保存独立判断，再主动展开。')}catch(e){alert(String(e))}};
$('complete').onclick=()=>{if(!$('reviewer_name').value.trim()||!$('reviewer_role').value.trim()){alert('请先填写身份或代号和专业背景。');return}state.review_completed_at=stamp();state.reviewer_identity={name:$('reviewer_name').value,role:$('reviewer_role').value};save();alert('填写完成时间已记录，请下载并保存文件。未答题仍为空，不会自动判为已通过。')};
$('download').onclick=()=>{save();const exported={...state,exported_at:stamp()};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(exported,null,2)+'\n'],{type:'application/json'}));a.download='external-review-responses.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = build(args.protocol)
    publish(outputs, check=args.check)
    print(json.dumps({"status": "reproduction_checked" if args.check else "packet_prepared_human_review_not_performed",
                      "questions": 24, "human_reviews_completed": 0, "files": len(outputs)}))


if __name__ == "__main__":
    main()
