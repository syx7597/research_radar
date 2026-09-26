"""
S3 多 agent LLM 抽取：Agent① 实体侦察 → Agent② 关系抽取（+ Agent④ schema 提案收集）。

设计要点（每条有实验依据）：
- 两段式（KGGen 式）：head 只能从①盘点出的型号里选，治"张冠李戴"（金标错误①）
- 关系子集按章节路由（ODKE+ 式），prompt 由 schema.py 自动编译
- 只抽 24 个实体关系；数值属性本轮只信 S2 结构化源（v2 文本数值层最弱的教训）
- Agent④ 零成本：②的输出里带 proposals 字段收集 schema 装不下的关系，人工终审
- 每 chunk 结果落盘 s3_raw.jsonl，断点续跑；LLM 层还有磁盘缓存双保险

输入:  work/chunks.jsonl (kind=text)
输出:  work/s3_raw.jsonl  / work/s3_proposals.jsonl
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "pipeline" / "v3"))

import llm_client as L  # noqa: E402
import schema as S      # noqa: E402

WORK     = ROOT / "pipeline" / "v3" / "work"
RAW_OUT  = WORK / "s3_raw.jsonl"
PROP_OUT = WORK / "s3_proposals.jsonl"
WORKERS  = 6
MIN_TEXT = 80

SCOUT_SYSTEM = """你是军事雷达情报文本的实体侦察员。从给定文本中列出**文本里实际出现**的实体，只输出 JSON：
{"radars": ["文本中出现的雷达/电子系统型号，按原文写法"],
 "platforms": ["舰艇/飞机/车辆平台"],
 "manufacturers": ["厂商/研制机构"],
 "countries": ["国家"]}
规则：只列文本中字面出现的名称；不确定是不是雷达型号的不要放进 radars；不要推断、不要补全。"""

EXTRACT_SYSTEM = """你是雷达知识图谱抽取员。从文本中抽取三元组，只输出 JSON：
{"triples": [{"head": "...", "relation": "...", "tail": "...", "evidence": "≤30词的原文引文"}],
 "proposals": [{"head": "...", "relation": "自拟关系名", "tail": "...", "evidence": "原文引文"}]}

硬规则：
1. relation 只能从下面给定的关系清单中选；清单外但文本明确表达的真实关系放入 proposals（不要丢弃也不要硬塞）
2. head 必须从给定的雷达型号清单中选，按清单写法
3. tail 必须是文本中出现的表述，照抄原文写法
4. evidence 必须是包含该事实的原文连续片段（≤30词），不得改写
5. 文本没有明确表达的关系不要输出；宁缺毋滥
6. 例外——列表章节：如果章节标题表明了语义（如 Operators/Users/Variants），
   章节里的裸列表条目（"- Serbia"）默认与**页面主体**构成该章节含义的关系
   （Operators 列表项 → 页面主体 operatedBy 列表项）；带注释的条目
   （"- Switzerland : TAFLIR"）若给出本地型号，head 用页面主体仍成立"""

EXTRACT_USER_TPL = """## 可用关系清单
{rel_block}

## 雷达型号清单（head 只能从这里选）
{radars}

## 页面主体（文本用代词或省略主语时通常指它）
{hint}

## 文本（章节：{heading}）
{text}"""


def scout(chunk: dict) -> dict:
    resp = L.chat([{"role": "system", "content": SCOUT_SYSTEM},
                   {"role": "user", "content": chunk["text"][:3000]}],
                  max_tokens=500)
    data = L.parse_json(resp, {})
    if not isinstance(data, dict):        # LLM 偶尔返回裸数组
        data = {}
    radars = []
    for r in data.get("radars", []):
        r = str(r).strip()
        if not (2 <= len(r) <= 60) or r in S.NON_RADAR_ENTITIES:
            continue
        radars.append(r)
    hint = chunk["radar_hint"]
    hint_listed = any(r.lower() == hint.lower() for r in radars)
    # WEG 条目主体是导弹/防空系统而非雷达 → 不强制 hint 做 head，纯靠侦察
    if not hint_listed and chunk.get("source_kind") != "weg":
        radars.append(hint)          # 页面主体始终可作 head；S4 会按是否在文中降层
    return {"radars": radars[:15],
            "platforms": [str(x).strip() for x in data.get("platforms", [])][:15],
            "countries": [str(x).strip() for x in data.get("countries", [])][:20]}


def extract(chunk: dict, inventory: dict) -> tuple[list, list]:
    rels = S.route_relations(chunk.get("heading"), chunk["source_kind"])
    user = EXTRACT_USER_TPL.format(
        rel_block=S.prompt_block(rels),
        radars="\n".join(f"- {r}" for r in inventory["radars"]),
        hint=chunk["radar_hint"],
        heading=chunk.get("heading") or "（导语）",
        text=chunk["text"][:3000])
    resp = L.chat([{"role": "system", "content": EXTRACT_SYSTEM},
                   {"role": "user", "content": user}], max_tokens=1400)
    data = L.parse_json(resp, {})
    if isinstance(data, list):            # LLM 直接给了 triples 数组
        data = {"triples": data}
    elif not isinstance(data, dict):
        data = {}
    allowed = {r.lower() for r in inventory["radars"]}
    triples, proposals = [], []
    for t in data.get("triples", []):
        if not isinstance(t, dict):
            continue
        head, rel = str(t.get("head", "")).strip(), str(t.get("relation", "")).strip()
        tail, ev  = str(t.get("tail", "")).strip(), str(t.get("evidence", "")).strip()
        if not head or not tail or rel not in S.EDGE_RELATIONS:
            continue
        if head.lower() not in allowed:          # 两段式硬约束
            continue
        triples.append({"head": head, "relation": rel, "tail": tail,
                        "evidence": ev[:300], "chunk_id": chunk["chunk_id"],
                        "doc_id": chunk["doc_id"], "source_kind": chunk["source_kind"],
                        "heading": chunk.get("heading"), "lang": chunk["lang"]})
    for p in data.get("proposals", []):
        if isinstance(p, dict) and p.get("relation"):
            p["chunk_id"] = chunk["chunk_id"]
            proposals.append(p)
    return triples, proposals


WEG_SYSTEM = """本文本来自装备指南，描述一个防空/导弹武器系统「{system}」。请抽取，只输出 JSON：
{{"radar_edges": [{{"head": "系统中的雷达型号", "relation": "partOfSystem|hasFunction|hasFrequencyBand|hasMode|hasTechType|hasVariant|deployedOn", "tail": "...", "evidence": "≤30词原文"}}],
 "system_facts": [{{"relation": "countryOfOrigin|developedBy", "tail": "国家或研制方", "evidence": "≤30词原文"}}]}}

规则：
1. radar_edges 的 head 只能从下面的雷达清单选；partOfSystem 的 tail 一律写「{system}」
2. 只把**明确属于本系统**的雷达用 partOfSystem 连接（排除文中对比提及的其它系统雷达）
3. system_facts 是系统「{system}」本身的国别/研制方
4. tail 照抄原文写法；evidence 必须是原文片段；文本没明说的不要输出"""


def extract_weg(chunk: dict, inv: dict) -> tuple[list, list]:
    system = chunk["radar_hint"]
    allowed = {r.lower() for r in inv["radars"]}
    resp = L.chat([{"role": "system", "content": WEG_SYSTEM.format(system=system)},
                   {"role": "user", "content":
                    "雷达清单:\n" + "\n".join(f"- {r}" for r in inv["radars"])
                    + f"\n\n文本:\n{chunk['text'][:3000]}"}], max_tokens=1200)
    data = L.parse_json(resp, {})
    if not isinstance(data, dict):
        return [], []
    base = {"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"],
            "source_kind": "weg", "heading": chunk.get("heading"), "lang": chunk["lang"]}
    out = []
    for t in data.get("radar_edges", []):
        if not isinstance(t, dict):
            continue
        head, rel = str(t.get("head", "")).strip(), str(t.get("relation", "")).strip()
        tail, ev = str(t.get("tail", "")).strip(), str(t.get("evidence", "")).strip()
        if not head or rel not in S.EDGE_RELATIONS or head.lower() not in allowed:
            continue
        if rel == "partOfSystem":
            tail = system                      # 统一 tail 为条目系统名
        elif not tail:
            continue
        out.append({**base, "head": head, "relation": rel, "tail": tail,
                    "evidence": ev[:300]})
    for t in data.get("system_facts", []):     # 系统本身的国别/研制方（head=系统）
        if not isinstance(t, dict):
            continue
        rel, tail = str(t.get("relation", "")).strip(), str(t.get("tail", "")).strip()
        ev = str(t.get("evidence", "")).strip()
        if rel in ("countryOfOrigin", "developedBy") and tail:
            out.append({**base, "head": system, "relation": rel, "tail": tail,
                        "evidence": ev[:300], "head_is_system": True})
    return out, []


def process_chunk(chunk: dict) -> dict:
    inv = scout(chunk)
    if chunk.get("source_kind") == "weg":
        triples, proposals = extract_weg(chunk, inv)
    else:
        triples, proposals = extract(chunk, inv)
    return {"chunk_id": chunk["chunk_id"], "inventory": inv,
            "triples": triples, "proposals": proposals}


def main():
    chunks = [json.loads(l) for l in (WORK / "chunks.jsonl").open(encoding="utf-8")]
    chunks = [c for c in chunks if c["kind"] == "text" and len(c["text"]) >= MIN_TEXT]

    done = set()
    if RAW_OUT.exists():
        for l in RAW_OUT.open(encoding="utf-8"):
            try:
                done.add(json.loads(l)["chunk_id"])
            except Exception:
                pass
    todo = [c for c in chunks if c["chunk_id"] not in done]
    print(f"[S3] chunks: {len(chunks)} total, {len(done)} done, {len(todo)} to run")

    raw_f  = RAW_OUT.open("a", encoding="utf-8")
    prop_f = PROP_OUT.open("a", encoding="utf-8")
    n = n_triples = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_chunk, c): c for c in todo}
        for fut in as_completed(futs):
            try:
                res = fut.result()
            except Exception as e:
                print(f"  !! {futs[fut]['chunk_id']}: {e}")
                continue
            raw_f.write(json.dumps(res, ensure_ascii=False) + "\n")
            for p in res["proposals"]:
                prop_f.write(json.dumps(p, ensure_ascii=False) + "\n")
            n += 1
            n_triples += len(res["triples"])
            if n % 50 == 0:
                raw_f.flush()
                print(f"[S3] {n}/{len(todo)} chunks, {n_triples} triples, "
                      f"LLM calls {L.STATS['calls']} (cache {L.STATS['cache_hits']})")
    raw_f.close(); prop_f.close()
    print(f"[S3] done: +{n} chunks, +{n_triples} triples")
    print(f"[S3] LLM: {L.STATS}")


if __name__ == "__main__":
    main()
