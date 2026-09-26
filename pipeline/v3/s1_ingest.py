"""
S1 分块路由器：corpus.json → work/chunks.jsonl

- 每 doc 产出 0/1 条 struct 记录（infobox/RT 规格表）+ N 条 text 块
- wiki 按 "== 章节 ==" 切块（块 ≤CHUNK_CAP，超长按段落再切，块带章节标题）
- RT 卡整卡一块；GS 按段落聚块
- zh 正文一并入块（lang=zh）
- radar_hint = 页面主型号（仅提示，S3 不得强制作 head）
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS = ROOT / "radar_corpus" / "corpus.json"
OUT    = ROOT / "pipeline" / "v3" / "work" / "chunks.jsonl"

CHUNK_CAP = 2500
_SEC_RX   = re.compile(r"^(={2,4}) (.+?) \1$", re.M)


def _split_paras(text: str, cap: int) -> list[str]:
    """按段落聚合到 cap 以内；单段超长按句切。"""
    paras, out, buf = [p for p in text.split("\n") if p.strip()], [], ""
    for p in paras:
        if len(p) > cap:                       # 超长单段按句切
            for sent in re.split(r"(?<=[.!?。！？]) ", p):
                if len(buf) + len(sent) > cap and buf:
                    out.append(buf); buf = ""
                buf += (" " if buf else "") + sent
            continue
        if len(buf) + len(p) > cap and buf:
            out.append(buf); buf = ""
        buf += ("\n" if buf else "") + p
    if buf.strip():
        out.append(buf)
    return out


def _sections(text: str) -> list[tuple[str | None, str]]:
    """按 == 标题 == 切成 (heading, body) 列表；导语 heading=None。"""
    pieces, last, heading = [], 0, None
    for m in _SEC_RX.finditer(text):
        body = text[last:m.start()].strip()
        if body:
            pieces.append((heading, body))
        heading, last = m.group(2).strip(), m.end()
    tail = text[last:].strip()
    if tail:
        pieces.append((heading, tail))
    return pieces


def chunk_doc(doc: dict) -> list[dict]:
    did  = doc["id"]
    kind = ("rt" if did.endswith("__rt") else "gs" if did.endswith("__gs")
            else "weg" if did.endswith("__weg") else "wiki")
    hint = doc["en_title"]
    recs = []

    if doc.get("infobox"):
        recs.append({"chunk_id": f"{did}#ibx", "doc_id": did, "source_kind": kind,
                     "kind": "struct", "radar_hint": hint, "lang": "en",
                     "heading": "infobox", "struct": doc["infobox"], "text": ""})

    for lang, field in (("en", "raw_text_en"), ("zh", "raw_text_zh")):
        text = (doc.get(field) or "").strip()
        if not text:
            continue
        n = 0
        for heading, body in _sections(text):
            for piece in _split_paras(body, CHUNK_CAP):
                recs.append({"chunk_id": f"{did}#{lang}{n}", "doc_id": did,
                             "source_kind": kind, "kind": "text", "lang": lang,
                             "radar_hint": hint, "heading": heading,
                             "struct": None, "text": piece})
                n += 1
    return recs


def main():
    docs = json.loads(CORPUS.read_text(encoding="utf-8"))
    only = set(sys.argv[sys.argv.index("--docs") + 1].split(",")) \
        if "--docs" in sys.argv else None
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_struct = n_text = 0
    with OUT.open("w", encoding="utf-8") as f:
        for doc in docs:
            if only and doc["id"] not in only:
                continue
            for rec in chunk_doc(doc):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if rec["kind"] == "struct":
                    n_struct += 1
                else:
                    n_text += 1
    print(f"[S1] struct {n_struct} + text {n_text} chunks -> {OUT}")


if __name__ == "__main__":
    main()
