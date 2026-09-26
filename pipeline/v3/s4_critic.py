"""
S4 Agent③ 忠实度裁判（窄批判，已验证 +5pp；宽批判已证伪，此处禁止评论 schema）。

两级门控：
  1) 词法硬门控（零成本）：tail（含别名变体）须在块文本中出现；数值须数字命中
  2) LLM 窄裁判（只跑一级未过但引文真实存在的）：只判「引文是否支持论断」
另记录 head 接地状态：head 在文中 → grounded；head 只是页面主体 → hinted（降层）

输入:  work/s3_raw.jsonl + work/chunks.jsonl
输出:  work/s4_gated.jsonl / work/s4_rejected.jsonl / work/s4_report.json
"""

import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "pipeline" / "v3"))

import llm_client as L  # noqa: E402
import schema as S      # noqa: E402

WORK    = ROOT / "pipeline" / "v3" / "work"
WORKERS = 6

JUDGE_SYSTEM = """你是抽取忠实度裁判。给定【原文片段】和【论断】，只判断：原文片段是否支持该论断？
- 只依据片段文字本身，不使用你的世界知识
- 不要评价关系类型选得是否合理，只判断事实是否被原文表达
只输出 JSON：{"support": true} 或 {"support": false}"""


def _norm(s: str) -> str:
    return re.sub(r"[\s\-_/\.]+", "", s.lower())


def tail_in_text(tail: str, text: str, tail_type: str) -> bool:
    tn, xn = _norm(tail), _norm(text)
    if tn and tn in xn:
        return True
    # 别名变体：规范名→查它的所有表面形式是否在文中（反向同理）
    canon = S.resolve_alias(tail, tail_type)
    if canon != tail and _norm(canon) in xn:
        return True
    for (kind, surface), c in S._ALIAS_IDX.items():
        if kind == tail_type and c == canon and _norm(surface) in xn:
            return True
    # 数值型 tail：所有数字组都出现即可
    nums = re.findall(r"\d+(?:\.\d+)?", tail)
    if nums and all(n in text for n in nums):
        return True
    return False


def judge(evidence: str, triple: dict) -> bool:
    spec = S.RELATIONS.get(triple["relation"], {})
    tpl  = spec.get("triple_text_zh", "{head} {rel} {tail}")
    try:
        assertion = tpl.format(head=triple["head"], tail=triple["tail"])
    except Exception:
        assertion = f"{triple['head']} {triple['relation']} {triple['tail']}"
    resp = L.chat([{"role": "system", "content": JUDGE_SYSTEM},
                   {"role": "user", "content":
                    f"【原文片段】{evidence}\n【论断】{assertion}"}], max_tokens=50)
    data = L.parse_json(resp, {})
    return bool(isinstance(data, dict) and data.get("support"))


def gate(t: dict, text: str, hint: str) -> tuple[str, str]:
    """返回 (verdict, tier)。verdict ∈ pass/judge_pass/reject:<reason>"""
    tail_type = sorted(S.TYPE_SIG[t["relation"]][1])[0] if S.TYPE_SIG[t["relation"]][1] else "Entity"
    head_grounded = _norm(t["head"]) in _norm(text)
    hinted = not head_grounded and _norm(t["head"]) == _norm(hint)
    if not head_grounded and not hinted:
        return "reject:head_not_in_text", ""

    if tail_in_text(t["tail"], text, tail_type):
        return "pass", "v3_llm_hinted" if hinted else "v3_llm_grounded"

    ev = t.get("evidence", "").strip()
    # 引文必须真实存在于原文（防裁判被幻觉引文骗过）
    if ev and _norm(ev)[:60] in _norm(text):
        if judge(ev, t):
            return "judge_pass", "v3_judge_passed"
        return "reject:judge_unsupported", ""
    return "reject:tail_not_in_text", ""


def main():
    chunks = {c["chunk_id"]: c for c in
              (json.loads(l) for l in (WORK / "chunks.jsonl").open(encoding="utf-8"))
              if c["kind"] == "text"}
    triples = []
    for l in (WORK / "s3_raw.jsonl").open(encoding="utf-8"):
        triples.extend(json.loads(l)["triples"])
    print(f"[S4] gating {len(triples)} triples")

    gated_f, rej_f = (WORK / "s4_gated.jsonl").open("w", encoding="utf-8"), \
                     (WORK / "s4_rejected.jsonl").open("w", encoding="utf-8")
    stats = Counter()

    def work(t):
        c = chunks.get(t["chunk_id"])
        if not c:
            return t, "reject:chunk_missing", ""
        return t, *gate(t, c["text"], c["radar_hint"])

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for fut in as_completed([ex.submit(work, t) for t in triples]):
            t, verdict, tier = fut.result()
            stats[verdict.split(":")[0]] += 1
            if verdict.startswith("reject"):
                t["reject_reason"] = verdict.split(":", 1)[1]
                rej_f.write(json.dumps(t, ensure_ascii=False) + "\n")
                stats[verdict] += 1
            else:
                t["tier"] = tier
                gated_f.write(json.dumps(t, ensure_ascii=False) + "\n")
    gated_f.close(); rej_f.close()

    total = len(triples) or 1
    passed = stats["pass"] + stats["judge_pass"]
    report = {"total": len(triples), "passed": passed,
              "pass_rate": round(passed / total, 4),
              "lexical_pass": stats["pass"], "judge_pass": stats["judge_pass"],
              **{k: v for k, v in stats.items() if k.startswith("reject:")}}
    (WORK / "s4_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[S4] {report}")
    print(f"[S4] LLM: {L.STATS}")


if __name__ == "__main__":
    main()
