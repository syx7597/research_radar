"""
LLM-ASSISTED pre-annotation of annotation_facts.csv (NOT the gold label).
=========================================================================
Asks DeepSeek to judge each verifiable fact as 1/0/? with a one-line reason, to
speed up the human pass. HONEST CAVEAT: an LLM judging an (often LLM-extracted) KG
is partly circular; treat this as a PRE-LABEL to be reviewed by a human — focus
review on the 0s and ?s. The gold precision in the thesis must be human-verified.

Writes annotation_facts.csv (or _llm.csv if the first is open/locked) with the
`correct(1/0/?)` column pre-filled and the reason in `note` (prefixed [LLM]).
Key read from apikey.txt (never printed/committed).

Run:  PYTHONIOENCODING=utf-8 python experiments/kg_eval/llm_annotate.py
"""
import os, re, sys, csv, time
from pathlib import Path
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SHEET = HERE / "annotation_facts.csv"

GLOSS = {
    "developedBy": "研制方/厂商", "manufacturedBy": "制造商", "countryOfOrigin": "原产国",
    "operatedBy": "使用国", "deployedOn": "部署平台", "hasFrequencyBand": "工作频段",
    "hasTechType": "技术体制", "hasMode": "工作模式", "hasFunction": "功能",
    "frequency_GHz": "工作频率(GHz)", "range_km": "作用距离(km)", "compatibleWith": "兼容武器",
    "upgradeOf": "升级自(前代型号)", "derivedFrom": "衍生自", "decade": "年代",
    "affiliatedTo": "隶属国", "hasSubsystem": "子系统", "hasComponent": "部件",
    "peak_power_kW": "峰值功率(kW)", "weight_kg": "重量(kg)",
}
SYS = ("你是雷达装备领域的知识核查员。给定一条三元组(主体, 关系, 值)与抽取证据，"
       "判断该事实是否正确。优先依据证据(若证据是支持该结论的原文)，否则用你的领域知识。"
       "严格只输出一行：1(确信正确) / 0(确信错误) / ?(无法确定真伪)。"
       "【关键】若你并不真正掌握该具体型号的确切事实或参数(如冷僻型号的频率/距离等数值)，"
       "必须输出 ? ，绝不允许猜测。空格后跟≤20字理由。")


def load_key():
    if os.getenv("DEEPSEEK_API_KEY"):
        return os.environ["DEEPSEEK_API_KEY"]
    for i, ln in enumerate((ROOT / "apikey.txt").read_text(encoding="utf-8").splitlines()):
        if "api.deepseek.com" in ln:
            for nxt in (ROOT / "apikey.txt").read_text(encoding="utf-8").splitlines()[i:i+4]:
                m = re.search(r'api[_]?key\s*[=:]\s*"?([A-Za-z0-9\-]{16,})"?', nxt)
                if m:
                    return m.group(1)
    sys.exit("no DeepSeek key")


KEY = load_key()


def judge(head, rel, tail, ev):
    g = GLOSS.get(rel, rel)
    user = f"主体: {head}\n关系: {rel} ({g})\n值: {tail}\n证据: {ev}\n请判断。"
    try:
        r = requests.post("https://api.deepseek.com/v1/chat/completions",
                          headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
                          json={"model": "deepseek-chat", "temperature": 0, "max_tokens": 60,
                                "messages": [{"role": "system", "content": SYS},
                                             {"role": "user", "content": user}]}, timeout=30)
        out = r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return "?", f"err:{e}"
    m = re.match(r"\s*([10?])\s*(.*)", out)
    return (m.group(1), m.group(2).strip()[:30]) if m else ("?", out[:30])


def load_rows(path):
    raw = [ln for ln in open(path, encoding="utf-8-sig")]
    head_i = next(i for i, ln in enumerate(raw) if ln.lstrip().startswith("id,"))
    rows = list(csv.DictReader(raw[head_i:]))
    return raw[:head_i], list(csv.reader(raw[head_i:head_i+1]))[0], rows


def main():
    comment, header, rows = load_rows(SHEET)
    print(f"LLM pre-annotating {len(rows)} facts via DeepSeek...\n")
    dist = {"1": 0, "0": 0, "?": 0}
    for i, r in enumerate(rows, 1):
        lab, why = judge(r["head"], r["relation"], r["tail"], r.get("evidence", ""))
        r["correct(1/0/?)"] = lab
        r["note"] = f"[LLM] {why}"
        dist[lab if lab in dist else "?"] += 1
        if i % 20 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}  (1:{dist['1']} 0:{dist['0']} ?:{dist['?']})", flush=True)
        time.sleep(0.05)
    out = SHEET
    try:
        f = open(out, "w", newline="", encoding="utf-8-sig")
    except PermissionError:
        out = HERE / "annotation_facts_llm.csv"; f = open(out, "w", newline="", encoding="utf-8-sig")
    with f:
        w = csv.writer(f)
        for ln in comment:
            f.write(ln if ln.endswith("\n") else ln + "\n")
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(h, "") for h in header])
    print(f"\nwrote LLM pre-labels -> {out.name}")
    print(f"distribution: 1(对)={dist['1']}  0(错)={dist['0']}  ?(不确定)={dist['?']}")
    print("NEXT: review (esp. the 0s and ?s), correct, then run score_annotation.py")


if __name__ == "__main__":
    main()
