# -*- coding: utf-8 -*-
"""G2-(a):可判分题型的模板生成器(冒烟批)。
从 kg_v3 确定性生成带金标的题,每型一小批,并【自校验】金标(独立重算对得上)。
只读 kg_v3。输出 ca_agraphrag/data/probe_smoke.jsonl + 抽查摘要。

题型(均可自动判分,可进 RL reward):
  single_hop / two_hop / count / enumerate / comparison / negation / multi_constraint
"""
import json
import re
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "ca_agraphrag" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rng = random.Random(42)
PER = 40  # 每型题数(冒烟)

E = json.load(open(ROOT / "kg_v3" / "edges.json", encoding="utf-8"))
Nodes = json.load(open(ROOT / "kg_v3" / "entities.json", encoding="utf-8"))
type_of = {n["name"]: n.get("type", "") for n in Nodes}

# --- 干净实体名过滤(排除数字/字段词/超长) ---
FIELD = re.compile(r"^(功能|体制|频段|研制|装备|现状|作用距离|天线|发射机|接收机)")
def clean(name):
    if not name or len(name) > 45:
        return False
    if re.fullmatch(r"[\d.,\s]+(km|m|kg|GHz|MHz)?", name, re.I):
        return False
    if FIELD.match(name):
        return False
    return bool(re.search(r"[A-Za-z0-9一-鿿]", name))

# --- 高置信过滤:金标只从可信边生成(治 KG 数据噪声,呼应 trust 主题) ---
# 单值关系(应只有一个真值)在手册按使用国分章时被污染成多值(操作国当原产国),
# 故单值关系的金标【必须多源印证】——印证的那条才是真值;多值关系高层级即可。
# countryOfOrigin 已在 KG 层修复(操作国污染→operatedBy),不再需强制印证;
# developedBy/partOfSystem 仍可有多值冲突(分包/更替),金标保守要求印证。
HI_TIER = {"v3_manual", "v3_wikidata", "v3_struct"}
SINGLE_VAL = {"developedBy", "partOfSystem", "headquarteredIn"}
def high_conf(e):
    if e["relation"] in SINGLE_VAL:
        return bool(e.get("corroborated"))     # 单值关系:必须印证(治操作国污染)
    return e.get("tier") in HI_TIER or e.get("corroborated")

# --- 频段值白名单(治脏 tail) ---
BAND_OK = re.compile(r"^(X|S|C|L|K|Ku|Ka|Ka?u|UHF|VHF|HF|EHF|SHF|P|W|D|E|F|G|H|I|J)([\s/-]|$)", re.I)
# --- 国家白名单(闭集,治海用手册国别字段混入的系列名/文献/规格) ---
VALID_COUNTRY = {
    "美国", "美國", "俄罗斯", "蘇聯", "苏联", "英国", "英國", "法国", "加拿大", "意大利",
    "瑞典", "以色列", "德国", "日本", "中国", "中国台湾", "荷兰", "印度", "澳大利亚", "波兰",
    "西班牙", "丹麦", "伊朗", "乌克兰", "韩国", "土耳其", "捷克", "比利时", "挪威", "罗马尼亚",
    "巴西", "瑞士", "南非", "巴基斯坦", "保加利亚", "泰国",
    "American", "United Kingdom of Great Britain and Ireland", "Bulgaria", "Kazakhstan",
    "Estonia", "Slovakia", "German Democratic Republic", "Republic of the Congo",
    "Central African Republic", "Benin",
}

# --- 索引(只用高置信边) ---
hr2t = defaultdict(set)          # (head, rel) -> tails
rt2h = defaultdict(set)          # (rel, tail) -> heads
by_head = defaultdict(list)      # head -> edges
n_all, n_hi = 0, 0
for e in E:
    n_all += 1
    if not (clean(e["head"]) and clean(e["tail"])):
        continue
    if not high_conf(e):
        continue
    if e["relation"] == "hasFrequencyBand" and not BAND_OK.match(e["tail"].strip()):
        continue                                  # 跳过脏频段值
    if e["relation"] == "countryOfOrigin" and e["tail"] not in VALID_COUNTRY:
        continue                                  # 国别只认白名单(治系列名/文献/规格污染)
    n_hi += 1
    hr2t[(e["head"], e["relation"])].add(e["tail"])
    rt2h[(e["relation"], e["tail"])].add(e["head"])
    by_head[e["head"]].append(e)
print(f"[高置信过滤] {n_all} 边 -> 可用于金标 {n_hi} 边 ({n_hi/n_all:.0%})")

REL_ZH = {
    "developedBy": "的研制方是", "countryOfOrigin": "的原产国是",
    "hasFrequencyBand": "工作在哪个频段", "deployedOn": "部署在哪个平台上",
    "operatedBy": "被哪个国家/军种使用", "hasFunction": "的功能是",
    "hasMode": "支持哪种工作模式", "derivedFrom": "衍生自哪个型号",
    "hasVariant": "有哪个变体", "replaces": "替代了哪个型号",
    "partOfSystem": "属于哪个武器系统", "headquarteredIn": "总部位于",
}
SINGLE_HOP_RELS = ["developedBy", "countryOfOrigin", "hasFrequencyBand",
                   "deployedOn", "operatedBy", "derivedFrom", "partOfSystem"]
GROUP_RELS = {"countryOfOrigin": "原产于{}", "hasFrequencyBand": "工作在{}频段",
              "developedBy": "由{}研制", "operatedBy": "被{}使用"}

out = []
qid = 0

def add(qtype, question, gold, gold_kind, support, tools):
    global qid
    qid += 1
    out.append({"qid": f"{qtype}-{qid:04d}", "type": qtype, "question": question,
                "gold_answer": gold, "gold_kind": gold_kind,
                "gold_support": support, "expected_tools": tools})

# --- 1. single_hop ---
pool = [(h, r, sorted(t)) for (h, r), t in hr2t.items()
        if r in SINGLE_HOP_RELS and type_of.get(h) == "Radar"]
for h, r, t in rng.sample(pool, min(PER, len(pool))):
    add("single_hop", f"{h}{REL_ZH.get(r, '的'+r+'是')}？", t,
        "set" if len(t) > 1 else "scalar",
        [{"head": h, "rel": r, "tail": t}], ["graph_lookup"])

# --- 2. two_hop (derivedFrom/hasVariant ∘ developedBy|countryOfOrigin) ---
two = []
for e1 in E:
    if e1["relation"] not in ("derivedFrom", "replaces", "hasVariant"):
        continue
    mid = e1["tail"]
    for r2 in ("developedBy", "countryOfOrigin"):
        tails = hr2t.get((mid, r2))
        if tails and clean(e1["head"]) and clean(mid):
            two.append((e1["head"], e1["relation"], mid, r2, sorted(tails)))
R1_ZH = {"derivedFrom": "衍生自的源型号", "replaces": "所替代的型号", "hasVariant": "的某个变体型号"}
for h, r1, mid, r2, t in rng.sample(two, min(PER, len(two))):
    q = f"{h}{R1_ZH[r1]}，其{REL_ZH.get(r2,r2)[1:]}？"
    add("two_hop", q, t, "set" if len(t) > 1 else "scalar",
        [{"head": h, "rel": r1, "tail": mid}, {"head": mid, "rel": r2, "tail": t}],
        ["graph_lookup", "subgraph"])

# --- 3. count & 4. enumerate (同源:分组关系反向) ---
for r, tmpl in GROUP_RELS.items():
    groups = [(tail, sorted(h for h in hs if type_of.get(h) == "Radar"))
              for (rr, tail), hs in rt2h.items() if rr == r]
    groups = [(tail, hs) for tail, hs in groups if 3 <= len(hs) <= 30]
    for tail, hs in rng.sample(groups, min(PER // len(GROUP_RELS) + 1, len(groups))):
        add("count", f"{tmpl.format(tail)}的雷达共有多少款？", len(hs), "count",
            [{"rel": r, "tail": tail, "members": hs}], ["set_op", "count"])
        add("enumerate", f"列出所有{tmpl.format(tail)}的雷达。", hs, "set",
            [{"rel": r, "tail": tail, "members": hs}], ["set_op", "enumerate"])

# --- 5. comparison (仅取两边都有原产国的实体,避免缺数据误判) ---
NEG_ZH = {"deployedOn": "部署在 {} 上", "operatedBy": "被 {} 使用",
          "hasFrequencyBand": "工作在 {} 频段"}
by_maker = defaultdict(list)
for (r, tail), hs in rt2h.items():
    if r == "developedBy":
        for h in hs:
            if type_of.get(h) == "Radar" and hr2t.get((h, "countryOfOrigin")):
                by_maker[tail].append(h)
pairs = []
for tail, hs in by_maker.items():
    if len(hs) >= 2:
        for _ in range(min(3, len(hs))):
            pairs.append(tuple(rng.sample(hs, 2)))
for a, b in rng.sample(pairs, min(PER, len(pairs))):
    va, vb = sorted(hr2t.get((a, "countryOfOrigin"), [])), sorted(hr2t.get((b, "countryOfOrigin"), []))
    add("comparison", f"{a} 和 {b} 的原产国是否相同？",
        {"same": va == vb, "a": va, "b": vb}, "compare",
        [{"a": a, "va": va}, {"b": b, "vb": vb}], ["graph_lookup"])

# --- 6. negation (一半真一半假,专用否定措辞) ---
neg_rels = ["deployedOn", "operatedBy", "hasFrequencyBand"]
radars = [n["name"] for n in Nodes if n.get("type") == "Radar" and clean(n["name"])]
for _ in range(PER):
    r = rng.choice(neg_rels)
    h = rng.choice(radars)
    real = sorted(hr2t.get((h, r), []))
    if real and rng.random() < 0.5:
        tail = rng.choice(real); gold = True
    else:
        alltails = list({t for (rr, t) in rt2h if rr == r})
        if not alltails:
            continue
        tail = rng.choice(alltails); gold = tail in real
    add("negation", f"{h} 是否{NEG_ZH[r].format(tail)}？", gold, "bool",
        [{"head": h, "rel": r, "candidate": tail, "actual": real}], ["graph_lookup"])

# --- 7. multi_constraint (两关系交,金标=交集) ---
mc = []
for (r1, t1), h1 in rt2h.items():
    if r1 != "countryOfOrigin":
        continue
    for (r2, t2), h2 in rt2h.items():
        if r2 != "hasFrequencyBand":
            continue
        inter = sorted({h for h in (h1 & h2) if type_of.get(h) == "Radar"})
        if 2 <= len(inter) <= 20:
            mc.append((t1, t2, inter))
for t1, t2, inter in rng.sample(mc, min(PER, len(mc))):
    add("multi_constraint", f"原产于{t1}且工作在{t2}频段的雷达有哪些？", inter, "set",
        [{"c1": ("countryOfOrigin", t1), "c2": ("hasFrequencyBand", t2), "members": inter}],
        ["set_op", "enumerate"])

# --- 落盘 ---
(OUT / "probe_smoke.jsonl").write_text(
    "\n".join(json.dumps(o, ensure_ascii=False) for o in out), encoding="utf-8")

# --- 自校验:独立重算金标 ---
def recompute(o):
    t, g = o["type"], o["gold_answer"]
    s = o["gold_support"]
    if t == "single_hop":
        return sorted(hr2t.get((s[0]["head"], s[0]["rel"]), [])) == (g if isinstance(g, list) else [g])
    if t == "count":
        return len(sorted(h for h in rt2h.get((s[0]["rel"], s[0]["tail"]), []) if type_of.get(h) == "Radar")) == g
    if t == "negation":
        return (s[0]["candidate"] in s[0]["actual"]) == g
    if t == "multi_constraint":
        c1, c2 = s[0]["c1"], s[0]["c2"]
        inter = sorted({h for h in (rt2h.get(tuple(c1), set()) & rt2h.get(tuple(c2), set())) if type_of.get(h) == "Radar"})
        return inter == g
    return None  # 其余型抽查即可

from collections import Counter
cnt = Counter(o["type"] for o in out)
checks = [recompute(o) for o in out]
ok = sum(1 for c in checks if c is True)
bad = sum(1 for c in checks if c is False)
print(f"生成 {len(out)} 题 -> ca_agraphrag/data/probe_smoke.jsonl")
print("各型:", dict(cnt))
print(f"自校验(可重算的型): 通过 {ok} / 失败 {bad}")
print("\n=== 抽查(每型一例) ===")
seen = set()
for o in out:
    if o["type"] in seen:
        continue
    seen.add(o["type"])
    g = o["gold_answer"]
    gs = (str(g)[:60] + "…") if len(str(g)) > 60 else g
    print(f"[{o['type']}] {o['question']}")
    print(f"   金标({o['gold_kind']}): {gs}")
