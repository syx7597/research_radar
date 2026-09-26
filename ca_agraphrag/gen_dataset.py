# -*- coding: utf-8 -*-
"""放量生成正式题集 + 实体级切分(防泄露,测泛化)。
复用 gen_probe 验证过的过滤/金标逻辑,提高各型题量,按【实体/组 → 桶】切 train/dev/test。
实体级切分:同一雷达(锚定题)或同一分组(聚合题)只落一个桶,test 与 train 不共享实体。

  python ca_agraphrag/gen_dataset.py
输出 ca_agraphrag/data/{train,dev,test}.jsonl
"""
import json
import re
import random
import hashlib
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "ca_agraphrag" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rng = random.Random(42)

# 每型目标(总量,切分后约 80/10/10)
CAP = {"single_hop": 1500, "two_hop": 1200, "count": 800, "enumerate": 800,
       "comparison": 900, "negation": 1000, "multi_constraint": 500}

E = json.load(open(ROOT / "kg_v3" / "edges.json", encoding="utf-8"))
Nodes = json.load(open(ROOT / "kg_v3" / "entities.json", encoding="utf-8"))
type_of = {n["name"]: n.get("type", "") for n in Nodes}

# ---------- 过滤(与 gen_probe 一致) ----------
FIELD = re.compile(r"^(功能|体制|频段|研制|装备|现状|作用距离|天线|发射机|接收机)")
def clean(name):
    if not name or len(name) > 45:
        return False
    if re.fullmatch(r"[\d.,\s]+(km|m|kg|GHz|MHz)?", name, re.I):
        return False
    if FIELD.match(name):
        return False
    if re.search(r"[：:]|<br", name):        # 字段粘连/规格残留(head 或 tail 均过滤)
        return False
    return bool(re.search(r"[A-Za-z0-9一-鿿]", name))

HI_TIER = {"v3_manual", "v3_wikidata", "v3_struct"}
SINGLE_VAL = {"developedBy", "partOfSystem", "headquarteredIn"}
LINEAGE = {"derivedFrom", "hasVariant", "replaces"}   # 谱系关系放宽接受 llm_grounded(0.85,可靠且多跳需要)
def high_conf(e):
    r = e["relation"]
    if r in SINGLE_VAL:
        return bool(e.get("corroborated"))
    if r in LINEAGE:
        return e.get("tier") in (HI_TIER | {"v3_llm_grounded"}) or e.get("corroborated")
    return e.get("tier") in HI_TIER or e.get("corroborated")

BAND_OK = re.compile(r"^(X|S|C|L|K|Ku|Ka|Ka?u|UHF|VHF|HF|EHF|SHF|P|W|D|E|F|G|H|I|J)([\s/-]|$)", re.I)
VALID_COUNTRY = {
    "美国", "美國", "俄罗斯", "蘇聯", "苏联", "英国", "英國", "法国", "加拿大", "意大利",
    "瑞典", "以色列", "德国", "日本", "中国", "中国台湾", "荷兰", "印度", "澳大利亚", "波兰",
    "西班牙", "丹麦", "伊朗", "乌克兰", "韩国", "土耳其", "捷克", "比利时", "挪威", "罗马尼亚",
    "巴西", "瑞士", "南非", "巴基斯坦", "保加利亚", "泰国",
    "American", "United Kingdom of Great Britain and Ireland", "Bulgaria", "Kazakhstan",
    "Estonia", "Slovakia", "German Democratic Republic", "Republic of the Congo",
    "Central African Republic", "Benin",
}

# ===== 上游 KG 脏值清洗(人工核验发现的 4 类系统性问题) =====
known_models = {n["name"] for n in Nodes
                if n.get("type") in ("Radar", "RadarSystem", "Aircraft",
                                      "NavalVessel", "GroundVehicle", "Platform")}
DIRTY = re.compile(r"[：:]|<br\s*/?>")                         # ①字段粘连:全/半角冒号或<br>
DESIG = re.compile(r"^(AN/|APG|APS|APQ|SPG|SPS|SPY|MPQ|TPS|TPY|MiG-|F-\d)", re.I)
AIRCRAFT_TOK = re.compile(r"(F4D|Skyray|Tornado|Mirage|MiG-|F-1[56])", re.I)

def norm_band(b):                                             # ②频段书写归一
    b = re.sub(r"\s*(band|波段)\s*$", "", b.strip(), flags=re.I)
    b = re.sub(r"[一-鿿]", "", b)                    # 去中文残留(如"低")
    b = re.sub(r"\bK\s+u\b", "Ku", b, flags=re.I)           # "K u" -> "Ku"
    b = re.sub(r"\s*/\s*", "/", b)                           # 斜杠周围空格
    return re.sub(r"\s+", "", b).strip("/")

def is_company(tail):                                         # ③developedBy 非公司混入
    return not (tail in known_models or DESIG.match(tail) or AIRCRAFT_TOK.search(tail))

COMPANY_ALIAS = {"westinghouse electrics": "westinghouse electric"}
DEPT_SUFFIX = re.compile(
    r"\s+(avionica|navigation systems.*|norden systems?|electronic systems|"
    r"maritime sensors.*|systems canada)$", re.I)
def canon_company(c):                                         # ④公司名归一(供比较判同)
    k = re.sub(r"\s+", " ", c.strip()).lower()
    k = COMPANY_ALIAS.get(k, k)
    return DEPT_SUFFIX.sub("", k).strip()

hr2t = defaultdict(set)
rt2h = defaultdict(set)
for e in E:
    if not (clean(e["head"]) and clean(e["tail"])):
        continue
    if not high_conf(e):
        continue
    rel, tail = e["relation"], e["tail"].strip()
    if DIRTY.search(tail):                                    # ①跳过字段粘连脏值
        continue
    if rel == "hasFrequencyBand":
        tail = norm_band(tail)                                # ②归一后再校验
        if not tail or not BAND_OK.match(tail):
            continue
    if rel == "countryOfOrigin" and tail not in VALID_COUNTRY:
        continue
    if rel == "developedBy" and not is_company(tail):         # ③剔除非公司研制方
        continue
    hr2t[(e["head"], rel)].add(tail)
    rt2h[(rel, tail)].add(e["head"])

# ---------- 实体/组 → 桶(稳定哈希, 80/10/10) ----------
def bucket(key):
    h = int(hashlib.md5(str(key).encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("dev" if h < 90 else "test")

REL_ZH = {"developedBy": "的研制方是", "countryOfOrigin": "的原产国是",
          "hasFrequencyBand": "工作在哪个频段", "deployedOn": "部署在哪个平台上",
          "operatedBy": "被哪个国家/军种使用", "derivedFrom": "衍生自哪个型号",
          "partOfSystem": "属于哪个武器系统"}
NEG_ZH = {"deployedOn": "部署在 {} 上", "operatedBy": "被 {} 使用", "hasFrequencyBand": "工作在 {} 频段"}
R1_ZH = {"derivedFrom": "衍生自的源型号", "replaces": "所替代的型号", "hasVariant": "的某个变体型号"}
GROUP = {"countryOfOrigin": "原产于{}", "hasFrequencyBand": "工作在{}频段",
         "developedBy": "由{}研制", "operatedBy": "被{}使用",
         "hasMode": "支持{}模式", "hasFunction": "功能为{}",
         "hasTechType": "采用{}体制", "deployedOn": "部署在{}上"}
# 多约束/比较可用的约束关系(tail 相对干净)
CONSTRAINT_RELS = ["countryOfOrigin", "hasFrequencyBand", "hasMode",
                   "hasFunction", "hasTechType", "deployedOn", "operatedBy"]
CONS_ZH = {"countryOfOrigin": "原产于{}", "hasFrequencyBand": "工作在{}频段",
           "hasMode": "支持{}模式", "hasFunction": "功能为{}",
           "hasTechType": "采用{}体制", "deployedOn": "部署在{}上", "operatedBy": "被{}使用"}
CMP_ZH = {"countryOfOrigin": "原产国", "developedBy": "研制方",
          "hasFrequencyBand": "工作频段", "deployedOn": "部署平台"}
SINGLE_HOP_RELS = ["developedBy", "countryOfOrigin", "hasFrequencyBand",
                   "deployedOn", "operatedBy", "derivedFrom", "partOfSystem"]

buckets = {"train": [], "dev": [], "test": []}
seen_q = set()
qid = 0
def add(anchor, qtype, question, gold, kind, support, tools):
    global qid
    if question in seen_q:
        return
    seen_q.add(question)
    qid += 1
    b = bucket(anchor)
    buckets[b].append({"qid": f"{qtype}-{qid:05d}", "type": qtype, "question": question,
                       "gold_answer": gold, "gold_kind": kind, "gold_support": support,
                       "expected_tools": tools, "split": b, "anchor": anchor})

# 1 single_hop
pool = [(h, r, sorted(t)) for (h, r), t in hr2t.items()
        if r in SINGLE_HOP_RELS and type_of.get(h) == "Radar"]
rng.shuffle(pool)
for h, r, t in pool[:CAP["single_hop"]]:
    add(h, "single_hop", f"{h}{REL_ZH.get(r, '的'+r+'是')}？", t,
        "set" if len(t) > 1 else "scalar", [{"head": h, "rel": r, "tail": t}], ["graph_lookup"])

# 2 two_hop —— 两跳都走过滤视图 hr2t(否则环境工具走不通第一跳)
two = []
for (h, r1), mids in hr2t.items():
    if r1 not in ("derivedFrom", "replaces", "hasVariant") or len(mids) != 1:
        continue                              # 只取单一源型号的头,避免"衍生自"多义
    mid = next(iter(mids))
    for r2 in ("developedBy", "countryOfOrigin"):
        tl = hr2t.get((mid, r2))
        if tl:
            two.append((h, r1, mid, r2, sorted(tl)))
rng.shuffle(two)
for h, r1, mid, r2, t in two[:CAP["two_hop"]]:
    add(h, "two_hop", f"{h}{R1_ZH[r1]}，其{REL_ZH.get(r2, r2)[1:]}？", t,
        "set" if len(t) > 1 else "scalar",
        [{"head": h, "rel": r1, "tail": mid}, {"head": mid, "rel": r2, "tail": t}],
        ["graph_lookup", "subgraph"])

# 3+4 count / enumerate
for r, tmpl in GROUP.items():
    groups = [(tail, sorted(h for h in hs if type_of.get(h) == "Radar"))
              for (rr, tail), hs in rt2h.items() if rr == r]
    groups = [(tail, hs) for tail, hs in groups if 3 <= len(hs) <= 30]
    for tail, hs in groups:
        add((r, tail), "count", f"{tmpl.format(tail)}的雷达共有多少款？", len(hs), "count",
            [{"rel": r, "tail": tail, "members": hs}], ["set_op", "count"])
        add((r, tail), "enumerate", f"列出所有{tmpl.format(tail)}的雷达。", hs, "set",
            [{"rel": r, "tail": tail, "members": hs}], ["set_op", "enumerate"])

# 5 comparison (多维度: 原产国 / 研制方 / 频段 / 平台)
radars_with = defaultdict(list)
for (h, r), t in hr2t.items():
    if r in CMP_ZH and type_of.get(h) == "Radar":
        radars_with[r].append(h)
cmp_items = []
for r, rs in radars_with.items():
    if len(rs) < 2:
        continue
    for _ in range(min(len(rs) * 3, CAP["comparison"])):
        a, b = rng.sample(rs, 2)
        cmp_items.append((r, a, b))
rng.shuffle(cmp_items)
for r, a, b in cmp_items[:CAP["comparison"]]:
    va, vb = sorted(hr2t.get((a, r), [])), sorted(hr2t.get((b, r), []))
    if not va or not vb:
        continue
    if r == "developedBy":                       # 公司名归一后判同(治拼写/子公司)
        same = set(map(canon_company, va)) == set(map(canon_company, vb))
    else:
        same = (va == vb)
    add(a, "comparison", f"{a} 和 {b} 的{CMP_ZH[r]}是否相同？",
        {"same": same, "a": va, "b": vb, "dim": r}, "compare",
        [{"a": a, "va": va}, {"b": b, "vb": vb}], ["graph_lookup"])

# 6 negation
neg_rels = ["deployedOn", "operatedBy", "hasFrequencyBand"]
radars = [n["name"] for n in Nodes if n.get("type") == "Radar" and clean(n["name"])]
for _ in range(CAP["negation"] * 2):
    if sum(1 for x in buckets.values() for q in x if q["type"] == "negation") >= CAP["negation"]:
        break
    r = rng.choice(neg_rels); h = rng.choice(radars)
    real = sorted(hr2t.get((h, r), []))
    if real and rng.random() < 0.5:
        tail = rng.choice(real); gold = True
    else:
        allt = list({t for (rr, t) in rt2h if rr == r})
        if not allt:
            continue
        tail = rng.choice(allt); gold = tail in real
    if r == "hasFrequencyBand" and not BAND_OK.match(tail.strip()):
        continue
    add(h, "negation", f"{h} 是否{NEG_ZH[r].format(tail)}？", gold, "bool",
        [{"head": h, "rel": r, "candidate": tail, "actual": real}], ["graph_lookup"])

# 7 multi_constraint (实体画像取任意两不同关系约束,天然可满足,组合丰富)
head_cons = defaultdict(list)
for (h, r), ts in hr2t.items():
    if r in CONSTRAINT_RELS and type_of.get(h) == "Radar":
        for t in ts:
            head_cons[h].append((r, t))
mc_seen, mc_items = set(), []
heads = [h for h in head_cons if len({r for r, _ in head_cons[h]}) >= 2]
rng.shuffle(heads)
for h in heads:
    cons = head_cons[h][:]
    rng.shuffle(cons)
    made = False
    for i in range(len(cons)):
        for j in range(i + 1, len(cons)):
            (r1, t1), (r2, t2) = cons[i], cons[j]
            if r1 == r2:
                continue
            key = tuple(sorted([(r1, t1), (r2, t2)]))
            if key in mc_seen:
                continue
            inter = sorted({x for x in (rt2h.get((r1, t1), set()) & rt2h.get((r2, t2), set()))
                            if type_of.get(x) == "Radar"})
            if 2 <= len(inter) <= 20:
                mc_seen.add(key)
                mc_items.append((r1, t1, r2, t2, inter))
                made = True
                break
        if made:
            break
    if len(mc_items) >= CAP["multi_constraint"] * 2:
        break
rng.shuffle(mc_items)
for r1, t1, r2, t2, inter in mc_items[:CAP["multi_constraint"]]:
    q = f"{CONS_ZH[r1].format(t1)}且{CONS_ZH[r2].format(t2)}的雷达有哪些？"
    add((r1, t1, r2, t2), "multi_constraint", q, inter, "set",
        [{"c1": (r1, t1), "c2": (r2, t2), "members": inter}], ["set_op", "enumerate"])

# ---------- 落盘 + 报告 ----------
for b, items in buckets.items():
    (OUT / f"{b}.jsonl").write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in items), encoding="utf-8")

print("=== 放量 + 实体级切分完成 ===")
tot = sum(len(v) for v in buckets.values())
print(f"总题 {tot}  ->  train {len(buckets['train'])} / dev {len(buckets['dev'])} / test {len(buckets['test'])}")
for b in ("train", "dev", "test"):
    c = Counter(q["type"] for q in buckets[b])
    print(f"  [{b:5}] " + "  ".join(f"{t}:{c[t]}" for t in CAP))

# 泄露检查:锚定实体是否跨桶
anchor_bucket = defaultdict(set)
for b, items in buckets.items():
    for q in items:
        anchor_bucket[str(q["anchor"])].add(b)
leak = sum(1 for v in anchor_bucket.values() if len(v) > 1)
print(f"\n泄露检查:跨桶锚点 {leak}(应为 0,实体级切分保证不泄露)")
