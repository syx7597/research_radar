# -*- coding: utf-8 -*-
"""G1 硬门:题型 × 能否自动判分(→可做 RL 奖励) × 可生成量级。
只读 kg_v3。判"哪些题型能从图确定性算出金标 → 可进 reward",并估各自可生成量。
核心矛盾:最想主攻的复杂/任务型问题,恰恰最难给可验证奖励。
"""
import json
from collections import defaultdict, Counter
from pathlib import Path
from itertools import combinations

ROOT = Path(__file__).resolve().parent.parent
E = json.load(open(ROOT / "kg_v3" / "edges.json", encoding="utf-8"))
Nodes = json.load(open(ROOT / "kg_v3" / "entities.json", encoding="utf-8"))

# 属性型关系(住实体的,不算实体间边)——这些进属性不进关系空间
# 实际 edges.json 只含实体间关系;数值属性在 entities.attributes。此处 E 均为实体间边。
rel_count = Counter(e["relation"] for e in E)
type_of = {n["name"]: n.get("type", "") for n in Nodes}

# 分组关系(count/enumerate 用):按 tail 分组统计 head 数
GROUP_RELS = ["countryOfOrigin", "hasFrequencyBand", "developedBy", "hasMode",
              "operatedBy", "hasTechType", "deployedOn", "hasFunction"]
# 谱系/多跳关系
CHAIN_RELS = ["derivedFrom", "hasVariant", "replaces", "replacedBy", "partOfSystem", "upgradeOf"]

# --- 单跳: 实体间边总数 ---
single_hop = len(E)

# --- 2跳可组合路径: r1.tail 作为 r2.head ---
by_head = defaultdict(list)
for e in E:
    by_head[e["head"]].append(e)
two_hop = 0
two_hop_examples = Counter()
for e1 in E:
    mid = e1["tail"]
    for e2 in by_head.get(mid, []):
        if e2["tail"] != e1["head"]:
            two_hop += 1
            two_hop_examples[f'{e1["relation"]}∘{e2["relation"]}'] += 1

# --- count / enumerate: 各分组关系下,组大小在 [3,30] 的组数(可判分且有区分度) ---
group_volume = {}
for r in GROUP_RELS:
    groups = defaultdict(set)
    for e in E:
        if e["relation"] == r:
            groups[e["tail"]].add(e["head"])
    usable = sum(1 for v in groups.values() if 3 <= len(v) <= 30)
    total_groups = len(groups)
    group_volume[r] = (usable, total_groups)

# --- 比较题: 同类型实体对,共享某关系 ---
# 估: 同 developedBy tail 下的雷达两两可比 → 组合数(取样即可,量级足够)
radar_by_maker = defaultdict(list)
for e in E:
    if e["relation"] == "developedBy" and type_of.get(e["head"]) == "Radar":
        radar_by_maker[e["tail"]].append(e["head"])
compare_pairs = sum(len(v) * (len(v) - 1) // 2 for v in radar_by_maker.values() if len(v) >= 2)

# --- 冲突题 ---
SINGLE_VAL = {"developedBy", "countryOfOrigin", "partOfSystem"}
byhr = defaultdict(set)
for e in E:
    if e["relation"] in SINGLE_VAL:
        byhr[(e["head"], e["relation"])].add(e["tail"])
conflicts = sum(1 for t in byhr.values() if len(t) > 1)

print("=" * 70)
print("G1: 题型 × 能否自动判分(→可做 RL 奖励) × 可生成量级")
print("=" * 70)

rows = [
    ("单跳事实", "✅ 可(EM)", f"{single_hop:,}", "每条实体间边=一题;可做 reward"),
    ("2跳桥接", "✅ 可(EM+路径核对)", f"{two_hop:,}", "可组合路径充足;去重后仍数千"),
    ("计数(count)", "✅ 可(精确数)", f"{sum(u for u,_ in group_volume.values()):,} 组", "分组关系的[3,30]组;见下明细"),
    ("枚举(set)", "✅ 可(集合F1)", "同上", "与计数同源,金标=分组成员集"),
    ("多约束过滤", "✅ 可(集合F1)*", "组合爆炸,取样", "*金标须限'有该属性/关系的覆盖子集'(缺失71%教训)"),
    ("否定/布尔", "✅ 可(是/否)", f"~{single_hop:,}+", "任一事实取反+负采样,量极大"),
    ("比较(2实体)", "✅ 可(结构化)", f"{compare_pairs:,} 对", "同厂商雷达两两;取样即可"),
    ("冲突识别", "🔶 半可(识别=是/否可判;倾向值有争议)", f"{conflicts}", "→ trust 奖励主场,非普通correctness"),
    ("复杂/抽象/任务型/报告", "❌ 不可(无确定金标)", "—", "只能 LLM-judge/人工 → 进评测不进 reward"),
]
w = max(len(r[0]) for r in rows)
for name, judge, vol, note in rows:
    print(f"\n【{name}】")
    print(f"  判分: {judge}")
    print(f"  量级: {vol}")
    print(f"  备注: {note}")

print("\n" + "-" * 70)
print("分组关系明细(count/enumerate 的金标来源, usable=组大小∈[3,30]):")
for r, (u, t) in sorted(group_volume.items(), key=lambda x: -x[1][0]):
    print(f"  {r:20} 可用组 {u:4} / 总组 {t:4}")

print("\n2跳组合 top 模板:")
for k, c in two_hop_examples.most_common(8):
    print(f"  {k:40} {c}")

print("\n" + "=" * 70)
print("G1 结论")
print("=" * 70)
auto = single_hop + two_hop + sum(u for u, _ in group_volume.values()) + compare_pairs
print(f"• 可自动判分题型(→可进 RL reward):单跳/2跳/计数/枚举/多约束/否定/比较")
print(f"  → 金标全部可从 KG 确定性算出,可生成量级 万级以上,RL 训练题量充足")
print(f"• trust 奖励主场:{conflicts} 冲突题 + 任意答案的证据 tier/印证标记(题量另算)")
print(f"• ❌ 复杂/抽象/任务型:无可验证奖励 → 只能当【泛化评测集】,不能进训练 reward")
print(f"• 对策:在可判分题型上训策略 → 测其对复杂题的【零样本泛化】;复杂题用 LLM-judge 评测")
