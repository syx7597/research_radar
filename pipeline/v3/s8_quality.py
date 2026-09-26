"""
S8 全局质量校验 —— 融合后的自查层。

把之前逐轮人肉发现的质量问题模式编码成自动检测器,任何新数据源接入后跑一遍
即可自动暴露问题(数字当型号/字段词当实体/写法碎片/悬空引用/类型违规/孤立分析),
不再依赖人肉逐个发现。只读 kg_v3,输出 kg_v3/quality_report.md,不改数据。

  python pipeline/v3/s8_quality.py
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "pipeline" / "v3"))
import schema as S  # noqa: E402
sys.path.insert(0, str(ROOT / "pipeline" / "v3"))
from s5_fuse import maker_core, _norm, _norm_platform  # noqa: E402

KGV3 = ROOT / "kg_v3"
FIELD_WORDS = re.compile(
    r"^(功能|体制|频段|研制厂商|研制|装备|现状|作用距离|探测距离|工作频率|工作方式|"
    r"天线|发射机|接收机|波束|扫描|重复频率|脉冲|处理机|冷却|质量|尺寸|体积|天馈线|"
    r"覆盖范围|方位|分辨力|MTBF|峰值功率|平均功率|数据|电源|显示|噪声|增益|接口)[\s：:]")


def check_garbage_heads(edges, ents):
    """数字/字段词/超长 被当成雷达型号(转录/解析 bug 的信号)。"""
    bad = defaultdict(list)
    for e in ents:
        if e["type"] != "Radar":
            continue
        n = e["name"]
        if re.fullmatch(r"[\d.,]+\s*(km|m|kg|GHz|MHz|°|dB|海里)?", n, re.I):
            bad["数值当型号"].append(n)
        elif FIELD_WORDS.match(n):
            bad["规格字段当型号"].append(n)
        elif len(n) > 50:
            bad["超长串当型号"].append(n)
        elif not re.search(r"[A-Za-z0-9一-鿿]", n):
            bad["乱码"].append(n)
    return bad


def check_fragmented(edges, rel, keyfn, label):
    """同一实体的多写法碎片(本该是 hub 却被拆散)。"""
    by_key = defaultdict(set)
    for e in edges:
        if e["relation"] == rel:
            k = keyfn(e["tail"])
            if k and len(k) >= 3:
                by_key[k].add(e["tail"].strip())
    frag = {k: sorted(v) for k, v in by_key.items() if len(v) > 1}
    return frag


def check_dangling(edges, ents):
    """tail 是雷达类型但不在实体表(悬空引用)。"""
    names = {e["name"] for e in ents}
    dang = set()
    for e in edges:
        if e["tail_type"] in ("Radar", "RadarSystem") and e["tail"] not in names:
            dang.add(e["tail"])
    return dang


def check_type_violation(edges):
    """边的 head/tail 类型不符 schema 签名。"""
    viol = Counter()
    for e in edges:
        rel = e["relation"]
        if rel not in S.TYPE_SIG:
            continue
        _, tails = S.TYPE_SIG[rel]
        if tails and e["tail_type"] not in tails and e["tail_type"] not in (
                "Aircraft", "NavalVessel", "GroundVehicle", "Platform"):
            viol[f"{rel}: tail={e['tail_type']} 应∈{sorted(tails)}"] += 1
    return viol


def main():
    edges = json.loads((KGV3 / "edges.json").read_text(encoding="utf-8"))
    ents = json.loads((KGV3 / "entities.json").read_text(encoding="utf-8"))
    deg = Counter()
    for e in edges:
        deg[e["head"]] += 1
        deg[e["tail"]] += 1

    garbage = check_garbage_heads(edges, ents)
    maker_frag = check_fragmented(edges, "developedBy", maker_core, "厂商")
    plat_frag = check_fragmented(edges, "deployedOn", _norm_platform, "平台")
    dangling = check_dangling(edges, ents)
    viol = check_type_violation(edges)
    iso_types = Counter(e["type"] for e in ents if deg[e["name"]] == 1)

    L = ["# KG v3 全局质量报告", "",
         f"- 边 **{len(edges)}** / 实体 **{len(ents)}** / 平均度 "
         f"**{round(2*len(edges)/len(ents), 2)}**", ""]

    tot_garbage = sum(len(v) for v in garbage.values())
    L += [f"## 1. 垃圾型号节点 (转录/解析 bug 信号): **{tot_garbage}**"]
    for k, v in garbage.items():
        L.append(f"- {k}: {len(v)}  例: {v[:5]}")
    if not tot_garbage:
        L.append("- ✅ 无")

    L += ["", f"## 2. 厂商写法碎片 (本该 hub): **{len(maker_frag)}** 组"]
    for k, v in sorted(maker_frag.items(), key=lambda x: -len(x[1]))[:8]:
        L.append(f"- {v[:4]}")

    L += ["", f"## 3. 平台写法碎片: **{len(plat_frag)}** 组"]
    for k, v in sorted(plat_frag.items(), key=lambda x: -len(x[1]))[:6]:
        L.append(f"- {v[:4]}")

    L += ["", f"## 4. 悬空雷达引用 (tail 不在实体表): **{len(dangling)}**"]
    if dangling:
        L.append(f"- 例: {sorted(dangling)[:8]}")

    L += ["", f"## 5. 类型签名违规: **{sum(viol.values())}**"]
    for k, v in viol.most_common(8):
        L.append(f"- {k}: {v}")

    L += ["", "## 6. 孤立叶子 (degree=1) 分类型",
          f"- 总计 {sum(iso_types.values())} ({sum(iso_types.values())/len(ents):.0%})"]
    for t, n in iso_types.most_common():
        L.append(f"- {t}: {n}")

    (KGV3 / "quality_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[S8] 质量报告 -> kg_v3/quality_report.md")
    print(f"[S8] 垃圾型号 {tot_garbage} | 厂商碎片 {len(maker_frag)}组 | "
          f"平台碎片 {len(plat_frag)}组 | 悬空 {len(dangling)} | 类型违规 {sum(viol.values())}")


if __name__ == "__main__":
    main()
