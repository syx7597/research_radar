# -*- coding: utf-8 -*-
"""EOB countermeasure wargaming — battlefield-level planning over a set of enemy
radars (an Electronic Order of Battle), not just per-radar advice.

Given a list of enemy radar models it:
  1. profiles + ranks them by threat priority,
  2. recommends countermeasures per threat (via EWAdvisor),
  3. solves an integrated **asset package** via greedy weighted set-cover — the
     minimal set of friendly EW systems that covers the most (priority-weighted)
     threats, given each system's technique+band coverage,
  4. reports technique economy (which jamming mode covers the most threats),
     capability gaps, and SEAD priority targets.

The set-cover gives a concrete, defensible "carry these N systems to cover M/K
threats" output. All effectiveness remains doctrine-level heuristic ("非实测").
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ew_advisor import EWAdvisor
from threat_profile import profile_for


class EWWargame:
    def __init__(self, advisor=None, resolver=None):
        self.adv = advisor or EWAdvisor()
        self.resolver = resolver           # optional name->canonical (KG resolve)
        self.sys_meta = {s["id"]: s for s in self.adv.systems}

    # ---------- assess an EOB ----------
    def assess(self, models, friendly_sides=None):
        threats = []
        for m in models:
            m = (m or "").strip()
            if not m:
                continue
            prof = profile_for(m, resolver=self.resolver)
            if not prof:
                threats.append({"model": m, "found": False})
                continue
            rec = self.adv.recommend(prof, friendly_sides=friendly_sides)
            threats.append({"model": prof.get("name", m), "found": True,
                            "profile": prof, "rec": rec, "priority": rec["threat_priority"]})
        threats.sort(key=lambda t: (-t.get("priority", 0), t.get("model", "")))
        plan = self._integrate(threats)
        return {"threats": threats, "plan": plan}

    # ---------- integrated planning ----------
    def _integrate(self, threats):
        engaged = [t for t in threats if t.get("found")]
        # threat -> set of system ids that can engage it (any viable technique + band, or ARM)
        # and threat -> {system_id: [technique names]} for annotating the package
        threat_systems = []
        threat_sys_tech = []
        for t in engaged:
            sids = set()
            sys_tech = {}
            for tid, syslist in t["rec"].get("equipment", {}).items():
                tname = self.adv.jam.get(tid, {}).get("name_zh", tid)
                for s in syslist:
                    sids.add(s["id"])
                    sys_tech.setdefault(s["id"], []).append(tname)
            for s in t["rec"].get("arm", []):
                sids.add(s["id"])
                sys_tech.setdefault(s["id"], []).append("反辐射")
            threat_systems.append(sids)
            threat_sys_tech.append(sys_tech)

        # ---- greedy weighted set-cover for a minimal asset package ----
        coverable = {i for i, s in enumerate(threat_systems) if s}
        weight = {i: engaged[i]["priority"] for i in range(len(engaged))}
        uncovered = set(coverable)
        package = []
        while uncovered:
            best, best_gain, best_cov = None, 0, set()
            for sid in self.sys_meta:
                cov = {i for i in uncovered if sid in threat_systems[i]}
                gain = sum(weight[i] for i in cov)
                if gain > best_gain:
                    best, best_gain, best_cov = sid, gain, cov
            if not best:
                break
            techs = set()
            for i in best_cov:
                techs.update(threat_sys_tech[i].get(best, []))
            package.append({"system": best, "name": self.sys_meta[best]["name_zh"],
                            "country": self.sys_meta[best].get("country", ""),
                            "via": sorted(techs),
                            "covers": [engaged[i]["model"] for i in sorted(best_cov)]})
            uncovered -= best_cov

        # ---- technique economy: which jamming mode covers the most threats ----
        tech_cover = {}
        for t in engaged:
            for item in t["rec"]["recommended"]:
                tech_cover.setdefault(item["id"], {"name": item["name"], "threats": []})
                tech_cover[item["id"]]["threats"].append(t["model"])
        tech_rank = sorted(tech_cover.values(), key=lambda x: -len(x["threats"]))

        # ---- gaps & SEAD ----
        gaps_noasset = [engaged[i]["model"] for i in coverable.symmetric_difference(range(len(engaged)))
                        if not threat_systems[i]]
        gaps_jam = [t["model"] for t in engaged
                    if not t["rec"]["recommended"]]
        sead = [t["model"] for t in engaged
                if t["priority"] >= 4 and t["rec"].get("arm")]

        total = len(engaged)
        covered = len(coverable - uncovered)
        w_total = sum(weight.values()) or 1
        w_covered = sum(weight[i] for i in (coverable - uncovered))
        return {
            "n_threats": len(threats), "n_engaged": total,
            "package": package,
            "coverage": {"threats": f"{covered}/{total}",
                         "weighted_pct": round(w_covered / w_total * 100)},
            "technique_economy": tech_rank,
            "gaps_no_asset": gaps_noasset,
            "gaps_all_countered": gaps_jam,
            "sead_targets": sead,
            "unknown": [t["model"] for t in threats if not t.get("found")],
        }

    # ---------- markdown report ----------
    def render(self, result):
        threats, plan = result["threats"], result["plan"]
        lines = ["# 战场对抗推演（EOB）", "",
                 f"**威胁数**：{plan['n_engaged']} 部已识别"
                 + (f"（{len(plan['unknown'])} 部图谱未收录）" if plan["unknown"] else ""), ""]

        # threat ranking
        lines += ["## 威胁排序", "", "| # | 型号 | 威胁度 | 用途 | 跟踪体制 | 首选对抗 |",
                  "|---|---|---|---|---|---|"]
        rank = 0
        for t in threats:
            if not t.get("found"):
                continue
            rank += 1
            p = t["profile"]
            purp = "、".join(p.get("purpose") or []) or "—"
            trk = p.get("tracking_method", "—")
            top = t["rec"]["recommended"][0]["name"] if t["rec"]["recommended"] else "—（手段受限）"
            lines.append(f"| {rank} | {t['model']} | {t['priority']}/5 | {purp} | {trk} | {top} |")

        # asset package (set-cover)
        lines += ["", "## 🎒 推荐装备包（最小覆盖）",
                  f"携带下列 **{len(plan['package'])}** 套装备即可覆盖 "
                  f"**{plan['coverage']['threats']}** 个威胁"
                  f"（按威胁度加权 **{plan['coverage']['weighted_pct']}%**）："]
        for a in plan["package"]:
            via = ("（经由：" + "、".join(a["via"][:4]) + "）") if a.get("via") else ""
            lines.append(f"- **{a['name']}**（{a['country']}）{via} → 应对：{ '、'.join(a['covers']) }")
        if not plan["package"]:
            lines.append("_无可匹配装备（型号/频段信息不足）_")

        # technique economy
        if plan["technique_economy"]:
            lines += ["", "## 🎛 干扰样式经济性（一招多用）"]
            for te in plan["technique_economy"][:5]:
                lines.append(f"- **{te['name']}**：可用于 {len(te['threats'])} 个威胁（{ '、'.join(te['threats'][:6]) }）")

        # SEAD
        if plan["sead_targets"]:
            lines += ["", "## 🎯 SEAD 优先目标（反辐射硬摧毁）",
                      "、".join(plan["sead_targets"])]

        # gaps
        if plan["gaps_no_asset"] or plan["gaps_all_countered"]:
            lines += ["", "## ⚠ 能力缺口"]
            if plan["gaps_all_countered"]:
                lines.append(f"- 软杀伤受限（推荐手段被对方抗干扰抵消或信息不足）：{ '、'.join(plan['gaps_all_countered']) }")
            if plan["gaps_no_asset"]:
                lines.append(f"- 无匹配装备：{ '、'.join(plan['gaps_no_asset']) }")
        if plan["unknown"]:
            lines += ["", f"> 注：{ '、'.join(plan['unknown']) } 未被图谱收录，建议联网补全后再评估。"]

        lines += ["", "> ⚠ 本推演为 **doctrine 级启发式（非实测）**，装备为开源参考；"
                      "效能与覆盖判断供研究/教学参考，不构成作战计划依据。"]
        return "\n".join(lines)


def load_wargame():
    return EWWargame()
