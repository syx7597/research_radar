# -*- coding: utf-8 -*-
"""EW countermeasure advisor — deterministic doctrine reasoning.

Given a threat radar's *profile* (purpose / tracking method / scan / agility /
the ECCM it employs), recommend friendly jamming techniques to counter it, the
techniques to AVOID, and which recommendations the threat's own ECCM would
neutralise — each item carrying its doctrine rule id + source + confidence.

This is the auditable core of the "radar countermeasure" capability: the
reasoning is a pure match over `data/ew/doctrine_map.json` (no LLM in the trust
path). Effectiveness is doctrine-level heuristic, explicitly flagged "非实测".
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCTRINE_PATH = ROOT / "data" / "ew" / "doctrine_map.json"
SYSTEMS_PATH = ROOT / "data" / "ew" / "ew_systems.json"


# band letter -> coarse frequency class (handles mixed IEEE + NATO designations)
BAND_CLASS = {
    "HF": "low", "VHF": "low", "UHF": "low", "P": "low", "A": "low", "B": "low",
    "L": "mid", "S": "mid", "D": "mid", "E": "mid", "F": "mid",
    "C": "high", "X": "high", "G": "high", "H": "high", "I": "high", "J": "high",
    "Ku": "vhigh", "Ka": "vhigh", "K": "vhigh",
}
BAND_GUIDE = {
    "low": "米波/分米波（VHF/UHF，长波长）：抗隐身/远程警戒常用；雷达天线大、测角精度低，"
           "但机载小平台干扰孔径受限——宜大功率压制或远距支援干扰(SOJ)，难自卫硬压。",
    "mid": "L/S 波段：警戒/目标指示主力；噪声压制与拦阻有效，需兼顾功率密度与覆盖。",
    "high": "C/X 波段：火控/跟踪/多功能主力波段；自卫干扰吊舱与 DRFM 欺骗最有效，主流干扰机均覆盖。",
    "vhigh": "Ku/Ka 波段：末制导/高分辨；作用距离近、波束窄，重点用欺骗/拖曳诱饵与抵近干扰。",
}


def _matches(cond, profile):
    """A rule `if` block matches when every key is satisfied (AND).
    A condition value may be a scalar (equality) or a list (membership)."""
    for key, want in cond.items():
        have = profile.get(key)
        if have is None:
            return False
        if isinstance(want, list):
            # profile value (scalar or list) must intersect the wanted set
            haves = have if isinstance(have, list) else [have]
            if not any(h in want for h in haves):
                return False
        elif isinstance(want, bool):
            if bool(have) != want:
                return False
        else:
            haves = have if isinstance(have, list) else [have]
            if want not in haves:
                return False
    return True


class EWAdvisor:
    def __init__(self, doctrine_path=DOCTRINE_PATH, systems_path=SYSTEMS_PATH):
        d = json.load(open(doctrine_path, encoding="utf-8"))
        self.meta = d["_meta"]
        self.jam = {j["id"]: j for j in d["jamming_techniques"]}
        self.eccm = {e["id"]: e for e in d["eccm_techniques"]}
        self.rules = d["recommendation_rules"]
        self.prio_rules = d.get("threat_priority_rules", [])
        self.systems = []
        if Path(systems_path).exists():
            self.systems = json.load(open(systems_path, encoding="utf-8")).get("systems", [])

    # ---------- map recommended techniques -> concrete EW equipment ----------
    @staticmethod
    def _band_ok(sys_rec, threat_bands):
        if sys_rec.get("broadband"):
            return True
        if not threat_bands:
            return True   # threat band unknown -> don't exclude (flagged elsewhere)
        return bool(set(sys_rec.get("bands", [])) & set(threat_bands))

    def match_equipment(self, technique_ids, profile, friendly_sides=None):
        """Return {technique_id: [systems...]} plus an ARM list, filtered by band
        coverage (and optionally by friendly country/side)."""
        bands = profile.get("bands") or []
        purposes = set(profile.get("purpose") or [])

        def allowed(s):
            return (friendly_sides is None) or (s.get("country") in friendly_sides)

        by_tech = {}
        for tid in technique_ids:
            hits = [s for s in self.systems
                    if tid in s.get("techniques", []) and allowed(s) and self._band_ok(s, bands)]
            if hits:
                by_tech[tid] = hits
        arms = [s for s in self.systems
                if s.get("ew_type") == "arm" and allowed(s) and self._band_ok(s, bands)
                and (purposes & set(s.get("suppresses_purpose", [])))]
        return by_tech, arms

    # ---------- threat priority ----------
    def priority(self, profile):
        best, why = 0, []
        for r in self.prio_rules:
            if _matches(r["if"], profile):
                p = r.get("priority", r.get("priority_min", 0))
                if p > best:
                    best, why = p, [r.get("note", "")]
                elif p == best and r.get("note"):
                    why.append(r["note"])
        return best, why

    # ---------- per-radar engagement guidance (from this radar's own numbers) ----------
    @staticmethod
    def engagement_notes(profile):
        """Parameter-specific tactical notes derived from THIS radar's quantitative
        fields (band / frequency / peak power / range), so two radars in the same
        doctrine category still get differentiated guidance. All from real KG values."""
        notes = []
        bands = profile.get("bands") or []
        classes = []
        for b in bands:
            cls = BAND_CLASS.get(b)
            if cls and cls not in classes:
                classes.append(cls)
        for cls in classes:
            notes.append({"kind": "band", "text": BAND_GUIDE[cls]})

        freq = profile.get("frequency_GHz")
        pk = profile.get("peak_power_kW")
        rng = profile.get("range_km")
        params = []
        if freq is not None:
            params.append(f"频率≈{freq:g}GHz")
        if pk is not None:
            params.append(f"峰值功率≈{pk:g}kW")
        if rng is not None:
            params.append(f"探测距离≈{rng:g}km")
        if pk is not None or rng is not None:
            hard = (pk is not None and pk >= 100) or (rng is not None and rng >= 300)
            pstr = "（" + "、".join(params) + "）" if params else ""
            if hard:
                notes.append({"kind": "burnthrough",
                              "text": f"高能/远程{pstr}：烧穿距离较大，单纯噪声压制需高 ERP 或抵近(stand-in)；"
                                      f"建议欺骗/无源诱饵 + 远距支援(SOJ) 结合，避免硬拼功率。"})
            else:
                notes.append({"kind": "burnthrough",
                              "text": f"功率/作用距离有限{pstr}：常规自卫干扰即可有效压制。"})
        elif params:
            notes.append({"kind": "param", "text": "本机参数（" + "、".join(params) + "）。"})
        return notes

    # ---------- core recommendation ----------
    def recommend(self, profile, friendly_sides=None):
        """Return a structured, provenance-carrying recommendation.
        friendly_sides: optional set of country labels to restrict equipment to."""
        recommend, avoid, notes = {}, {}, []
        for rule in self.rules:
            if not _matches(rule["if"], profile):
                continue
            cite = {"rule_id": rule["id"], "source": rule["source"],
                    "confidence": rule["confidence"], "rationale": rule["rationale"]}
            for jid in rule.get("recommend", []):
                recommend.setdefault(jid, []).append(cite)
            for jid in rule.get("avoid", []):
                avoid.setdefault(jid, []).append(cite)
            if rule.get("note"):
                notes.append({"rule_id": rule["id"], "note": rule["note"],
                              "source": rule["source"], "confidence": rule["confidence"]})

        # an explicit `avoid` from any fired rule overrides a `recommend`
        for jid in avoid:
            recommend.pop(jid, None)

        # threat's own ECCM neutralises some recommended techniques
        employs = profile.get("employs_eccm") or []
        neutralised = {}   # jam_id -> [eccm_id, ...]
        for e in employs:
            for jid in self.eccm.get(e, {}).get("defeats", []):
                neutralised.setdefault(jid, []).append(e)

        def pack(jid, cites):
            j = self.jam.get(jid, {})
            conf = max((c["confidence"] for c in cites), default=0.0)
            blocked = neutralised.get(jid)
            if blocked:
                conf = round(conf * 0.4, 3)   # heavily discounted if countered
            return {
                "id": jid, "name": j.get("name_zh", jid),
                "category": j.get("category", ""),
                "effective_against": j.get("effective_against", []),
                "geometry_note": j.get("geometry_note", ""),
                "confidence": conf,
                "cites": cites,
                "neutralised_by": [self.eccm[e]["name_zh"] for e in (blocked or [])],
            }

        viable = [pack(j, c) for j, c in recommend.items() if j not in neutralised]
        countered = [pack(j, c) for j, c in recommend.items() if j in neutralised]
        viable.sort(key=lambda x: -x["confidence"])
        countered.sort(key=lambda x: -x["confidence"])
        avoid_items = [{"id": j, "name": self.jam.get(j, {}).get("name_zh", j),
                        "cites": c} for j, c in avoid.items()]
        prio, prio_why = self.priority(profile)

        # concrete equipment that can deliver the viable techniques (+ ARM for SEAD)
        equip_by_tech, arms = self.match_equipment([v["id"] for v in viable], profile, friendly_sides)

        return {
            "profile": profile,
            "threat_priority": prio,
            "priority_reason": prio_why,
            "recommended": viable,
            "countered_by_eccm": countered,
            "avoid": avoid_items,
            "notes": notes,
            "equipment": equip_by_tech,
            "arm": arms,
            "engagement": self.engagement_notes(profile),
            "evidence_type": self.meta.get("evidence_type", "doctrine_heuristic"),
        }

    # ---------- markdown report (auditable, '非实测' flagged) ----------
    def render(self, rec, title=None):
        p = rec["profile"]
        pname = title or p.get("name") or p.get("model") or "未命名威胁"
        ev = p.get("_evidence", {})
        TIER_MARK = {"prior": "推测", "llm": "文献", "keyword": "", "": ""}
        prof_bits, used_prior = [], False
        for k, label in (("purpose", "用途"), ("tracking_method", "跟踪体制"),
                         ("scan_type", "扫描"), ("bands", "频段"),
                         ("freq_agile", "频率捷变"), ("lpi", "LPI")):
            if p.get(k) not in (None, [], False):
                v = "、".join(p[k]) if isinstance(p[k], list) else p[k]
                tier = (ev.get(k) or {}).get("tier", "")
                mark = TIER_MARK.get(tier, "")
                if tier == "prior":
                    used_prior = True
                prof_bits.append(f"{label}={v}" + (f"（{mark}）" if mark else ""))
        for k, label, unit in (("frequency_GHz", "频率", "GHz"), ("peak_power_kW", "峰值功率", "kW"),
                               ("range_km", "探测距离", "km")):
            if p.get(k) is not None:
                prof_bits.append(f"{label}={p[k]:g}{unit}")
        if p.get("employs_eccm"):
            prof_bits.append("已知抗干扰=" + "、".join(self.eccm.get(e, {}).get("name_zh", e)
                                                  for e in p["employs_eccm"]))
        lines = [f"# 对抗建议：{pname}", "",
                 f"**威胁画像**：{'；'.join(prof_bits) or '（信息不足）'}",
                 f"**威胁等级**：{rec['threat_priority']}/5"
                 + (f"（{'；'.join(filter(None, rec['priority_reason']))}）" if rec["priority_reason"] else ""),
                 ""]

        def cite_str(cites):
            c = cites[0]
            return f"〔依据:{c['rule_id']}，来源:{c['source']}，置信:{c['confidence']:.2f}〕"

        case = p.get("documented_case")
        if case:
            lines += ["", "## 📜 实战记载（真实战史·有出处）",
                      f"_{p.get('role','')}_　**{case.get('conflict','')}**"]
            if case.get("countermeasures_used"):
                lines.append("- **实际采用的对抗**：" + "；".join(
                    c["what"] for c in case["countermeasures_used"]))
            if case.get("counter_counter"):
                lines.append("- **该雷达/操作方的反制**：" + "；".join(case["counter_counter"]))
            if case.get("outcome"):
                lines.append(f"- **结果**：{case['outcome']}")
            if case.get("lesson"):
                lines.append(f"- **启示**：{case['lesson']}")
            if case.get("sources"):
                lines.append("- 来源：" + "；".join(case["sources"]))

        if not case and p.get("sources"):
            sysline = f"（{p.get('system','')}）" if p.get("system") else ""
            lines += ["", f"## 📚 威胁来源（开源参考）",
                      f"_{p.get('role','')}_{sysline}",
                      "- 来源：" + "；".join(p["sources"])]

        if rec.get("engagement"):
            lines += ["", "## 🧭 交战研判（按本机参数）"]
            for n in rec["engagement"]:
                lines.append(f"- {n['text']}")

        if rec["recommended"]:
            lines.append("## ✅ 推荐干扰手段")
            for r in rec["recommended"]:
                lines.append(f"- **{r['name']}**（{r['category']}，综合置信 {r['confidence']:.2f}）"
                             f" {cite_str(r['cites'])}")
                if r["geometry_note"]:
                    lines.append(f"    - 战术附注：{r['geometry_note']}")
                lines.append(f"    - 依据：{r['cites'][0]['rationale']}")
        else:
            lines.append("## ✅ 推荐干扰手段\n_无（可能信息不足或全部被对方抗干扰抵消）_")

        if rec["countered_by_eccm"]:
            lines += ["", "## ⚠ 会被对方抗干扰抵消（慎用）"]
            for r in rec["countered_by_eccm"]:
                lines.append(f"- **{r['name']}**：被对方「{'、'.join(r['neutralised_by'])}」克制，"
                             f"残余置信仅 {r['confidence']:.2f} {cite_str(r['cites'])}")

        if rec.get("equipment") or rec.get("arm"):
            lines += ["", "## 🛰 可用对抗装备（按手段，开源参考）"]
            for tid, syslist in rec["equipment"].items():
                tname = self.jam.get(tid, {}).get("name_zh", tid)
                names = "；".join(f"{s['name_zh']}({s['country']})" for s in syslist)
                lines.append(f"- **{tname}** ← {names}")
            if rec["arm"]:
                names = "；".join(f"{s['name_zh']}({s['country']})" for s in rec["arm"])
                lines.append(f"- **反辐射硬摧毁(SEAD)** ← {names}")

        if rec["avoid"]:
            lines += ["", "## ⛔ 应避免"]
            for r in rec["avoid"]:
                lines.append(f"- **{r['name']}**：{r['cites'][0]['rationale']} {cite_str(r['cites'])}")

        if rec["notes"]:
            lines += ["", "## 📝 提示"]
            for n in rec["notes"]:
                lines.append(f"- {n['note']} 〔{n['rule_id']}，置信:{n['confidence']:.2f}〕")

        lines += ["", f"> ⚠ 以上为 **doctrine 级启发式判断（非实测）**，"
                      f"effectiveness 来自公开 EW 教科书规则，供决策参考与教学，不构成作战火控依据。"]
        if used_prior:
            lines.append("> ⚠ 画像中标「推测」的字段由 doctrine 先验推断（非文献实证），相关推荐的不确定性更高，建议核验。")
        return "\n".join(lines)


def load_advisor():
    return EWAdvisor()
