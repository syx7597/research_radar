# -*- coding: utf-8 -*-
"""Evaluate the EW countermeasure advisor.

Part A — DOCTRINE CASES: hand-curated (threat profile -> expected technique)
  cases drawn from EW textbooks (independent ground truth, NOT the engine's own
  rules). Each asserts that the advisor's recommendation INCLUDES the doctrinally
  correct technique(s) and EXCLUDES the wrong one(s). This measures whether the
  system gives textbook answers.

Part B — ENGINE INVARIANTS over all real profiles: a recommended technique is
  never also avoided / countered; every cited rule id exists; equipment always
  band-matches. This measures implementation soundness.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ew_advisor import EWAdvisor
from threat_profile import build_profiles, OUT

adv = EWAdvisor()


def rec_sets(profile):
    r = adv.recommend(profile)
    viable = {x["id"] for x in r["recommended"]}
    avoid = {x["id"] for x in r["avoid"]}
    countered = {x["id"] for x in r["countered_by_eccm"]}
    return r, viable, avoid, countered


# ---------------- Part A: curated doctrine cases ----------------
# each: name, profile, must_include (any-of groups), must_exclude (from viable)
CASES = [
    {"name": "单脉冲火控→交叉眼/交叉极化，忌倒相增益",
     "profile": {"purpose": ["fire_control"], "tracking_method": "monopulse", "bands": ["X"]},
     "include_any": [["cross_eye", "cross_pol", "terrain_bounce"]],
     "exclude": ["inverse_gain"]},
    {"name": "圆锥扫描跟踪→倒相增益有效",
     "profile": {"purpose": ["track"], "tracking_method": "conical_scan", "bands": ["X"]},
     "include_any": [["inverse_gain"]], "exclude": []},
    {"name": "波瓣切换→倒相增益",
     "profile": {"purpose": ["fire_control"], "tracking_method": "lobe_switching"},
     "include_any": [["inverse_gain"]], "exclude": []},
    {"name": "频率捷变搜索→拦阻/扫频，忌瞄准式",
     "profile": {"purpose": ["search"], "freq_agile": True, "bands": ["S"]},
     "include_any": [["barrage_noise", "sweep_noise"]], "exclude": ["spot_noise"]},
    {"name": "非捷变搜索→瞄准或拦阻噪声",
     "profile": {"purpose": ["search"], "bands": ["A"]},
     "include_any": [["spot_noise", "barrage_noise"]], "exclude": []},
    {"name": "TWS截获→DRFM假目标",
     "profile": {"purpose": ["acquisition"], "tracking_method": "TWS", "bands": ["S"]},
     "include_any": [["false_targets_drfm"]], "exclude": []},
    {"name": "CW照射制导→速度门拖引",
     "profile": {"purpose": ["SAM_guidance"], "tracking_method": "cw_illuminator", "bands": ["X"]},
     "include_any": [["vgpo"]], "exclude": []},
    {"name": "距离波门火控→距离门拖引",
     "profile": {"purpose": ["fire_control"], "tracking_method": "range_gate", "bands": ["X"]},
     "include_any": [["rgpo"]], "exclude": []},
    {"name": "火控雷达→无源诱饵(诱饵/箔条)",
     "profile": {"purpose": ["fire_control"], "tracking_method": "monopulse", "bands": ["X"]},
     "include_any": [["decoy_towed", "chaff"]], "exclude": []},
    {"name": "带旁瓣对消的搜索→噪声被抵消(不进可行集)",
     "profile": {"purpose": ["search"], "bands": ["S"], "employs_eccm": ["sidelobe_canceller"]},
     "include_any": [], "exclude": ["spot_noise", "barrage_noise"]},
    {"name": "带MTI的火控→箔条被抵消",
     "profile": {"purpose": ["fire_control"], "tracking_method": "monopulse",
                 "bands": ["X"], "employs_eccm": ["mti_pulse_doppler"]},
     "include_any": [], "exclude": ["chaff"]},
    {"name": "频率捷变(作为ECCM)→瞄准式被否决",
     "profile": {"purpose": ["search"], "freq_agile": True,
                 "bands": ["S"], "employs_eccm": ["freq_agility"]},
     "include_any": [["barrage_noise", "sweep_noise"]], "exclude": ["spot_noise"]},
    {"name": "单脉冲→编队/闪烁(多平台)也可选",
     "profile": {"purpose": ["track"], "tracking_method": "monopulse", "bands": ["X"]},
     "include_any": [["formation_jamming", "blink_jamming"]], "exclude": ["inverse_gain", "agc_jamming"]},
    {"name": "圆锥扫描→AGC干扰也有效",
     "profile": {"purpose": ["fire_control"], "tracking_method": "conical_scan", "bands": ["X"]},
     "include_any": [["agc_jamming", "inverse_gain"]], "exclude": []},
    {"name": "PRF参差→距离门拖入被否决",
     "profile": {"purpose": ["fire_control"], "tracking_method": "range_gate",
                 "bands": ["X"], "prf_agile": True},
     "include_any": [["rgpo"]], "exclude": ["rgpi"]},
    {"name": "距离波门→拖出/拖入/掩护脉冲多手段",
     "profile": {"purpose": ["fire_control"], "tracking_method": "range_gate", "bands": ["X"]},
     "include_any": [["rgpo"], ["rgpi"], ["cover_pulse"]], "exclude": []},
]


def run_part_a():
    passed, rows = 0, []
    for c in CASES:
        r, viable, avoid, countered = rec_sets(c["profile"])
        ok = True
        why = []
        for grp in c["include_any"]:
            if not (set(grp) & viable):
                ok = False; why.append(f"缺少{grp}")
        for ex in c["exclude"]:
            # excluded technique must NOT be in the viable recommended set
            if ex in viable:
                ok = False; why.append(f"误含{ex}")
        passed += ok
        rows.append((c["name"], ok, "；".join(why)))
    return passed, rows


# ---------------- Part B: engine invariants over real profiles ----------------
def run_part_b():
    profs = json.load(open(OUT, encoding="utf-8")) if OUT.exists() else build_profiles()
    rule_ids = {ru["id"] for ru in adv.rules}
    n = 0
    viol = {"avoid_in_viable": 0, "countered_in_viable": 0,
            "bad_rule_cite": 0, "equip_band_mismatch": 0}
    for prof in profs.values():
        if not (prof.get("purpose") or prof.get("tracking_method")):
            continue
        n += 1
        r = adv.recommend(prof)
        viable = {x["id"] for x in r["recommended"]}
        avoid = {x["id"] for x in r["avoid"]}
        countered = {x["id"] for x in r["countered_by_eccm"]}
        if viable & avoid:
            viol["avoid_in_viable"] += 1
        if viable & countered:
            viol["countered_in_viable"] += 1
        for x in r["recommended"] + r["avoid"]:
            for c in x["cites"]:
                if c["rule_id"] not in rule_ids:
                    viol["bad_rule_cite"] += 1
        bands = set(prof.get("bands") or [])
        for syslist in r.get("equipment", {}).values():
            for s in syslist:
                if not s.get("broadband") and bands and not (set(s.get("bands", [])) & bands):
                    viol["equip_band_mismatch"] += 1
    return n, viol


if __name__ == "__main__":
    pa, rows = run_part_a()
    print("=" * 60)
    print(f"Part A — doctrine cases: {pa}/{len(CASES)} passed "
          f"({pa/len(CASES)*100:.0f}%)")
    for name, ok, why in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {why}" if why else ""))
    n, viol = run_part_b()
    print("=" * 60)
    print(f"Part B — engine invariants over {n} real profiles:")
    total_viol = sum(viol.values())
    for k, v in viol.items():
        print(f"  {k:24s}: {v}")
    print(f"  => {'ALL INVARIANTS HOLD' if total_viol == 0 else f'{total_viol} VIOLATIONS'}")
