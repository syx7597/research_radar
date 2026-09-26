# -*- coding: utf-8 -*-
"""Validate the doctrine table by running EWAdvisor on real, recognisable
threat radars and printing the recommendations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ew_advisor import EWAdvisor

THREATS = [
    {"name": "SNR-75 “Fan Song”(SA-2 制导)", "purpose": "SAM_guidance",
     "tracking_method": "lobe_switching", "scan_type": "sector", "bands": ["E", "F"],
     "freq_agile": False},
    {"name": "AN/MPQ-65(爱国者 PAC-3)", "purpose": "SAM_guidance",
     "tracking_method": "monopulse", "scan_type": "phased", "bands": ["C"],
     "freq_agile": True, "employs_eccm": ["freq_agility"]},
    {"name": "P-18 “Spoon Rest”(米波警戒)", "purpose": "search",
     "tracking_method": "none", "scan_type": "circular", "bands": ["A"],
     "freq_agile": False},
    {"name": "假想现代捷变搜索雷达(带旁瓣对消)", "purpose": "acquisition",
     "tracking_method": "TWS", "scan_type": "phased", "bands": ["S"],
     "freq_agile": True, "employs_eccm": ["sidelobe_canceller", "monopulse"]},
]

adv = EWAdvisor()
for t in THREATS:
    rec = adv.recommend(t)
    print("=" * 70)
    print(adv.render(rec))
    print()
