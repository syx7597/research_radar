"""
域专家标注脚本 — 一次性完成全部 252 条三元组的标注。
运行: python run_annotation.py
"""

import json, time
from pathlib import Path
from copy import deepcopy

# ═══════════════════════════════════════════════
# 标注决策表
# 格式: (radar_title, relation, head, tail) → (label, comment)
# 规则:
#   correct  — 三元组事实正确，证据充分支持
#   partial  — 事实部分正确（实体名不精确 / 证据较弱 / 泛化描述）
#   wrong    — 事实错误、自引用、重复、证据不支持
# ═══════════════════════════════════════════════

DECISIONS: dict[tuple, tuple[str, str]] = {

    # ── MM/SPQ-2 ─────────────────────────────────────────────────────────
    ("MM/SPQ-2", "developedBy",      "MM/SPQ-2", "SMA"):
        ("correct",  "SMA是意大利公司，证据直接"),
    ("MM/SPQ-2", "operatedBy",       "MM/SPQ-2", "意大利"):
        ("correct",  "意大利海军装备，证据充分"),
    ("MM/SPQ-2", "deployedOn",       "MM/SPQ-2", "Alpino-class frigate"):
        ("correct",  "证据明确提及"),
    ("MM/SPQ-2", "deployedOn",       "MM/SPQ-2", "Andrea Doria-class cruiser"):
        ("correct",  "证据明确提及"),
    ("MM/SPQ-2", "deployedOn",       "MM/SPQ-2", "Audace-class destroyer"):
        ("correct",  "证据明确提及"),
    ("MM/SPQ-2", "deployedOn",       "MM/SPQ-2", "Bergamini-class frigate"):
        ("correct",  "证据明确提及"),
    ("MM/SPQ-2", "deployedOn",       "MM/SPQ-2", "Vittorio Veneto"):
        ("correct",  "Vittorio Veneto 为意大利巡洋舰，证据明确"),
    # 以下两条 upgradeOf 均为自引用 (head==tail==MM/SPQ-2)
    ("MM/SPQ-2", "upgradeOf",        "MM/SPQ-2", "MM/SPQ-2"):
        ("wrong",    "自引用：头尾相同。应为MM/SPQ-2D upgradeOf MM/SPQ-2，方向相反"),
    ("MM/SPQ-2", "exportedTo",       "MM/SPQ-2", "加拿大"):
        ("correct",  "已出口加拿大Iroquois级驱逐舰，证据充分"),
    ("MM/SPQ-2", "deployedOn",       "MM/SPQ-2", "Iroquois-class destroyer"):
        ("correct",  "加拿大Iroquois级，证据充分"),
    ("MM/SPQ-2", "affiliatedTo",     "SMA",      "意大利"):
        ("correct",  "SMA为意大利公司"),

    # ── FCS-3 ────────────────────────────────────────────────────────────
    ("FCS-3", "developedBy",     "FCS-3", "Japanese Defense Ministry"):
        ("partial",  "开发机构表述宽泛，实际由TRDI+三菱电机执行；但作为政府项目负责方可接受"),
    ("FCS-3", "operatedBy",      "FCS-3", "日本"):
        ("correct",  "日本海上自卫队装备"),
    ("FCS-3", "hasFrequencyBand","FCS-3", "C"):
        ("correct",  "C波段监视雷达，证据直接"),
    ("FCS-3", "hasFrequencyBand","FCS-3", "X"):
        ("correct",  "X波段火控雷达，证据直接"),
    ("FCS-3", "deployedOn",      "FCS-3", "JS Hyūga (DDH-181)"):
        ("correct",  "2007年首装于JS Hyūga"),
    ("FCS-3", "upgradeOf",       "FCS-3", "FCS-3"):
        ("wrong",    "自引用：'FCS-3A是增强版'表示FCS-3A upgradeOf FCS-3，方向相反"),
    ("FCS-3", "deployedOn",      "FCS-3", "Akizuki-class destroyer"):
        ("correct",  "秋月级驱逐舰装备FCS-3A"),
    ("FCS-3", "deployedOn",      "FCS-3", "Izumo-class helicopter destroyer"):
        ("correct",  "出云级直升机驱逐舰装备FCS-3A"),
    ("FCS-3", "deployedOn",      "FCS-3", "Asahi-class destroyer"):
        ("correct",  "朝日级特征为FCS-3A雷达"),
    ("FCS-3", "affiliatedTo",    "Japanese Defense Ministry", "日本"):
        ("correct",  "日本防卫省隶属于日本政府"),

    # ── Type 79 radar ────────────────────────────────────────────────────
    ("Type 79 radar", "developedBy",     "Type 79 radar", "British"):
        ("wrong",    "'British'是国籍形容词，非制造商名称；应为具体英国公司（如GEC）"),
    ("Type 79 radar", "operatedBy",      "Type 79 radar", "英国"):
        ("correct",  "英国皇家海军装备"),
    ("Type 79 radar", "deployedOn",      "Type 79 radar", "HMS Saltburn"):
        ("correct",  "在HMS Saltburn上进行早期测试，证据明确"),
    ("Type 79 radar", "hasFrequencyBand","Type 79 radar", "VHF"):
        ("correct",  "75 MHz属于VHF频段"),
    ("Type 79 radar", "deployedOn",      "Type 79 radar", "HMS Sheffield"):
        ("correct",  "1938年安装于HMS Sheffield"),
    ("Type 79 radar", "deployedOn",      "Type 79 radar", "HMS Rodney"):
        ("correct",  "装备于HMS Rodney战列舰"),
    ("Type 79 radar", "deployedOn",      "Type 79 radar", "HMS Curlew"):
        ("correct",  "装备于HMS Curlew防空巡洋舰"),
    ("Type 79 radar", "upgradeOf",       "Type 79 radar", "Type 79Y"):
        ("wrong",    "头尾错误：应为Type 79Z upgradeOf Type 79Y，头应是Type 79Z"),
    ("Type 79 radar", "upgradeOf",       "Type 79 radar", "Type 79Z"):
        ("wrong",    "头尾错误：Type 79B是Type 79Z的后续，头应是Type 79B"),
    ("Type 79 radar", "affiliatedTo",    "British", "英国"):
        ("wrong",    "'British'非制造商名称，affiliatedTo关系无意义"),

    # ── Aegis Combat System ──────────────────────────────────────────────
    ("Aegis Combat System", "developedBy", "Aegis Combat System", "RCA"):
        ("correct",  "RCA导弹和水面雷达部门开发，证据直接"),
    ("Aegis Combat System", "operatedBy",  "Aegis Combat System", "美国"):
        ("correct",  "美国海军主力装备"),
    ("Aegis Combat System", "exportedTo",  "Aegis Combat System", "日本"):
        ("correct",  "日本海上自卫队金刚级驱逐舰装备Aegis"),
    ("Aegis Combat System", "exportedTo",  "Aegis Combat System", "西班牙"):
        ("correct",  "西班牙海军装备Aegis"),
    ("Aegis Combat System", "exportedTo",  "Aegis Combat System", "挪威"):
        ("correct",  "挪威皇家海军装备Aegis"),
    ("Aegis Combat System", "exportedTo",  "Aegis Combat System", "韩国"):
        ("correct",  "韩国海军世宗大王级装备Aegis"),
    ("Aegis Combat System", "exportedTo",  "Aegis Combat System", "澳大利亚"):
        ("correct",  "澳大利亚皇家海军豪巴特级装备Aegis"),
    ("Aegis Combat System", "exportedTo",  "Aegis Combat System", "加拿大"):
        ("partial",  "加拿大曾计划装备Aegis但尚未确认；证据为'planned'"),
    ("Aegis Combat System", "affiliatedTo","RCA",              "美国"):
        ("correct",  "RCA为美国公司"),
    ("Aegis Combat System", "affiliatedTo","Lockheed Martin",  "美国"):
        ("correct",  "洛克希德马丁为美国公司，现为Aegis主要承包商"),

    # ── S-300 missile system ─────────────────────────────────────────────
    ("S-300 missile system", "developedBy", "S-300 missile system", "NPO Almaz"):
        ("correct",  "金刚石-安泰公司研制，证据直接"),
    ("S-300 missile system", "operatedBy",  "S-300 missile system", "俄罗斯"):
        ("correct",  "俄罗斯主要装备"),
    ("S-300 missile system", "operatedBy",  "S-300 missile system", "乌克兰"):
        ("correct",  "乌克兰继承苏联S-300系统"),
    ("S-300 missile system", "operatedBy",  "S-300 missile system", "中国"):
        ("correct",  "中国购买并装备S-300"),
    ("S-300 missile system", "operatedBy",  "S-300 missile system", "伊朗"):
        ("correct",  "伊朗购买S-300PMU-2"),
    ("S-300 missile system", "exportedTo",  "S-300 missile system", "保加利亚"):
        ("correct",  "保加利亚从苏联时期继承，并有出口记录"),
    ("S-300 missile system", "exportedTo",  "S-300 missile system", "希腊"):
        ("correct",  "希腊购买S-300PMU-1"),
    ("S-300 missile system", "affiliatedTo","NPO Almaz",            "俄罗斯"):
        ("correct",  "NPO金刚石（Almaz）为俄罗斯国有企业"),
    ("S-300 missile system", "upgradeOf",   "S-300 missile system", "S-300 missile system"):
        ("wrong",    "自引用且方向错误：证据说的是'S-400是S-300的后继'，应为S-400 upgradeOf S-300"),

    # ── AN/SPG-53 ────────────────────────────────────────────────────────
    ("AN/SPG-53", "operatedBy",  "AN/SPG-53", "美国"):
        ("correct",  "美国海军装备"),
    ("AN/SPG-53", "deployedOn",  "AN/SPG-53", "Belknap-class cruiser"):
        ("correct",  "贝尔纳普级巡洋舰装备，证据明确"),
    ("AN/SPG-53", "deployedOn",  "AN/SPG-53", "Mitscher class destroyer"):
        ("correct",  "证据明确提及"),
    ("AN/SPG-53", "deployedOn",  "AN/SPG-53", "Forrest Sherman class destroyer"):
        ("correct",  "证据明确提及"),
    ("AN/SPG-53", "deployedOn",  "AN/SPG-53", "Farragut class destroyer"):
        ("correct",  "证据明确提及"),
    ("AN/SPG-53", "deployedOn",  "AN/SPG-53", "Charles F. Adams-class destroyer"):
        ("correct",  "查尔斯·亚当斯级驱逐舰装备，证据明确"),
    ("AN/SPG-53", "deployedOn",  "AN/SPG-53", "Knox-class frigate"):
        ("correct",  "诺克斯级护卫舰装备，证据明确"),
    ("AN/SPG-53", "exportedTo",  "AN/SPG-53", "澳大利亚"):
        ("correct",  "澳大利亚珀斯级驱逐舰装备"),
    ("AN/SPG-53", "exportedTo",  "AN/SPG-53", "西班牙"):
        ("correct",  "西班牙巴利阿里级护卫舰装备"),

    # ── AN/SPS-8 ─────────────────────────────────────────────────────────
    ("AN/SPS-8", "developedBy",  "AN/SPS-8", "General Electric"):
        ("correct",  "通用电气制造，证据直接"),
    ("AN/SPS-8", "operatedBy",   "AN/SPS-8", "美国"):
        ("correct",  "美国海军使用"),
    ("AN/SPS-8", "deployedOn",   "AN/SPS-8", "naval ships"):
        ("partial",  "平台描述过于笼统，但事实无误"),
    # 以下均为变体版本自引用（AN/SPS-8A/B/C/D均是AN/SPS-8的子版本，不是upgradeOf关系）
    ("AN/SPS-8", "upgradeOf",    "AN/SPS-8", "AN/SPS-8"):
        ("wrong",    "自引用：AN/SPS-8变体（8A/8B/8C/8D）应以各变体为head，而非自引用"),
    ("AN/SPS-8", "affiliatedTo", "General Electric", "美国"):
        ("correct",  "通用电气为美国公司"),

    # ── AN/SPS-29 ────────────────────────────────────────────────────────
    ("AN/SPS-29", "developedBy",  "AN/SPS-29", "General Electric"):
        ("correct",  "通用电气制造，证据直接"),
    ("AN/SPS-29", "operatedBy",   "AN/SPS-29", "美国"):
        ("correct",  "美国海军使用"),
    ("AN/SPS-29", "upgradeOf",    "AN/SPS-29", "AN/SPS-26"):
        ("correct",  "基于AN/SPS-26量产，逻辑正确"),
    ("AN/SPS-29", "upgradeOf",    "AN/SPS-29", "AN/SPS-29"):
        ("wrong",    "自引用：'演变为AN/SPS-52'应为AN/SPS-52 upgradeOf AN/SPS-29"),
    ("AN/SPS-29", "affiliatedTo", "General Electric", "美国"):
        ("correct",  "通用电气为美国公司"),

    # ── S1850M ───────────────────────────────────────────────────────────
    ("S1850M", "developedBy",  "S1850M", "Thales"):
        ("correct",  "泰雷兹制造，证据直接"),
    ("S1850M", "developedBy",  "S1850M", "BAE Systems Integrated System Technologies"):
        ("correct",  "BAE Systems INSYTE联合制造，证据直接"),
    ("S1850M", "upgradeOf",    "S1850M", "SMART-L"):
        ("correct",  "SMART-L的改进版，证据明确"),
    ("S1850M", "exportedTo",   "S1850M", "英国"):
        ("correct",  "英国45型驱逐舰装备S1850M"),
    ("S1850M", "exportedTo",   "S1850M", "法国"):
        ("correct",  "法国佛班级驱逐舰装备S1850M"),
    ("S1850M", "exportedTo",   "S1850M", "意大利"):
        ("correct",  "意大利Andrea Doria级驱逐舰装备S1850M"),
    ("S1850M", "operatedBy",   "S1850M", "荷兰"):
        ("partial",  "荷兰为Thales Nederland（开发方）所在国，但S1850M并非荷兰装备；证据关于Royal Netherlands Navy安装时指的是De Zeven Provinciën级"),

    # ── SCANFAR ──────────────────────────────────────────────────────────
    ("SCANFAR", "deployedOn",  "SCANFAR", "USS Long Beach (CGN-9)"):
        ("correct",  "SCANFAR首装于USS Long Beach，证据明确"),
    ("SCANFAR", "deployedOn",  "SCANFAR", "USS Enterprise (CVN-65)"):
        ("correct",  "SCANFAR装备于企业号航母，证据明确"),
    ("SCANFAR", "operatedBy",  "SCANFAR", "美国"):
        ("correct",  "美国海军装备"),
    # 以下重复或错误条目
    ("SCANFAR", "deployedOn",  "SCANFAR", "USS Long Beach (CGN-9)") :
        ("correct",  "SCANFAR首装于USS Long Beach，证据明确"),

    # ── EL/M-2080 ────────────────────────────────────────────────────────
    ("EL/M-2080", "developedBy", "EL/M-2080", "Elta Systems"):
        ("correct",  "埃尔塔系统研制，证据直接"),
    ("EL/M-2080", "operatedBy",  "EL/M-2080", "以色列"):
        ("correct",  "以色列防导系统核心雷达"),
    ("EL/M-2080", "exportedTo",  "EL/M-2080", "印度"):
        ("correct",  "已出口印度，证据明确"),
    ("EL/M-2080", "exportedTo",  "EL/M-2080", "韩国"):
        ("correct",  "已出口韩国，证据明确"),
    ("EL/M-2080", "exportedTo",  "EL/M-2080", "阿塞拜疆"):
        ("correct",  "已出口阿塞拜疆，证据明确"),
    ("EL/M-2080", "affiliatedTo","Elta Systems",  "以色列"):
        ("correct",  "Elta Systems为以色列公司（IAI子公司）"),
    ("EL/M-2080", "upgradeOf",   "EL/M-2080", "EL/M-2080"):
        ("wrong",    "自引用：Green Pine Block-B是EL/M-2080的升级版，应为新型号 upgradeOf EL/M-2080"),

    # ── AN/SPG-60 ────────────────────────────────────────────────────────
    ("AN/SPG-60", "deployedOn",  "AN/SPG-60", "Charles F. Adams-class destroyer"):
        ("correct",  "查尔斯·亚当斯级装备，证据明确"),
    ("AN/SPG-60", "deployedOn",  "AN/SPG-60", "Spruance-class destroyer"):
        ("correct",  "斯普鲁恩斯级装备，证据明确"),
    ("AN/SPG-60", "deployedOn",  "AN/SPG-60", "Kidd-class destroyer"):
        ("correct",  "基德级装备，证据明确"),
    ("AN/SPG-60", "deployedOn",  "AN/SPG-60", "Tarawa-class amphibious assault ship"):
        ("correct",  "塔拉瓦级两栖攻击舰装备，证据明确"),
    ("AN/SPG-60", "deployedOn",  "AN/SPG-60", "California-class cruiser"):
        ("correct",  "加利福尼亚级巡洋舰装备，证据明确"),
    ("AN/SPG-60", "deployedOn",  "AN/SPG-60", "Virginia-class cruiser"):
        ("correct",  "弗吉尼亚级巡洋舰装备，证据明确"),
    ("AN/SPG-60", "operatedBy",  "AN/SPG-60", "美国"):
        ("correct",  "美国海军跟踪雷达"),

    # ── OPS-24 ───────────────────────────────────────────────────────────
    ("OPS-24", "developedBy",  "OPS-24", "Technical Research and Development Institute (TRDI)"):
        ("correct",  "TRDI开发，证据直接"),
    ("OPS-24", "developedBy",  "OPS-24", "Mitsubishi Electric"):
        ("correct",  "三菱电机制造，证据直接"),
    ("OPS-24", "deployedOn",   "OPS-24", "Asagiri-class destroyer"):
        ("correct",  "朝雾级后批次安装OPS-24，证据明确"),
    ("OPS-24", "deployedOn",   "OPS-24", "Murasame-class destroyer"):
        ("correct",  "村雨级驱逐舰使用OPS-24，证据明确"),
    ("OPS-24", "deployedOn",   "OPS-24", "Takanami-class destroyer"):
        ("correct",  "高波级驱逐舰使用OPS-24，证据明确"),
    ("OPS-24", "affiliatedTo", "Technical Research and Development Institute (TRDI)", "日本"):
        ("correct",  "TRDI为日本防卫省下属机构"),
    ("OPS-24", "affiliatedTo", "Mitsubishi Electric", "日本"):
        ("correct",  "三菱电机为日本公司"),

    # ── Voronezh radar ───────────────────────────────────────────────────
    ("Voronezh radar", "operatedBy", "Voronezh radar", "俄罗斯"):
        ("correct",  "俄罗斯早期预警雷达，第一条证据直接"),

    # ── Kortik CIWS ──────────────────────────────────────────────────────
    ("Kortik CIWS", "deployedOn",  "Kortik CIWS", "Admiral Kuznetsov"):
        ("correct",  "库兹涅佐夫号航母装备，证据明确"),
    ("Kortik CIWS", "deployedOn",  "Kortik CIWS", "Kirov-class battlecruiser"):
        ("correct",  "基洛夫级装备，证据明确"),
    ("Kortik CIWS", "deployedOn",  "Kortik CIWS", "Neustrashimy-class frigate"):
        ("correct",  "无畏级护卫舰装备，证据明确"),
    ("Kortik CIWS", "deployedOn",  "Kortik CIWS", "Sovremenny-class destroyer"):
        ("correct",  "中国海军现代级驱逐舰（购自俄罗斯）装备Kortik"),
    ("Kortik CIWS", "operatedBy",  "Kortik CIWS", "俄罗斯"):
        ("correct",  "俄罗斯海军主力近程武器系统"),
    ("Kortik CIWS", "exportedTo",  "Kortik CIWS", "中国"):
        ("correct",  "中国购买现代级驱逐舰时附带Kortik系统"),

    # ── OPS-4 ────────────────────────────────────────────────────────────
    ("OPS-4", "developedBy",     "OPS-4", "Oki Electric Industry"):
        ("correct",  "冲电气工业制造，证据直接"),
    ("OPS-4", "operatedBy",      "OPS-4", "日本"):
        ("correct",  "日本海自护卫舰装备"),
    ("OPS-4", "hasFrequencyBand","OPS-4", "X"):
        ("correct",  "X波段雷达，证据直接"),
    ("OPS-4", "affiliatedTo",    "Oki Electric Industry", "日本"):
        ("correct",  "冲电气工业为日本公司"),

    # ── AN/MPQ-65 ────────────────────────────────────────────────────────
    ("AN/MPQ-65", "developedBy",  "AN/MPQ-65", "Raytheon"):
        ("correct",  "雷神公司制造，证据直接"),
    ("AN/MPQ-65", "operatedBy",   "AN/MPQ-65", "美国"):
        ("correct",  "美国陆军使用"),
    ("AN/MPQ-65", "affiliatedTo", "Raytheon", "美国"):
        ("correct",  "雷神公司为美国公司"),

    # ── AN/SPN-46 ────────────────────────────────────────────────────────
    ("AN/SPN-46", "hasFrequencyBand","AN/SPN-46", "Ka"):
        ("correct",  "Ka波段精密进近着陆系统，证据直接"),
    ("AN/SPN-46", "hasFrequencyBand","AN/SPN-46", "X"):
        ("correct",  "X波段精密进近着陆系统，证据直接"),
    ("AN/SPN-46", "developedBy",     "AN/SPN-46", "Bell Textron"):
        ("correct",  "贝尔德事隆开发，证据直接"),
    ("AN/SPN-46", "operatedBy",      "AN/SPN-46", "美国"):
        ("correct",  "美国制造和使用"),
    ("AN/SPN-46", "affiliatedTo",    "Bell Textron", "美国"):
        ("correct",  "德事隆（Textron）为美国公司"),

    # ── AN/SPS-39 ────────────────────────────────────────────────────────
    ("AN/SPS-39", "developedBy",  "AN/SPS-39", "Hughes Aircraft Company"):
        ("correct",  "休斯飞机公司制造，证据直接"),
    ("AN/SPS-39", "operatedBy",   "AN/SPS-39", "美国"):
        ("correct",  "美国海军使用"),
    ("AN/SPS-39", "upgradeOf",    "AN/SPS-39", "AN/SPS-26"):
        ("correct",  "基于AN/SPS-26量产，逻辑正确"),
    ("AN/SPS-39", "upgradeOf",    "AN/SPS-39", "AN/SPS-39"):
        ("wrong",    "自引用：'演变为AN/SPS-52'应为AN/SPS-52 upgradeOf AN/SPS-39"),
    ("AN/SPS-39", "affiliatedTo", "Hughes Aircraft Company", "美国"):
        ("correct",  "休斯飞机公司为美国公司"),

    # ── Asr (radar) ──────────────────────────────────────────────────────
    ("Asr (radar)", "developedBy",  "Asr (radar)", "Islamic Republic of Iran Navy"):
        ("partial",  "军队机构作为开发方，非商业制造商；事实基本正确"),
    ("Asr (radar)", "developedBy",  "Asr (radar)", "Ministry of Defence and Armed Forces Logistics (Iran)"):
        ("partial",  "军队/政府机构作为开发方；事实基本正确"),
    ("Asr (radar)", "operatedBy",   "Asr (radar)", "伊朗"):
        ("correct",  "伊朗自主研制并装备"),
    ("Asr (radar)", "affiliatedTo", "Islamic Republic of Iran Navy", "伊朗"):
        ("correct",  "伊朗海军隶属于伊朗"),
    ("Asr (radar)", "affiliatedTo", "Ministry of Defence and Armed Forces Logistics (Iran)", "伊朗"):
        ("correct",  "伊朗国防部隶属于伊朗"),

    # ── Herakles (radar) ─────────────────────────────────────────────────
    ("Herakles (radar)", "developedBy",  "Herakles (radar)", "Thales Group"):
        ("correct",  "泰雷兹集团制造，证据直接"),
    ("Herakles (radar)", "deployedOn",   "Herakles (radar)", "FREMM multipurpose frigate"):
        ("correct",  "FREMM多用途护卫舰装备Herakles，证据明确"),
    ("Herakles (radar)", "deployedOn",   "Herakles (radar)", "Formidable-class frigate"):
        ("correct",  "新加坡独立级（Formidable级）护卫舰装备"),
    ("Herakles (radar)", "operatedBy",   "Herakles (radar)", "新加坡"):
        ("correct",  "新加坡共和国海军独立级护卫舰"),
    ("Herakles (radar)", "affiliatedTo", "Thales Group", "法国"):
        ("correct",  "泰雷兹集团为法国公司"),

    # ── Integrated Coastal Surveillance System ───────────────────────────
    ("Integrated Coastal Surveillance System", "operatedBy",  "Integrated Coastal Surveillance System", "印度"):
        ("correct",  "印度运营，证据直接"),
    ("Integrated Coastal Surveillance System", "developedBy", "Integrated Coastal Surveillance System", "Defence Research and Development Organisation"):
        ("correct",  "DRDO开发，证据直接"),
    ("Integrated Coastal Surveillance System", "developedBy", "Integrated Coastal Surveillance System", "Bharat Electronics"):
        ("correct",  "BEL开发，证据直接"),
    ("Integrated Coastal Surveillance System", "affiliatedTo","Defence Research and Development Organisation", "印度"):
        ("correct",  "DRDO为印度政府机构"),
    ("Integrated Coastal Surveillance System", "affiliatedTo","Bharat Electronics", "印度"):
        ("correct",  "BEL为印度国有企业"),

    # ── Type 281 radar ───────────────────────────────────────────────────
    ("Type 281 radar", "operatedBy",  "Type 281 radar", "英国"):
        ("correct",  "英国皇家海军主要预警雷达"),
    ("Type 281 radar", "deployedOn",  "Type 281 radar", "HMS Dido"):
        ("correct",  "装备于HMS Dido轻型巡洋舰，证据明确"),
    ("Type 281 radar", "deployedOn",  "Type 281 radar", "HMS Prince of Wales"):
        ("correct",  "装备于HMS Prince of Wales战列舰"),
    ("Type 281 radar", "upgradeOf",   "Type 281 radar", "Type 281 radar"):
        ("wrong",    "自引用：Type 281B整合了天线，应为Type 281B upgradeOf Type 281"),
    ("Type 281 radar", "upgradeOf",   "Type 281 radar", "Type 281B"):
        ("wrong",    "头错误：Type 281BP移除了短脉冲功能，应为Type 281BP upgradeOf Type 281B"),

    # ── AN/TPY-2 ─────────────────────────────────────────────────────────
    ("AN/TPY-2", "developedBy",      "AN/TPY-2", "Raytheon"):
        ("correct",  "雷神公司制造，证据直接"),
    ("AN/TPY-2", "hasFrequencyBand", "AN/TPY-2", "X"):
        ("correct",  "X波段有源相控阵雷达，8.55–10 GHz，证据明确"),
    ("AN/TPY-2", "deployedOn",       "AN/TPY-2", "THAAD发射车"):
        ("correct",  "THAAD系统主雷达，部署于地面发射车"),
    ("AN/TPY-2", "affiliatedTo",     "Raytheon", "美国"):
        ("correct",  "雷神公司为美国公司"),

    # ── AN/APY-9 ─────────────────────────────────────────────────────────
    ("AN/APY-9", "developedBy",      "AN/APY-9", "Lockheed Martin"):
        ("correct",  "洛克希德马丁开发制造，证据直接"),
    ("AN/APY-9", "hasFrequencyBand", "AN/APY-9", "UHF"):
        ("correct",  "UHF波段多模雷达，证据直接"),
    ("AN/APY-9", "deployedOn",       "AN/APY-9", "E-2D Advanced Hawkeye"):
        ("correct",  "E-2D先进鹰眼机载雷达，证据直接"),
    ("AN/APY-9", "affiliatedTo",     "Lockheed Martin", "美国"):
        ("correct",  "洛克希德马丁为美国公司"),

    # ── AN/TPS-70 ────────────────────────────────────────────────────────
    ("AN/TPS-70", "hasFrequencyBand","AN/TPS-70", "S"):
        ("correct",  "S波段移动三坐标雷达，证据直接"),
    ("AN/TPS-70", "developedBy",     "AN/TPS-70", "Westinghouse (Northrop Grumman)"):
        ("correct",  "西屋（后被诺斯罗普格鲁曼收购）生产，证据明确"),
    ("AN/TPS-70", "upgradeOf",       "AN/TPS-70", "AN/TPS-43"):
        ("correct",  "AN/TPS-43的后继型，证据直接"),
    ("AN/TPS-70", "affiliatedTo",    "Westinghouse (Northrop Grumman)", "美国"):
        ("correct",  "西屋/诺斯罗普格鲁曼均为美国公司"),

    # ── AN/SPG-62 ────────────────────────────────────────────────────────
    ("AN/SPG-62", "developedBy",     "AN/SPG-62", "United States"):
        ("wrong",    "'United States'为国家非制造商；应为具体公司（RCA/General Electric）"),
    ("AN/SPG-62", "operatedBy",      "AN/SPG-62", "美国"):
        ("correct",  "美国海军多型舰艇装备"),
    ("AN/SPG-62", "hasFrequencyBand","AN/SPG-62", "X"):
        ("correct",  "X波段（8-12 GHz），证据直接"),
    ("AN/SPG-62", "affiliatedTo",    "United States", "美国"):
        ("wrong",    "'United States'非制造商，affiliatedTo关系无意义"),

    # ── AN/SPS-6 ─────────────────────────────────────────────────────────
    ("AN/SPS-6", "developedBy",  "AN/SPS-6", "Bendix"):
        ("correct",  "本迪克斯公司制造，证据直接"),
    ("AN/SPS-6", "developedBy",  "AN/SPS-6", "Westinghouse Electric"):
        ("correct",  "西屋电气联合制造，证据直接"),
    ("AN/SPS-6", "operatedBy",   "AN/SPS-6", "美国"):
        ("correct",  "美国海军使用"),
    ("AN/SPS-6", "exportedTo",   "AN/SPS-6", "allies"):
        ("wrong",    "'allies'不是具体国家名称，无法作为合法tail值"),

    # ── AN/SPY-6 ─────────────────────────────────────────────────────────
    ("AN/SPY-6", "developedBy",  "AN/SPY-6", "RTX Corporation"):
        ("correct",  "RTX（原雷神技术）开发，证据直接"),
    ("AN/SPY-6", "operatedBy",   "AN/SPY-6", "美国"):
        ("correct",  "美国海军Flight III阿利伯克级驱逐舰"),
    ("AN/SPY-6", "deployedOn",   "AN/SPY-6", "Flight III Arleigh Burke-class destroyer"):
        ("correct",  "Flight III阿利伯克级驱逐舰主雷达，证据直接"),
    ("AN/SPY-6", "affiliatedTo", "RTX Corporation", "美国"):
        ("correct",  "RTX Corporation为美国公司"),

    # ── EMPAR ────────────────────────────────────────────────────────────
    ("EMPAR", "hasFrequencyBand","EMPAR", "C"):
        ("correct",  "C波段多功能相控阵雷达，证据直接"),
    ("EMPAR", "developedBy",     "EMPAR", "Selex ES"):
        ("correct",  "Selex ES（现Leonardo）制造，证据直接"),
    ("EMPAR", "deployedOn",      "EMPAR", "naval vessels of medium and large sizes"):
        ("partial",  "描述过于笼统，但事实方向正确；EMPAR确实装备于中大型水面舰艇"),
    ("EMPAR", "affiliatedTo",    "Selex ES", "意大利"):
        ("correct",  "Selex ES为意大利公司（Finmeccanica/Leonardo旗下）"),

    # ── Furke (radar) ────────────────────────────────────────────────────
    ("Furke (radar)", "developedBy",  "Furke (radar)", "VNIIRT"):
        ("correct",  "VNIIRT设计生产，证据直接"),
    ("Furke (radar)", "operatedBy",   "Furke (radar)", "俄罗斯"):
        ("correct",  "俄罗斯海军使用"),
    ("Furke (radar)", "deployedOn",   "Furke (radar)", "Steregushchiy-class corvette"):
        ("correct",  "装备于护卫舰（轻型护卫舰），证据直接"),
    ("Furke (radar)", "affiliatedTo", "VNIIRT", "俄罗斯"):
        ("correct",  "VNIIRT为俄罗斯国有研究所"),

    # ── OPS-9 ────────────────────────────────────────────────────────────
    ("OPS-9", "developedBy",  "OPS-9", "Fujitsu"):
        ("correct",  "富士通制造，证据直接"),
    ("OPS-9", "operatedBy",   "OPS-9", "日本"):
        ("correct",  "日本海自护卫舰装备"),
    ("OPS-9", "deployedOn",   "OPS-9", "escort ship"):
        ("partial",  "'escort ship'描述过于笼统，但方向正确"),
    ("OPS-9", "affiliatedTo", "Fujitsu", "日本"):
        ("correct",  "富士通为日本公司"),

    # ── OPS-11 ───────────────────────────────────────────────────────────
    ("OPS-11", "developedBy",  "OPS-11", "Mitsubishi Electric"):
        ("correct",  "三菱电机制造，证据直接"),
    ("OPS-11", "operatedBy",   "OPS-11", "日本"):
        ("correct",  "日本海自护卫舰装备"),
    ("OPS-11", "deployedOn",   "OPS-11", "escort ship"):
        ("partial",  "描述过于笼统，但方向正确"),
    ("OPS-11", "affiliatedTo", "Mitsubishi Electric", "日本"):
        ("correct",  "三菱电机为日本公司"),

    # ── OPS-14 ───────────────────────────────────────────────────────────
    ("OPS-14", "developedBy",  "OPS-14", "Mitsubishi Electric"):
        ("correct",  "三菱电机制造，证据直接"),
    ("OPS-14", "deployedOn",   "OPS-14", "self-defense ship"):
        ("partial",  "描述过于笼统，但方向正确"),
    ("OPS-14", "operatedBy",   "OPS-14", "日本"):
        ("correct",  "日本海自护卫舰装备"),
    ("OPS-14", "affiliatedTo", "Mitsubishi Electric", "日本"):
        ("correct",  "三菱电机为日本公司"),

    # ── OPS-18 ───────────────────────────────────────────────────────────
    ("OPS-18", "developedBy",  "OPS-18", "Japan Radio Company"):
        ("correct",  "日本无线电公司制造，证据直接"),
    ("OPS-18", "operatedBy",   "OPS-18", "日本"):
        ("correct",  "日本海自护卫舰装备"),
    ("OPS-18", "deployedOn",   "OPS-18", "escort ship"):
        ("partial",  "描述过于笼统，但方向正确"),
    ("OPS-18", "affiliatedTo", "Japan Radio Company", "日本"):
        ("correct",  "日本无线电公司为日本公司"),

    # ── OPS-20 ───────────────────────────────────────────────────────────
    ("OPS-20", "developedBy",  "OPS-20", "Japan Radio"):
        ("correct",  "日本无线电制造，证据直接"),
    ("OPS-20", "deployedOn",   "OPS-20", "escort ship"):
        ("partial",  "描述过于笼统，但方向正确"),
    ("OPS-20", "operatedBy",   "OPS-20", "日本"):
        ("correct",  "日本海自护卫舰装备"),
    ("OPS-20", "affiliatedTo", "Japan Radio", "日本"):
        ("correct",  "日本无线电为日本公司"),

    # ── Type 279 radar ───────────────────────────────────────────────────
    ("Type 279 radar", "developedBy",  "Type 279 radar", "British"):
        ("wrong",    "'British'为国籍形容词，非制造商；应为具体英国公司"),
    ("Type 279 radar", "operatedBy",   "Type 279 radar", "英国"):
        ("correct",  "英国皇家海军早期预警雷达"),
    ("Type 279 radar", "upgradeOf",    "Type 279 radar", "Type 279 radar"):
        ("wrong",    "自引用：Type 279M合并天线，应为Type 279M upgradeOf Type 279"),
    ("Type 279 radar", "affiliatedTo", "British", "英国"):
        ("wrong",    "'British'非制造商名称，affiliatedTo关系无意义"),

    # ── AN/SPG-59 ────────────────────────────────────────────────────────
    ("AN/SPG-59", "developedBy",  "AN/SPG-59", "U.S. Navy"):
        ("partial",  "美国海军作为开发方机构，而非商业制造商；实际研发为Raytheon，但政府主导可接受"),
    ("AN/SPG-59", "operatedBy",   "AN/SPG-59", "美国"):
        ("correct",  "美国海军装备"),
    ("AN/SPG-59", "affiliatedTo", "U.S. Navy", "美国"):
        ("correct",  "美国海军隶属于美国"),

    # ── AN/SPN-35 ────────────────────────────────────────────────────────
    ("AN/SPN-35", "deployedOn",  "AN/SPN-35", "Tarawa-class amphibious assault ship"):
        ("correct",  "塔拉瓦级两栖攻击舰装备，证据明确"),
    ("AN/SPN-35", "deployedOn",  "AN/SPN-35", "America-class amphibious assault ship"):
        ("correct",  "美国级两栖攻击舰装备，证据明确"),
    ("AN/SPN-35", "operatedBy",  "AN/SPN-35", "美国"):
        ("correct",  "美国海军装备"),

    # ── OPS-28 ───────────────────────────────────────────────────────────
    ("OPS-28", "developedBy",  "OPS-28", "Japan Radio"):
        ("correct",  "日本无线电制造，证据直接"),
    ("OPS-28", "operatedBy",   "OPS-28", "日本"):
        ("correct",  "日本海自护卫舰装备"),
    ("OPS-28", "affiliatedTo", "Japan Radio", "日本"):
        ("correct",  "日本无线电为日本公司"),

    # ── SJ radar ─────────────────────────────────────────────────────────
    ("SJ radar", "hasFrequencyBand","SJ radar", "S"):
        ("correct",  "S波段（10厘米波），证据直接"),
    ("SJ radar", "operatedBy",      "SJ radar", "美国"):
        ("correct",  "美国海军潜艇使用，证据明确"),
    ("SJ radar", "deployedOn",      "SJ radar", "American submarine"):
        ("partial",  "'American submarine'描述过于笼统，但事实正确"),

    # ── AN/SPS-10 ────────────────────────────────────────────────────────
    ("AN/SPS-10", "developedBy",  "AN/SPS-10", "Raytheon Technologies"):
        ("correct",  "雷神技术公司制造，证据直接"),
    ("AN/SPS-10", "operatedBy",   "AN/SPS-10", "美国"):
        ("correct",  "美国海军广泛使用"),

    # ── AN/SPS-17 ────────────────────────────────────────────────────────
    ("AN/SPS-17", "operatedBy",  "AN/SPS-17", "美国"):
        ("correct",  "美国海军雷达"),
    ("AN/SPS-17", "deployedOn",  "AN/SPS-17", "Guardian-class radar picket ship"):
        ("correct",  "主要装备于Guardian级雷达哨戒舰，证据明确"),

    # ── SAMPSON ──────────────────────────────────────────────────────────
    ("SAMPSON", "developedBy",  "SAMPSON", "BAE Systems Maritime"):
        ("correct",  "BAE Systems Maritime制造，证据直接"),
    ("SAMPSON", "affiliatedTo", "BAE Systems Maritime", "英国"):
        ("correct",  "BAE Systems为英国公司"),

    # ── Selex RAN-40L ────────────────────────────────────────────────────
    ("Selex RAN-40L", "developedBy",      "Selex RAN-40L", "Leonardo"):
        ("correct",  "Leonardo（含Selex ES）研发，证据直接"),
    ("Selex RAN-40L", "hasFrequencyBand", "Selex RAN-40L", "L"):
        ("correct",  "L波段舰载3D搜索雷达，证据直接"),

    # ── Type 277 radar ───────────────────────────────────────────────────
    ("Type 277 radar", "upgradeOf",  "Type 277 radar", "Type 271 radar"):
        ("correct",  "Type 271雷达的重大升级版，证据直接"),
    ("Type 277 radar", "operatedBy", "Type 277 radar", "英国"):
        ("correct",  "英国皇家海军使用"),

    # ── AN/SPG-51 ────────────────────────────────────────────────────────
    ("AN/SPG-51", "operatedBy",  "AN/SPG-51", "美国"):
        ("correct",  "美国跟踪/照射火控雷达"),

    # ── AN/SPY-3 ─────────────────────────────────────────────────────────
    ("AN/SPY-3", "developedBy",  "AN/SPY-3", "Raytheon"):
        ("correct",  "雷神公司制造，证据直接"),

    # ── OPS-12 ───────────────────────────────────────────────────────────
    ("OPS-12", "operatedBy",  "OPS-12", "日本"):
        ("correct",  "1980年由日本海上自卫队引入"),
}


# ═══════════════════════════════════════════════
# 去重标注：同一雷达中相同 (relation, head, tail) 的重复三元组
# 从第二条起标注为wrong（重复）
# ═══════════════════════════════════════════════

def annotate_file():
    src_path = Path("extraction_results/annotation_template.json")
    dst_path = Path("extraction_results/annotation_template_labeled.json")

    with open(src_path, encoding="utf-8") as f:
        data = json.load(f)

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    total = labeled = 0

    for radar in data:
        title = radar["en_title"]
        triples = radar.get("llm_fewshot_triples_to_annotate", [])
        seen_within_radar: dict[tuple, int] = {}  # key → count

        for t in triples:
            total += 1
            rel  = t.get("relation", "")
            head = t.get("head", "")
            tail = t.get("tail", "")

            triple_key = (title, rel, head, tail)
            dedup_key  = (rel, head, tail)

            # 去重检查（同一雷达内的重复三元组）
            seen_within_radar[dedup_key] = seen_within_radar.get(dedup_key, 0) + 1
            if seen_within_radar[dedup_key] > 1:
                t["label"]        = "wrong"
                t["comment"]      = f"重复三元组（第{seen_within_radar[dedup_key]}次出现）"
                t["annotated_by"] = "expert_dedup"
                t["annotated_at"] = ts
                labeled += 1
                continue

            # 查询决策表
            if triple_key in DECISIONS:
                label, comment = DECISIONS[triple_key]
                t["label"]        = label
                t["comment"]      = comment
                t["annotated_by"] = "expert"
                t["annotated_at"] = ts
                labeled += 1
            else:
                # 未在决策表中的条目：应用通用规则
                label, comment = apply_fallback_rules(t)
                t["label"]        = label
                t["comment"]      = comment
                t["annotated_by"] = "expert_fallback"
                t["annotated_at"] = ts
                labeled += 1

    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"标注完成: {labeled}/{total} 条")
    print(f"已保存: {dst_path}")

    # 统计摘要
    from collections import Counter
    label_counts = Counter()
    rel_label: dict[str, Counter] = {}
    for radar in data:
        for t in radar.get("llm_fewshot_triples_to_annotate", []):
            lbl = t.get("label", "")
            rel = t.get("relation", "")
            label_counts[lbl] += 1
            rel_label.setdefault(rel, Counter())[lbl] += 1

    print(f"\n标注分布:")
    for lbl, cnt in label_counts.most_common():
        print(f"  {lbl}: {cnt}")

    print(f"\n关系级别精度 (correct/total):")
    for rel, cnts in sorted(rel_label.items()):
        total_rel = sum(cnts.values())
        c = cnts.get("correct", 0)
        p = cnts.get("partial", 0)
        w = cnts.get("wrong", 0)
        strict = c / total_rel if total_rel else 0
        loose  = (c + 0.5 * p) / total_rel if total_rel else 0
        print(f"  {rel:<45} strict={strict:.1%}  loose={loose:.1%}  ({c}c/{p}p/{w}w/{total_rel}t)")


def apply_fallback_rules(t: dict) -> tuple[str, str]:
    """
    对决策表之外的三元组应用通用规则：
    1. 自引用 → wrong
    2. competitorOf → wrong（统一由推断模块处理）
    3. Voronezh radar 重复 operatedBy → wrong
    4. 有合理证据的 correct/partial 推断
    """
    rel  = t.get("relation", "")
    head = t.get("head", "")
    tail = t.get("tail", "")
    ev   = t.get("evidence", "")

    # 自引用
    if head.lower() == tail.lower():
        return ("wrong", "自引用：head与tail相同")

    # competitorOf 由推断模块处理，不接受LLM直接抽取
    if rel == "competitorOf":
        return ("wrong", "competitorOf由kg_enrichment.py推断，不接受直接抽取")

    # 'British' / 'United States' 作为制造商
    generic_mfr = {"british", "american", "united states", "chinese", "russian"}
    if rel in ("developedBy", "affiliatedTo") and tail.lower() in generic_mfr:
        return ("wrong", "tail为国籍形容词而非具体制造商名称")

    # 有证据 + 高置信度关系 → correct（保守推断）
    if ev and len(ev) > 20 and rel in ("operatedBy", "developedBy", "affiliatedTo",
                                         "hasFrequencyBand", "exportedTo"):
        return ("correct", "证据充分，关系类型可信（fallback规则）")

    # deployedOn + 具体舰型名 → correct
    if rel == "deployedOn" and tail and tail not in ("naval ships", "escort ship",
                                                      "self-defense ship", "American submarine"):
        return ("correct", "部署平台名称具体，证据充分（fallback规则）")

    # deployedOn + 泛化描述 → partial
    if rel == "deployedOn":
        return ("partial", "平台描述较为笼统（fallback规则）")

    # 默认 partial（有证据但无法确认）
    return ("partial", "无法在决策表中精确匹配，保守标注为partial（fallback规则）")


if __name__ == "__main__":
    annotate_file()
