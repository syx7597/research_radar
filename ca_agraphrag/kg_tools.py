# -*- coding: utf-8 -*-
"""KG 工具层 —— agent 的动作空间(读 kg_v3,确定性执行)。
6 算子能力在此落地为 agent 工具(不叫"算子",避免与 agent/composition.py 混)。
纯 KG 查询,不含向量检索(hybrid_search/text_search 后续单独接)。
"""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ==== 与 gen_dataset 一致的边过滤(保证工具查询视图 == 金标视图,环境才自洽)====
_FIELD = re.compile(r"^(功能|体制|频段|研制|装备|现状|作用距离|天线|发射机|接收机)")
_HI_TIER = {"v3_manual", "v3_wikidata", "v3_struct"}
_SINGLE_VAL = {"developedBy", "partOfSystem", "headquarteredIn"}
_LINEAGE = {"derivedFrom", "hasVariant", "replaces"}   # 与 gen_dataset 一致:谱系放宽
_BAND_OK = re.compile(r"^(X|S|C|L|K|Ku|Ka|UHF|VHF|HF|EHF|SHF|P|W|D|E|F|G|H|I|J)([\s/-]|$)", re.I)
_DIRTY = re.compile(r"[：:]|<br\s*/?>")
_VALID_COUNTRY = {
    "美国", "美國", "俄罗斯", "蘇聯", "苏联", "英国", "英國", "法国", "加拿大", "意大利",
    "瑞典", "以色列", "德国", "日本", "中国", "中国台湾", "荷兰", "印度", "澳大利亚", "波兰",
    "西班牙", "丹麦", "伊朗", "乌克兰", "韩国", "土耳其", "捷克", "比利时", "挪威", "罗马尼亚",
    "巴西", "瑞士", "南非", "巴基斯坦", "保加利亚", "泰国",
    "American", "United Kingdom of Great Britain and Ireland", "Bulgaria", "Kazakhstan",
    "Estonia", "Slovakia", "German Democratic Republic", "Republic of the Congo",
    "Central African Republic", "Benin"}

def _clean(name):
    if not name or len(name) > 45:
        return False
    if re.fullmatch(r"[\d.,\s]+(km|m|kg|GHz|MHz)?", name, re.I):
        return False
    if _FIELD.match(name) or _DIRTY.search(name):
        return False
    return bool(re.search(r"[A-Za-z0-9一-鿿]", name))

def _norm_band(b):
    b = re.sub(r"\([^)]*\)", "", b.strip())
    b = re.sub(r"\s*(band|波段)\s*", "", b, flags=re.I)
    b = re.sub(r"[一-鿿]", "", b)
    b = re.sub(r"\bK\s+u\b", "Ku", b, flags=re.I)
    b = re.sub(r"\s*/\s*", "/", b)
    return re.sub(r"\s+", "", b).strip("/")


class KGTools:
    def __init__(self, edges_path=None, filtered=True):
        """filtered=True: 用与金标一致的过滤视图(RL 环境用);False: 全边(通用)。"""
        edges_path = edges_path or (ROOT / "kg_v3" / "edges.json")
        E = json.loads(Path(edges_path).read_text(encoding="utf-8"))
        N = json.loads((ROOT / "kg_v3" / "entities.json").read_text(encoding="utf-8"))
        self.type_of = {n["name"]: n.get("type", "") for n in N}
        self.hr2t = defaultdict(set)
        self.rt2h = defaultdict(set)
        kept = 0
        for e in E:
            h, r, t = e["head"], e["relation"], e["tail"].strip()
            if filtered:
                if not (_clean(h) and _clean(t)):
                    continue
                if r in _SINGLE_VAL:
                    if not e.get("corroborated"):
                        continue
                elif r in _LINEAGE:
                    if not (e.get("tier") in (_HI_TIER | {"v3_llm_grounded"}) or e.get("corroborated")):
                        continue
                elif not (e.get("tier") in _HI_TIER or e.get("corroborated")):
                    continue
                if r == "hasFrequencyBand":
                    t = _norm_band(t)
                    if not t or not _BAND_OK.match(t):
                        continue
                if r == "countryOfOrigin" and t not in _VALID_COUNTRY:
                    continue
            self.hr2t[(h, r)].add(t)
            self.rt2h[(r, t)].add(h)
            kept += 1
        self.n_edges = kept

    # ---- 工具(每个返回 JSON-able 观察) ----
    def lookup(self, entity, relation):
        """取 entity 在 relation 上的尾实体集(单跳/多跳链/否定判定用)。"""
        return sorted(self.hr2t.get((entity, relation), set()))

    def find(self, relation, value, type=None):
        """反向:取所有满足 (·, relation, value) 的头实体(计数/枚举/多约束/比较用)。"""
        hs = self.rt2h.get((relation, value), set())
        if type:
            hs = {h for h in hs if self.type_of.get(h) == type}
        return sorted(hs)

    def intersect(self, *sets):
        # 空集是交集的吸收元，不能过滤掉；None 与其他集合工具一致视为空集。
        s = [set(x or []) for x in sets]
        return sorted(set.intersection(*s)) if s else []

    def union(self, *sets):
        out = set()
        for x in sets:
            out |= set(x or [])
        return sorted(out)

    def difference(self, a, b):
        return sorted(set(a or []) - set(b or []))

    @staticmethod
    def count(items):
        return len(items or [])

    # 工具注册表(供环境/策略生成动作 schema)
    SCHEMA = {
        "lookup": {"args": ["entity", "relation"], "desc": "取实体在某关系上的尾实体集"},
        "find": {"args": ["relation", "value", "type?"], "desc": "取满足(·,关系,值)的所有头实体"},
        "intersect": {"args": ["sets"], "desc": "多个集合求交(多约束)"},
        "union": {"args": ["sets"], "desc": "多个集合求并"},
        "difference": {"args": ["a", "b"], "desc": "集合差 a-b(否定/排除)"},
        "count": {"args": ["items"], "desc": "计数"},
        "finish": {"args": ["answer"], "desc": "给出最终答案,结束"},
    }

    def call(self, tool, args):
        """按 (tool, args) 执行一个工具,返回结果(finish 由环境处理)。"""
        if tool == "lookup":
            return self.lookup(args["entity"], args["relation"])
        if tool == "find":
            return self.find(args["relation"], args["value"], args.get("type"))
        if tool == "intersect":
            return self.intersect(*args["sets"])
        if tool == "union":
            return self.union(*args["sets"])
        if tool == "difference":
            return self.difference(args["a"], args["b"])
        if tool == "count":
            return self.count(args["items"])
        raise ValueError(f"unknown tool: {tool}")
