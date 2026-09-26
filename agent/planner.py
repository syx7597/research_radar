# -*- coding: utf-8 -*-
"""Composition planner: LLM translates a question into a JSON operator-composition
plan tree (the grammar executed by composition.py). The LLM only emits the plan;
execution is deterministic. A thin LLM answerer renders the executed result +
trace into a natural-language answer.
"""
import os, re, json, sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qa_strategy_pipeline import llm_call, get_relation_specs
from composition import CompositionExecutor, PlanError, is_degenerate, empty_leaves

# Clean, well-populated numeric attributes (for aggregate/calculator). Each entity
# that has the attribute stores a parseable number; whole-KB coverage is partial
# (~20-25% of radars at most), so aggregate results must report coverage.
NUMERIC_ATTRS = {
    "range_km": "探测距离(km)", "frequency_GHz": "工作频率(GHz)",
    "peak_power_kW": "峰值功率(kW)", "avg_power_W": "平均功率(W)",
    "weight_kg": "重量(kg)", "antenna_gain_dB": "天线增益(dB)",
    "antenna_size_m": "天线尺寸(m)", "mtbf_hours": "平均无故障时间(h)",
    "price_usd_million": "价格(百万美元)", "lru_count": "LRU数量",
    "pulse_width_us": "脉冲宽度(us)", "prf_kHz": "脉冲重复频率(kHz)",
    "resolution_m": "分辨率(m)", "beam_width_deg": "波束宽度(度)", "year": "年份",
}


def _numeric_specs() -> str:
    return "\n".join(f"  {k:20s} {v}" for k, v in NUMERIC_ATTRS.items())

PLAN_SYSTEM = """你是雷达知识图谱的「查询计划器」。把问题翻译成一棵 JSON 算子组合计划树。
你**只输出计划树 JSON**，不执行、不回答。执行由确定性引擎完成。

可用关系（schema id (head_type -> tail_type)：中文标签/同义词）：
{relation_specs}

可用数值属性（仅用于 aggregate 的 attribute；名字必须照抄）：
{numeric_specs}

计划树节点语法（可嵌套组合）：
- {{"op":"constraint","relation":"<rel>","value":"<取值>"}}        → 满足 (?, rel, value) 的实体集合
- {{"op":"intersect","args":[节点,节点,...]}}                      → 多个集合求交（AND）
- {{"op":"union","args":[...]}}                                    → 求并（OR）
- {{"op":"difference","args":[集合A,集合B]}}                       → A 减 B（A 但不满足 B）
- {{"op":"path","start":"<实体>","chain":["r1","r2^-1",...]}}      → 从实体沿关系链走；r^-1 表反向
- {{"op":"count","arg":集合节点}}                                  → 集合大小（计数题）
- {{"op":"enumerate","arg":集合节点}}                              → 列出集合（列举题）
- {{"op":"aggregate","func":"avg|sum|max|min","attribute":"<数值rel>","over":集合节点}} → 数值聚合
- {{"op":"compare","a":"<实体A>","b":"<实体B>","relation":"<rel>"}} → 两实体某关系的异同对比
- {{"op":"contains","set":集合节点,"member":"<实体>"}}             → 判断成员是否在集合里（是否/否定题）

规则：
- 关系名必须用上面列表里的英文 schema id；中文同义词映射过去。
- 国家用中文标准形式（美国/中国/俄罗斯…）；频段用单字母（S/X/L，不带"波段"）。
- 「多少款/几种」→ count；「列出/有哪些」→ enumerate；「是否/有没有」→ contains；
  「平均/总/最大」+数值 → aggregate；「A 和 B 的异同」→ compare；「X 的 Y 的 Z」→ path 多跳。
- 组合查询要嵌套：例如「中国研制的S波段雷达有多少种」=
  {{"op":"count","arg":{{"op":"intersect","args":[
     {{"op":"constraint","relation":"countryOfOrigin","value":"中国"}},
     {{"op":"constraint","relation":"hasFrequencyBand","value":"S"}}]}}}}
- 反向枚举用 path + r^-1：例如「Raytheon 研制了哪些雷达」=
  {{"op":"enumerate","arg":{{"op":"path","start":"Raytheon","chain":["developedBy^-1"]}}}}
- 数值聚合用 aggregate，attribute 必须从上面"数值属性"列表里选：例如「美国雷达的平均探测距离」=
  {{"op":"aggregate","func":"avg","attribute":"range_km","over":{{"op":"constraint","relation":"countryOfOrigin","value":"美国"}}}}
- **只添加问题明确要求的约束**，不要自己加"在役/现役"等问题没说的过滤条件，以免交集为空。

只输出 JSON，不要解释，不要 markdown 包裹。"""

EXAMPLES = [
    ("美国一共研制了多少款雷达？",
     {"op": "count", "arg": {"op": "constraint", "relation": "countryOfOrigin", "value": "美国"}}),
    ("列出工作在 S 波段的雷达",
     {"op": "enumerate", "arg": {"op": "constraint", "relation": "hasFrequencyBand", "value": "S"}}),
    ("AN/TPY-2 的研制方属于哪个国家？",
     {"op": "path", "start": "AN/TPY-2", "chain": ["developedBy", "affiliatedTo"]}),
    ("AN/TPY-2 是否被中国使用？",
     {"op": "contains", "set": {"op": "path", "start": "AN/TPY-2", "chain": ["operatedBy"]}, "member": "中国"}),
    ("AN/SPY-1 和 AN/TPY-2 的研制方相同吗？",
     {"op": "compare", "a": "AN/SPY-1", "b": "AN/TPY-2", "relation": "developedBy"}),
]


def _fewshot_block():
    lines = []
    for q, p in EXAMPLES:
        lines.append(f"问题：{q}\n计划：{json.dumps(p, ensure_ascii=False)}")
    return "\n\n".join(lines)


def plan(question: str) -> dict:
    sys_p = PLAN_SYSTEM.format(relation_specs=get_relation_specs(), numeric_specs=_numeric_specs())
    user = _fewshot_block() + f"\n\n问题：{question}\n计划："
    raw = llm_call([{"role": "system", "content": sys_p},
                    {"role": "user", "content": user}], max_tokens=400)
    mobj = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not mobj:
        raise PlanError(f"no JSON plan in: {raw[:200]}")
    return json.loads(mobj.group(0))


def deterministic_summary(value, trace) -> str:
    """Authoritative answer string built ONLY from the executed value + trace.
    Never passes numbers through an LLM — this is the auditable answer core."""
    op = trace.get("op")
    if op == "count":
        return f"共 {value} 款。"
    if op == "aggregate":
        func, attr, cov = trace["func"], trace["attribute"], trace["coverage"]
        if value is None:
            return f"无足够数据计算（{cov}）。"
        return f"{func}({attr}) = {value:.4g}（{cov}）。"
    if op == "contains":
        return f"{'是' if value else '否'}（目标实体「{trace['member']}」{'在' if value else '不在'}已知集合中，集合大小 {trace['set_n']}）。"
    if op == "compare":
        same, oa, ob = trace["same"], trace["only_a"], trace["only_b"]
        if trace["identical"]:
            return f"相同：{trace['a']} 与 {trace['b']} 在 {trace['relation']} 上一致（{ '、'.join(same) }）。"
        return (f"不同。共同：{ '、'.join(same) or '无' }；"
                f"{trace['a']} 独有：{ '、'.join(oa) or '无' }；{trace['b']} 独有：{ '、'.join(ob) or '无' }。")
    if op == "enumerate" or isinstance(value, set):
        items = sorted(value) if isinstance(value, set) else trace.get("items", [])
        head = "、".join(items[:40])
        return f"共 {len(items)} 项：{head}" + (" …" if len(items) > 40 else "")
    return str(value)


ANSWER_SYSTEM = """你是雷达情报分析助手。下面「权威结论」由确定性引擎算出，是唯一可信的事实。
请用一句自然、流畅的中文复述它来回答用户问题。**不得修改、增减任何数字、实体名或覆盖率**，
不得添加权威结论之外的事实。可以让语言更自然，但事实必须与权威结论完全一致。"""


def answer(question: str, det: str) -> str:
    user = f"问题：{question}\n权威结论：{det}\n请用一句自然中文回答（数字与事实不得改动）："
    out = llm_call([{"role": "system", "content": ANSWER_SYSTEM},
                    {"role": "user", "content": user}], max_tokens=400)
    return out if out and not out.startswith("[LLM_ERROR") else det


REFLECT_SYS = """上一个查询计划执行后返回了空/无数据，很可能是计划有误（不是真没有答案）。
请诊断原因并输出**修正后的计划树 JSON**。

常见原因与修正：
- 取值没对上图谱写法：约束的 value 在图谱里查不到（如写"在役"但图谱存"服役中"，写"USA"但存"美国"）
  → 参考下面给出的"该关系在图谱中的真实取值"，改用图谱里的写法。
- 实体名没匹配：path 的 start 在图谱里不存在或写法不同 → 换更标准/更短的型号写法。
- 关系选错：换一个语义相近的关系。
- 约束过严：交集为空时，去掉问题没明确要求的约束。

计划树语法与上一轮相同（constraint/intersect/union/difference/path/count/enumerate/aggregate/compare/contains）。
只输出修正后的计划树 JSON，不要解释。"""


def reflect(question, prev_plan, trace, ex):
    """Diagnose an empty/degenerate result and produce a revised plan, using the
    KG's actual values for the failed leaves as grounding hints."""
    hints = []
    for leaf in empty_leaves(trace):
        if leaf["kind"] == "constraint":
            vals = ex.distinct_tails(leaf["relation"], 25)
            hints.append(f"- 关系 {leaf['relation']} 的真实取值示例：{vals}（你写的 value=「{leaf['value']}」未命中）")
        else:
            hints.append(f"- path 起点「{leaf['start']}」在图谱中未找到或链 {leaf['chain']} 走空")
    hint_block = "\n".join(hints) if hints else "（未定位到具体空叶子；可能是约束过严或关系选错）"
    user = (f"问题：{question}\n上一轮计划：{json.dumps(prev_plan, ensure_ascii=False)}\n"
            f"诊断线索：\n{hint_block}\n\n请输出修正后的计划树 JSON：")
    raw = llm_call([{"role": "system", "content": REFLECT_SYS},
                    {"role": "user", "content": user}], max_tokens=400)
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class CompositionAgent:
    """Inner analytical core: question -> plan -> deterministic execute -> answer.
    The deterministic_summary is the auditable answer; the LLM only paraphrases it.
    On a degenerate (empty) result, a reflection node diagnoses and re-plans."""
    def __init__(self, executor: CompositionExecutor, max_reflect: int = 2):
        self.ex = executor
        self.max_reflect = max_reflect

    def run(self, question: str, paraphrase: bool = True) -> dict:
        p = plan(question)
        value, trace = self.ex.evaluate(p)
        reflections = []
        rounds = 0
        while is_degenerate(value, trace) and rounds < self.max_reflect:
            rounds += 1
            rp = reflect(question, p, trace, self.ex)
            if not rp or rp == p:
                break
            try:
                rv, rt = self.ex.evaluate(rp)
            except PlanError:
                break
            reflections.append({"round": rounds, "revised_plan": rp,
                                "fixed": not is_degenerate(rv, rt)})
            p, value, trace = rp, rv, rt
            if not is_degenerate(value, trace):
                break
        det = deterministic_summary(value, trace)
        nl = answer(question, det) if paraphrase else det
        return {"question": question, "plan": p, "value": value, "trace": trace,
                "det_answer": det, "answer": nl, "reflections": reflections}
