# -*- coding: utf-8 -*-
"""Outer ReAct orchestration loop (hand-rolled, dependency-light).

The agent emits ONE JSON action per step; we execute the tool, feed back the
observation, and loop until it emits a `final` action (or hits max_iters).
Tools wrap the assets we already built:
  composition_query  -> inner CompositionAgent (the analytical core)
  entity_dossier     -> provenance-cited equipment dossier
  comparison_report  -> provenance-cited comparison
  web_search         -> (stub here; real version added next)
Multi-turn: pass prior turns in `history`.
"""
import os, re, json, sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qa_strategy_pipeline import llm_call
from composition import load_executor
from planner import CompositionAgent
from report import RadarReporter
import json as _json


class RadarAgent:
    def __init__(self, web_search_fn=None, enable_web=True):
        ex = load_executor()
        self.core = CompositionAgent(ex)
        triples = _json.load(open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8"))
        self.reporter = RadarReporter(triples)
        if web_search_fn is None and enable_web:
            try:
                from web_tool import WebTool
                web_search_fn = lambda q: WebTool().search_extract(q).get("text", "")
            except Exception:
                web_search_fn = None
        self.web_search_fn = web_search_fn
        self.artifacts = {}     # name -> full markdown (reports), surfaced in final
        self._last_detail = None  # per-step structured detail (e.g. inner plan tree)
        self._advisor = None    # EW countermeasure advisor (lazy)
        self._wargame = None    # EOB wargame planner (lazy)

    # ---- tools: each returns a SHORT observation string for the loop ----
    def _t_composition_query(self, args):
        q = args.get("question", "")
        r = self.core.run(q, paraphrase=False)
        val = r.get("value")
        items = sorted(val) if isinstance(val, (set, frozenset)) else None
        self._last_detail = {"kind": "composition", "plan": r["plan"],
                             "det_answer": r["det_answer"],
                             "reflections": r.get("reflections") or [],
                             "value_items": items}      # 实体集合，供数据流绑定
        return f"答案：{r['det_answer']}（plan={_json.dumps(r['plan'], ensure_ascii=False)[:120]}）"

    def _t_entity_dossier(self, args):
        d = self.reporter.dossier(args.get("entity", ""))
        if not d.get("found"):
            self._last_detail = {"kind": "gap", "entity": args.get("entity", "")}
            return d.get("note", "未收录该实体。")
        key = f"dossier::{d['entity']}"
        self.artifacts[key] = d["markdown"]
        self._last_detail = {"kind": "report", "title": f"{d['entity']} 装备档案",
                             "n_claims": d["n_claims"], "n_low_conf": d.get("n_low_conf", 0)}
        return f"已生成《{d['entity']} 装备档案》（{d['n_claims']} 条带证据事实）。摘要：{d.get('summary','')[:120]}"

    def _t_comparison_report(self, args):
        c = self.reporter.compare(args.get("a", ""), args.get("b", ""))
        key = f"compare::{c['a']}::{c['b']}"
        self.artifacts[key] = c["markdown"]
        self._last_detail = {"kind": "report", "title": f"{c['a']} vs {c['b']} 对比报告",
                             "missing": c.get("missing") or []}
        note = ("（注：" + "、".join(c["missing"]) + " 未收录）") if c.get("missing") else ""
        return f"已生成《{c['a']} vs {c['b']} 对比报告》{note}。"

    def _t_web_search(self, args):
        if self.web_search_fn is None:
            return "web_search 暂未启用（图谱外信息本次无法补全）。"
        self._last_detail = {"kind": "web", "query": args.get("query", "")}
        return self.web_search_fn(args.get("query", ""))

    def _t_counter_advisor(self, args):
        """Given an enemy radar model, derive its threat profile from the KG and
        recommend doctrine-level countermeasures (jam/avoid/anti-ECCM)."""
        model = args.get("radar") or args.get("entity") or args.get("question", "")
        from threat_profile import profile_for
        prof = profile_for(model, resolver=self.reporter.kg.resolve)
        if not prof:
            self._last_detail = {"kind": "gap", "entity": model}
            return (f"未找到「{model}」的威胁画像（图谱缺跟踪体制/用途等字段）。"
                    f"可改用 web_search 补全，或手动给定画像。")
        if self._advisor is None:
            from ew_advisor import load_advisor
            self._advisor = load_advisor()
        rec = self._advisor.recommend(prof)
        name = prof.get("name", model)
        md = self._advisor.render(rec, title=name)
        self.artifacts[f"counter::{name}"] = md
        n = len(rec["recommended"])
        top = rec["recommended"][0]["name"] if rec["recommended"] else "—"
        self._last_detail = {"kind": "counter", "title": f"{name} 对抗建议",
                             "threat_priority": rec["threat_priority"],
                             "n_recommend": n, "profile": prof}
        return (f"已生成《{name} 对抗建议》：威胁等级 {rec['threat_priority']}/5，"
                f"推荐 {n} 项干扰手段（首选：{top}）。")

    def _t_eob_wargame(self, args):
        """Given several enemy radars (an EOB), produce an integrated countermeasure
        plan: threat ranking + minimal asset package + technique economy + gaps."""
        raw = args.get("radars") or args.get("eob") or args.get("question", "")
        if isinstance(raw, str):
            models = [m.strip() for m in re.split(r"[,\n、;]+", raw) if m.strip()]
        else:
            models = [str(m).strip() for m in raw if str(m).strip()]
        if not models:
            return "未解析到敌方雷达型号列表。请用逗号/换行分隔多个型号。"
        if self._wargame is None:
            from ew_wargame import EWWargame
            if self._advisor is None:
                from ew_advisor import load_advisor
                self._advisor = load_advisor()
            self._wargame = EWWargame(advisor=self._advisor, resolver=self.reporter.kg.resolve)
        res = self._wargame.assess(models)
        md = self._wargame.render(res)
        self.artifacts["wargame::EOB"] = md
        plan = res["plan"]
        self._last_detail = {"kind": "wargame", "n_engaged": plan["n_engaged"],
                             "n_package": len(plan["package"]),
                             "coverage": plan["coverage"], "n_unknown": len(plan["unknown"])}
        return (f"已生成《战场对抗推演》：识别 {plan['n_engaged']} 部威胁，"
                f"推荐装备包 {len(plan['package'])} 套覆盖 {plan['coverage']['threats']}"
                f"（加权 {plan['coverage']['weighted_pct']}%）。")

    TOOLS = {
        "composition_query": ("回答任何关于雷达的事实/计数/列举/数值聚合/对比/多跳/多约束筛选问题，"
                              "返回确定性答案。这是主分析工具，绝大多数问题用它。", "_t_composition_query"),
        "entity_dossier": ("生成某一型号雷达的可溯源装备档案（每条事实带来源+置信度）。"
                           "当用户要某型号的『档案/概览/介绍』时用。", "_t_entity_dossier"),
        "comparison_report": ("生成两型号雷达的可溯源对比报告。当用户要『对比/比较』两个具体型号时用。",
                              "_t_comparison_report"),
        "web_search": ("当 composition_query 返回『未收录/空』、或问题涉及图谱可能没有的新型号/最新动态时，"
                       "联网检索补全。", "_t_web_search"),
        "counter_advisor": ("给定一部【敌方雷达型号】，给出对抗建议（推荐哪些干扰样式、应避免哪些、"
                            "怎么应对其抗干扰），基于威胁画像 + EW doctrine，结论可溯源。"
                            "当用户问『怎么对抗/干扰/压制 X 雷达』『X 该用什么干扰』时用。", "_t_counter_advisor"),
        "eob_wargame": ("给定【多部敌方雷达】(逗号或换行分隔)，做战场对抗推演：威胁排序 + 最小装备包"
                        "(集合覆盖) + 干扰样式经济性 + 能力缺口 + SEAD。当用户给一组敌方布控/EOB、"
                        "问『这些雷达怎么应对』『整体对抗方案』时用，radars 填型号列表。", "_t_eob_wargame"),
    }

    def call_tool(self, tool, args):
        """Directly invoke a single tool (manual mode — bypasses the ReAct planner).
        Returns {ok, tool, args, observation, detail, report_md}. report_md holds the
        full markdown of any report artifact produced by this call."""
        spec = self.TOOLS.get(tool)
        if not spec:
            return {"ok": False, "tool": tool, "args": args, "detail": None,
                    "report_md": "", "observation": f"未知工具 {tool}。"}
        self._last_detail = None
        before = set(self.artifacts)
        try:
            obs = getattr(self, spec[1])(args or {})
            ok = True
        except Exception as e:
            obs, ok = f"工具执行出错：{type(e).__name__}: {e}", False
        new_keys = [k for k in self.artifacts if k not in before]
        report_md = "\n\n".join(self.artifacts[k] for k in new_keys)
        return {"ok": ok, "tool": tool, "args": args, "observation": obs,
                "detail": self._last_detail, "report_md": report_md}

    def _system_prompt(self):
        tool_desc = "\n".join(f"- {n}({_args(n)})：{d}" for n, (d, _) in self.TOOLS.items())
        return (
            "你是雷达情报分析助手。通过调用工具来完成用户的分析请求，支持多步：先收集证据，再综合。\n\n"
            "可用工具：\n" + tool_desc + "\n- final(answer)：给出最终回答，结束。\n\n"
            "每一步**只输出一个 JSON**，格式：\n"
            '{"thought":"简短推理","action":"工具名","args":{...}}\n'
            "或结束时：\n"
            '{"thought":"...","action":"final","args":{"answer":"最终中文回答"}}\n\n'
            "规则：\n"
            "- 简单事实/计数/对比问题：通常一次 composition_query 就够，然后 final。\n"
            "- 复杂分析（如『对比中美舰载雷达』『生成X的报告』）：分多步调用工具收集，再 final。\n"
            "- 对抗类问题（『怎么对抗/干扰/压制 X 雷达』『X 该用什么干扰』）：用 counter_advisor，radar 填型号。\n"
            "- 多部敌方雷达/整体对抗方案/EOB：用 eob_wargame，radars 填型号列表（逗号分隔）。\n"
            "- composition_query 或 entity_dossier 说某实体未收录时，用 web_search 补全；"
            "**web_search 的 query 只用型号本身（如 AN/SPY-1），不要加『装备档案/参数/介绍』等词**。\n"
            "- web_search 返回联网事实后，直接据此 final 回答（并注明信息来自联网、未经图谱核验）。\n"
            "- 工具返回『已生成报告』后，**直接 final，不要重复调用同一工具**。\n"
            "- **final 的 answer 只能复述工具观察里出现过的事实，严禁添加图谱未提供的技术参数/数字/型号**"
            "（报告全文与逐条证据会自动附在你的回答后，你只需写一句引导语，如『已生成对比报告，详见下方』）。\n"
            "- 收集够了就 final，不要无谓多调。最多 6 步。\n"
            "- 只输出 JSON，不要 markdown 包裹、不要解释。"
        )

    def _resolve_coref(self, question, history):
        """Rewrite a follow-up that uses pronouns / ellipsis into a self-contained
        question, using the conversation history (decontextualization)."""
        if not history:
            return question, question
        if not re.search(r"它|他|她|它们|他们|这个|那个|这款|那款|这型|该|此|前者|后者|呢$|呢？$", question):
            return question, question
        hist = "\n".join(f"{h['role']}: {h['content'][:200]}" for h in history[-4:])
        sysp = ("把用户的最新追问改写成一个自包含的完整问题：补全被指代(它/这个/该/前者…)"
                "或省略的实体/主语，依据对话历史。若已自包含则原样返回。只输出改写后的问题，不要解释。")
        out = llm_call([{"role": "system", "content": sysp},
                        {"role": "user", "content": f"对话历史：\n{hist}\n\n最新追问：{question}\n\n改写："}],
                       max_tokens=120)
        rewritten = (out or "").strip().strip('"').strip()
        if not rewritten or rewritten.startswith("[LLM_ERROR") or len(rewritten) > 200:
            return question, question
        return rewritten, question

    def run(self, question, history=None, max_iters=6, verbose=False):
        """Non-streaming convenience wrapper: drains run_stream() and returns the
        final result dict (used by eval scripts / programmatic callers)."""
        final = None
        for ev in self.run_stream(question, history=history, max_iters=max_iters):
            if verbose:
                if ev["type"] == "coref":
                    print(f"  [coref] {ev['original']!r} -> {ev['resolved']!r}")
                elif ev["type"] == "step":
                    print(f"  [step {ev['step']}] {ev['action']} {ev['args']}")
            if ev["type"] == "final":
                final = ev
        return {"answer": final["answer"], "trace": final["trace"],
                "artifacts": final["artifacts"],
                "resolved_question": final["resolved_question"],
                "original_question": final["original_question"]}

    # ---------------- Plan-and-Execute ----------------
    MAX_FOREACH = 6        # cap fan-out of a foreach step

    def _make_plan(self, question):
        """LLM decomposes the request into a step plan with dataflow. Returns a list
        of steps [{id, tool, args, foreach?}]. Falls back to a single composition
        step if planning fails. The LLM only PLANS; tools execute deterministically."""
        tool_desc = "\n".join(f"  - {n}({_args(n)})：{d.splitlines()[0]}" for n, (d, _) in self.TOOLS.items())
        sysp = (
            "你是雷达情报分析的**任务规划器**。把用户请求分解为一串可执行步骤(JSON)，"
            "由确定性引擎执行。你只规划、不回答。\n\n"
            "可用工具：\n" + tool_desc + "\n\n"
            "步骤格式：{\"id\":\"s1\",\"tool\":\"工具名\",\"args\":{...}}\n"
            "**数据流**：后面步骤可引用前面步骤的结果：\n"
            "  · args 里写 \"$s1\" 表示用步骤 s1 的结果(通常是实体集合/列表)；\n"
            "  · 若某步要对一个集合**逐个**处理，加 \"foreach\":\"$s1\"，并在 args 里用 \"$item\" 表示当前元素。\n\n"
            "规则：\n"
            "  · 简单事实/计数/列举/对比/单型号问题 → 单步即可。\n"
            "  · 组合任务(先查出一批，再对每个做分析) → 用 foreach 串联，如"
            "‘列出X再分别给对抗建议’= s1 列举 + s2 foreach $s1 调 counter_advisor。\n"
            "  · counter_advisor 针对一部雷达；eob_wargame 针对一组(radars 可填 \"$s1\")。\n"
            "  · 型号未收录可加一步 web_search。\n"
            "  · 步数尽量少、够用即可。只输出 JSON 数组，不要解释、不要 markdown 包裹。\n\n"
            "示例：\n"
            "问题：列出中国研制的舰载雷达，并分别给出对抗建议\n"
            "计划：[{\"id\":\"s1\",\"tool\":\"composition_query\",\"args\":{\"question\":\"列出中国研制的舰载雷达\"}},"
            "{\"id\":\"s2\",\"tool\":\"counter_advisor\",\"foreach\":\"$s1\",\"args\":{\"radar\":\"$item\"}}]\n"
            "问题：对比 AN/TPY-2 和 AN/MPQ-65\n"
            "计划：[{\"id\":\"s1\",\"tool\":\"comparison_report\",\"args\":{\"a\":\"AN/TPY-2\",\"b\":\"AN/MPQ-65\"}}]"
        )
        raw = llm_call([{"role": "system", "content": sysp},
                        {"role": "user", "content": f"问题：{question}\n计划："}], max_tokens=500)
        m = re.search(r"\[.*\]", raw or "", flags=re.DOTALL)
        if m:
            try:
                steps = _json.loads(m.group(0))
                steps = [s for s in steps if isinstance(s, dict) and s.get("tool") in self.TOOLS]
                if steps:
                    return steps
            except Exception:
                pass
        return [{"id": "s1", "tool": "composition_query", "args": {"question": question}}]

    @staticmethod
    def _resolve_arg(v, env, item=None):
        if v == "$item":
            return item
        if isinstance(v, str):
            mm = re.fullmatch(r"\$(\w+)", v)
            if mm:
                return env.get(mm.group(1))
            return v
        return v

    def _bindable(self):
        """The value of the just-run tool, for dataflow binding (entity list / None)."""
        d = self._last_detail or {}
        return d.get("value_items")

    def _run_one(self, tool, args):
        self._last_detail = None
        try:
            obs = getattr(self, self.TOOLS[tool][1])(args or {})
        except Exception as e:
            obs = f"工具执行出错：{type(e).__name__}: {e}"
        return obs, self._last_detail, self._bindable()

    def _exec_steps(self, steps, env, trace, flags):
        """Execute a step plan deterministically with dataflow binding; yield step
        events; mutate env/trace; record gap flags (empty foreach) into `flags`."""
        for st in steps:
            tool = st.get("tool")
            if tool not in self.TOOLS:
                continue
            raw_args = st.get("args", {}) or {}
            if st.get("foreach"):
                items = self._resolve_arg(st["foreach"], env) or []
                if not isinstance(items, list):
                    items = [items]
                items = [x for x in items if x][: self.MAX_FOREACH]
                if not items:
                    flags.setdefault("empty_foreach", []).append(st.get("id", "?"))
                bound = []
                for it in items:
                    a = {k: self._resolve_arg(v, env, item=it) for k, v in raw_args.items()}
                    obs, det, _ = self._run_one(tool, a)
                    entry = {"step": len(trace), "action": tool, "args": a,
                             "observation": obs[:300], "thought": f"foreach {st.get('id','')}={it}",
                             "detail": det}
                    trace.append(entry); yield {"type": "step", **entry}
                    bound.append(it)
                env[st.get("id", f"s{len(trace)}")] = bound
            else:
                a = {k: self._resolve_arg(v, env) for k, v in raw_args.items()}
                obs, det, val = self._run_one(tool, a)
                entry = {"step": len(trace), "action": tool, "args": a,
                         "observation": obs[:300], "thought": "", "detail": det}
                trace.append(entry); yield {"type": "step", **entry}
                env[st.get("id", f"s{len(trace)}")] = val

    # signals of a degenerate / incomplete result (deterministic, not LLM self-judge)
    _GAP_MARKERS = ("未找到", "未收录", "联网未找到", "未解析到", "无足够数据",
                    "共 0 项", "共 0 款", "暂未启用")

    def _verify(self, trace, flags):
        """Deterministic checks over the executed trace. Returns a list of issues."""
        issues = []
        for e in trace:
            o = e.get("observation", "")
            if o.startswith("工具执行出错"):
                issues.append({"kind": "error", "action": e["action"], "text": o[:90]})
            elif any(m in o for m in self._GAP_MARKERS):
                issues.append({"kind": "empty", "action": e["action"], "args": e.get("args"), "text": o[:90]})
        for sid in flags.get("empty_foreach", []):
            issues.append({"kind": "empty_foreach", "text": f"步骤 {sid} 的集合为空，逐项分析未产生结果"})
        return issues

    def _revise_plan(self, question, steps, trace, issues):
        """Given verification issues, ask the LLM to produce a REVISED plan (e.g. add
        web_search for an unresolved model, rephrase an empty enumeration). One round."""
        issue_txt = "；".join(i.get("text", i["kind"]) for i in issues)
        obs_txt = "\n".join(f"- [{e['action']}] {e['observation'][:80]}" for e in trace)
        tool_desc = "\n".join(f"  - {n}({_args(n)})" for n in self.TOOLS)
        sysp = (
            "上一版计划执行后存在缺口(空结果/未收录/出错)。请输出**修正后的步骤计划**(JSON 数组)以弥补缺口。\n"
            "可用工具：\n" + tool_desc + "\n"
            "常见修正：① 某型号图谱未收录 → 加 web_search(query=型号本身) 补全；"
            "② 列举/计数为空 → 放宽或改写问题(去掉过严约束)；③ 工具出错 → 改参数或换工具。\n"
            "数据流语法同前($sN、foreach $sN、$item)。只输出 JSON 数组，不要解释。"
        )
        raw = llm_call([{"role": "system", "content": sysp},
                        {"role": "user", "content": f"问题：{question}\n原计划：{_json.dumps(steps, ensure_ascii=False)}\n"
                                                     f"缺口：{issue_txt}\n各步观察：\n{obs_txt}\n\n修正计划："}],
                       max_tokens=500)
        m = re.search(r"\[.*\]", raw or "", flags=re.DOTALL)
        if not m:
            return None
        try:
            rev = _json.loads(m.group(0))
            rev = [s for s in rev if isinstance(s, dict) and s.get("tool") in self.TOOLS]
            return rev or None
        except Exception:
            return None

    def run_stream(self, question, history=None, max_iters=6):
        """Generator (Plan-and-Execute): yields events so a UI can render the agent's
        reasoning live. Event `type` ∈ {coref, plan, step, final}. The LLM decomposes
        the task into a dataflow plan; deterministic tools execute it; facts come only
        from tool outputs (LLM stays out of the trust path)."""
        resolved, original = self._resolve_coref(question, history)
        if resolved != original:
            yield {"type": "coref", "original": original, "resolved": resolved}
        question = resolved

        steps = self._make_plan(question)
        yield {"type": "plan", "steps": steps}

        env, trace, flags = {}, [], {}
        yield from self._exec_steps(steps, env, trace, flags)

        # ---- ③ deterministic verification + one self-revision round ----
        issues = self._verify(trace, flags)
        if issues:
            yield {"type": "verify", "issues": issues}
            rev = self._revise_plan(question, steps, trace, issues)
            if rev and rev != steps:
                yield {"type": "plan", "steps": rev, "revised": True}
                yield from self._exec_steps(rev, env, trace, flags)

        # ---- synthesis: LLM summarises observations (facts only); artifacts appended verbatim ----
        obs_block = "\n".join(f"- [{e['action']}] {e['observation']}" for e in trace)
        if len(trace) <= 1 and trace and (trace[0]["detail"] or {}).get("kind") == "composition":
            answer = self._finalize(trace[0]["detail"].get("det_answer", obs_block))
        else:
            syn = llm_call([
                {"role": "system", "content": "你是雷达情报分析助手。基于下列各步骤的确定性观察，"
                 "写一段简洁的中文综述回答用户问题。**只复述观察中出现的事实，严禁添加未出现的数字/型号/参数**。"
                 "报告全文会自动附在后面，你只需综述要点。"},
                {"role": "user", "content": f"问题：{question}\n各步观察：\n{obs_block}\n\n综述："}],
                max_tokens=500)
            if not syn or syn.startswith("[LLM_ERROR"):
                syn = obs_block
            answer = self._finalize(syn)
        yield {"type": "final", "answer": answer, "trace": trace, "artifacts": dict(self.artifacts),
               "resolved_question": resolved, "original_question": original}

    def _finalize(self, llm_answer):
        """Append the deterministic, provenance-cited artifacts verbatim so the
        auditable deliverable is always what the user sees — not the LLM's prose."""
        if not self.artifacts:
            return llm_answer
        parts = [llm_answer.strip(), "\n\n---\n"]
        for name, md in self.artifacts.items():
            parts.append(md + "\n")
        return "\n".join(parts)


def _args(name):
    return {"composition_query": "question", "entity_dossier": "entity",
            "comparison_report": "a, b", "web_search": "query",
            "counter_advisor": "radar", "eob_wargame": "radars"}.get(name, "...")


def _parse_action(raw):
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None
