# -*- coding: utf-8 -*-
"""雷达情报分析 Agent — 多页面 Streamlit 应用。

  streamlit run agent/app.py

左侧导航在多个专属功能页之间切换：
  💬 智能问答     — 双层 ReAct Agent，实时展开推理轨迹
  🎯 对抗建议     — 选一部敌方雷达 → 对抗手段/装备（doctrine 可溯源·非实测）
  🗺️ 对抗推演     — 选一组敌方 EOB → 威胁排序 + 最小装备包(集合覆盖) + 缺口
  📋 情报报告     — 可溯源装备档案 / 对比
  🌐 联网检索     — 图谱缺口联网补全
"""
import os, sys, json
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title="雷达情报分析 Agent", page_icon="📡", layout="wide")

# ---------------- styling ----------------
st.markdown("""
<style>
  .block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1200px; }
  /* dark sidebar with forced LIGHT text so labels are always readable */
  section[data-testid="stSidebar"] { background: #11151c !important; }
  section[data-testid="stSidebar"] *:not(svg):not(path) { color: #e8eaed !important; }
  section[data-testid="stSidebar"] .stButton > button {
      color: #e8eaed !important; background: transparent !important;
      border: 1px solid transparent !important; text-align: left; justify-content: flex-start;
      font-size: 0.95rem; padding: 0.45rem 0.65rem; width: 100%; }
  section[data-testid="stSidebar"] .stButton > button:hover {
      background: rgba(255,255,255,0.10) !important; border-color: rgba(255,255,255,0.12) !important; }
  section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
      background: rgba(56,139,253,0.28) !important; border-color: rgba(56,139,253,0.55) !important;
      font-weight: 600; }
  div[data-testid="stMetric"] {
      background: rgba(128,128,128,0.10); border-radius: 10px; padding: 0.6rem 0.85rem; }
  .pill { display:inline-block; padding:2px 9px; margin:2px 4px 2px 0; border-radius:11px;
          background:rgba(56,139,253,0.15); color:#3b82f6; font-size:0.8rem; }
  .pill-warn { background:rgba(234,160,0,0.16); color:#b8860b; }
  h1, h2, h3 { letter-spacing: .3px; }
</style>
""", unsafe_allow_html=True)


# ---------------- cached resources ----------------
@st.cache_resource(show_spinner="正在加载知识图谱、算子引擎与对抗知识库…")
def get_core():
    from agent_loop import RadarAgent
    from ew_advisor import load_advisor
    from ew_wargame import EWWargame
    from threat_profile import all_profile_names
    agent = RadarAgent()
    adv = load_advisor()
    wg = EWWargame(advisor=adv, resolver=agent.reporter.kg.resolve)
    names = all_profile_names()
    return agent, adv, wg, names


TOOL_META = {
    "composition_query": ("🧮", "算子组合查询"), "entity_dossier": ("📋", "装备档案"),
    "comparison_report": ("⚖️", "对比报告"), "web_search": ("🌐", "联网检索"),
    "counter_advisor": ("🎯", "对抗建议"), "eob_wargame": ("🗺️", "对抗推演"),
}


# ---------------- shared: ReAct step rendering (chat page) ----------------
def render_step(ev, container):
    tool = ev.get("action", "")
    emoji, name = TOOL_META.get(tool, ("🔧", tool))
    args = ev.get("args") or {}
    arg_str = "、".join(f"{k}={v}" for k, v in args.items())
    with container:
        st.markdown(f"**{emoji} 调用工具 `{name}`**　<span style='color:#888'>{arg_str}</span>",
                    unsafe_allow_html=True)
        if ev.get("thought"):
            st.markdown(f"<span style='color:#888'>💭 {ev['thought']}</span>", unsafe_allow_html=True)
        det = ev.get("detail") or {}
        kind = det.get("kind")
        if kind == "composition":
            st.caption("内层算子组合计划树（确定性求值）：")
            st.code(json.dumps(det["plan"], ensure_ascii=False, indent=2), language="json")
            if det.get("reflections"):
                st.markdown(f"🔁 触发反思重规划 {len(det['reflections'])} 轮")
            st.success(f"确定性答案：{det.get('det_answer', '')}")
        elif kind == "counter":
            st.markdown(f"🎯 **{det.get('title','')}**　威胁等级 **{det.get('threat_priority','?')}/5**"
                        f"　推荐 {det.get('n_recommend','?')} 项")
        elif kind == "wargame":
            cov = det.get("coverage", {})
            st.markdown(f"🗺️ **战场对抗推演**　{det.get('n_engaged','?')} 部威胁 → 装备包 "
                        f"{det.get('n_package','?')} 套覆盖 {cov.get('threats','?')}（加权 {cov.get('weighted_pct','?')}%）")
        elif kind == "report":
            badges = [f"📎 {det.get('n_claims','?')} 条带证据事实"]
            if det.get("n_low_conf"):
                badges.append(f"⚠ {det['n_low_conf']} 条低置信")
            st.markdown(f"**生成《{det.get('title','')}》**　" + "　".join(badges))
        elif kind == "web":
            st.markdown(f"🌐 联网检索：`{det.get('query','')}`")
        st.markdown(f"<span style='color:#1a7f37'>👁 观察：</span>{ev.get('observation','')}",
                    unsafe_allow_html=True)
        st.divider()


# ================= PAGE: 智能问答 =================
def page_chat(agent):
    st.title("💬 智能问答")
    st.caption("双层智能体：外层 ReAct 自主编排 · 内层算子组合 · 可溯源 · 多轮对话。问事实/统计/对比/对抗都行。")

    examples = ["美国研制且工作在 S 波段的雷达有多少款？", "对比 AN/TPY-2 和 AN/MPQ-65",
                "怎么对抗 AN/SPG-62 雷达？", "敌方部署了 AN/SPG-62、AN/APY-9、Type 281，整体对抗方案？"]
    cols = st.columns(len(examples))
    clicked = None
    for c, q in zip(cols, examples):
        if c.button(q, use_container_width=True):
            clicked = q

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "assistant":
                if m.get("resolved") and m["resolved"] != m.get("original"):
                    st.info(f"🔄 指代消解：「{m['original']}」→「{m['resolved']}」")
                if m.get("trace"):
                    with st.expander(f"🧠 推理轨迹（{len([s for s in m['trace'] if 'action' in s])} 步）"):
                        for s in m["trace"]:
                            if "action" in s:
                                render_step(s, st.container())
                st.markdown(m["content"])
            else:
                st.markdown(m["content"])

    prompt = st.chat_input("问我关于雷达的事实、统计、对比、对抗…") or clicked
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            history = [{"role": m["role"], "content": m["content"]}
                       for m in st.session_state.messages[:-1] if m["role"] in ("user", "assistant")][-6:]
            log = st.status("🤖 Agent 正在分析…", expanded=True)
            final_ev, trace, resolved, original, n = None, [], prompt, prompt, 0
            try:
                for ev in agent.run_stream(prompt, history=history):
                    if ev["type"] == "coref":
                        resolved, original = ev["resolved"], ev["original"]
                        with log:
                            st.info(f"🔄 指代消解：「{original}」→「{resolved}」")
                    elif ev["type"] == "plan":
                        with log:
                            steps_txt = "　".join(
                                f"{s.get('id','')}.{TOOL_META.get(s.get('tool'),('🔧',s.get('tool')))[1]}"
                                + (f"(foreach {s['foreach']})" if s.get("foreach") else "")
                                for s in ev["steps"])
                            tag = "🔁 **修订计划**" if ev.get("revised") else f"🧭 **任务分解（{len(ev['steps'])} 步）**"
                            st.markdown(f"{tag}：{steps_txt}")
                    elif ev["type"] == "verify":
                        with log:
                            st.warning("🔎 自检发现缺口：" + "；".join(i.get("text", i["kind"])[:50] for i in ev["issues"])
                                       + " → 自动重规划")
                    elif ev["type"] == "step":
                        n += 1
                        log.update(label=f"🤖 第 {n} 步：{TOOL_META.get(ev['action'], ('🔧', ev['action']))[1]}…")
                        trace.append(ev); render_step(ev, log)
                    elif ev["type"] == "final":
                        final_ev = ev
                answer = final_ev["answer"] if final_ev else "（无结果）"
                log.update(label=f"✅ 完成（{n} 步）", state="complete", expanded=False)
            except Exception as e:
                answer = f"出错：{type(e).__name__}: {e}"
                log.update(label="❌ 出错", state="error")
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer, "trace": trace,
                                          "resolved": resolved, "original": original})
    if st.session_state.messages:
        if st.button("🗑 清空对话"):
            st.session_state.messages = []
            st.rerun()


# ================= PAGE: 对抗建议 =================
def page_counter(adv, names):
    from threat_profile import profile_for
    st.title("🎯 对抗建议")
    st.caption("选一部敌方雷达 → 威胁画像(取自图谱) → 推荐干扰样式 / 应避免 / 抗干扰 / 具体装备。"
               "结论基于 EW doctrine，可溯源、标注「非实测」。")

    c1, c2 = st.columns([3, 1])
    pick = c1.selectbox("敌方雷达型号", options=names,
                        index=names.index("AN/SPG-62") if "AN/SPG-62" in names else 0)
    custom = c2.text_input("或手填型号", placeholder="AN/APG-66")
    target = custom.strip() or pick
    go = st.button("▶ 生成对抗建议", type="primary")
    if not go:
        return
    prof = profile_for(target, resolver=getattr(adv, "_resolver", None))
    if not prof:
        st.warning(f"图谱未收录「{target}」的威胁画像，换个型号或先用「联网检索」补全。")
        return
    rec = adv.recommend(prof)
    _render_counter(adv, rec, prof, target)


def _profile_pills(adv, prof):
    ev = prof.get("_evidence", {})
    TIER = {"prior": "推测", "llm": "文献", "keyword": ""}
    bits = []
    for k, lab in (("purpose", "用途"), ("tracking_method", "跟踪体制"), ("scan_type", "扫描"),
                   ("bands", "频段"), ("freq_agile", "频率捷变"), ("lpi", "LPI")):
        if prof.get(k) not in (None, [], False):
            v = "、".join(prof[k]) if isinstance(prof[k], list) else prof[k]
            tier = (ev.get(k) or {}).get("tier", "")
            mk = TIER.get(tier, "")
            cls = "pill pill-warn" if tier == "prior" else "pill"
            bits.append(f"<span class='{cls}'>{lab}: {v}{'（'+mk+'）' if mk else ''}</span>")
    for k, lab, unit in (("frequency_GHz", "频率", "GHz"), ("peak_power_kW", "峰值功率", "kW"),
                         ("range_km", "探测距离", "km")):
        if prof.get(k) is not None:
            bits.append(f"<span class='pill'>{lab}: {prof[k]:g}{unit}</span>")
    return "".join(bits)


def _render_counter(adv, rec, prof, title):
    c1, c2, c3 = st.columns(3)
    c1.metric("威胁等级", f"{rec['threat_priority']}/5")
    c2.metric("推荐手段", len(rec["recommended"]))
    c3.metric("可用装备类", len(rec.get("equipment", {})) + (1 if rec.get("arm") else 0))
    st.markdown("**威胁画像**　" + _profile_pills(adv, prof), unsafe_allow_html=True)
    case = prof.get("documented_case")
    if case:
        with st.container(border=True):
            st.markdown(f"**📜 实战记载（真实战史·有出处）**　<span class='pill'>{case.get('conflict','')}</span>",
                        unsafe_allow_html=True)
            if prof.get("role"):
                st.caption(prof["role"])
            if case.get("countermeasures_used"):
                st.markdown("**实际采用的对抗**：" + "；".join(c["what"] for c in case["countermeasures_used"]))
            if case.get("counter_counter"):
                st.markdown("**该雷达方的反制**：" + "；".join(case["counter_counter"]))
            if case.get("outcome"):
                st.markdown(f"**结果**：{case['outcome']}")
            if case.get("lesson"):
                st.info("💡 " + case["lesson"])
            if case.get("sources"):
                st.caption("来源：" + " ｜ ".join(case["sources"]))
    elif prof.get("sources"):
        with st.container(border=True):
            sysline = f"（{prof.get('system','')}）" if prof.get("system") else ""
            st.markdown(f"**📚 威胁来源（开源参考）**{sysline}")
            if prof.get("role"):
                st.caption(prof["role"])
            st.caption("来源：" + " ｜ ".join(prof["sources"]))
    if rec.get("engagement"):
        with st.container(border=True):
            st.markdown("**🧭 交战研判（按本机参数）**")
            for n in rec["engagement"]:
                st.markdown(f"- {n['text']}")
    st.divider()

    if rec["recommended"]:
        st.markdown("#### ✅ 推荐干扰手段")
        for r in rec["recommended"]:
            with st.container(border=True):
                cite = r["cites"][0]
                st.markdown(f"**{r['name']}**　<span class='pill'>{r['category']}</span>"
                            f"　置信 {r['confidence']:.2f}", unsafe_allow_html=True)
                st.caption(f"依据：{cite['rationale']}　〔{cite['rule_id']} · {cite['source']}〕")
                if r["geometry_note"]:
                    st.caption(f"战术附注：{r['geometry_note']}")
    if rec.get("equipment") or rec.get("arm"):
        st.markdown("#### 🛰 可用对抗装备（开源参考）")
        for tid, syslist in rec["equipment"].items():
            tname = adv.jam.get(tid, {}).get("name_zh", tid)
            st.markdown(f"- **{tname}** ← " + "；".join(f"{s['name_zh']}({s['country']})" for s in syslist))
        if rec["arm"]:
            st.markdown("- **反辐射硬摧毁(SEAD)** ← " + "；".join(f"{s['name_zh']}({s['country']})" for s in rec["arm"]))
    cc = st.columns(2)
    if rec["countered_by_eccm"]:
        with cc[0]:
            st.markdown("#### ⚠ 会被对方抗干扰抵消")
            for r in rec["countered_by_eccm"]:
                st.markdown(f"- {r['name']}（被「{'、'.join(r['neutralised_by'])}」克制，残余 {r['confidence']:.2f}）")
    if rec["avoid"]:
        with cc[1]:
            st.markdown("#### ⛔ 应避免")
            for r in rec["avoid"]:
                st.markdown(f"- {r['name']}：{r['cites'][0]['rationale']}")
    st.divider()
    md = adv.render(rec, title=title)
    st.download_button("📥 下载对抗建议(Markdown)", md, file_name=f"对抗建议_{title}.md".replace("/", "_"))
    st.caption("⚠ doctrine 级启发式（非实测），供研究/教学参考。")


# ================= PAGE: 对抗推演 =================
def page_wargame(wg, names):
    st.title("🗺️ 对抗推演（多威胁 EOB）")
    st.caption("选择/填写一组敌方雷达布控 → 威胁排序 + 最小装备包(贪心加权集合覆盖) + 干扰样式经济性 + 能力缺口 + SEAD。")

    default = [m for m in ["AN/SPG-62", "AN/APY-9", "EMPAR", "Type 281"] if m in names]
    sel = st.multiselect("从图谱选择敌方雷达", options=names, default=default)
    extra = st.text_area("追加手填型号（每行一个，可选）", height=80, placeholder="AN/MPQ-65")
    models = list(sel) + [x.strip() for x in extra.splitlines() if x.strip()]
    st.caption(f"当前 EOB：{len(models)} 部 → " + "、".join(models[:10]) if models else "请先选择或填写敌方雷达")
    if not st.button("▶ 生成对抗推演", type="primary"):
        return
    if not models:
        st.warning("请先选择或填写至少一部敌方雷达。")
        return
    with st.spinner("推演中（画像 → 对抗 → 集合覆盖装备分配）…"):
        res = wg.assess(models)
    plan, threats = res["plan"], res["threats"]

    m = st.columns(4)
    m[0].metric("识别威胁", plan["n_engaged"])
    m[1].metric("装备包", f"{len(plan['package'])} 套")
    m[2].metric("加权覆盖", f"{plan['coverage']['weighted_pct']}%")
    m[3].metric("SEAD 目标", len(plan["sead_targets"]))
    if plan["unknown"]:
        st.info("图谱未收录（已跳过）：" + "、".join(plan["unknown"]))

    st.markdown("#### 📊 威胁排序")
    rows = []
    for t in threats:
        if not t.get("found"):
            continue
        p = t["profile"]
        rows.append({"型号": t["model"], "威胁度": f"{t['priority']}/5",
                     "用途": "、".join(p.get("purpose") or []),
                     "跟踪体制": p.get("tracking_method", "—"),
                     "频段": "、".join(p.get("bands") or []) or "—",
                     "首选对抗": t["rec"]["recommended"][0]["name"] if t["rec"]["recommended"] else "—（受限）"})
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown(f"#### 🎒 推荐装备包（最小覆盖 · 携 {len(plan['package'])} 套覆盖 {plan['coverage']['threats']}）")
    for a in plan["package"]:
        with st.container(border=True):
            via = ("　<span class='pill'>经由：" + "、".join(a["via"][:4]) + "</span>") if a.get("via") else ""
            st.markdown(f"**{a['name']}**　{a['country']}{via}", unsafe_allow_html=True)
            st.caption("应对：" + "、".join(a["covers"]))
    if not plan["package"]:
        st.write("_无可匹配装备（型号/频段信息不足）_")

    cc = st.columns(2)
    with cc[0]:
        if plan["technique_economy"]:
            st.markdown("#### 🎛 干扰样式经济性")
            for te in plan["technique_economy"][:5]:
                st.markdown(f"- **{te['name']}** · {len(te['threats'])} 个威胁")
    with cc[1]:
        if plan["sead_targets"]:
            st.markdown("#### 🎯 SEAD 优先目标")
            st.write("、".join(plan["sead_targets"]))
        if plan["gaps_no_asset"] or plan["gaps_all_countered"]:
            st.markdown("#### ⚠ 能力缺口")
            if plan["gaps_all_countered"]:
                st.markdown("- 软杀伤受限：" + "、".join(plan["gaps_all_countered"]))
            if plan["gaps_no_asset"]:
                st.markdown("- 无匹配装备：" + "、".join(plan["gaps_no_asset"]))

    st.divider()
    md = wg.render(res)
    st.download_button("📥 下载对抗推演报告(Markdown)", md, file_name="EOB_对抗推演.md")
    with st.expander("查看完整 Markdown 报告"):
        st.markdown(md)
    st.caption("⚠ doctrine 级启发式（非实测），装备为开源参考，不构成作战计划依据。")


# ================= PAGE: 情报报告 =================
def page_reports(agent, names):
    st.title("📋 情报报告")
    st.caption("可溯源情报产品：装备档案（每条事实带来源+置信，低置信标 ⚠）/ 两型号对比。")
    tab1, tab2 = st.tabs(["📋 装备档案", "⚖️ 对比报告"])
    with tab1:
        e = st.selectbox("型号", options=names, key="dossier_sel",
                         index=names.index("AN/TPY-2") if "AN/TPY-2" in names else 0)
        ec = st.text_input("或手填型号", key="dossier_custom", placeholder="AN/APG-66")
        if st.button("▶ 生成装备档案", type="primary"):
            d = agent.reporter.dossier((ec.strip() or e))
            if not d.get("found"):
                st.warning(d.get("note", "未收录该实体。"))
            else:
                if d.get("summary"):
                    st.info("概述：" + d["summary"])
                st.markdown(d["markdown"])
                st.download_button("📥 下载档案", d["markdown"], file_name=f"档案_{d['entity']}.md".replace("/", "_"))
    with tab2:
        c1, c2 = st.columns(2)
        a = c1.selectbox("型号 A", options=names, key="cmp_a",
                         index=names.index("AN/TPY-2") if "AN/TPY-2" in names else 0)
        b = c2.selectbox("型号 B", options=names, key="cmp_b",
                         index=names.index("AN/MPQ-65") if "AN/MPQ-65" in names else 1)
        if st.button("▶ 生成对比", type="primary"):
            c = agent.reporter.compare(a, b)
            st.markdown(c["markdown"])
            st.download_button("📥 下载对比", c["markdown"], file_name="对比报告.md")


# ================= PAGE: 联网检索 =================
def page_web(agent):
    st.title("🌐 联网检索")
    st.caption("图谱未收录的型号 → 联网(Wikipedia)抽取 + 实体链接，结果标注「未经图谱核验」。")
    q = st.text_input("型号 / 查询", placeholder="AN/SPY-1")
    if st.button("▶ 联网检索", type="primary") and q.strip():
        with st.spinner("联网检索中…"):
            r = agent.call_tool("web_search", {"query": q.strip()})
        st.markdown(r.get("observation", ""))


# ---------------- nav ----------------
PAGES = [("chat", "💬", "智能问答"), ("counter", "🎯", "对抗建议"),
         ("wargame", "🗺️", "对抗推演"), ("reports", "📋", "情报报告"),
         ("web", "🌐", "联网检索")]

if "page" not in st.session_state:
    st.session_state.page = "chat"

with st.sidebar:
    st.markdown("## 📡 雷达情报 Agent")
    st.caption("图谱问答 + 对抗决策辅助")
    st.divider()
    for key, emoji, label in PAGES:
        if st.button(f"{emoji}　{label}", use_container_width=True,
                     type="primary" if st.session_state.page == key else "secondary"):
            st.session_state.page = key
            st.rerun()
    st.divider()
    st.caption("双层架构：外层 ReAct · 内层算子组合\n\n对抗：威胁画像→doctrine推理→装备匹配\n\n"
               "关键结论由确定性 trace 生成，LLM 不进信任路径；效能为 doctrine 启发式（非实测）。")

agent, adv, wg, names = get_core()
adv._resolver = agent.reporter.kg.resolve   # let counter page resolve aliases

page = st.session_state.page
if page == "chat":
    page_chat(agent)
elif page == "counter":
    page_counter(adv, names)
elif page == "wargame":
    page_wargame(wg, names)
elif page == "reports":
    page_reports(agent, names)
elif page == "web":
    page_web(agent)
