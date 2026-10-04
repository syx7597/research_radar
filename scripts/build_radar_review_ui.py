#!/usr/bin/env python3
"""Build a local, answer-free review worksheet for the existing 12 candidates.

No review is completed by this script. Only a human's browser action can export
a separate draft CSV; it never overwrites the repository's review decisions.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_radar_review_workflow as workflow

DEST = ROOT / "artifacts/thesis_direction_review/radar_review_ui_v1"
SCREEN = ROOT / "artifacts/thesis_direction_review/radar_primary_screening_001_003/ai_screening.json"
CAPTURE = SCREEN.with_name("capture_manifest_v1.json")
LABELS = {"subject_check": "主体归属", "variant_check": "型号版本", "event_check": "事件类型",
          "value_check": "数值／文本读法", "unit_check": "单位", "condition_check": "条件与范围"}


def escaped(value):
    return html.escape(str(value), quote=True)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build():
    audit = workflow.audit(workflow.DEFAULT_WORKFLOW)
    facts, _ = workflow.load_packet()
    decisions = workflow.read_csv(workflow.DEFAULT_WORKFLOW / "review_decisions.csv", workflow.DECISION_HEADERS)
    if any(row["decision"] != "pending" for row in decisions):
        raise ValueError("This first-review worksheet cannot replace existing decisions")
    screening = json.loads(SCREEN.read_text())
    supplements = {row["fact_id"]: row for row in screening["records"]}
    sources = {row["source_id"]: row for row in screening["sources"]}
    captures = {row["source_id"]: row for row in json.loads(CAPTURE.read_text())["sources"] if row["capture_status"] == "captured"} if CAPTURE.exists() else {}
    cards = []
    for fact in facts:
        fid = fact["fact_id"]
        values = {k: fact[k] for k in ("variant", "event_type", "value_kind", "value_status", "value", "min_value", "max_value", "unit_std", "condition_status", "condition_raw") if fact[k]}
        loc = json.loads(fact["locator"])
        controls = "".join(
            f'<label>{escaped(label)}<select data-field="{field}"><option value="">未填写</option>'
            '<option value="confirmed">与本快照一致</option><option value="not_applicable">不适用</option>'
            '<option value="not_specified_in_source">本来源未明确</option></select></label>'
            for field, label in LABELS.items())
        supplemental = ""
        if fid in supplements:
            screen = supplements[fid]
            links = []
            for sid in screen["candidate_source_ids"]:
                source = sources[sid]
                local = ""
                if sid in captures:
                    caption = "本地全文定位" if captures[sid]["quote_binding_verified"] else "本地响应（未找到原引文，不作证据）"
                    local = f' · <a href="../../../{escaped(captures[sid]["text"]["file"])}" target="_blank" rel="noopener">{caption}</a>'
                links.append(f'<li><a href="{escaped(source["url"])}" target="_blank" rel="noopener">{escaped(source["title"])}</a>{local}<br>{escaped(source["does_not_establish"])}</li>')
            supplemental = f'<details><summary>补充机构来源与 AI 预审（仍待你核查）</summary><p>{escaped(screen["reason_zh"])}</p><ul>{"".join(links)}</ul></details>'
        cards.append(f'''<article data-fact="{fid}">
<h2>{fid[-2:]} · {escaped(fact['entity_name'])} · {escaped(fact['attribute'])}</h2>
<p><b>旧属性原值：</b>{escaped(fact['value_raw'])}</p>
<p><b>来源快照短摘录：</b><q>{escaped(fact['evidence_text'])}</q></p>
<p><a href="{escaped(fact['source_uri'])}" target="_blank" rel="noopener">打开网页</a> · <a href="../../../{escaped(fact['source_file'])}" target="_blank" rel="noopener">打开本地原始快照</a> · 定位 <code>{escaped(loc['json_pointer'])}</code> [{loc['char_start']}, {loc['char_end']})</p>
<details><summary>AI 候选结构与待核查说明</summary><pre>{escaped(json.dumps(values, ensure_ascii=False, indent=2))}</pre><p>{escaped(fact['review_notes'])}</p><p class="small">来源 SHA-256：{escaped(fact['source_sha256'])}</p></details>
{supplemental}
<div class="checks">{controls}</div>
<label>本条决定<select data-field="decision"><option value="pending">待审（默认）</option><option value="accepted">确认本快照的读法</option><option value="needs_changes">需要修订候选解释</option><option value="rejected">不接受候选解释</option></select></label>
<label>理由／需修订内容（完成审阅必填）<textarea data-field="reason" rows="3" placeholder="例如：原文描述整套系统，不能赋给雷达型号；或者该区间的两个端点与条件均与本快照一致。"></textarea></label>
<label class="ack"><input type="checkbox" data-field="ack">我已对照原文完成本条审阅；这里只确认资料读法，不证明现实参数真实。</label>
</article>''')
    payload = json.dumps({"headers": workflow.DECISION_HEADERS, "rows": decisions}, ensure_ascii=False).replace("<", "\\u003c")
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>首批雷达事实审阅 · 12 条</title><style>
body{font-family:system-ui,sans-serif;max-width:980px;margin:32px auto;padding:0 18px;color:#172b3a;background:#f6f8fa;line-height:1.6}h1{font-size:26px}h2{font-size:19px}article,.intro{background:white;border:1px solid #d4dde4;border-radius:10px;padding:20px;margin:18px 0}label{display:block;margin:10px 0}select,textarea,input[type=text]{font:inherit;padding:8px;border:1px solid #9caebc;border-radius:4px;box-sizing:border-box;width:100%}.checks{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.small{font-size:12px;overflow-wrap:anywhere}pre{white-space:pre-wrap}q{background:#f1f5f9}a{color:#125a95}button{padding:12px 20px;font:inherit;background:#125a95;color:white;border:0;border-radius:6px;cursor:pointer}.ack{font-weight:600}.status{color:#8c2b12}footer{margin:25px 0 60px}@media(max-width:600px){.checks{grid-template-columns:1fr}}@media print{button,footer{display:none}article{break-inside:avoid}body{background:white}}
</style><h1>首批雷达事实审阅 · 12 条</h1>
<section class="intro"><p><b>第一轮任务：核对现有来源快照的读法。</b>由你先审阅，后续再请同学交叉复核。所有决定初始均为待审，没有预填“确认”。可以分批完成。</p>
<ol><li>打开原文，先辨认主体是系统、雷达、平台还是部件，再检查事件、值、单位、条件。</li><li>“本来源未明确”不是“现实中不存在”。选择确认只表示读法相符；不确定或归属不对时选择需要修订并说明。</li><li>填写真实署名和理由，勾选本条已核查，再导出 CSV。导出不会上传，也不会修改仓库。网页关闭会丢失未导出的填写。</li></ol>
<p>网页与历史快照可能不同，请优先核对本地快照。全文仅保存在本工作区，GitHub 上的本地全文链接不会提供下载。已归档的机构来源仅辅助 01–03 的主体／事件核查，不能据此填替代年份。</p>
<p><b>本轮导出不属于独立事实验收，也不是独立问答金标。</b>后续仍需一手来源与必要事实修订。12 道既有开发题永久不进入最终测试。</p>
<label>你的真实署名<input id="reviewer" type="text" autocomplete="name" placeholder="请自行填写，AI 不代填"></label></section>
''' + "\n".join(cards) + '''<footer><button id="export">导出已填写的审阅 CSV</button><p id="status" class="status" role="status"></p></footer>
<script id="review-data" type="application/json">''' + payload + r'''</script>
<script>
const payload=JSON.parse(document.getElementById('review-data').textContent);
document.getElementById('export').addEventListener('click',()=>{
  const name=document.getElementById('reviewer').value.trim();
  const rows=payload.rows.map(x=>({...x})); let completed=0;
  try {
    for(const row of rows){
      const card=document.querySelector('article[data-fact="'+row.fact_id+'"]');
      const value=key=>card.querySelector('[data-field="'+key+'"]').value.trim();
      const decision=value('decision');
      if(decision==='pending') continue;
      if(!name || /(?:^|[^a-z0-9])(?:ai|gpt\d*|chatgpt|codex|assistant|model)(?:$|[^a-z0-9])/i.test(name)) throw Error('请填写你本人的真实署名。');
      if(!value('reason') || !card.querySelector('[data-field="ack"]').checked) throw Error(row.fact_id+'：请填写理由，并确认已对照原文核查。');
      const checks=['subject_check','variant_check','event_check','value_check','unit_check','condition_check'];
      if(decision==='accepted' && checks.some(key=>!value(key))) throw Error(row.fact_id+'：确认读法需要填写全部六项检查。');
      Object.assign(row,{decision,decision_scope:'source_snapshot_reading',reviewer_kind:'human',reviewer:name,reviewed_at:new Date().toISOString(),reason:value('reason')});
      checks.forEach(key=>row[key]=value(key)); completed++;
    }
    if(!completed) throw Error('尚未完成任何一条审阅；没有导出或修改文件。');
    const quote=x=>'"'+String(x??'').replace(/"/g,'""')+'"';
    const csv=[payload.headers.map(quote).join(','),...rows.map(row=>payload.headers.map(h=>quote(row[h])).join(','))].join('\n')+'\n';
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
    const a=document.createElement('a');a.href=url;a.download='radar_review_decisions_draft.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    document.getElementById('status').textContent='已导出 '+completed+' 条审阅，其余保留待审。文件尚未导入仓库，独立事实验收仍未完成。';
  } catch(e){document.getElementById('status').textContent=e.message;}
});
</script></html>'''
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "index.html").write_text(page)
    inputs = [workflow.PACKET / "facts.csv", workflow.DEFAULT_WORKFLOW / "review_decisions.csv", SCREEN, Path(__file__)]
    if CAPTURE.exists():
        inputs.append(CAPTURE)
    manifest = {"version": "radar_review_ui_v1", "facts": len(facts), "human_review_status": "pending",
                "human_review_arrangement": "user first pass; peer cross-review planned",
                "accepted_independent_facts": 0, "export_scope": "source_snapshot_reading",
                "changes_repository_on_export": False, "all_decisions_initially_pending": True,
                "inputs_sha256": {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in inputs},
                "html_sha256": digest((DEST / "index.html").read_bytes()),
                "source_bindings_verified": audit["exact_source_bindings"]}
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"html": str((DEST / "index.html").relative_to(ROOT)), "cards": len(cards), "decisions_completed": 0}, ensure_ascii=False))


if __name__ == "__main__":
    build()
