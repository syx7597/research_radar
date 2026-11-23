"""Build six editable 4:3 slides for a radar-research exchange.

Reuses the supplied deck's blue/teal visual language and school branding.
All numerical claims are read from existing aggregate results. No model runs.
"""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'雷达技术资料知识组织与可追溯问答_研究交流_5加1页.pptx'
SOURCES={
 'representation':'results/radar_domain/coverage_v2/semantic_summary.json',
 'citations':'results/radar_domain/coverage_v2_citation_sensitivity_v1/semantic_summary.json',
 'freeze':'results/radar_domain/frozen_20261107/summary.json',
 'composition':'results/radar_domain/compositional_supplement_20261108/summary.json',
 'case':'results/radar_domain/programs_dev_20261107/development_case.json',
 'public':'results/agent_feedback/holdout_v1_analysis.json',
 'readings':'artifacts/thesis_direction_review/radar_readings_v1/snapshot_audit.json',
}
data={k:json.loads((ROOT/v).read_text()) for k,v in SOURCES.items()}
stats=data['representation']['adjudicated']['overall']
cit=data['citations']['uniform_resolver']['overall']
pub=data['public']['primary'];case=data['case'];splits=data['freeze']['question_counts']
assert stats['flat']['denominator']==stats['bound']['denominator']==96
assert (stats['flat']['answer_correct'],stats['bound']['answer_correct'])==(73,80)
assert (stats['flat']['binding_error_any'],stats['bound']['binding_error_any'])==(17,11)
assert (cit['flat']['joint_correct'],cit['bound']['joint_correct'])==(55,68)
assert case['kind']=='complete_reference_program_not_model_prediction' and case['call_count']==5
assert data['freeze']['model_runs']==0

def color(v):return RGBColor.from_string(v)
NAVY=color('183657');BLUE=color('2F6491');TEAL=color('4F7D7B');INK=color('24303D')
GRAY=color('657482');LINE=color('D9E2E9');PALE=color('EDF3F8');PALETEAL=color('EDF5F3')
WHITE=color('FFFFFF');GOLD=color('A8792E');PALEGOLD=color('FAF3E8')
FONT='Microsoft YaHei'
prs=Presentation(str(HERE/'assets/source_theme.pptx'));prs.slide_width=Inches(10);prs.slide_height=Inches(7.5)
prs.core_properties.title='雷达技术资料的条件化知识组织与可追溯问答'
prs.core_properties.subject='研究所交流材料：5页主内容＋1页公开方法补充'
prs.core_properties.author='';prs.core_properties.last_modified_by=''
texts=[];notes=[];bounds=[]


def rect(slide,x,y,w,h,fill=WHITE,border=LINE,kind=MSO_SHAPE.RECTANGLE):
    s=slide.shapes.add_shape(kind,Inches(x),Inches(y),Inches(w),Inches(h))
    s.fill.solid();s.fill.fore_color.rgb=fill
    if border is None:s.line.fill.background()
    else:s.line.color.rgb=border;s.line.width=Pt(.8)
    s.shadow.inherit=False
    return s


def text(slide,value,x,y,w,h,size=18,bold=False,fill=INK,align=PP_ALIGN.LEFT,valign=MSO_ANCHOR.TOP):
    s=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=s.text_frame
    tf.clear();tf.word_wrap=True;tf.margin_left=tf.margin_right=Inches(.025)
    tf.margin_top=tf.margin_bottom=Inches(.015);tf.vertical_anchor=valign
    for i,line in enumerate(str(value).split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.alignment=align
        p.space_before=Pt(0);p.space_after=Pt(0);p.line_spacing=1.12
        r=p.add_run();r.text=line;r.font.name=FONT;r.font.size=Pt(size);r.font.bold=bold;r.font.color.rgb=fill
        props=r._r.get_or_add_rPr()
        for tag in ['a:latin','a:ea','a:cs']:
            n=props.find(qn(tag))
            if n is None:n=props.makeelement(qn(tag),{});props.append(n)
            n.set('typeface','Arial' if tag=='a:latin' else FONT)
    texts.append({'slide':len(prs.slides),'text':str(value),'font_pt':size})
    bounds.append((len(prs.slides),x,y,w,h,str(value)[:45]))
    return s


def box(slide,value,x,y,w,h,size=18,fill=PALE,bold=False,ink=INK):
    rect(slide,x,y,w,h,fill)
    text(slide,value,x+.09,y+.07,w-.18,h-.14,size,bold,ink,PP_ALIGN.CENTER,MSO_ANCHOR.MIDDLE)


def arrow(slide,x,y,w=.22,h=.16):
    rect(slide,x,y,w,h,BLUE,None,MSO_SHAPE.RIGHT_ARROW)


def slide(kicker,title,n,footer='华东师范大学 · 雷达资料知识问答研究',status=None):
    s=prs.slides.add_slide(prs.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=WHITE
    rect(s,0,0,.075,7.5,NAVY,None)
    text(s,kicker,.48,.23,6.1,.25,11,True,TEAL)
    s.shapes.add_picture(str(HERE/'assets/ecnu_brand.png'),Inches(7.35),Inches(.18),width=Inches(2.15))
    text(s,title,.48,.68,9.0,.75,25,True,NAVY)
    rect(s,.48,1.43,9.04,.018,LINE,None)
    text(s,footer,.48,7.12,8.4,.19,9.5,False,GRAY)
    text(s,f'{n:02d}',9.10,7.10,.42,.23,10,True,GRAY,PP_ALIGN.RIGHT)
    if status:box(s,status,7.54,6.65,1.95,.29,11,PALETEAL,True,TEAL)
    return s


def add_notes(s,script,scope,source):
    value='讲述建议\n'+script+'\n\n数据与表述口径\n'+scope+'\n\n来源\n'+source
    s.notes_slide.notes_text_frame.text=value
    notes.append({'slide':len(prs.slides),'script':script,'scope':scope,'source':source})


# 1. A content-first opening that can be inserted into a supervisor's deck.
s=slide('研究交流｜需求场景','雷达技术资料的知识组织与可追溯问答',1)
text(s,'把型号、部件、参数、适用条件与来源连接为可执行问答。',.55,1.72,8.9,.54,19,False,INK)
for x,num,title,body in [
 (.55,'01','参数定位','指定型号与工况下，\n参数的量值、单位\n和适用对象是什么？'),
 (3.62,'02','条件比较','相同属性对应不同\n部件或配置时，\n怎样保持正确归属？'),
 (6.69,'03','组合核查','联合读取多个记录，\n完成筛选与比较，\n返回可核查的出处。')]:
    rect(s,x,2.55,2.77,2.65,PALE)
    text(s,num,x+.16,2.72,.62,.37,15,True,TEAL)
    text(s,title,x+.16,3.20,2.44,.42,22,True,NAVY)
    text(s,body,x+.16,3.87,2.44,1.05,17)
for i,label in enumerate(['公开技术资料','条件化知识','查询与比较','答案＋出处']):
    x=.60+i*2.30;box(s,label,x,5.65,2.00,.60,17,PALETEAL if i==3 else WHITE,True,NAVY)
    if i<3:arrow(s,x+2.05,5.87,.20,.15)
text(s,'已有基础：来源知识整理、条件表示对照与查询执行验证',.62,6.45,8.8,.35,15,True,TEAL)
add_notes(s,'本研究围绕雷达技术资料的查找、比较和核对。问题通常涉及型号、部件及工况，因此我们将参数和它的适用条件、出处共同组织，随后通过可执行操作形成答案。下面介绍已经完成的资料与问答研究。','交流材料的场景为公开技术资料利用。这里展示任务需求与已有研究流程。原开题PPT中的国家/波段统计示例替换为参数与条件查询。','当前六章研究稿第5章；旧PPT第2、5、8页的流程框架。')

# 2. Current bounded resource, replacing the old full-graph inventory.
s=slide('已有基础｜知识资源','让参数与型号、部件和工况保持对应',2)
for x,value,label in [(.55,'8','制造商来源组'),(3.62,'502','来源读法'),(6.69,'96','中文评价问题')]:
    rect(s,x,1.80,2.77,1.13,PALE)
    text(s,value,x+.15,1.92,1.1,.60,33,True,BLUE)
    text(s,label,x+1.20,2.15,1.40,.47,15,False,GRAY)
labels=['公开资料','结构化读法','条件与归属','来源核对','查询视图']
for i,label in enumerate(labels):
    x=.55+i*1.84;box(s,label,x,3.43,1.62,.62,16,PALETEAL if i==4 else WHITE,True,NAVY)
    if i<4:arrow(s,x+1.65,3.66,.16,.14)
rect(s,.55,4.53,8.91,1.04,PALETEAL)
text(s,'主体  ·  部件  ·  属性  ·  量值形式  ·  适用条件  ·  原文出处',.73,4.72,8.54,.61,18,True,NAVY,PP_ALIGN.CENTER)
text(s,'原值、范围、离散选项与配对参数分别保存，支持条件筛选与来源追溯。',.65,5.98,8.75,.63,17)
text(s,'502为来源读法条数；96题采用AI交叉审阅参考。',.65,6.70,8.75,.25,11.5,False,GRAY)
add_notes(s,'目前在八个制造商来源组内整理了502条来源读法，并形成96道中文评价问题。资料经过记录整理、条件关联和来源核对，提供图式访问与类型化查询视图。每条记录同时保存量值含义、适用对象和原文位置。','502指来源读法；多型号可共享读法。这里采用限定资料范围的当前资源规模。旧PPT的10744实体、22241关系及173人工抽样指标不作为本页评价依据。参考及语义评分采用AI交叉审阅。',SOURCES['readings']+'；'+SOURCES['representation'])

# 3. Same facts and same model; representation effect is explicitly named.
s=slide('已完成实验｜条件表示','条件关联组织改善了资料问答效果',3)
text(s,'同一3B模型、同一批语义记录：平铺表示与条件绑定对照',.55,1.72,8.9,.47,17)
rect(s,.55,2.35,4.30,3.03,PALE);rect(s,5.15,2.35,4.30,3.03,PALETEAL)
text(s,'正文正确（共96题）',.78,2.55,3.82,.44,19,True,NAVY)
for y,label,value,c in [(3.23,'平铺',stats['flat']['answer_correct'],GRAY),(4.13,'条件绑定',stats['bound']['answer_correct'],TEAL)]:
    text(s,label,.80,y,1.25,.36,16,False,INK)
    rect(s,2.02,y+.08,2.04*(value/96),.28,c,None)
    text(s,str(value),4.14,y-.02,.46,.43,24,True,c,PP_ALIGN.RIGHT)
text(s,'净增7题  ·  提高7.3个百分点',.82,4.88,3.80,.28,14,True,BLUE)
text(s,'参数错绑（题数）',5.38,2.55,3.82,.44,19,True,NAVY)
text(s,'17',5.49,3.40,1.15,.72,40,True,GRAY,PP_ALIGN.CENTER)
text(s,'→',6.72,3.53,.77,.55,28,False,TEAL,PP_ALIGN.CENTER)
text(s,'11',7.59,3.40,1.15,.72,40,True,TEAL,PP_ALIGN.CENTER)
text(s,'平铺',5.65,4.17,.85,.32,15,False,GRAY,PP_ALIGN.CENTER)
text(s,'条件绑定',7.43,4.17,1.48,.32,15,False,TEAL,PP_ALIGN.CENTER)
text(s,'错绑减少6题',5.42,4.88,3.70,.28,15,True,TEAL,PP_ALIGN.CENTER)
rect(s,.55,5.78,8.90,.73,WHITE,LINE)
text(s,'答案与来源共同正确：55 → 68 / 96',.73,5.91,8.54,.36,18,True,NAVY,PP_ALIGN.CENTER)
text(s,'上行为统一引用解析后的事后对照；评价采用AI交叉审阅。',.65,6.68,8.72,.27,11.5,False,GRAY)
add_notes(s,'这里比较资料的组织方式。两组使用同一批语义记录与相同模型，条件绑定的正文正确从73题提高到80题，参数错绑从17题减少到11题。在统一引用解析的事后对照中，答案与来源共同正确由55题增加到68题。这说明保持条件归属有助于专业资料回答。','分母均为96。73→80、17→11属于资料表示实验；55→68包含同一引用解析后的事后处理，正文评分未变。参考和语义判分为AI交叉审阅。恢复训练的雷达域内A/C实验仍待开展。',SOURCES['representation']+' /adjudicated/overall；'+SOURCES['citations']+' /uniform_resolver/overall')

# 4. A complete, source-backed reference execution example.
s=slide('执行验证｜条件查询案例','从条件检索到可核查的参数答案',4)
text(s,'例：−20℃时，JMA-5200Mk2各部件是否落在资料\n给定的温度范围内？',.57,1.71,8.85,.78,19,True,INK)
xs=[.55,4.45,6.95];ws=[3.90,2.50,2.50]
for x,w,label in zip(xs,ws,['部件','原文温度范围','范围判断']):
    rect(s,x,2.82,w,.50,NAVY,NAVY);text(s,label,x+.13,2.94,w-.26,.29,16,True,WHITE)
rows=[('扫描器','−25 ～ 55℃','范围内',PALETEAL,TEAL),('处理器、显示器与键盘','−15 ～ 55℃','范围外',PALEGOLD,GOLD)]
for i,(component,value,status,fill,ink) in enumerate(rows):
    y=3.32+i*.82
    for x,w,v in zip(xs,ws,[component,value,status]):
        rect(s,x,y,w,.82,fill)
        text(s,v,x+.13,y+.18,w-.26,.48,17,False,ink if x==xs[-1] else INK,PP_ALIGN.LEFT,MSO_ANCHOR.MIDDLE)
for i,label in enumerate(['读取记录','选取范围','判断条件','组装答案','提交']):
    x=.56+i*1.84;box(s,label,x,5.42,1.60,.61,16,WHITE,True,NAVY)
    if i<4:arrow(s,x+1.65,5.65,.16,.13)
text(s,'参考程序经5次调用完成，答案保留部件、原值、条件与出处。',.61,6.30,8.80,.41,15.5,True,TEAL)
text(s,'资料：JRC JMA-5200Mk2产品册，第6页；此页展示参考程序执行验证。',.61,6.79,8.80,.22,10.5,False,GRAY)
add_notes(s,'这个例子来自JRC产品册。资料对扫描器与处理器、显示器、键盘给出了不同的温度范围。查询先取得两类记录，再判断负20摄氏度是否落在各自范围内，最终返回对应部件、温度范围和出处。参考程序通过五次调用完成。','这是已有开发参考程序的真实执行案例。范围判断对应资料给定规格，保持部件区别；本页不作为已训练模型的单题预测展示。',SOURCES['case']+'；https://www.jrc.co.jp/hubfs/jrc-corp/assets/pdf/en/product/JMA-5200Mk2.pdf 第6页')

# 5. Specific bounded follow-up and useful discussion topics.
s=slide('后续验证｜交流议题','围绕真实资料与典型查询开展有限验证',5)
items=[
 ('已有基础','来源记录、类型查询接口\n与可执行参考程序已准备',PALE),
 ('研究验证','普通续训与恢复续训对照\n评价答案、条件、来源与成本',PALETEAL),
 ('拟交流内容','资料字段与工况表达\n典型查询需求及专家核对方式',PALEGOLD),
]
for i,(title,body,fill) in enumerate(items):
    y=1.83+i*1.32;rect(s,.55,y,8.90,1.09,fill)
    text(s,title,.78,y+.25,1.70,.47,20,True,NAVY)
    text(s,body,2.70,y+.17,6.45,.75,17)
text(s,'已冻结任务材料',.63,5.98,2.3,.35,15,True,TEAL)
text(s,f'{splits["train"]} 训练题   /   {splits["dev"]} 开发题   /   24基础题＋12组合题',.63,6.36,8.70,.40,18,True,NAVY)
text(s,'后续进行域内模型对照；最终评价题在评分前固定参考与核查规则。',.63,6.82,8.70,.22,10.5,False,GRAY)
add_notes(s,'当前已有来源隔离的任务、参考程序和训练评价入口。下一步在固定范围内比较普通续训与恢复续训，评价完整答案、条件、来源和成本。与研究所交流时，希望讨论资料字段和工况的表达方式、实际高频查询，以及专家核对可以怎样组织。','576为合成训练题，108为开发题；24基础题与12组合题共享四个新来源家族并分别评价。新域内模型实验待开展。交流议题为建议讨论内容，不表示已建立合作或已获对方数据。',SOURCES['freeze']+'；'+SOURCES['composition']+'；用户2026-11-23修订开题报告。')

# 6. Optional methodological support, unmistakably public benchmark evidence.
s=slide('补充页｜公开基准方法','恢复监督：让小模型接续实际偏离历史',6,footer='补充页 · 公开KQA Pro项目留出评价')
text(s,'保留真实执行历史，映射参考后续步骤，重放核验后用于训练。',.55,1.75,8.90,.58,18)
for i,label in enumerate(['真实失败历史','结果依赖映射','后缀执行核验','配对续训']):
    x=.57+i*2.30;box(s,label,x,2.61,2.00,.73,17,PALE if i!=3 else PALETEAL,True,NAVY)
    if i<3:arrow(s,x+2.05,2.89,.20,.15)
rect(s,.55,3.90,8.90,1.75,PALE)
for x,label,value in [(1.00,'普通续训',pub['A_mean_accuracy']*100),(4.03,'恢复续训',pub['C_mean_accuracy']*100)]:
    text(s,label,x,4.10,2.40,.38,17,True,GRAY,PP_ALIGN.CENTER)
    text(s,f'{value:.2f}%',x,4.63,2.40,.68,34,True,BLUE if x==1 else TEAL,PP_ALIGN.CENTER)
text(s,'+4.75',7.11,4.33,1.95,.63,32,True,TEAL,PP_ALIGN.CENTER)
text(s,'个百分点',7.11,5.00,1.95,.35,16,True,TEAL,PP_ALIGN.CENTER)
text(s,'2,000题项目留出集；两个共享初始模型的续训种子',.65,6.02,8.70,.39,16,True,NAVY)
ci='[3.53, 5.98]'
text(s,f'题级配对95%区间：{ci}个百分点。雷达应用沿用同一构造方法开展后续评价。',.65,6.57,8.70,.42,11.5,False,GRAY)
add_notes(s,'这页作为方法补充，按交流需要使用。初始模型产生实际失败历史，恢复监督通过中间结果映射接回参考步骤，并对后续动作训练。在公开KQA Pro项目留出的2000题上，两个续训种子的平均准确率由78.75%提高到83.50%，形成当前方法研究的主要实证。','公开基准结果；两个续训种子共享初始模型与训练语料。置信区间按题配对、条件于已评价检查点。此项4.75个百分点与雷达条件表示的7题增量分别报告。',SOURCES['public']+' /primary；同方案Qwen2.5-3B-Instruct＋LoRA。')

for page,x,y,w,h,t in bounds:
    assert x>=0 and y>=0 and x+w<=10.001 and y+h<=7.501,(page,t)
assert len(prs.slides)==6
assert all('不是' not in t['text'] and '而是' not in t['text'] for t in texts)
prs.save(OUTPUT)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(HERE/'slide_content.json').write_text(json.dumps({'date':'2026-11-23','slide_size':'4:3','slides':6,'main_slides':5,'optional_public_method_slide':6,'text_items':texts,'speaker_notes':notes},ensure_ascii=False,indent=2)+'\n')
manifest={'date':'2026-11-23','status':'generated_editable_pptx','file':str(OUTPUT.relative_to(ROOT)),'pptx_sha256':sha(OUTPUT),'slides':6,'aspect':'4:3','editable_elements':'Native PowerPoint text, shapes, tables represented by editable text/rectangles, and bars; school branding is an unchanged image','sources_sha256':{v:sha(ROOT/v) for v in SOURCES.values()},'case_type':case['kind'],'new_model_runs':0,'server_accessed':False,'human_evaluation_added':0,'source_branding_sha256':sha(HERE/'assets/ecnu_brand.png'),'builder_sha256':sha(Path(__file__)),'content_sha256':sha(HERE/'slide_content.json'),'legacy_slides_reused_conceptually':[2,5,6,8,9],'legacy_claims_replaced':['RL as the main research route','Old full-graph counts as current evaluated resource size','Unverified human-review quality statistic (n=173)','Old diagnostic gains as current method results','MetaQA and four-arm RL experiment promises'],'geometry_bounds_check':'passed','visual_review':'pending'}
(HERE/'build_record.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'pptx':str(OUTPUT.relative_to(ROOT)),'slides':6,'editable':True},ensure_ascii=False))
