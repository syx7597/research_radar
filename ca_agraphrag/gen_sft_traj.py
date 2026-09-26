# -*- coding: utf-8 -*-
"""SFT 冷启动轨迹生成 —— 为训练集每题构造金标工具序列(think→tool→observe→finish)。
finish 用工具实际算出的结果(已 oracle 验证 == 金标),轨迹自洽可执行。
每条轨迹回放进环境确认 reward=1,才写盘。输出 ca_agraphrag/data/sft_train.jsonl。

  python ca_agraphrag/gen_sft_traj.py
"""
import json
from pathlib import Path
from ca_agraphrag.kg_tools import KGTools
from ca_agraphrag.env import KGQAEnv, score_answer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ca_agraphrag" / "data"
tools = KGTools(filtered=True)
env = KGQAEnv(tools)

SYS = (
    "你是雷达知识图谱问答 agent。通过多轮调用工具回答问题,每步只输出一个 JSON:"
    '{"thought":"简短推理","tool":"工具名","args":{...}}, 得到答案后 '
    '{"thought":"...","tool":"finish","args":{"answer":<答案>}}。\n'
    "工具:lookup(entity,relation) 取尾实体集;find(relation,value,type) 取满足的头实体;"
    "intersect(sets)/union(sets)/difference(a,b) 集合运算;count(items) 计数;finish(answer) 结束。"
)


def build_actions(item):
    """返回 [(thought, action)...] 以 finish 结尾;action 与 env 动作一致。用工具实际取值。"""
    t, s = item["type"], item["gold_support"]
    A = []
    if t == "single_hop":
        h, r = s[0]["head"], s[0]["rel"]
        res = tools.lookup(h, r)
        A.append((f"查 {h} 的 {r}", {"tool": "lookup", "args": {"entity": h, "relation": r}}))
        A.append(("得到结果,作答", {"tool": "finish", "args": {"answer": res}}))
    elif t == "two_hop":
        h, r1, mid, r2 = s[0]["head"], s[0]["rel"], s[0]["tail"], s[1]["rel"]
        A.append((f"先查 {h} 的 {r1} 得到源型号", {"tool": "lookup", "args": {"entity": h, "relation": r1}}))
        res = tools.lookup(mid, r2)
        A.append((f"再查源型号 {mid} 的 {r2}", {"tool": "lookup", "args": {"entity": mid, "relation": r2}}))
        A.append(("得到答案", {"tool": "finish", "args": {"answer": res}}))
    elif t in ("count", "enumerate"):
        r, v = s[0]["rel"], s[0]["tail"]
        members = tools.find(r, v, "Radar")
        A.append((f"找出所有 {r}={v} 的雷达", {"tool": "find", "args": {"relation": r, "value": v, "type": "Radar"}}))
        if t == "count":
            A.append(("对结果计数", {"tool": "count", "args": {"items": members}}))
            A.append(("给出数量", {"tool": "finish", "args": {"answer": len(members)}}))
        else:
            A.append(("列出全部", {"tool": "finish", "args": {"answer": members}}))
    elif t == "multi_constraint":
        c1, c2 = s[0]["c1"], s[0]["c2"]
        s1 = tools.find(c1[0], c1[1], "Radar")
        s2 = tools.find(c2[0], c2[1], "Radar")
        inter = tools.intersect(s1, s2)
        A.append((f"找满足约束1 {c1[0]}={c1[1]} 的雷达", {"tool": "find", "args": {"relation": c1[0], "value": c1[1], "type": "Radar"}}))
        A.append((f"找满足约束2 {c2[0]}={c2[1]} 的雷达", {"tool": "find", "args": {"relation": c2[0], "value": c2[1], "type": "Radar"}}))
        A.append(("两约束求交", {"tool": "intersect", "args": {"sets": [s1, s2]}}))
        A.append(("给出交集", {"tool": "finish", "args": {"answer": inter}}))
    elif t == "negation":
        h, r, cand = s[0]["head"], s[0]["rel"], s[0]["candidate"]
        real = tools.lookup(h, r)
        A.append((f"查 {h} 的 {r} 看是否含 {cand}", {"tool": "lookup", "args": {"entity": h, "relation": r}}))
        A.append(("判断是否命中", {"tool": "finish", "args": {"answer": cand in real}}))
    elif t == "comparison":
        dim = item["gold_answer"].get("dim")
        a, b = s[0]["a"], s[1]["b"]
        va, vb = tools.lookup(a, dim), tools.lookup(b, dim)
        A.append((f"查 {a} 的 {dim}", {"tool": "lookup", "args": {"entity": a, "relation": dim}}))
        A.append((f"查 {b} 的 {dim}", {"tool": "lookup", "args": {"entity": b, "relation": dim}}))
        same = set(va) == set(vb) and bool(va)
        A.append(("比较两者是否相同", {"tool": "finish", "args": {"answer": {"same": same}}}))
    return A


def to_messages(item, actions):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": item["question"]}]
    for thought, act in actions:
        msgs.append({"role": "assistant",
                     "content": json.dumps({"thought": thought, **act}, ensure_ascii=False)})
        if act["tool"] != "finish":
            result = tools.call(act["tool"], act["args"])
            msgs.append({"role": "tool", "content": json.dumps({"result": result}, ensure_ascii=False)})
    return msgs


def replay_ok(item, actions):
    """回放进环境,确认 finish 拿 reward correctness=1。"""
    env.reset(item)
    r = done = info = None
    for _, act in actions:
        obs, r, done, info = env.step(act)
        if done:
            break
    return info and info.get("correctness") == 1.0


def main():
    items = [json.loads(l) for l in (DATA / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out, bad = [], 0
    from collections import Counter
    steps_by_type = Counter()
    for it in items:
        acts = build_actions(it)
        if not acts or not replay_ok(it, acts):
            bad += 1
            continue
        steps_by_type[it["type"]] += len(acts)
        out.append({"qid": it["qid"], "type": it["type"],
                    "messages": to_messages(it, acts), "n_turns": len(acts)})
    (DATA / "sft_train.jsonl").write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in out), encoding="utf-8")
    print(f"SFT 轨迹 {len(out)}/{len(items)} 条(回放不达标丢弃 {bad}) -> sft_train.jsonl")
    tc = Counter(o["type"] for o in out)
    print("各型:", dict(tc))
    avg = sum(o["n_turns"] for o in out) / max(len(out), 1)
    print(f"平均轮数 {avg:.1f}")
    print("\n=== 样例轨迹(multi_constraint) ===")
    ex = next(o for o in out if o["type"] == "multi_constraint")
    for m in ex["messages"]:
        print(f"  [{m['role']}] {m['content'][:100]}")


if __name__ == "__main__":
    main()
