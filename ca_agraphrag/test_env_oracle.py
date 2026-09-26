# -*- coding: utf-8 -*-
"""Oracle 回放:用 KG 工具按金标构造正确解,验证环境自洽。
(1) 工具输出能否复现金标成员(工具视图 == 金标视图)
(2) 环境对正确答案是否给 correctness=1
逐题型报命中率。全高 => 环境+工具+奖励可用于 RL。
"""
import json
from collections import defaultdict
from pathlib import Path
from ca_agraphrag.kg_tools import KGTools
from ca_agraphrag.env import KGQAEnv, score_answer

ROOT = Path(__file__).resolve().parent.parent
tools = KGTools(filtered=True)
env = KGQAEnv(tools)


def oracle_answer(item):
    """用工具算出该题的答案(一个正确 agent 会得到的)。"""
    t, s, g = item["type"], item["gold_support"], item["gold_answer"]
    if t == "single_hop":
        return tools.lookup(s[0]["head"], s[0]["rel"])
    if t == "two_hop":
        mids = tools.lookup(s[0]["head"], s[0]["rel"])
        out = set()
        for m in mids:
            out |= set(tools.lookup(m, s[1]["rel"]))
        return sorted(out)
    if t == "count":
        return tools.count(tools.find(s[0]["rel"], s[0]["tail"], "Radar"))
    if t == "enumerate":
        return tools.find(s[0]["rel"], s[0]["tail"], "Radar")
    if t == "multi_constraint":
        c1, c2 = s[0]["c1"], s[0]["c2"]
        return tools.intersect(tools.find(c1[0], c1[1], "Radar"),
                               tools.find(c2[0], c2[1], "Radar"))
    if t == "negation":
        return s[0]["candidate"] in tools.lookup(s[0]["head"], s[0]["rel"])
    if t == "comparison":
        dim = g.get("dim")
        va = tools.lookup(s[0]["a"], dim)
        vb = tools.lookup(s[1]["b"], dim)
        return {"same": set(va) == set(vb) and bool(va)}
    return None


def main():
    items = []
    for name in ("train", "dev", "test"):
        items += [json.loads(l) for l in (ROOT / "ca_agraphrag" / "data" / f"{name}.jsonl")
                  .read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"回放 {len(items)} 题")

    by_type = defaultdict(lambda: [0, 0])   # type -> [correct, total]
    fails = []
    for it in items:
        ans = oracle_answer(it)
        corr = score_answer(ans, it["gold_answer"], it["gold_kind"])
        by_type[it["type"]][0] += (corr == 1.0)
        by_type[it["type"]][1] += 1
        if corr != 1.0 and len(fails) < 12:
            fails.append((it["qid"], it["type"], str(it["gold_answer"])[:40], str(ans)[:40]))

    print("\n=== 各题型 oracle 命中率(工具复现金标 & 环境给满分) ===")
    tot = [0, 0]
    for t, (c, n) in sorted(by_type.items()):
        print(f"  {t:16} {c}/{n} = {c/n:.1%}")
        tot[0] += c; tot[1] += n
    print(f"  {'合计':16} {tot[0]}/{tot[1]} = {tot[0]/tot[1]:.1%}")
    if fails:
        print("\n--- 未满分样例(qid | type | gold | oracle) ---")
        for f in fails:
            print(f"  {f[0]:20} {f[1]:14} gold={f[2]}  oracle={f[3]}")

    # 环境 step 流程冒烟(一条)
    print("\n=== 环境 reset/step 冒烟 ===")
    it = items[0]
    env.reset(it)
    obs, r, done, info = env.step({"tool": "finish", "args": {"answer": oracle_answer(it)}})
    print(f"  {it['type']} | reward={r:.3f} correctness={info.get('correctness')}")


if __name__ == "__main__":
    main()
