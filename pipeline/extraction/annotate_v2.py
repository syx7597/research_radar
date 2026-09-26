"""
半自动化标注流水线 v2
功能：
  1. Stage 1 — 自动接受高置信度、高精度关系的三元组（节省~60%人工时间）
  2. Stage 2 — 交叉验证 deployedOn（查询平台的 Wikipedia infobox 确认）
  3. Stage 3 — 按优先级排序人工审核队列（低精度关系优先）
  4. Cohen's Kappa 计算（用于论文报告标注一致性）

运行方式：
    # 全自动处理（生成待人工审核的优先队列）
    python annotate_v2.py --auto

    # 交互式人工标注（从优先队列开始）
    python annotate_v2.py --interactive

    # 计算两个标注文件间的 Cohen's Kappa
    python annotate_v2.py --kappa file1.json file2.json
"""

import json
import re
import time
import argparse
import logging
from pathlib import Path
from typing import Optional
from copy import deepcopy

import requests

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ═══════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════

ANNOTATION_TEMPLATE_PATH = Path("extraction_results/annotation_template.json")
LABELED_OUTPUT_PATH      = Path("extraction_results/annotation_v2_labeled.json")
PRIORITY_QUEUE_PATH      = Path("extraction_results/annotation_priority_queue.json")

PROXIES = {
    "http":  "http://127.0.0.1:7897",
    "https": "http://127.0.0.1:7897",
}

# Stage 1：自动接受条件
AUTO_ACCEPT_RELATIONS   = {"operatedBy", "developedBy", "affiliatedTo"}
AUTO_ACCEPT_MIN_CONF    = 0.95

# Stage 3：人工审核排优先级
REVIEW_PRIORITY_ORDER = [
    "derivedFrom",    # 历史精度 8%，最优先
    "upgradeOf",      # 历史精度 55.6%
    "competitorOf",   # 历史精度 0%
    "deployedOn",     # 77.4%
    "hasFrequencyBand",
    "exportedTo",
    "developedBy",
    "operatedBy",
    "affiliatedTo",
    "hasFunction",
    "coDeployedWith",
]

# 已知合法的正则值集合（用于自动验证）
VALID_FREQUENCY_BANDS = {"HF","VHF","UHF","L","S","C","X","Ku","Ka","W"}
VALID_COUNTRIES_ZH = {
    "美国","中国","俄罗斯","英国","法国","德国","意大利",
    "以色列","日本","荷兰","瑞典","印度","韩国","澳大利亚",
}


# ═══════════════════════════════════════════════════════
#  Stage 1：自动接受高精度三元组
# ═══════════════════════════════════════════════════════

def _is_canonical_tail(relation: str, tail: str) -> bool:
    """验证 tail 是否为已知合法值。"""
    if relation in ("operatedBy", "exportedTo"):
        return tail in VALID_COUNTRIES_ZH
    if relation == "hasFrequencyBand":
        return tail in VALID_FREQUENCY_BANDS
    if relation in ("developedBy", "affiliatedTo"):
        # 制造商名称：非空且不是纯中文（制造商名通常含英文）
        return bool(tail) and len(tail) > 2
    return bool(tail)


def auto_accept_stage(template: list[dict]) -> tuple[list[dict], int]:
    """
    对每个雷达的三元组执行自动接受规则。
    返回更新后的模板和自动接受的三元组总数。
    """
    total_accepted = 0
    updated = deepcopy(template)

    for radar in updated:
        for t in radar.get("llm_fewshot_triples_to_annotate", []):
            if t.get("label"):   # 已有标注则跳过
                continue
            relation = t.get("relation", "")
            conf     = t.get("confidence", 0.0)
            tail     = t.get("tail", "")

            if (
                relation in AUTO_ACCEPT_RELATIONS and
                conf >= AUTO_ACCEPT_MIN_CONF and
                _is_canonical_tail(relation, tail)
            ):
                t["label"]        = "correct"
                t["annotated_by"] = "auto_stage1"
                t["annotated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                total_accepted += 1

    return updated, total_accepted


# ═══════════════════════════════════════════════════════
#  Stage 2：交叉验证 deployedOn（查 Wikipedia infobox）
# ═══════════════════════════════════════════════════════

def _fetch_wikipedia_text(title: str, max_chars: int = 3000) -> str:
    """获取 Wikipedia 文章的纯文本（用于验证 deployedOn）。"""
    params = {
        "action": "query", "titles": title,
        "prop": "extracts", "exintro": True,
        "explaintext": True, "format": "json",
    }
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params=params, timeout=15,
            proxies=PROXIES,
            headers={"User-Agent": "RadarKG-annotator/2.0"},
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            return (page.get("extract", "") or "")[:max_chars]
    except Exception as e:
        log.debug(f"Wikipedia 查询失败 ({title}): {e}")
    return ""


def cross_validate_deployed_on(template: list[dict],
                                sleep_secs: float = 1.5) -> tuple[list[dict], dict]:
    """
    对 deployedOn 三元组进行维基百科交叉验证。
    若平台页面中包含雷达名称，则自动接受；若明确不含，则降低置信度。
    返回更新后的模板和验证统计。
    """
    updated = deepcopy(template)
    stats   = {"validated": 0, "rejected": 0, "unknown": 0}

    for radar in updated:
        radar_name = radar.get("en_title", "")
        for t in radar.get("llm_fewshot_triples_to_annotate", []):
            if t.get("label"):
                continue
            if t.get("relation") != "deployedOn":
                continue

            platform = t.get("tail", "")
            if not platform:
                continue

            time.sleep(sleep_secs)
            platform_text = _fetch_wikipedia_text(platform).lower()
            radar_lower   = radar_name.lower()

            if not platform_text:
                stats["unknown"] += 1
                continue

            if radar_lower in platform_text:
                t["label"]        = "correct"
                t["annotated_by"] = "auto_stage2_wiki"
                t["annotated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                stats["validated"] += 1
            else:
                # 未找到，降低置信度并标记为待人工审核
                t["confidence"] = min(t.get("confidence", 0.5) * 0.7, 0.5)
                t["comment"]    = f"Wiki cross-check: '{radar_name}' NOT found in {platform} article"
                stats["rejected"] += 1

    log.info(f"Stage 2 验证: {stats}")
    return updated, stats


# ═══════════════════════════════════════════════════════
#  Stage 3：生成人工审核优先队列
# ═══════════════════════════════════════════════════════

def build_priority_queue(template: list[dict]) -> list[dict]:
    """
    提取所有未标注三元组，按优先级排序后返回人工审核队列。
    优先级：低精度关系 > 低置信度 > 短 evidence > 其他
    """
    queue: list[dict] = []

    rel_priority = {rel: i for i, rel in enumerate(REVIEW_PRIORITY_ORDER)}

    for radar in template:
        radar_id   = radar.get("radar_id", "")
        radar_name = radar.get("en_title", "")
        for t in radar.get("llm_fewshot_triples_to_annotate", []):
            if t.get("label"):   # 已标注的跳过
                continue
            relation = t.get("relation", "")
            conf     = t.get("confidence", 0.5)
            evidence = t.get("evidence", "")

            priority_score = (
                rel_priority.get(relation, 99) * 10 +
                (1 - conf) * 5 +
                (1 if len(evidence) < 30 else 0) * 3
            )
            queue.append({
                "radar_id":      radar_id,
                "radar_name":    radar_name,
                "triple":        t,
                "priority_score": round(priority_score, 2),
            })

    queue.sort(key=lambda x: x["priority_score"])
    log.info(f"人工审核队列: {len(queue)} 条待标注三元组")
    return queue


# ═══════════════════════════════════════════════════════
#  交互式标注器
# ═══════════════════════════════════════════════════════

def interactive_annotate(template: list[dict],
                          max_items: int = 200) -> list[dict]:
    """
    交互式人工标注。从优先队列开始，依次显示三元组，等待用户输入。
    标签选项：c=correct, p=partial, w=wrong, s=skip, q=quit
    """
    updated  = deepcopy(template)
    queue    = build_priority_queue(updated)[:max_items]

    # 建立索引：(radar_id, head, relation, tail) → 三元组引用
    triple_index: dict[tuple, dict] = {}
    for radar in updated:
        for t in radar.get("llm_fewshot_triples_to_annotate", []):
            key = (radar.get("radar_id",""), t.get("head",""),
                   t.get("relation",""), t.get("tail",""))
            triple_index[key] = t

    annotated = 0
    print(f"\n{'='*60}")
    print(f"  交互式标注  ({len(queue)} 条，输入 q 退出)")
    print(f"  c=correct  p=partial  w=wrong  s=skip  q=quit")
    print(f"{'='*60}\n")

    for item in queue:
        t    = item["triple"]
        key  = (item["radar_id"], t.get("head",""), t.get("relation",""), t.get("tail",""))
        real = triple_index.get(key, t)

        print(f"[{annotated+1}/{len(queue)}] 雷达: {item['radar_name']}")
        print(f"  三元组: {t.get('head','')} --[{t.get('relation','')}]--> {t.get('tail','')}")
        print(f"  置信度: {t.get('confidence', 0):.2f}")
        print(f"  证据:   {t.get('evidence','')}")
        print(f"  优先级分: {item['priority_score']}")

        while True:
            choice = input("  标注 [c/p/w/s/q]: ").strip().lower()
            if choice in ("c", "p", "w", "s", "q"):
                break
            print("  请输入 c/p/w/s/q")

        if choice == "q":
            print("已退出标注。")
            break
        if choice == "s":
            continue

        label_map = {"c": "correct", "p": "partial", "w": "wrong"}
        real["label"]        = label_map[choice]
        real["annotated_by"] = "human"
        real["annotated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        annotated += 1
        print(f"  → 已标注为: {real['label']}\n")

    print(f"\n本次标注: {annotated} 条")
    return updated


# ═══════════════════════════════════════════════════════
#  Cohen's Kappa 计算
# ═══════════════════════════════════════════════════════

def compute_kappa(file1: str, file2: str) -> float:
    """
    计算两个标注文件间的 Cohen's Kappa（衡量标注一致性）。
    仅对两者都已标注的三元组计算。
    标签空间：correct / partial / wrong
    """
    def load_labels(path: str) -> dict[str, str]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        labels: dict[str, str] = {}
        for radar in data:
            for t in radar.get("llm_fewshot_triples_to_annotate", []):
                if not t.get("label"):
                    continue
                key = f"{radar.get('radar_id','')}|{t.get('head','')}|{t.get('relation','')}|{t.get('tail','')}"
                labels[key] = t["label"]
        return labels

    lab1 = load_labels(file1)
    lab2 = load_labels(file2)
    common_keys = set(lab1) & set(lab2)

    if not common_keys:
        print("两个文件没有共同标注的三元组。")
        return 0.0

    LABELS = ["correct", "partial", "wrong"]
    n = len(common_keys)

    # 统计
    agree = sum(1 for k in common_keys if lab1[k] == lab2[k])
    po = agree / n  # 实际一致率

    # 期望一致率
    from collections import Counter
    c1 = Counter(lab1[k] for k in common_keys)
    c2 = Counter(lab2[k] for k in common_keys)
    pe = sum((c1[l] / n) * (c2[l] / n) for l in LABELS)

    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    print(f"\nCohen's Kappa 计算结果:")
    print(f"  共同标注三元组: {n} 条")
    print(f"  实际一致率 Po: {po:.3f}")
    print(f"  期望一致率 Pe: {pe:.3f}")
    print(f"  Cohen's Kappa:  {kappa:.3f}")
    if kappa >= 0.80:
        print("  解读: 极强一致性 (Almost perfect)")
    elif kappa >= 0.60:
        print("  解读: 强一致性 (Substantial)")
    elif kappa >= 0.40:
        print("  解读: 中等一致性 (Moderate)")
    else:
        print("  解读: 一致性较低，建议重新讨论标注规范")
    return kappa


# ═══════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════

def run_auto_pipeline(save_path: Path = LABELED_OUTPUT_PATH) -> list[dict]:
    """运行完整自动化流水线（Stage 1 + Stage 2），生成优先队列。"""
    if not ANNOTATION_TEMPLATE_PATH.exists():
        print(f"找不到标注模板: {ANNOTATION_TEMPLATE_PATH}")
        print("请先运行 triple_extraction_v2.py 生成模板")
        raise SystemExit(1)

    with open(ANNOTATION_TEMPLATE_PATH, encoding="utf-8") as f:
        template = json.load(f)

    total = sum(
        len(r.get("llm_fewshot_triples_to_annotate", []))
        for r in template
    )
    print(f"原始三元组总数: {total} 条（来自 {len(template)} 个雷达）")

    # Stage 1
    print("\n[Stage 1] 自动接受高置信度三元组...")
    template, accepted1 = auto_accept_stage(template)
    print(f"  自动接受: {accepted1} 条")

    # Stage 2
    print("\n[Stage 2] 交叉验证 deployedOn（需要网络）...")
    do_wiki = input("  是否执行 Wikipedia 交叉验证？[y/N]: ").strip().lower() == "y"
    if do_wiki:
        template, stats2 = cross_validate_deployed_on(template)
        print(f"  验证通过: {stats2['validated']} | 降权: {stats2['rejected']} | 未知: {stats2['unknown']}")

    # Stage 3：生成优先队列
    print("\n[Stage 3] 生成人工审核优先队列...")
    queue = build_priority_queue(template)
    with open(PRIORITY_QUEUE_PATH, "w", encoding="utf-8") as f:
        json.dump(queue, f, ensure_ascii=False, indent=2)
    print(f"  待人工审核: {len(queue)} 条 → {PRIORITY_QUEUE_PATH}")

    # 保存中间结果
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    print(f"\n已保存中间结果: {save_path}")

    # 统计
    labeled = sum(
        1 for r in template
        for t in r.get("llm_fewshot_triples_to_annotate", [])
        if t.get("label")
    )
    print(f"当前标注进度: {labeled}/{total} ({labeled/max(total,1):.1%})")
    return template


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="半自动化标注流水线 v2")
    parser.add_argument("--auto",        action="store_true", help="运行自动化流水线")
    parser.add_argument("--interactive", action="store_true", help="交互式人工标注")
    parser.add_argument("--kappa",       nargs=2,             help="计算两文件 Cohen's Kappa")
    args = parser.parse_args()

    if args.kappa:
        compute_kappa(args.kappa[0], args.kappa[1])
    elif args.interactive:
        if LABELED_OUTPUT_PATH.exists():
            with open(LABELED_OUTPUT_PATH, encoding="utf-8") as f:
                template = json.load(f)
        else:
            template = run_auto_pipeline()
        updated = interactive_annotate(template)
        with open(LABELED_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(updated, f, ensure_ascii=False, indent=2)
        print(f"标注结果已保存: {LABELED_OUTPUT_PATH}")
    else:
        run_auto_pipeline()
