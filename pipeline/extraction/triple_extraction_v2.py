"""
雷达知识图谱 - 三元组抽取实验 v2.0
包含三种方法的对比实验：
  方法A：规则抽取（正则+词典）
  方法B：LLM零样本抽取
  方法C：LLM少样本抽取

【v2.0 主要改动】
  1. RuleBasedExtractor
     - 修复 _facts_to_triples: platform tail_type 改为从 RELATION_TYPES 推断
     - 增加 exportedTo / competitorOf / affiliatedTo 三类规则
     - 增加 _normalize_country(): 将 "United States" 等归一化为 "美国"
     - 增加 _normalize_band(): 统一频段写法
     - _match_sentence 不再每句只取第一个关系匹配，改为全关系扫描
     - _deduplicate 改为保留置信度最高的那条（原来保留最早出现的）

  2. ZeroShotLLMExtractor
     - System prompt 独立为常量，拆出 user prompt
     - 使用 Anthropic system 参数而不是把 prompt 塞进 user（减少 token 浪费）
     - _call_anthropic 支持 system 字段
     - 增加 _post_process(): 归一化 country / band，补全缺失 tail_type
     - _validate 频段校验兼容 "X-Band"/"X_Band"/"X band" 等写法

  3. FewShotLLMExtractor
     - 增加第三个 few-shot 示例（欧洲雷达，覆盖 exportedTo / affiliatedTo）
     - 复用 ZeroShotLLMExtractor._post_process()（抽成父类方法）
     - prompt 末尾增加 chain-of-thought 引导语

  4. 验证与后处理（统一提取为 BaseLLMExtractor）
     - 新增 BaseLLMExtractor 基类，避免 B/C 两个类重复代码
     - _validate 逻辑合并，只有一处维护

  5. generate_annotation_template
     - 同时展示方法A的结果（原来只展示B/C），方便人工比对规则 vs LLM
     - 增加 "rule_triples_for_reference" 字段

  6. compute_metrics
     - 增加分方法统计（支持同时评测 B 和 C）
     - 增加 per_relation 精确率分布，方便找出哪类关系最弱

依赖：
    pip install requests python-dotenv

使用前配置：
    在 .env 文件里设置 ANTHROPIC_API_KEY（或 OPENAI_API_KEY）
    或直接在下方 LLM_CONFIG 里填写
"""

import json
import re
import os
import time
import logging
from pathlib import Path
from typing import Optional
from copy import deepcopy

# ═══════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════

CORPUS_PATH = Path("radar_corpus/corpus.json")
OUTPUT_DIR  = Path("extraction_results")
OUTPUT_DIR.mkdir(exist_ok=True)

LLM_CONFIG = {
    # ── 选择使用的LLM ──────────────────────────────────────
    # 可选值: "deepseek" | "openai" | "anthropic" | "ollama"
    "provider": "deepseek",

    # ── DeepSeek 配置 ──────────────────────────────────────
    # API Key 从 https://platform.deepseek.com 获取
    # 推荐模型：
    #   deepseek-chat        → DeepSeek-V3，性价比最高，适合批量抽取
    #   deepseek-reasoner    → DeepSeek-R1，推理更强但更慢更贵
    "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    "deepseek_model":   "deepseek-chat",

    # ── OpenAI 配置 ────────────────────────────────────────
    "openai_api_key":  os.getenv("OPENAI_API_KEY", "your-key-here"),
    "openai_model":    "gpt-4o-mini",

    # ── Anthropic 配置 ─────────────────────────────────────
    "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", "your-key-here"),
    "anthropic_model":   "claude-3-5-haiku-20241022",

    # ── Ollama 本地配置 ────────────────────────────────────
    "ollama_url":   "http://localhost:11434",
    "ollama_model": "qwen2.5:7b",

    # ── 代理（按需修改，不需要则设为 None）────────────────
    # "proxies": {
    #     "http":  "http://127.0.0.1:7890",
    #     "https": "http://127.0.0.1:7890",
    # },
    "proxies": None,   # 不需要代理时取消注释这行，注释掉上面4行

    "max_tokens":    1500,
    "sleep_between": 1.0,   # DeepSeek 免费额度限速较严，建议保持 1.0 以上
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "extraction.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
#  本体定义
# ═══════════════════════════════════════════════════════

ENTITY_TYPES = {
    "RadarSystem":      "雷达系统",
    "Country":          "国家",
    "Manufacturer":     "制造商",
    "NavalVessel":      "水面舰艇",
    "AircraftPlatform": "航空平台",
    "GroundPlatform":   "地面平台",
    "FrequencyBand":    "频段",
    "FunctionDomain":   "功能域",
}

RELATION_TYPES = {
    "deployedOn":      ("RadarSystem",  "Platform",       "部署于"),
    "developedBy":     ("RadarSystem",  "Manufacturer",   "研制方"),
    "operatedBy":      ("RadarSystem",  "Country",        "装备于"),
    "exportedTo":      ("RadarSystem",  "Country",        "出口至"),
    "upgradeOf":       ("RadarSystem",  "RadarSystem",    "升级自"),
    "derivedFrom":     ("RadarSystem",  "RadarSystem",    "衍生自"),
    "competitorOf":    ("RadarSystem",  "RadarSystem",    "竞争型号"),
    "coDeployedWith":  ("RadarSystem",  "RadarSystem",    "同平台配套"),
    "hasFrequencyBand":("RadarSystem",  "FrequencyBand",  "工作频段"),
    "hasFunction":     ("RadarSystem",  "FunctionDomain", "具备功能"),
    "affiliatedTo":    ("Manufacturer", "Country",        "隶属于"),
}

FREQUENCY_BAND_VALUES = {
    "HF": ["HF", "high frequency", "3-30 MHz"],
    "VHF": ["VHF", "very high frequency", "30-300 MHz"],
    "UHF": ["UHF", "ultra high frequency", "300-1000 MHz"],
    "L":   ["L-band", "L band", "1-2 GHz"],
    "S":   ["S-band", "S band", "2-4 GHz"],
    "C":   ["C-band", "C band", "4-8 GHz"],
    "X":   ["X-band", "X band", "8-12 GHz"],
    "Ku":  ["Ku-band", "Ku band", "12-18 GHz"],
    "Ka":  ["Ka-band", "Ka band", "26.5-40 GHz"],
    "W":   ["W-band", "millimeter wave", "mmW"],
}

# [修改1] 国家归一化表：英文表述 -> 中文统一写法
COUNTRY_NORMALIZE = {
    "united states": "美国", "u.s.": "美国", "us navy": "美国",
    "us air force": "美国", "us army": "美国", "american": "美国",
    "russia": "俄罗斯", "russian": "俄罗斯", "soviet union": "俄罗斯",
    "ussr": "俄罗斯", "russian navy": "俄罗斯",
    "china": "中国", "chinese": "中国", "pla": "中国", "plan": "中国",
    "people's republic": "中国",
    "united kingdom": "英国", "uk": "英国", "royal navy": "英国",
    "british": "英国",
    "france": "法国", "french": "法国", "marine nationale": "法国",
    "germany": "德国", "german": "德国", "bundeswehr": "德国",
    "italy": "意大利", "italian": "意大利",
    "israel": "以色列", "israeli": "以色列",
    "japan": "日本", "japanese": "日本", "jmsdf": "日本",
    "netherlands": "荷兰", "dutch": "荷兰",
    "sweden": "瑞典", "swedish": "瑞典",
    "india": "印度", "indian": "印度",
    "south korea": "韩国", "republic of korea": "韩国",
    "australia": "澳大利亚", "australian": "澳大利亚",
}

# [修改1] 频段归一化表：各种写法 -> 标准单字母
BAND_NORMALIZE = {
    r"\bHF\b":                        "HF",
    r"\bVHF\b":                       "VHF",
    r"\bUHF\b":                       "UHF",
    r"\bL[\s\-_]?[Bb]and\b":         "L",
    r"\bS[\s\-_]?[Bb]and\b":         "S",
    r"\bC[\s\-_]?[Bb]and\b":         "C",
    r"\bX[\s\-_]?[Bb]and\b":         "X",
    r"\bKu[\s\-_]?[Bb]and\b":        "Ku",
    r"\bKa[\s\-_]?[Bb]and\b":        "Ka",
    r"\bW[\s\-_]?[Bb]and\b":         "W",
    r"\bmillimeter[\s\-]?wave\b":     "W",
    r"(?<!\d)1[\s]?[-–][\s]?2\s*GHz": "L",
    r"(?<!\d)2[\s]?[-–][\s]?4\s*GHz": "S",
    r"(?<!\d)4[\s]?[-–][\s]?8\s*GHz": "C",
    r"(?<!\d)8[\s]?[-–][\s]?12\s*GHz":"X",
    r"(?<!\d)12[\s]?[-–][\s]?18\s*GHz":"Ku",
}


# ═══════════════════════════════════════════════════════
#  方法A：规则抽取
# ═══════════════════════════════════════════════════════

class RuleBasedExtractor:

    RELATION_PATTERNS = {
        "deployedOn": [
            r"(?:installed|deployed|fitted|equipped|mounted)\s+(?:on|in|aboard)\s+([A-Z][^,.]{3,60}(?:class|destroyer|frigate|cruiser|carrier|corvette)[^,.]*)",
            r"([A-Z][^,.]{3,60}(?:class|destroyer|frigate|cruiser|carrier)[^,.]*)\s+(?:is|are|was|were)\s+equipped\s+with",
            r"used\s+(?:on|aboard|in)\s+([A-Z][^,.]{3,60}(?:class|destroyer|frigate|cruiser|carrier)[^,.]*)",
            # [修改2] 新增：飞机平台匹配
            r"(?:installed|carried|fitted)\s+(?:on|in|aboard)\s+([A-Z][^,.]{3,50}(?:aircraft|airplane|fighter|bomber|helicopter|UAV)[^,.]*)",
        ],
        "developedBy": [
            r"developed\s+by\s+([A-Z][A-Za-z\s&/,]+?)(?:\.|,|\s+and|\s+for|\s+in)",
            r"manufactured\s+by\s+([A-Z][A-Za-z\s&/,]+?)(?:\.|,|\s+and|\s+for)",
            r"produced\s+by\s+([A-Z][A-Za-z\s&/,]+?)(?:\.|,|\s+and|\s+for)",
            r"([A-Z][A-Za-z\s&]+?)\s+(?:developed|designed|produced|manufactured)\s+the",
            # [修改2] 新增：build/create 动词
            r"built\s+by\s+([A-Z][A-Za-z\s&/,]+?)(?:\.|,|\s+and|\s+for)",
        ],
        "operatedBy": [
            r"(?:operated|used|employed)\s+by\s+(?:the\s+)?([A-Z][A-Za-z\s]+?(?:Navy|Air Force|Army|Military|Armed Forces))",
            r"in\s+service\s+with\s+(?:the\s+)?([A-Z][A-Za-z\s]+?(?:Navy|Air Force|Army))",
            r"(?:serves?|serving)\s+(?:with|in)\s+(?:the\s+)?([A-Z][A-Za-z\s]+?(?:Navy|Air Force|Army))",
        ],
        "upgradeOf": [
            # 尾实体须为雷达系统名（AN/前缀、Type前缀、或已知雷达名模式）
            r"(?:upgrade|successor|replacement)\s+(?:of|to|for)\s+(?:the\s+)?(AN/[A-Z]+-\w+|Type\s+\d+[A-Za-z]*|[A-Z][A-Za-z0-9\-/]+\s+radar)",
            r"(?:replaces?|succeeded?)\s+(?:the\s+)?(AN/[A-Z]+-\w+|Type\s+\d+[A-Za-z]*\s+radar)",
            r"improved\s+version\s+of\s+(?:the\s+)?(AN/[A-Z]+-\w+|Type\s+\d+[A-Za-z]*)",
        ],
        "derivedFrom": [
            # derivedFrom 须包含共源关键词，规则抽取仅匹配最强信号
            r"(?:shares?\s+(?:the\s+)?(?:same\s+)?(?:\w+\s+)?subsystem|common\s+(?:\w+\s+)?module)\s+(?:with|as)\s+(?:the\s+)?(AN/[A-Z]+-\w+|Type\s+\d+[A-Za-z]*)",
            r"(?:derived)\s+from\s+(?:the\s+)?(AN/[A-Z]+-\w+|Type\s+\d+[A-Za-z]*)",
        ],
        "exportedTo": [
            r"exported?\s+to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"sold\s+to\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"(?:purchased|acquired|ordered)\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        ],
        # [修改2] 新增：竞争关系和厂商归属
        "competitorOf": [
            r"compet(?:es?|ing|itor)\s+(?:with|against|to)\s+(?:the\s+)?(AN/[A-Z]+-\w+|[A-Z][A-Za-z0-9\-/]+)",
            r"rival\s+(?:of|to)\s+(?:the\s+)?(AN/[A-Z]+-\w+|[A-Z][A-Za-z0-9\-/]+)",
        ],
        "affiliatedTo": [
            r"([A-Z][A-Za-z\s&]+?),?\s+(?:a\s+)?(?:US|American|British|French|Russian|Chinese|Israeli|Dutch|Swedish)\s+(?:company|firm|corporation|defense contractor)",
            r"([A-Z][A-Za-z\s&]+?)\s+is\s+(?:a\s+)?subsidiary\s+of",
        ],
    }

    COUNTRY_DICT = {
        "美国":   ["United States", "US Navy", "US Air Force", "US Army", "American"],
        "俄罗斯": ["Russia", "Soviet Union", "USSR", "Russian Navy"],
        "中国":   ["China", "Chinese", "People's Republic", "PLA", "PLAN"],
        "英国":   ["United Kingdom", "UK", "Royal Navy", "British"],
        "法国":   ["France", "French Navy", "Marine nationale"],
        "德国":   ["Germany", "German", "Bundeswehr"],
        "意大利": ["Italy", "Italian Navy", "Marina Militare"],
        "以色列": ["Israel", "Israeli", "IDF"],
        "日本":   ["Japan", "Japanese", "JMSDF"],
        "荷兰":   ["Netherlands", "Dutch", "Royal Netherlands Navy"],
        "瑞典":   ["Sweden", "Swedish"],
        "印度":   ["India", "Indian Navy"],
        "韩国":   ["South Korea", "Republic of Korea Navy"],
    }

    def extract(self, radar: dict) -> list[dict]:
        triples = []
        name  = radar.get("en_title", "")
        text  = radar.get("raw_text_en", "")
        facts = radar.get("known_facts", {})

        if not name or not text:
            return triples

        triples.extend(self._facts_to_triples(name, facts))

        sentences = self._split_sentences(text)
        for sent in sentences:
            triples.extend(self._match_sentence(name, sent))

        for band in facts.get("frequencyBand", []):
            norm = self._normalize_band(band)
            if norm:
                triples.append({
                    "head":       name,
                    "head_type":  "RadarSystem",
                    "relation":   "hasFrequencyBand",
                    "tail":       norm,
                    "tail_type":  "FrequencyBand",
                    "confidence": 0.95,
                    "source":     "rule_facts",
                    "evidence":   f"frequencyBand from known_facts: {band}",
                })

        triples = self._deduplicate(triples)
        return triples

    def _facts_to_triples(self, name: str, facts: dict) -> list[dict]:
        result = []

        if facts.get("country"):
            result.append({
                "head": name, "head_type": "RadarSystem",
                "relation": "operatedBy",
                "tail": facts["country"], "tail_type": "Country",
                "confidence": 0.90, "source": "rule_facts",
                "evidence": f"country inferred: {facts['country']}",
            })

        if facts.get("manufacturer"):
            mfr = facts["manufacturer"]
            result.append({
                "head": name, "head_type": "RadarSystem",
                "relation": "developedBy",
                "tail": mfr, "tail_type": "Manufacturer",
                "confidence": 0.85, "source": "rule_facts",
                "evidence": f"manufacturer extracted: {mfr}",
            })
            # [修改3] 从 manufacturer + country 推断 affiliatedTo
            if facts.get("country"):
                result.append({
                    "head": mfr, "head_type": "Manufacturer",
                    "relation": "affiliatedTo",
                    "tail": facts["country"], "tail_type": "Country",
                    "confidence": 0.70, "source": "rule_facts",
                    "evidence": f"inferred from co-occurrence of manufacturer and country",
                })

        for platform in facts.get("platform", []):
            # [修改4] tail_type 根据平台关键词推断，不再硬写 "Platform"
            tail_type = self._infer_platform_type(platform)
            result.append({
                "head": name, "head_type": "RadarSystem",
                "relation": "deployedOn",
                "tail": platform, "tail_type": tail_type,
                "confidence": 0.80, "source": "rule_facts",
                "evidence": f"platform extracted from text: {platform}",
            })

        return result

    # [修改4] 新增：平台类型推断
    def _infer_platform_type(self, platform_name: str) -> str:
        name_lower = platform_name.lower()
        naval_kw   = ["destroyer", "frigate", "cruiser", "carrier",
                      "corvette", "ship", "vessel", "ddg", "ffg", "cg"]
        air_kw     = ["aircraft", "airplane", "fighter", "bomber",
                      "helicopter", "uav", "drone", "awacs", "aew"]
        ground_kw  = ["vehicle", "truck", "ground", "land", "mobile",
                      "fixed", "site"]
        for kw in naval_kw:
            if kw in name_lower:
                return "NavalVessel"
        for kw in air_kw:
            if kw in name_lower:
                return "AircraftPlatform"
        for kw in ground_kw:
            if kw in name_lower:
                return "GroundPlatform"
        return "Platform"  # 无法判断时用通用类型

    # [修改1] 新增：国家名归一化
    def _normalize_country(self, raw: str) -> Optional[str]:
        """将英文国家表述归一化为中文，返回 None 表示无法识别"""
        raw_lower = raw.lower().strip()
        if raw_lower in COUNTRY_NORMALIZE:
            return COUNTRY_NORMALIZE[raw_lower]
        # 尝试部分匹配
        for eng, zh in COUNTRY_NORMALIZE.items():
            if eng in raw_lower:
                return zh
        return None

    # [修改1] 新增：频段归一化
    def _normalize_band(self, raw: str) -> Optional[str]:
        """将各种频段写法归一化为标准单字母"""
        raw_strip = raw.strip()
        # 已经是标准值
        if raw_strip in FREQUENCY_BAND_VALUES:
            return raw_strip
        # 用正则匹配
        for pat, norm in BAND_NORMALIZE.items():
            if re.search(pat, raw_strip, re.IGNORECASE):
                return norm
        return None

    # [修改5] _match_sentence：改为扫描全部关系，不再 break
    def _match_sentence(self, radar_name: str, sentence: str) -> list[dict]:
        result = []
        for relation, patterns in self.RELATION_PATTERNS.items():
            for pat in patterns:
                for m in re.finditer(pat, sentence, re.IGNORECASE):
                    tail = m.group(1).strip()
                    if len(tail) < 2 or len(tail) > 80:
                        continue

                    # 国家类关系做归一化
                    if relation in ("operatedBy", "exportedTo"):
                        norm = self._normalize_country(tail)
                        if norm:
                            tail = norm
                        else:
                            continue  # 无法识别的国家跳过，避免噪声

                    head_type, tail_type, _ = RELATION_TYPES.get(
                        relation, ("RadarSystem", "Entity", "")
                    )
                    # affiliatedTo 的 head 是 Manufacturer，不是雷达
                    if relation == "affiliatedTo":
                        continue  # 规则层较难准确识别厂商名，跳过

                    result.append({
                        "head":       radar_name,
                        "head_type":  "RadarSystem",
                        "relation":   relation,
                        "tail":       tail,
                        "tail_type":  tail_type,
                        "confidence": 0.70,
                        "source":     "rule_pattern",
                        "evidence":   sentence[:200],
                    })
        return result

    def _split_sentences(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    # [修改6] _deduplicate：保留置信度最高的，而非最早出现的
    def _deduplicate(self, triples: list[dict]) -> list[dict]:
        best: dict[tuple, dict] = {}
        for t in triples:
            key = (t["head"], t["relation"], t["tail"])
            if key not in best or t["confidence"] > best[key]["confidence"]:
                best[key] = t
        return list(best.values())


# ═══════════════════════════════════════════════════════
#  LLM 调用封装
# ═══════════════════════════════════════════════════════

def call_llm(prompt: str, system: str = "") -> Optional[str]:
    """统一的LLM调用接口，支持 DeepSeek / OpenAI / Anthropic / Ollama。"""
    provider = LLM_CONFIG["provider"]
    try:
        if provider == "deepseek":
            return _call_deepseek(prompt, system)
        elif provider == "openai":
            return _call_openai(prompt, system)
        elif provider == "anthropic":
            return _call_anthropic(prompt, system)
        elif provider == "ollama":
            full = f"{system}\n\n{prompt}" if system else prompt
            return _call_ollama(full)
        else:
            log.error(f"未知的LLM provider: {provider}")
            return None
    except Exception as e:
        log.warning(f"LLM调用失败: {e}")
        return None


def _call_deepseek(prompt: str, system: str = "") -> Optional[str]:
    """
    DeepSeek API 调用。
    DeepSeek 兼容 OpenAI Chat Completions 格式，base_url 换成 DeepSeek 端点即可。
    """
    import requests
    headers = {
        "Authorization": f"Bearer {LLM_CONFIG['deepseek_api_key']}",
        "Content-Type":  "application/json",
    }
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model":      LLM_CONFIG["deepseek_model"],
        "max_tokens": LLM_CONFIG["max_tokens"],
        "messages":   messages,
    }
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",   # DeepSeek 官方端点
        headers=headers,
        json=body,
        proxies=LLM_CONFIG.get("proxies"),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# [修改7] _call_anthropic 支持独立 system 字段，减少 token 浪费
def _call_anthropic(prompt: str, system: str = "") -> Optional[str]:
    import requests
    headers = {
        "x-api-key":         LLM_CONFIG["anthropic_api_key"],
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }
    body = {
        "model":      LLM_CONFIG["anthropic_model"],
        "max_tokens": LLM_CONFIG["max_tokens"],
        "messages":   [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=body,
        proxies=LLM_CONFIG.get("proxies"),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def _call_openai(prompt: str, system: str = "") -> Optional[str]:
    import requests
    headers = {
        "Authorization": f"Bearer {LLM_CONFIG['openai_api_key']}",
        "Content-Type":  "application/json",
    }
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model":      LLM_CONFIG["openai_model"],
        "max_tokens": LLM_CONFIG["max_tokens"],
        "messages":   messages,
    }
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=body,
        proxies=LLM_CONFIG.get("proxies"),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_ollama(prompt: str) -> Optional[str]:
    import requests
    body = {
        "model":  LLM_CONFIG["ollama_model"],
        "prompt": prompt,
        "stream": False,
    }
    resp = requests.post(
        f"{LLM_CONFIG['ollama_url']}/api/generate",
        json=body,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def parse_llm_json(text: str) -> Optional[list]:
    """从LLM输出中提取JSON，处理常见格式问题。"""
    if not text:
        return None
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()
    start = text.find("[")
    end   = text.rfind("]")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


# ═══════════════════════════════════════════════════════
#  本体约束描述（B/C 共用）
# ═══════════════════════════════════════════════════════

# [修改8] 把本体描述从 prompt 里独立出来，作为 system prompt 的一部分
ONTOLOGY_SYSTEM = """你是军事雷达领域的知识图谱构建专家。严格按照以下本体约束从文本中抽取三元组。

【实体类型】
- RadarSystem：雷达系统型号，如"AN/SPY-1D"、"Type 346"
- Country：国家，统一用中文，如"美国"、"中国"、"俄罗斯"
- Manufacturer：制造商，如"Raytheon"、"CETC"、"Thales"
- NavalVessel：水面舰艇，如"阿利·伯克级驱逐舰"
- AircraftPlatform：航空平台，如"E-2D预警机"
- GroundPlatform：地面平台，如"THAAD发射车"
- FrequencyBand：频段，值只能是：HF/VHF/UHF/L/S/C/X/Ku/Ka/W

【关系类型】
- deployedOn：雷达部署在哪个平台（NavalVessel/AircraftPlatform/GroundPlatform）
- developedBy：雷达由哪个制造商（Manufacturer）研制
- operatedBy：雷达被哪个国家（Country）装备使用
- exportedTo：雷达出口到哪个国家（Country）
- upgradeOf：该雷达是哪个型号（RadarSystem）的升级版/改进版（同型号族内，如AN/SPY-1D是AN/SPY-1A的升级）
- derivedFrom：该雷达与另一型号（RadarSystem）共享核心子系统，具有明确的技术继承关系。
  ⚠️ 重要区分：
    • derivedFrom（衍生自）：两型号共用天线/信号处理/发射机等核心部件，技术上同宗同源
    • upgradeOf（升级自）：同型号族内的改进版本（如AN/SPY-1D升级自AN/SPY-1A）
    • 若仅为"inspired by"/"based on concept"或仅有战略/设计理念相似，不抽取derivedFrom
    • head ≠ tail（不得自引用）
    • evidence 中必须含有以下词之一：subsystem/shares/common/based on the same/共用/同源/相同子系统
- competitorOf：该雷达与哪个型号（RadarSystem）是竞争关系（请勿抽取，该关系由系统自动推断）
- hasFrequencyBand：工作频段（FrequencyBand），只能是：HF/VHF/UHF/L/S/C/X/Ku/Ka/W
- affiliatedTo：制造商（Manufacturer）隶属于哪个国家（Country）

【输出规则】
1. 只返回 JSON 数组，不加任何其他文字或 Markdown
2. Country 的 tail 值统一用中文（美国/中国/俄罗斯/英国/法国/德国/意大利/以色列/日本/荷兰/瑞典/印度/韩国）
3. hasFrequencyBand 的 tail 只能是上述标准字母（X 而不是 X-band）
4. 只抽取有原文依据的三元组，不推测，不捏造
5. confidence：直接陈述=0.90-0.98，需推断=0.60-0.80，隐含=0.40-0.60
6. competitorOf 关系请勿抽取（留给后处理模块推断）
7. derivedFrom 的 head 和 tail 不得相同"""

ONTOLOGY_USER_HEADER = """雷达系统名称：{radar_name}

描述文本（前{text_len}字）：
{text}

请抽取所有可识别的知识三元组，输出格式：
[
  {{
    "head": "实体名",
    "head_type": "实体类型",
    "relation": "关系类型",
    "tail": "实体名或值",
    "tail_type": "实体类型",
    "confidence": 0.0到1.0,
    "evidence": "原文片段（50字以内）"
  }}
]"""


# ═══════════════════════════════════════════════════════
#  [修改9] 基类：避免 B/C 重复代码
# ═══════════════════════════════════════════════════════

class BaseLLMExtractor:
    """B/C 两个 LLM 提取器的公共基类。"""

    source_tag = "llm_base"

    def _post_process(self, triples: list[dict], radar_name: str) -> list[dict]:
        """
        后处理：
        - 归一化 Country tail（英文→中文）
        - 归一化 FrequencyBand tail（X-band→X）
        - 补全缺失的 tail_type
        - 过滤 head 不是雷达名的（除 affiliatedTo 外）
        """
        rule = RuleBasedExtractor()
        processed = []
        for t in triples:
            relation = t.get("relation", "")

            # head 应是雷达名（affiliatedTo 除外）
            if relation != "affiliatedTo":
                if t.get("head", "").lower() != radar_name.lower():
                    t["head"] = radar_name

            # Country 归一化
            if relation in ("operatedBy", "exportedTo", "affiliatedTo"):
                norm = rule._normalize_country(t.get("tail", ""))
                if norm:
                    t["tail"] = norm
                elif relation != "affiliatedTo":
                    # 无法识别的国家，降低置信度
                    t["confidence"] = min(t.get("confidence", 0.5), 0.40)

            # FrequencyBand 归一化
            if relation == "hasFrequencyBand":
                norm = rule._normalize_band(t.get("tail", ""))
                if norm:
                    t["tail"] = norm
                else:
                    continue  # 无法归一化的频段直接丢弃

            # 补全 tail_type
            if not t.get("tail_type") and relation in RELATION_TYPES:
                t["tail_type"] = RELATION_TYPES[relation][1]

            # 补全 source
            t["source"] = self.source_tag
            processed.append(t)

        return processed

    # 证明derivedFrom有效所需的关键词（至少出现一个）
    _DERIVED_EVIDENCE_KEYWORDS = [
        "subsystem", "shares", "common", "based on the same",
        "common origin", "共用", "同源", "相同子系统", "同一天线",
        "derived", "技术继承",
    ]

    def _validate(self, triples: list[dict]) -> list[dict]:
        """过滤不符合本体约束的三元组。"""
        valid = []
        for t in triples:
            relation = t.get("relation", "")
            head = t.get("head", "")
            tail = t.get("tail", "")

            if relation not in RELATION_TYPES:
                continue
            if not head or not tail:
                continue
            if relation == "hasFrequencyBand":
                if tail not in FREQUENCY_BAND_VALUES:
                    continue
            # derivedFrom 额外验证：
            #   1. head ≠ tail（防止自引用）
            #   2. evidence 中需含有技术共源关键词
            if relation == "derivedFrom":
                if head.lower() == tail.lower():
                    continue
                evidence = t.get("evidence", "").lower()
                if not any(kw.lower() in evidence for kw in self._DERIVED_EVIDENCE_KEYWORDS):
                    continue
            # competitorOf 由 kg_enrichment.py 推断，不接受 LLM 直接抽取
            if relation == "competitorOf":
                continue
            valid.append(t)
        return valid

    def _run(self, radar: dict, prompt: str) -> list[dict]:
        """通用执行逻辑：调用LLM → 解析 → 后处理 → 验证"""
        name = radar.get("en_title", "")
        time.sleep(LLM_CONFIG["sleep_between"])
        response = call_llm(prompt, system=ONTOLOGY_SYSTEM)
        if not response:
            return []
        triples = parse_llm_json(response)
        if not triples:
            log.warning(f"  [{self.source_tag}] JSON解析失败: {name}")
            return []
        triples = self._post_process(triples, name)
        return self._validate(triples)


# ═══════════════════════════════════════════════════════
#  方法B：LLM零样本抽取
# ═══════════════════════════════════════════════════════

class ZeroShotLLMExtractor(BaseLLMExtractor):

    source_tag = "llm_zero_shot"

    def extract(self, radar: dict) -> list[dict]:
        name = radar.get("en_title", "")
        text = radar.get("raw_text_en", "")
        if not name or not text:
            return []

        # [修改8] system/user 分离，减少 token 消耗
        prompt = ONTOLOGY_USER_HEADER.format(
            radar_name = name,
            text       = text[:1500],
            text_len   = min(len(text), 1500),
        )
        return self._run(radar, prompt)


# ═══════════════════════════════════════════════════════
#  方法C：LLM少样本抽取
# ═══════════════════════════════════════════════════════

# [修改10] 扩充为3个示例，覆盖欧洲雷达 + exportedTo + affiliatedTo
FEW_SHOT_EXAMPLES = """
示例1（美国舰载相控阵）：
雷达名称：AN/SPY-1D
文本片段："The AN/SPY-1D is a passive electronically scanned array radar developed
by Raytheon. It operates in S-band and is deployed on Arleigh Burke-class destroyers
of the United States Navy. It is an upgrade of the AN/SPY-1A."
输出：
[
  {"head":"AN/SPY-1D","head_type":"RadarSystem","relation":"developedBy","tail":"Raytheon","tail_type":"Manufacturer","confidence":0.98,"evidence":"developed by Raytheon"},
  {"head":"AN/SPY-1D","head_type":"RadarSystem","relation":"hasFrequencyBand","tail":"S","tail_type":"FrequencyBand","confidence":0.98,"evidence":"operates in S-band"},
  {"head":"AN/SPY-1D","head_type":"RadarSystem","relation":"deployedOn","tail":"Arleigh Burke-class destroyer","tail_type":"NavalVessel","confidence":0.95,"evidence":"deployed on Arleigh Burke-class destroyers"},
  {"head":"AN/SPY-1D","head_type":"RadarSystem","relation":"operatedBy","tail":"美国","tail_type":"Country","confidence":0.95,"evidence":"United States Navy"},
  {"head":"AN/SPY-1D","head_type":"RadarSystem","relation":"upgradeOf","tail":"AN/SPY-1A","tail_type":"RadarSystem","confidence":0.95,"evidence":"upgrade of the AN/SPY-1A"},
  {"head":"Raytheon","head_type":"Manufacturer","relation":"affiliatedTo","tail":"美国","tail_type":"Country","confidence":0.85,"evidence":"developed by Raytheon ... United States Navy"}
]

示例2（中国双频舰载雷达）：
雷达名称：Type 346 radar
文本片段："The Type 346 radar, also known as the Dragon Eye, is an active phased array
radar developed by CETC for the Chinese Navy. It operates in S/X dual band and is
fitted on Type 052C destroyers. The Type 346A is an improved version."
输出：
[
  {"head":"Type 346 radar","head_type":"RadarSystem","relation":"developedBy","tail":"CETC","tail_type":"Manufacturer","confidence":0.97,"evidence":"developed by CETC"},
  {"head":"Type 346 radar","head_type":"RadarSystem","relation":"operatedBy","tail":"中国","tail_type":"Country","confidence":0.97,"evidence":"Chinese Navy"},
  {"head":"Type 346 radar","head_type":"RadarSystem","relation":"hasFrequencyBand","tail":"S","tail_type":"FrequencyBand","confidence":0.95,"evidence":"S/X dual band"},
  {"head":"Type 346 radar","head_type":"RadarSystem","relation":"hasFrequencyBand","tail":"X","tail_type":"FrequencyBand","confidence":0.95,"evidence":"S/X dual band"},
  {"head":"Type 346 radar","head_type":"RadarSystem","relation":"deployedOn","tail":"Type 052C destroyer","tail_type":"NavalVessel","confidence":0.95,"evidence":"fitted on Type 052C destroyers"},
  {"head":"CETC","head_type":"Manufacturer","relation":"affiliatedTo","tail":"中国","tail_type":"Country","confidence":0.85,"evidence":"developed by CETC for the Chinese Navy"},
  {"head":"Type 346A","head_type":"RadarSystem","relation":"upgradeOf","tail":"Type 346 radar","tail_type":"RadarSystem","confidence":0.88,"evidence":"Type 346A is an improved version"}
]

示例3（欧洲出口型雷达）：
雷达名称：SMART-L
文本片段："The SMART-L is a Dutch naval radar manufactured by Thales Nederland, a
subsidiary of Thales Group. It operates in L-band and is deployed aboard
De Zeven Provinciën-class frigates of the Royal Netherlands Navy. The system has
also been exported to Germany and Denmark, where it serves aboard Type 124 frigates."
输出：
[
  {"head":"SMART-L","head_type":"RadarSystem","relation":"developedBy","tail":"Thales Nederland","tail_type":"Manufacturer","confidence":0.97,"evidence":"manufactured by Thales Nederland"},
  {"head":"SMART-L","head_type":"RadarSystem","relation":"hasFrequencyBand","tail":"L","tail_type":"FrequencyBand","confidence":0.97,"evidence":"operates in L-band"},
  {"head":"SMART-L","head_type":"RadarSystem","relation":"deployedOn","tail":"De Zeven Provinciën-class frigate","tail_type":"NavalVessel","confidence":0.95,"evidence":"deployed aboard De Zeven Provinciën-class frigates"},
  {"head":"SMART-L","head_type":"RadarSystem","relation":"operatedBy","tail":"荷兰","tail_type":"Country","confidence":0.95,"evidence":"Royal Netherlands Navy"},
  {"head":"SMART-L","head_type":"RadarSystem","relation":"exportedTo","tail":"德国","tail_type":"Country","confidence":0.92,"evidence":"exported to Germany"},
  {"head":"SMART-L","head_type":"RadarSystem","relation":"exportedTo","tail":"丹麦","tail_type":"Country","confidence":0.92,"evidence":"exported to ... Denmark"},
  {"head":"Thales Nederland","head_type":"Manufacturer","relation":"affiliatedTo","tail":"荷兰","tail_type":"Country","confidence":0.80,"evidence":"Dutch naval radar manufactured by Thales Nederland"}
]

示例4（derivedFrom正例 vs upgradeOf区分）：
雷达名称：AN/SPY-6
文本片段："The AN/SPY-6 Air and Missile Defense Radar (AMDR) is developed by Raytheon.
It shares the same gallium nitride (GaN) subsystem technology as the AN/TPY-4 ground
radar, both using common transmit/receive modules. The AN/SPY-6(V)1 is an improved
version of the AN/SPY-6(V)0 baseline."
输出：
[
  {"head":"AN/SPY-6","head_type":"RadarSystem","relation":"developedBy","tail":"Raytheon","tail_type":"Manufacturer","confidence":0.98,"evidence":"developed by Raytheon"},
  {"head":"AN/SPY-6","head_type":"RadarSystem","relation":"derivedFrom","tail":"AN/TPY-4","tail_type":"RadarSystem","confidence":0.82,"evidence":"shares the same gallium nitride (GaN) subsystem technology ... common transmit/receive modules"},
  {"head":"AN/SPY-6(V)1","head_type":"RadarSystem","relation":"upgradeOf","tail":"AN/SPY-6(V)0","tail_type":"RadarSystem","confidence":0.95,"evidence":"improved version of the AN/SPY-6(V)0 baseline"}
]

⚠️ 负例说明（不应抽取derivedFrom的情形）：
文本片段："The XYZ radar was inspired by the Cold War-era radar design philosophy."
→ 不抽取（"inspired by"不代表共用子系统，无derivedFrom）

文本片段："The ABC radar replaced the older DEF system."
→ 不抽取derivedFrom；若有升级关系可抽upgradeOf。
"""

# [修改11] few-shot prompt 末尾加 CoT 引导语
FEW_SHOT_USER_TEMPLATE = """{header}

参考示例：
{examples}

请仔细阅读文本，按照示例格式逐步思考：
1. 找出所有雷达型号名、制造商名、国家名、平台名
2. 判断它们之间的关系类型
3. 确认每条三元组都有原文依据
4. 输出 JSON 数组（不要任何说明文字）

输出："""


class FewShotLLMExtractor(BaseLLMExtractor):

    source_tag = "llm_few_shot"

    def extract(self, radar: dict) -> list[dict]:
        name = radar.get("en_title", "")
        text = radar.get("raw_text_en", "")
        if not name or not text:
            return []

        header = ONTOLOGY_USER_HEADER.format(
            radar_name = name,
            text       = text[:1500],
            text_len   = min(len(text), 1500),
        )
        prompt = FEW_SHOT_USER_TEMPLATE.format(
            header   = header,
            examples = FEW_SHOT_EXAMPLES,
        )
        return self._run(radar, prompt)


# ═══════════════════════════════════════════════════════
#  主流程：批量抽取
# ═══════════════════════════════════════════════════════

def run_extraction(corpus: list[dict],
                   extractor,
                   method_name: str,
                   output_path: Path,
                   sample_size: int = None) -> list[dict]:
    existing = {}
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            prev = json.load(f)
        for item in prev:
            existing[item["radar_id"]] = item
        log.info(f"[{method_name}] 加载已有结果 {len(existing)} 条")

    targets     = corpus[:sample_size] if sample_size else corpus
    all_results = []
    new_count   = 0

    for i, radar in enumerate(targets):
        radar_id = radar.get("id", f"radar_{i}")

        if radar_id in existing:
            all_results.append(existing[radar_id])
            continue

        log.info(f"[{method_name}] [{i+1:3d}/{len(targets)}] {radar.get('en_title','')}")

        try:
            triples = extractor.extract(radar)
        except Exception as e:
            log.error(f"  抽取异常: {e}")
            triples = []

        result = {
            "radar_id":    radar_id,
            "en_title":    radar.get("en_title", ""),
            "triple_count": len(triples),
            "triples":     triples,
        }
        all_results.append(result)
        new_count += 1

        if new_count % 10 == 0:
            _save(all_results, output_path)
            log.info(f"  已保存 {len(all_results)} 条")

    _save(all_results, output_path)
    total_triples = sum(r["triple_count"] for r in all_results)
    log.info(f"[{method_name}] 完成: {len(all_results)} 个雷达, {total_triples} 条三元组")
    return all_results


def _save(data: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════
#  生成人工标注模板
# ═══════════════════════════════════════════════════════

def generate_annotation_template(corpus: list[dict],
                                  results_a: list[dict],
                                  results_b: list[dict],
                                  results_c: list[dict],
                                  n_samples: int = 80) -> list[dict]:
    """
    [修改12] 同时展示三种方法的结果，方便人工比对。
    标注模板字段说明：
      llm_fewshot_triples_to_annotate: 需要人工判断 correct/wrong_entity/wrong_relation/hallucinated
      rule_triples_for_reference:      规则抽取结果（参考用，不需要标注）
      human_added_triples:             LLM漏掉的，人工补充
    """
    import random
    random.seed(42)

    result_map_a = {r["radar_id"]: r for r in results_a}
    result_map_b = {r["radar_id"]: r for r in results_b}
    result_map_c = {r["radar_id"]: r for r in results_c}

    scored = []
    for radar in corpus:
        rid = radar.get("id", "")
        n_c = len(result_map_c.get(rid, {}).get("triples", []))
        scored.append((n_c, radar))
    scored.sort(key=lambda x: -x[0])

    high   = [r for n, r in scored if n >= 5][:30]
    medium = [r for n, r in scored if 2 <= n < 5][:30]
    low    = [r for n, r in scored if n < 2][:20]
    sampled = high + medium + low

    template = []
    for radar in sampled:
        rid = radar.get("id", "")
        triples_a = result_map_a.get(rid, {}).get("triples", [])
        triples_b = result_map_b.get(rid, {}).get("triples", [])
        triples_c = result_map_c.get(rid, {}).get("triples", [])

        # 主标注对象：few-shot 结果（通常质量更高）
        to_annotate = [
            {
                "head":     t["head"],
                "relation": t["relation"],
                "tail":     t["tail"],
                "evidence": t.get("evidence", ""),
                "source":   t.get("source", ""),
                # 人工填写：
                "label":   "",   # correct / wrong_entity / wrong_relation / hallucinated
                "comment": "",
            }
            for t in triples_c
        ]

        item = {
            "radar_id":   rid,
            "en_title":   radar.get("en_title", ""),
            "zh_title":   radar.get("zh_title", ""),
            "source_en":  radar.get("source_en", ""),
            "text_preview": radar.get("raw_text_en", "")[:300],

            # 主标注目标
            "llm_fewshot_triples_to_annotate": to_annotate,

            # [修改12] 参考对比：规则抽取 + 零样本结果
            "rule_triples_for_reference": [
                {"head": t["head"], "relation": t["relation"],
                 "tail": t["tail"], "confidence": t.get("confidence")}
                for t in triples_a
            ],
            "zero_shot_triples_for_reference": [
                {"head": t["head"], "relation": t["relation"],
                 "tail": t["tail"], "confidence": t.get("confidence")}
                for t in triples_b
            ],

            # 人工补充的漏召
            "human_added_triples": [],
            "annotated_by": "",
            "annotated_at": "",
        }
        template.append(item)

    path = OUTPUT_DIR / "annotation_template.json"
    _save(template, path)
    log.info(f"标注模板已生成: {path} ({len(template)} 条)")
    return template


# ═══════════════════════════════════════════════════════
#  对比报告
# ═══════════════════════════════════════════════════════

def generate_comparison_report(results_a, results_b, results_c) -> dict:
    def stats(results):
        total         = len(results)
        total_triples = sum(r["triple_count"] for r in results)
        has_triples   = sum(1 for r in results if r["triple_count"] > 0)
        avg_triples   = total_triples / total if total else 0

        rel_dist = {}
        confidences = []
        for r in results:
            for t in r["triples"]:
                rel = t.get("relation", "unknown")
                rel_dist[rel] = rel_dist.get(rel, 0) + 1
                confidences.append(t.get("confidence", 0))

        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        return {
            "total_radars":         total,
            "radars_with_triples":  has_triples,
            "total_triples":        total_triples,
            "avg_triples_per_radar": round(avg_triples, 2),
            "avg_confidence":        round(avg_conf, 3),
            "relation_distribution": dict(
                sorted(rel_dist.items(), key=lambda x: -x[1])
            ),
        }

    report = {
        "method_a_rule":      stats(results_a),
        "method_b_zero_shot": stats(results_b),
        "method_c_few_shot":  stats(results_c),
        "note": "Precision/Recall/F1 需完成标注后运行 compute_metrics() 计算。",
    }
    path = OUTPUT_DIR / "comparison_report.json"
    _save(report, path)
    return report


def print_comparison(report: dict):
    print("\n" + "═" * 60)
    print("  三种抽取方法对比（标注前统计）")
    print("═" * 60)
    for method, s in report.items():
        if method == "note":
            continue
        print(f"\n  {method}:")
        print(f"    覆盖雷达数:     {s['radars_with_triples']}/{s['total_radars']}")
        print(f"    总三元组数:     {s['total_triples']}")
        print(f"    平均每雷达:     {s['avg_triples_per_radar']}")
        print(f"    平均置信度:     {s['avg_confidence']}")
        top5 = list(s["relation_distribution"].items())[:5]
        print(f"    关系分布Top5:  {', '.join(f'{k}:{v}' for k,v in top5)}")
    print("\n  " + report["note"])
    print("═" * 60)


# ═══════════════════════════════════════════════════════
#  [修改13] 标注后计算指标（支持分关系类型统计）
# ═══════════════════════════════════════════════════════

def compute_metrics(annotation_path: str = None):
    """
    读取完成标注的 annotation_template.json，计算 P/R/F1。
    新增：per-relation 精确率，方便找出最弱的关系类型。
    """
    path = Path(annotation_path or OUTPUT_DIR / "annotation_template.json")
    with open(path, encoding="utf-8") as f:
        annotations = json.load(f)

    total = correct = wrong_entity = wrong_rel = hallucinated = unannotated = 0
    per_relation: dict[str, dict] = {}

    for item in annotations:
        for t in item.get("llm_fewshot_triples_to_annotate", []):
            label    = t.get("label", "")
            relation = t.get("relation", "unknown")

            total += 1
            per_relation.setdefault(relation, {"total": 0, "correct": 0})
            per_relation[relation]["total"] += 1

            if label == "correct":
                correct += 1
                per_relation[relation]["correct"] += 1
            elif label == "wrong_entity":
                wrong_entity += 1
            elif label == "wrong_relation":
                wrong_rel += 1
            elif label == "hallucinated":
                hallucinated += 1
            else:
                unannotated += 1

    human_added = sum(
        len(item.get("human_added_triples", []))
        for item in annotations
    )

    annotated = total - unannotated
    if annotated == 0:
        print("还没有完成标注，请先填写 label 字段")
        return

    precision  = correct / annotated if annotated else 0
    recall_den = correct + hallucinated + wrong_entity + wrong_rel + human_added
    recall     = correct / recall_den if recall_den else 0
    f1         = (2 * precision * recall / (precision + recall)
                  if (precision + recall) else 0)

    print("\n" + "═" * 55)
    print("  标注评测结果")
    print("═" * 55)
    print(f"  已标注三元组:   {annotated}")
    print(f"  正确:           {correct}  ({correct/annotated:.1%})")
    print(f"  实体错误:       {wrong_entity}")
    print(f"  关系错误:       {wrong_rel}")
    print(f"  幻觉:           {hallucinated}")
    print(f"  人工补充漏召:   {human_added}")
    print(f"\n  Precision:      {precision:.3f}")
    print(f"  Recall:         {recall:.3f}")
    print(f"  F1:             {f1:.3f}")

    print("\n  分关系类型精确率：")
    for rel, s in sorted(per_relation.items(),
                          key=lambda x: -x[1]["total"]):
        p = s["correct"] / s["total"] if s["total"] else 0
        print(f"    {rel:20s}: {p:.1%}  ({s['correct']}/{s['total']})")
    print("═" * 55)


# ═══════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)
    log.info(f"加载语料: {len(corpus)} 个雷达")

    # 方法A：规则抽取（全量）
    log.info("\n" + "═" * 55)
    log.info("运行方法A：规则抽取")
    log.info("═" * 55)
    extractor_a = RuleBasedExtractor()
    results_a   = run_extraction(
        corpus, extractor_a, "方法A",
        OUTPUT_DIR / "method_a_results.json"
    )

    # 方法B：LLM零样本（先跑50条）
    log.info("\n" + "═" * 55)
    log.info("运行方法B：LLM零样本抽取（前50条）")
    log.info("═" * 55)
    extractor_b = ZeroShotLLMExtractor()
    results_b   = run_extraction(
        corpus, extractor_b, "方法B",
        OUTPUT_DIR / "method_b_results.json",
        sample_size=50
    )

    # 方法C：LLM少样本（先跑50条）
    log.info("\n" + "═" * 55)
    log.info("运行方法C：LLM少样本抽取（前50条）")
    log.info("═" * 55)
    extractor_c = FewShotLLMExtractor()
    results_c   = run_extraction(
        corpus, extractor_c, "方法C",
        OUTPUT_DIR / "method_c_results.json",
        sample_size=50
    )

    # 生成对比报告
    report = generate_comparison_report(results_a, results_b, results_c)
    print_comparison(report)

    # 生成人工标注模板（同时传入三种结果）
    generate_annotation_template(corpus, results_a, results_b, results_c)

    log.info("\n✅ 抽取实验完成")
    log.info(f"结果目录: {OUTPUT_DIR.absolute()}")
    log.info("下一步: 填写 annotation_template.json 的 label 字段，")
    log.info("        然后运行 compute_metrics() 计算 Precision/Recall/F1")