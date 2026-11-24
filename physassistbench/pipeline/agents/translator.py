"""
translator.py — Lightweight LLM-based translation utility for bilingual benchmark generation.

Translates clinical text between English and Chinese.
Uses the shared llm_client so that any --generation_model override also applies here.
"""

import logging

from physassistbench.pipeline.agents.llm_client import llm_call, extract_json

logger = logging.getLogger(__name__)


_TRANSLATE_TO_ZH_PROMPT = (
    "You are a professional medical translator specializing in clinical EHR queries. "
    "Your ONLY task is to translate the given text from English to Chinese. "
    "NEVER answer questions, explain concepts, or add any content not present in the original. "
    "Preserve all medical terminology and clinical meaning exactly. "
    "Output ONLY the translated Chinese text, nothing else.\n\n"
    "CRITICAL RULE for clinical value queries:\n"
    "English 'What is [lab/vital/drug]?' in a clinical EHR context means asking for the VALUE, "
    "NOT asking for a definition. Translate using the pattern '[item]是多少？' or '[item]的值是多少？', "
    "NOT '什么是[item]？'.\n"
    "Examples:\n"
    "  'What is the creatinine?' → '肌酐是多少？'\n"
    "  'What is the hemoglobin A1c?' → '糖化血红蛋白是多少？'\n"
    "  'What is the potassium?' → '血钾是多少？'\n"
    "  'What is the latest WBC?' → '最新的白细胞计数是多少？'"
)

_TRANSLATE_TO_EN_PROMPT = (
    "You are a professional medical translator. "
    "Your ONLY task is to translate the given text from Chinese to English word-for-word. "
    "NEVER answer questions, explain concepts, or add any content not present in the original. "
    "If the input is a question, output the English translation of that question. "
    "If the input is a statement, output the English translation of that statement. "
    "Preserve all medical terminology and clinical meaning exactly. "
    "Output ONLY the translated English text, nothing else."
)


def translate_to_zh(text: str) -> str:
    """Translate English clinical text to Chinese."""
    if not text or not text.strip():
        return text
    try:
        return llm_call(
            messages=[
                {"role": "system", "content": _TRANSLATE_TO_ZH_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.1,
            max_tokens=4000,
        ) or text
    except Exception as e:
        logger.warning(f"translate_to_zh failed: {e} — returning original text")
        return text


def translate_to_en(text: str) -> str:
    """Translate Chinese clinical text to English."""
    if not text or not text.strip():
        return text
    try:
        return llm_call(
            messages=[
                {"role": "system", "content": _TRANSLATE_TO_EN_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.1,
            max_tokens=4000,
        ) or text
    except Exception as e:
        logger.warning(f"translate_to_en failed: {e} — returning original text")
        return text


def translate_messages(messages: list, target_language: str) -> list:
    """
    Translate user AND assistant message content to target_language.
    SEP markers (plain strings) are passed through unchanged.
    """
    translate_fn = translate_to_zh if target_language == "zh" else translate_to_en
    result = []
    for msg in messages:
        if isinstance(msg, str):
            result.append(msg)
        elif isinstance(msg, dict) and msg.get("role") in ("user", "assistant"):
            content = msg.get("content") or ""
            translated = translate_fn(content) if content.strip() else content
            result.append({**msg, "content": translated})
        else:
            result.append(msg)
    return result


# ── Round-trip translation with semantic verification (paper App. A.4) ────────
# "Chinese queries are produced via a round-trip translation pipeline with
#  semantic verification": EN → ZH, back-translate ZH → EN, then check that the
#  back-translation asks for the same thing as the original; retry on mismatch.

_VERIFY_EQUIVALENCE_PROMPT = (
    "You verify clinical query translations. You are given an ORIGINAL English "
    "physician query and a BACK-TRANSLATION (English → Chinese → English). "
    "Decide whether the two are semantically equivalent: same requested data items, "
    "same drugs / labs / values / units, same clinical intent, and the same degree of "
    "implicitness (a pronoun or omitted element must stay implicit; nothing may be "
    "added or dropped). Wording and word order may differ. "
    'Return ONLY JSON: {"equivalent": true|false, "issue": "<what differs, or empty>"}'
)

MAX_TRANSLATION_ATTEMPTS = 3


def verify_round_trip(original_en: str, back_translation_en: str) -> tuple[bool, str]:
    """Semantic check of a round-trip translation. Returns (equivalent, issue).

    Fails closed on a parse/LLM error so the caller retries the translation.
    """
    try:
        raw = llm_call(
            messages=[
                {"role": "system", "content": _VERIFY_EQUIVALENCE_PROMPT},
                {"role": "user", "content": (
                    f"ORIGINAL: {original_en}\nBACK-TRANSLATION: {back_translation_en}")},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        result = extract_json(raw)
        if isinstance(result, dict):
            return bool(result.get("equivalent", False)), str(result.get("issue", ""))
    except Exception as e:
        logger.warning(f"verify_round_trip failed: {e}")
    return False, "verification unavailable"


def translate_query_to_zh_verified(text: str) -> str:
    """Translate an English physician query to Chinese with round-trip verification.

    Up to MAX_TRANSLATION_ATTEMPTS translations are tried; the first whose
    back-translation is judged equivalent to the original is returned. If none
    verifies, the last candidate is returned and a warning is logged.
    """
    if not text or not text.strip():
        return text
    candidate = text
    issue = ""
    for attempt in range(MAX_TRANSLATION_ATTEMPTS):
        hint = (f"\n\n(A previous translation was rejected: {issue}. "
                "Translate again, preserving the meaning exactly.)") if issue else ""
        try:
            candidate = llm_call(
                messages=[
                    {"role": "system", "content": _TRANSLATE_TO_ZH_PROMPT},
                    {"role": "user", "content": text + hint},
                ],
                temperature=0.1 if attempt == 0 else 0.3,
                max_tokens=4000,
            ) or text
        except Exception as e:
            logger.warning(f"translate_query_to_zh_verified: translation failed: {e}")
            issue = "translation call failed"
            continue
        back = translate_to_en(candidate)
        ok, issue = verify_round_trip(text, back)
        if ok:
            return candidate
        logger.info(f"  round-trip check failed (attempt {attempt + 1}): {issue[:120]}")
    logger.warning(f"translate_query_to_zh_verified: no verified translation for {text[:60]!r}")
    return candidate


def add_zh_patient_responses(executed_actions: list) -> None:
    """Store Chinese versions of the pre-generated patient replies, in place.

    The paper provides patient-simulation responses in both languages; the
    evaluator replays `patient_response_zh` / `patient_responses_all_literacy_zh`
    for Chinese sessions.
    """
    for act in executed_actions or []:
        obs = act.get("observation")
        if not isinstance(obs, dict) or "patient_response" not in obs:
            continue
        if obs.get("patient_response") and not obs.get("patient_response_zh"):
            obs["patient_response_zh"] = translate_to_zh(obs["patient_response"])
        lit = obs.get("patient_responses_all_literacy")
        if isinstance(lit, dict) and not obs.get("patient_responses_all_literacy_zh"):
            obs["patient_responses_all_literacy_zh"] = {
                level: (obs["patient_response_zh"]
                        if text == obs.get("patient_response") and obs.get("patient_response_zh")
                        else translate_to_zh(text))
                for level, text in lit.items()
            }
