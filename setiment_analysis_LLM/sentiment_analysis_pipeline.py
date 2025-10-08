# -*- coding: utf-8 -*-
"""
annotate_sentiment_batched.py
-----------------------------
Simplified, fast batched annotator for WhatsApp group messages.

What’s simplified
- Always sends messages to the LLM in **batches of 10** (single HTTP call returns an array of 10 JSON objects).
- Concurrency is controlled by **--num-workers** (default=4).
- If an LLM batch parse fails, we **fallback to heuristics** for that batch (keeps things moving).

Still included
- Enum emotion + free-form `emotion_summary` (1–2 words).
- Evidence terms (LLM asked explicitly) + heuristic fallback.
- Reactions-aware polarity nudge, gratitude/help/info/stress detection, toxicity heuristic.
- Deterministic post-pass to smooth polarity and reconcile fields.
- Exports sidecar `.sentiment.json` per input and optional combined CSV.

Usage
-----
python annotate_sentiment_batched.py --input /path/to/chat.json --model gpt-4o-mini
python annotate_sentiment_batched.py --input ./folder_of_chats --combined-csv ./labels.csv --num-workers 8
python annotate_sentiment_batched.py --input chat.json --dry-run   # heuristics only, no LLM calls

Requires
--------
- Python 3.9+
- pip install openai
- export OPENAI_API_KEY=sk-...

"""

import argparse
import csv
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ----------------------------
# Configuration & Schema
# ----------------------------

BATCH_SIZE = 3  # fixed, per user request

SCHEMA_DEFAULTS = {
    "polarity": 0.0,  # -1..+1
    "emotion_primary": "neutral_info",  # enum
    "emotion_summary": "",  # free-form 1–2 words
    "stress_score": 0.0,  # 0..1
    "uncertainty_score": 0.0,  # 0..1
    "help_request": False,
    "helpfulness": 0.0,  # 0..1
    "gratitude": False,
    "toxicity_score": 0.0,  # 0..1
    "info_drop": False,
    "reaction_sentiment": None,  # {"positive": int, "neutral": int, "negative": int}
    "evidence_terms": [],  # up to 5 spans from the message
}

ALLOWED_EMOTIONS = {
    "stress",
    "gratitude",
    "confusion",
    "neutral_info",
    "humor",
    "anger",
    "excitement",
    "other",
}

EMOJI_POSITIVE = {  # Happy / Affection / Approval
    "😀",
    "😃",
    "😄",
    "😁",
    "😆",
    "😅",
    "😂",
    "🤣",
    "🥲",
    "☺️",
    "😊",
    "😇",
    "🙂",
    "😉",
    "😌",
    "😍",
    "🥰",
    "😘",
    "😗",
    "😙",
    "😚",
    "😋",
    "😎",
    "🤩",
    "🥳",
    # Playful / Silly (generally positive)
    "😛",
    "😝",
    "😜",
    "🤪",
    # Hugs / Salute / Support
    "🤗",
    "🫡",
    # Money / Cowboy good-vibes
    "🤑",
    "🤠",
    # Cats (positive ones)
    "😺",
    "😸",
    "😹",
    "😻",
    "😽",
    # Hearts & love cluster
    "❤️",
    "🩷",
    "🧡",
    "💛",
    "💚",
    "💙",
    "🩵",
    "💜",
    "🤍",
    "🤎",
    "💖",
    "💘",
    "💝",
    "💗",
    "💓",
    "💕",
    "💞",
    "❣️",
    "💟",
    "❤️‍🔥",
    "❤️‍🩹",
    # Gestures: good / celebrate
    "👌",
    "✌️",
    "🤞",
    "🫰",
    "🤟",
    "🤘",
    "👍",
    "👏",
    "🫶",
    "🙌",
    "👐",
    "🤝",
    "🙏",
    # Party & celebration
    "🎉",
    "🎊",
    "🥂",
    "🍾",
    "🏆",
    "💯",
    "✅",
    "✔️",
}


EMOJI_NEGATIVE = {  # Dislike / Anger / Sadness / Fear / Disgust
    "😒",
    "😞",
    "😔",
    "😟",
    "😕",
    "🙁",
    "☹️",
    "😣",
    "😖",
    "😫",
    "😩",
    "🥺",
    "😢",
    "😭",
    "😤",
    "😠",
    "😡",
    "🤬",
    "🥵",
    "🥶",
    "😱",
    "😨",
    "😰",
    "😥",
    "😓",
    "🤥",
    "🙄",
    "😬",
    # Ill/woozy/nausea
    "🥴",
    "🤢",
    "🤮",
    "🤧",
    # Devils/ogres/clown poop/death (commonly negative connotation)
    "😈",
    "👿",
    "👹",
    "👺",
    "🤡",
    "💩",
    "💀",
    "☠️",
    # Cats (negative ones)
    "🙀",
    "😿",
    "😾",
    # Explicit negative symbols
    "💔",
    "🚫",
    "⛔️",
    "🛑",
    "❌",
    "❎",
    "🔞",
    "🚭",
    "📵",
    "☣️",
    "☢️",
    "⚠️",
    # Thumbs down / middle finger
    "👎",
    "🖕",
}


THANKS_TOKENS_HE = ["תודה", "תודה רבה", "תודה לכולם", "תודהה", "תודההה"]
HELP_TOKENS_HE = [
    "עזרה",
    "יכול לעזור",
    "יכולה לעזור",
    "אשמח לעזרה",
    "מישהו יודע",
    "מישהי יודעת",
    "מישהו יכול",
    "מישהי יכולה",
]
INFO_TOKENS_HE = [
    "הודיעה",
    "פורום החדשות",
    "קישור",
    "מודל",
    "מבחן",
    "קוויז",
    "הגשה",
    "סילבוס",
]
STRESS_TOKENS_HE = [
    "דחוף",
    "דחופה",
    "לחוץ",
    "לחוצה",
    "נתקע",
    "נתקעה",
    "לא עובד",
    "לא עובדת",
    "תקוע",
]
HUMOR_TOKENS_HE = ["😂", "חח", "חחח", "חחחח", "😅"]

THANKS_TOKENS_EN = ["thanks", "thank you", "thx", "ty"]
HELP_TOKENS_EN = [
    "help",
    "anyone knows",
    "can someone",
    "could someone",
    "pls",
    "please",
]
INFO_TOKENS_EN = ["link", "deadline", "quiz", "assignment", "syllabus"]
STRESS_TOKENS_EN = ["urgent", "stuck", "not working", "blocked"]
HUMOR_TOKENS_EN = ["lol", "lmao", "haha", "😂", "😅"]

# Toxicity lexicon (minimal)
TOX_HE = ["מפגר", "דפוק", "טיפש", "סתום", "חרא", "מנייאק"]
TOX_EN = ["idiot", "stupid", "dumb", "moron", "wtf", "bs", "shit", "asshole", "fuck"]

# ----------------------------
# Prompting (batched)
# ----------------------------

SYSTEM_PROMPT = """You are a precise annotator for academic WhatsApp group chats (Hebrew & English).
Return a JSON ARRAY where each element corresponds to ONE input message in the same order.
Each element must be an object with fields:
- polarity: float in [-1.0, +1.0] (overall pleasantness).
- emotion_primary: one of {stress, gratitude, confusion, neutral_info, humor, anger, excitement, other}.
- emotion_summary: 1–2 words free-form (e.g., "stressed", "thankful", "confused", "calm", "excited").
- stress_score: float [0..1] (urgency/pressure).
- uncertainty_score: float [0..1] (doubt/confusion).
- help_request: boolean (explicit ask for help).
- helpfulness: float [0..1] (contribution to solving).
- gratitude: boolean (thanks or ❤️/🙏).
- toxicity_score: float [0..1] (hostile language).
- info_drop: boolean (links/dates/official notices).
- reaction_sentiment: leave as null; caller may fill.
- evidence_terms: up to 5 short spans copied from the message (e.g., "תודה", "❤️", "שאלה 5", "http", "???", "דחוף", "link").

Guidelines:
- Questions aren't inherently negative; use stress/uncertainty unless toxic.
- Gratitude implies positive polarity.
- Humor (😂/חח/lol) mildly positive but not "gratitude".
- Output strictly a JSON ARRAY of objects, no prose, no trailing commentary.
"""

USER_TEMPLATE_BATCH = """Annotate the following {n} messages. Return a JSON ARRAY of {n} objects in the same order.

{lines}
"""

JSON_ARRAY_RE = re.compile(r"\[.*\]\s*$", re.S)

# ----------------------------
# OpenAI client
# ----------------------------


def _load_openai_client():
    try:
        from openai import OpenAI  # type: ignore

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        return OpenAI()
    except Exception as e:
        logging.warning("OpenAI client not available: %s", e)
        return None


def call_llm_array(
    client,
    model: str,
    sys_prompt: str,
    user_prompt: str,
    temperature: float = 1.0,
    max_retries: int = 3,
) -> Optional[List[Dict[str, Any]]]:
    """Call Chat Completions and parse a JSON array; return list of dicts or None."""
    if client is None:
        return None
    last_err = None
    for i in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            content = resp.choices[0].message.content or ""
            m = JSON_ARRAY_RE.search(content.strip())
            if not m:
                raise ValueError("No JSON array found in response")
            arr = json.loads(m.group(0))
            if not isinstance(arr, list):
                raise ValueError("Parsed content is not a list")
            return arr
        except Exception as e:
            last_err = e
            logging.warning(
                "LLM batch call failed (attempt %d/%d): %s", i + 1, max_retries, e
            )
    logging.error("LLM batch ultimately failed: %s", last_err)
    return None


# ----------------------------
# Utilities & heuristics
# ----------------------------


def _norm_float(x: Any, lo=-1.0, hi=1.0, default=0.0) -> float:
    try:
        v = float(x)
        return max(lo, min(hi, v))
    except Exception:
        return float(default)


def _norm_float01(x: Any, default=0.0) -> float:
    return _norm_float(x, lo=0.0, hi=1.0, default=default)


def summarize_reactions(
    reacts: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, int]]:
    if not reacts:
        return None
    pos = sum(
        int(r.get("count", 0)) for r in reacts if r.get("emoji") in EMOJI_POSITIVE
    )
    neg = sum(
        int(r.get("count", 0)) for r in reacts if r.get("emoji") in EMOJI_NEGATIVE
    )
    neu = sum(
        int(r.get("count", 0))
        for r in reacts
        if r.get("emoji") not in (EMOJI_POSITIVE | EMOJI_NEGATIVE)
    )
    return {"positive": pos, "neutral": neu, "negative": neg}


def _tox_heuristic(text: str) -> float:
    t = text.lower()
    score = 0.0
    if any(w in t for w in TOX_EN) or any(w in text for w in TOX_HE):
        score = max(score, 0.5)
    if any(e in text for e in ("👎", "🤬", "💢")):
        score = max(score, 0.2)
    return score


ANCHOR_RE = re.compile(r"(?:שאלה|תרגיל)\s*\d+|quiz\s*\d+|Q\s*\d+|True|False", re.I)


def _has_concrete_anchor(text: str) -> bool:
    return bool(ANCHOR_RE.search(text or ""))


def _fallback_evidence_terms(out: Dict[str, Any], msg_text: str) -> List[str]:
    terms = out.get("evidence_terms") or []
    if terms:
        return terms[:5]
    cues: List[str] = []
    if "http" in msg_text:
        cues.append("http")
    if (
        any(tok in msg_text for tok in THANKS_TOKENS_HE)
        or any(tok in msg_text.lower() for tok in THANKS_TOKENS_EN)
        or any(e in msg_text for e in ("🙏", "❤️"))
    ):
        cues.append("תודה/❤️/🙏")
    if "?" in msg_text:
        cues.append("?")
    cues.extend(ANCHOR_RE.findall(msg_text)[:2])
    for w in STRESS_TOKENS_HE + STRESS_TOKENS_EN:
        if w in msg_text:
            cues.append(w)
            break
    for w in HUMOR_TOKENS_HE + HUMOR_TOKENS_EN:
        if w in msg_text:
            cues.append("humor/😂")
            break
    return list(dict.fromkeys(cues))[:5]


def apply_heuristics(base: Dict[str, Any], msg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(SCHEMA_DEFAULTS)
    out.update({k: v for k, v in (base or {}).items() if k in SCHEMA_DEFAULTS})

    text = (msg.get("body") or "").strip()
    reacts = msg.get("reactions") or []

    # Reaction sentiment -> small polarity nudge
    rs = summarize_reactions(reacts)
    if rs:
        out["reaction_sentiment"] = rs
        total = (rs["positive"] + rs["negative"]) or 1
        bump = 0.1 * (rs["positive"] - rs["negative"]) / total
        bump = max(-0.15, min(0.15, bump))
        out["polarity"] = max(-1.0, min(1.0, out["polarity"] + bump))

    # Gratitude detection
    if (
        any(tok in text for tok in THANKS_TOKENS_HE)
        or any(tok in text.lower() for tok in THANKS_TOKENS_EN)
        or any(e in text for e in {"❤️", "🙏"})
    ):
        out["gratitude"] = True
        out["polarity"] = max(out["polarity"], 0.6)

    # Info drop detection
    if (
        "http://" in text
        or "https://" in text
        or any(tok in text for tok in INFO_TOKENS_HE)
        or any(tok in text.lower() for tok in INFO_TOKENS_EN)
    ):
        out["info_drop"] = True
        out["helpfulness"] = max(out["helpfulness"], 0.4)

    # Help / uncertainty
    if (
        "?" in text
        or any(tok in text for tok in HELP_TOKENS_HE)
        or any(tok in text.lower() for tok in HELP_TOKENS_EN)
    ):
        out["uncertainty_score"] = max(out["uncertainty_score"], 0.6)
        if (
            text.strip().startswith(("מישהו", "מישהי"))
            or any(tok in text for tok in HELP_TOKENS_HE)
            or any(tok in text.lower() for tok in HELP_TOKENS_EN)
        ):
            out["help_request"] = True

    # Stress
    if (
        any(tok in text for tok in STRESS_TOKENS_HE)
        or any(tok in text.lower() for tok in STRESS_TOKENS_EN)
        or "!!!" in text
    ):
        out["stress_score"] = max(out["stress_score"], 0.6)
        out["polarity"] = min(out["polarity"], -0.2)

    # Humor
    if any(tok in text for tok in HUMOR_TOKENS_HE) or any(
        tok in text.lower() for tok in HUMOR_TOKENS_EN
    ):
        if out["emotion_primary"] == "neutral_info":
            out["emotion_primary"] = "humor"
        out["polarity"] = max(out["polarity"], 0.2)

    # Toxicity heuristic (rare but important)
    out["toxicity_score"] = max(
        _norm_float01(out.get("toxicity_score")), _tox_heuristic(text)
    )

    # Clamp numeric ranges
    out["polarity"] = _norm_float(out.get("polarity"))
    out["stress_score"] = _norm_float01(out.get("stress_score"))
    out["uncertainty_score"] = _norm_float01(out.get("uncertainty_score"))
    out["helpfulness"] = _norm_float01(out.get("helpfulness"))

    # evidence_terms fallback
    out["evidence_terms"] = _fallback_evidence_terms(out, text)

    # emotion_summary fallback
    if not out.get("emotion_summary"):
        if out.get("gratitude"):
            out["emotion_summary"] = "thankful"
        elif out.get("help_request"):
            out["emotion_summary"] = (
                "stressed" if out.get("stress_score", 0) >= 0.6 else "confused"
            )
        elif out.get("info_drop"):
            out["emotion_summary"] = "informative"
        elif out.get("emotion_primary") == "humor":
            out["emotion_summary"] = "playful"
        elif out.get("toxicity_score", 0) >= 0.5:
            out["emotion_summary"] = "hostile"
        else:
            out["emotion_summary"] = "neutral"

    return out


def _normalize_postpass(out: Dict[str, Any], msg_text: str) -> Dict[str, Any]:
    emo = out.get("emotion_primary") or "neutral_info"
    if emo not in ALLOWED_EMOTIONS:
        emo = "neutral_info"

    if out.get("gratitude"):
        emo = "gratitude"
        out["polarity"] = max(round(max(out.get("polarity", 0.0), 0.6), 2), 0.6)
        if not out.get("emotion_summary"):
            out["emotion_summary"] = "thankful"
    elif out.get("help_request"):
        emo = "stress" if out.get("stress_score", 0.0) >= 0.6 else "confusion"
        cap = 0.4 if _has_concrete_anchor(msg_text) else 0.2
        out["helpfulness"] = min(out.get("helpfulness", 0.0), cap)
        if out.get("toxicity_score", 0.0) < 0.2:
            out["polarity"] = max(-0.2, min(0.2, out.get("polarity", 0.0)))
        if not out.get("emotion_summary"):
            out["emotion_summary"] = "stressed" if emo == "stress" else "confused"

    pol = float(out.get("polarity", 0.0))
    lo = -1.0 if emo in {"anger"} else -0.8
    hi = 1.0 if emo in {"gratitude"} else 0.8
    out["polarity"] = round(max(lo, min(hi, pol)), 2)

    # trim emotion_summary to <=2 words
    es = (out.get("emotion_summary") or "").strip()
    out["emotion_primary"] = emo
    out["emotion_summary"] = " ".join(es.split()[:2]) if es else ""
    return out


# ----------------------------
# Batch machinery
# ----------------------------


def chunked(lst: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def build_batch_prompt(batch_msgs: List[Dict[str, Any]]) -> str:
    lines = []
    for i, m in enumerate(batch_msgs, 1):
        text = json.dumps(m.get("body") or "", ensure_ascii=False)
        reacts = json.dumps(m.get("reactions") or [], ensure_ascii=False)
        lines.append(
            f'{i}) id="{m.get("messageId","")}" time="{m.get("datetime","")}" text={text} reactions={reacts}'
        )
    return USER_TEMPLATE_BATCH.format(n=len(batch_msgs), lines="\n".join(lines))


def annotate_batch_with_llm(
    client, model: str, batch_msgs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Returns list of raw LLM dicts (len == batch size) or empty list on failure."""
    prompt = build_batch_prompt(batch_msgs)
    arr = call_llm_array(client, model, SYSTEM_PROMPT, prompt)
    if not arr or len(arr) != len(batch_msgs):
        return []  # signal failure
    # ensure dicts
    clean = []
    for obj in arr:
        if isinstance(obj, dict):
            clean.append(obj)
        else:
            clean.append({})
    return clean


def annotate_batch(
    dry_run: bool, client, model: str, batch_msgs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    # 1) Try LLM (unless dry-run)
    llm_outputs: List[Dict[str, Any]] = []
    if not dry_run:
        llm_outputs = annotate_batch_with_llm(client, model, batch_msgs)
    # 2) If failed or dry-run: fall back to empty dicts (heuristics will fill)
    if not llm_outputs:
        llm_outputs = [{} for _ in batch_msgs]

    # 3) Heuristics + post-pass + attach QA fields
    out_rows: List[Dict[str, Any]] = []
    for msg, base in zip(batch_msgs, llm_outputs):
        row = apply_heuristics(base, msg)
        row = _normalize_postpass(row, msg.get("body") or "")
        # Persist identifiers & minimal provenance
        row["message_id"] = msg.get("messageId") or ""
        row["timestamp"] = msg.get("datetime") or ""
        row["body"] = msg.get("body") or ""
        row["serial_number"] = msg.get("serialNumber")
        snd = msg.get("sender")
        row["sender_id"] = (snd.get("phone") if isinstance(snd, dict) else snd) or ""
        if msg.get("replyTo"):
            row["reply_to_ref"] = msg["replyTo"].get("ref", "")
            row["reply_to_quote"] = msg["replyTo"].get("body", "")
        else:
            row["reply_to_ref"] = ""
            row["reply_to_quote"] = ""
        out_rows.append(row)
    return out_rows


# ----------------------------
# IO & Orchestration
# ----------------------------


def _load_existing_annotations(out_path: Path) -> List[Dict[str, Any]]:
    if not out_path.exists():
        return []
    try:
        return json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _write_sorted_annotations(
    out_path: Path, rows_by_id: Dict[str, Dict[str, Any]], id_to_serial: Dict[str, int]
) -> List[Dict[str, Any]]:
    rows = list(rows_by_id.values())

    def _serial_of(r: Dict[str, Any]) -> int:
        sn = r.get("serial_number")
        if isinstance(sn, int):
            return sn
        mid = r.get("message_id")
        return int(id_to_serial.get(mid, 0))

    rows.sort(key=_serial_of)
    out_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rows


def process_chat_file(
    path: Path, model: str, num_workers: int, dry_run: bool, resume: bool = True
) -> Tuple[Path, List[Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    messages: List[Dict[str, Any]] = data.get("messages", [])

    # Build id->serial map for ordering and resume decisions
    id_to_serial: Dict[str, int] = {}
    for m in messages:
        mid = m.get("messageId")
        if mid:
            try:
                id_to_serial[mid] = int(m.get("serialNumber") or 0)
            except Exception:
                id_to_serial[mid] = 0

    # Output path naming: <input_stem>_<model>_sentiment.json in same directory
    def _safe_model_tag(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]", "-", s)

    model_tag = _safe_model_tag(model)
    out_path = path.parent / f"{path.stem}_{model_tag}_sentiment.json"
    legacy_out_path = path.with_suffix(".sentiment.json")

    # Seed with existing annotations (if resuming): prefer new-named file, but also merge legacy
    existing_rows_new: List[Dict[str, Any]] = (
        _load_existing_annotations(out_path) if resume else []
    )
    existing_rows_legacy: List[Dict[str, Any]] = (
        _load_existing_annotations(legacy_out_path) if resume else []
    )
    rows_by_id: Dict[str, Dict[str, Any]] = {}
    processed_ids: set = set()
    max_serial_seen: int = -1
    for r in existing_rows_new + existing_rows_legacy:
        mid = r.get("message_id")
        if not mid:
            continue
        rows_by_id[mid] = r
        processed_ids.add(mid)
        sn = r.get("serial_number")
        if isinstance(sn, int):
            max_serial_seen = max(max_serial_seen, sn)
        else:
            max_serial_seen = max(max_serial_seen, int(id_to_serial.get(mid, -1)))

    # Decide which messages to process next, preserving original order
    pending_messages: List[Dict[str, Any]] = []
    if processed_ids:
        for m in messages:
            mid = m.get("messageId")
            sn = int(m.get("serialNumber") or -1)
            if mid in processed_ids:
                continue
            if sn <= max_serial_seen:
                continue
            pending_messages.append(m)
    else:
        pending_messages = messages[:]

    if not pending_messages:
        # Nothing to do; ensure existing annotations are sorted and returned
        final_rows = (
            _write_sorted_annotations(out_path, rows_by_id, id_to_serial)
            if rows_by_id
            else existing_rows_new or existing_rows_legacy
        )
        return out_path, final_rows

    client = None if dry_run else _load_openai_client()

    # Build batches of 10 from pending only
    batches: List[List[Dict[str, Any]]] = list(chunked(pending_messages, BATCH_SIZE))

    def _job(idx: int, batch_msgs: List[Dict[str, Any]]):
        return idx, batch_msgs, annotate_batch(dry_run, client, model, batch_msgs)

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futs = [ex.submit(_job, i, b) for i, b in enumerate(batches)]
        for f in as_completed(futs):
            _, batch_msgs, rows = f.result()
            # Merge rows into accumulator and autosave
            for msg, row in zip(batch_msgs, rows or []):
                mid = row.get("message_id") or msg.get("messageId") or ""
                if not mid:
                    continue
                # ensure serial_number present for stable ordering
                if row.get("serial_number") is None:
                    row["serial_number"] = msg.get("serialNumber")
                rows_by_id[mid] = row
            _write_sorted_annotations(out_path, rows_by_id, id_to_serial)

    # Finalize and return
    final_rows = _write_sorted_annotations(out_path, rows_by_id, id_to_serial)
    return out_path, final_rows


def iter_input_paths(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
    else:
        for p in sorted(input_path.glob("*.json")):
            # Skip any generated sentiment outputs
            if p.name.endswith(".sentiment.json") or p.name.endswith("_sentiment.json"):
                continue
            yield p


def write_combined_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    # Flatten reaction_sentiment
    for r in rows:
        rs = r.get("reaction_sentiment") or {}
        r["reactions_pos"] = rs.get("positive", 0)
        r["reactions_neu"] = rs.get("neutral", 0)
        r["reactions_neg"] = rs.get("negative", 0)
        r.pop("reaction_sentiment", None)

    field_order = [
        "message_id",
        "timestamp",
        "sender_id",
        "polarity",
        "emotion_primary",
        "emotion_summary",
        "stress_score",
        "uncertainty_score",
        "help_request",
        "helpfulness",
        "gratitude",
        "toxicity_score",
        "info_drop",
        "reactions_pos",
        "reactions_neu",
        "reactions_neg",
        "evidence_terms",
        "reply_to_ref",
        "reply_to_quote",
        "body",
    ]

    for r in rows:
        for k in field_order:
            if k not in r:
                r[k] = (
                    ""
                    if k
                    in (
                        "message_id",
                        "timestamp",
                        "sender_id",
                        "emotion_primary",
                        "emotion_summary",
                        "evidence_terms",
                        "reply_to_ref",
                        "reply_to_quote",
                        "body",
                    )
                    else 0
                )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ----------------------------
# CLI
# ----------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Batched (10) WhatsApp sentiment annotator."
    )
    parser.add_argument(
        "--input", required=True, help="Path to chat.json or folder of JSON files"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini", help="OpenAI chat model to use"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of concurrent LLM batches (default=4)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Heuristics-only mode (no LLM calls)"
    )
    parser.add_argument(
        "--combined-csv",
        type=str,
        default="",
        help="Optional path to write a combined CSV of all labels",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume from existing .sentiment.json (default resumes)",
    )
    parser.set_defaults(resume=True)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    input_path = Path(args.input)
    if not input_path.exists():
        logging.error("Input path not found: %s", input_path)
        sys.exit(2)

    all_rows: List[Dict[str, Any]] = []
    for p in iter_input_paths(input_path):
        logging.info("Processing: %s", p)
        out_path, annots = process_chat_file(
            p,
            model=args.model,
            num_workers=args.num_workers,
            dry_run=args.dry_run,
            resume=args.resume,
        )
        logging.info("Wrote: %s", out_path)
        all_rows.extend(annots)

    if args.combined_csv:
        csv_path = Path(args.combined_csv)
        write_combined_csv(all_rows, csv_path)
        logging.info("Combined CSV: %s", csv_path)


if __name__ == "__main__":
    main()

    # USAGE:   python /Users/jonatanvider/Documents/HiddenRepliesGPTeval/annotate_sentiment_batched.py --input /Users/jonatanvider/Documents/HiddenRepliesGPTeval/DB_2025.json --model gpt-5

