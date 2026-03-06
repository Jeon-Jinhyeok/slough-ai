"""Contextual enrichment of decision-maker messages using LLM.

Implements Anthropic's "Contextual Retrieval" pattern adapted for Slack:
- LLM reads the full conversation flow
- Decision-maker messages are preserved verbatim
- Surrounding discussion is summarized as context
- Each output chunk is self-contained and semantically rich for embedding
"""

import logging
import re
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI

from src.config import settings

logger = logging.getLogger(__name__)

_WINDOW_SIZE = 100  # messages per LLM call
_WINDOW_OVERLAP = 20  # overlap between windows to avoid splitting conversations


_CONTEXTUALIZE_PROMPT = """\
아래는 Slack #{channel_name} 채널의 대화입니다.
의사결정자: {dm_name}

{conversation}

위 대화에서 의사결정자({dm_name})의 발언을 찾아, 각 발언(또는 같은 주제의 연속 발언)에 대해 아래 형식으로 출력하세요.

형식:
===
[대화 상황] (이 발언의 맥락을 2-3문장으로. 어떤 주제/안건에 대해 누가 어떤 의견/수치를 제시했는지, 핵심 정보 포함)
[참여자] (이 대화에 참여한 사람 이름, 쉼표로 구분)
[의사결정자 원문] (의사결정자 발언을 원문 그대로, 한 글자도 변경 없이. 연속 발언은 줄바꿈으로 구분)
===

규칙:
1. 의사결정자 발언은 반드시 원문 그대로 보존 (절대 수정/요약하지 않음)
2. 같은 주제에 대한 연속 발언은 하나의 블록으로 묶기
3. 대화 상황에 구체적 수치, 이름, A안/B안 등 핵심 내용 반드시 포함
4. 의사결정자 발언이 없는 구간은 출력하지 않음
5. 독립적인 지시/원칙 발언은 그대로 보존 (대화 상황: "채널에서 팀 전체에 전달한 원칙/지침")"""


def _format_conversation(
    messages: list[dict], user_names: dict[str, str],
) -> str:
    """Format raw Slack messages into readable conversation text."""
    lines = []
    for msg in messages:
        uid = msg.get("user", "unknown")
        name = user_names.get(uid, uid)
        ts = float(msg.get("ts", 0))
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        time_str = dt.strftime("%H:%M")
        text = msg.get("text", "")
        lines.append(f"[{time_str}] {name}: {text}")
    return "\n".join(lines)


def _parse_blocks(response_text: str) -> list[dict]:
    """Parse the LLM response into structured blocks."""
    blocks = []
    parts = re.split(r"={3,}", response_text)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        context_m = re.search(
            r"\[대화 상황\]\s*(.+?)(?=\[참여자\]|\[의사결정자 원문\]|$)",
            part, re.DOTALL,
        )
        participants_m = re.search(
            r"\[참여자\]\s*(.+?)(?=\[의사결정자 원문\]|$)",
            part, re.DOTALL,
        )
        verbatim_m = re.search(
            r"\[의사결정자 원문\]\s*(.+?)$",
            part, re.DOTALL,
        )

        if not verbatim_m:
            continue

        blocks.append({
            "context": context_m.group(1).strip() if context_m else "",
            "participants": participants_m.group(1).strip() if participants_m else "",
            "verbatim": verbatim_m.group(1).strip(),
        })

    return blocks


def _match_timestamp(
    verbatim: str,
    raw_messages: list[dict],
    decision_maker_id: str,
) -> str:
    """Try to find the original Slack timestamp for a verbatim message.

    Matches the first line of verbatim text against decision-maker messages.
    Falls back to empty string if no match found.
    """
    # Build lookup from message text -> ts
    text_to_ts: dict[str, str] = {}
    for msg in raw_messages:
        if msg.get("user") == decision_maker_id and msg.get("text"):
            text_to_ts[msg["text"].strip()] = msg["ts"]

    # Try matching the first line of verbatim text
    first_line = verbatim.split("\n")[0].strip()
    if first_line in text_to_ts:
        return text_to_ts[first_line]

    # Try matching the full verbatim (single message case)
    if verbatim.strip() in text_to_ts:
        return text_to_ts[verbatim.strip()]

    return ""


def _blocks_to_messages(
    blocks: list[dict],
    channel_id: str,
    channel_name: str,
    raw_messages: list[dict],
    decision_maker_id: str,
) -> list[dict]:
    """Convert parsed blocks into the ingestion message format."""
    messages = []
    for block in blocks:
        parts = []
        if channel_name:
            parts.append(f"[#{channel_name}]")
        if block["context"]:
            parts.append(f"[대화 상황] {block['context']}")
        if block["participants"]:
            parts.append(f"[참여자] {block['participants']}")
        parts.append(f"[의사결정자 원문] {block['verbatim']}")

        text = "\n".join(parts)
        ts = _match_timestamp(block["verbatim"], raw_messages, decision_maker_id)

        messages.append({
            "text": text,
            "channel": channel_id,
            "ts": ts,
        })
    return messages


def _fallback_messages(
    raw_messages: list[dict],
    decision_maker_id: str,
    channel_id: str,
    channel_name: str,
) -> list[dict]:
    """Fallback: save decision-maker messages without LLM context."""
    results = []
    for msg in raw_messages:
        if msg.get("user") == decision_maker_id and msg.get("text"):
            text = msg["text"]
            if channel_name:
                text = f"[#{channel_name}] {text}"
            results.append({
                "text": text,
                "channel": channel_id,
                "ts": msg["ts"],
            })
    return results


async def contextualize_messages(
    raw_messages: list[dict],
    decision_maker_id: str,
    user_names: dict[str, str],
    channel_id: str,
    channel_name: str = "",
) -> list[dict]:
    """Contextualize decision-maker messages using LLM.

    Reads the full conversation flow and produces self-contained chunks
    where each decision-maker statement includes surrounding discussion context.

    Args:
        raw_messages: All channel messages in chronological order.
        decision_maker_id: Slack user ID of the decision-maker.
        user_names: Mapping of user ID -> display name.
        channel_id: Channel ID for metadata.
        channel_name: Channel name for context prefix.

    Returns:
        List of contextualized messages ready for chunking/embedding.
        Format: [{"text": str, "channel": str, "ts": str}]
    """
    if not raw_messages:
        return []

    # Check if any decision-maker messages exist
    has_dm = any(msg.get("user") == decision_maker_id for msg in raw_messages)
    if not has_dm:
        return []

    dm_name = user_names.get(decision_maker_id, decision_maker_id)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=4000,
        api_key=settings.openai_api_key,
    )

    all_results: list[dict] = []
    seen_verbatim: set[str] = set()  # dedup across overlapping windows

    # Build windows
    total = len(raw_messages)
    if total <= _WINDOW_SIZE:
        windows = [(0, total)]
    else:
        windows = []
        start = 0
        while start < total:
            end = min(start + _WINDOW_SIZE, total)
            windows.append((start, end))
            if end >= total:
                break
            start = end - _WINDOW_OVERLAP

    for win_start, win_end in windows:
        window = raw_messages[win_start:win_end]

        # Skip windows with no decision-maker messages
        if not any(msg.get("user") == decision_maker_id for msg in window):
            continue

        conversation_text = _format_conversation(window, user_names)

        prompt = _CONTEXTUALIZE_PROMPT.format(
            channel_name=channel_name or channel_id,
            dm_name=dm_name,
            conversation=conversation_text,
        )

        try:
            response = await llm.ainvoke([{"role": "user", "content": prompt}])
            output = response.content or ""

            blocks = _parse_blocks(output)

            messages = _blocks_to_messages(
                blocks, channel_id, channel_name,
                window, decision_maker_id,
            )

            # Dedup: overlapping windows may produce duplicate blocks
            for msg in messages:
                verbatim_key = msg["text"]
                if verbatim_key not in seen_verbatim:
                    seen_verbatim.add(verbatim_key)
                    all_results.append(msg)

        except Exception:
            logger.exception(
                "Contextualization failed for window %d-%d in channel %s, "
                "falling back to raw messages",
                win_start, win_end, channel_id,
            )
            fallback = _fallback_messages(
                window, decision_maker_id, channel_id, channel_name,
            )
            for msg in fallback:
                if msg["text"] not in seen_verbatim:
                    seen_verbatim.add(msg["text"])
                    all_results.append(msg)

    logger.info(
        "Contextualized %d blocks from %d raw messages in #%s",
        len(all_results), len(raw_messages), channel_name or channel_id,
    )
    return all_results
