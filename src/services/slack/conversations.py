"""Slack conversation helpers — channel listing and message history fetching."""

import logging
import time

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

logger = logging.getLogger(__name__)

# Slack API rate limit: ~50 req/min for conversations.history (Tier 3)
# We add a small delay between paginated calls to stay safe.
_PAGE_DELAY_SECONDS = 0.5


def join_channel(client: WebClient, channel_id: str) -> bool:
    """Join a public channel. Returns True if successful or already joined."""
    try:
        client.conversations_join(channel=channel_id)
        logger.info("Bot joined channel %s", channel_id)
        return True
    except SlackApiError as e:
        if e.response.get("error") == "already_in_channel":
            return True
        logger.exception("Failed to join channel %s", channel_id)
        return False


def list_bot_channels(client: WebClient) -> list[dict]:
    """List all public channels the bot has been added to.

    Returns:
        List of {"id": str, "name": str} dicts.
    """
    channels = []
    cursor = None

    while True:
        try:
            resp = client.conversations_list(
                types="public_channel",
                exclude_archived=True,
                limit=200,
                cursor=cursor or "",
            )
        except SlackApiError:
            logger.exception("Failed to list channels")
            break

        for ch in resp.get("channels", []):
            if ch.get("is_member"):
                channels.append({"id": ch["id"], "name": ch["name"]})

        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(_PAGE_DELAY_SECONDS)

    logger.info("Found %d channels bot is a member of", len(channels))
    return channels


_SHORT_MSG_THRESHOLD = 80  # chars — short DM messages get surrounding context
_CONTEXT_WINDOW = 3  # preceding messages to include for context


def resolve_user_names(client: WebClient, user_ids: set[str]) -> dict[str, str]:
    """Resolve Slack user IDs to display names.

    Args:
        client: Slack WebClient with bot token.
        user_ids: Set of Slack user IDs to resolve.

    Returns:
        Mapping of user_id -> display name. Falls back to user_id on error.
    """
    names: dict[str, str] = {}
    for uid in user_ids:
        try:
            resp = client.users_info(user=uid)
            profile = resp["user"].get("profile", {})
            name = (
                profile.get("display_name")
                or profile.get("real_name")
                or resp["user"].get("name", uid)
            )
            names[uid] = name
        except Exception:
            names[uid] = uid
        time.sleep(0.1)  # rate limit courtesy
    logger.info("Resolved %d user names", len(names))
    return names


def fetch_channel_messages_raw(
    client: WebClient,
    channel_id: str,
    oldest: float = 0,
    limit_per_page: int = 200,
) -> list[dict]:
    """Fetch ALL messages from a channel in chronological order.

    Returns all non-system text messages (from all users), sorted oldest-first.
    Used by the contextualizer to read the full conversation flow.

    Args:
        client: Slack WebClient with bot token.
        channel_id: Channel to fetch from.
        oldest: Unix timestamp - only fetch messages after this time.
        limit_per_page: Messages per API call (max 200).

    Returns:
        List of raw Slack message dicts, sorted oldest-first.
    """
    all_raw: list[dict] = []
    cursor = None

    while True:
        try:
            kwargs: dict = {"channel": channel_id, "limit": limit_per_page}
            if oldest:
                kwargs["oldest"] = str(oldest)
            if cursor:
                kwargs["cursor"] = cursor
            resp = client.conversations_history(**kwargs)
        except SlackApiError as e:
            if e.response.get("error") == "not_in_channel":
                logger.warning("Bot not in channel %s, skipping", channel_id)
                return []
            logger.exception("Failed to fetch history for channel %s", channel_id)
            break

        all_raw.extend(resp.get("messages", []))

        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(_PAGE_DELAY_SECONDS)

    # Reverse to chronological order (API returns newest-first)
    all_raw.reverse()

    # Filter out system messages, keep only text messages
    filtered = [
        msg for msg in all_raw
        if msg.get("text") and not msg.get("subtype")
    ]
    logger.info(
        "Fetched %d raw messages from channel %s", len(filtered), channel_id,
    )
    return filtered


def fetch_channel_history(
    client: WebClient,
    channel_id: str,
    decision_maker_id: str,
    oldest: float = 0,
    limit_per_page: int = 200,
    channel_name: str = "",
) -> list[dict]:
    """Fetch all messages from a channel authored by the decision-maker.

    Paginates through the full history (or from `oldest` timestamp).
    Only includes messages with text content from the decision-maker.

    For short decision-maker messages (< 80 chars) in the main channel flow,
    preceding messages are included as context so that replies like "동의합니다"
    retain what was agreed to.  The decision-maker's message is always primary.

    Args:
        client: Slack WebClient with bot token.
        channel_id: Channel to fetch from.
        decision_maker_id: Slack user ID to filter by.
        oldest: Unix timestamp — only fetch messages after this time. 0 = all history.
        limit_per_page: Messages per API call (max 200).
        channel_name: Channel name to include as context prefix (e.g. "general").

    Returns:
        List of dicts matching the AI contract format:
        [{"text": str, "channel": str, "ts": str, "thread_ts"?: str}]
    """
    # First pass: collect ALL messages (not just DM) for context lookback.
    # conversations.history returns newest-first, so higher indices = older.
    all_raw: list[dict] = []
    cursor = None

    while True:
        try:
            kwargs = {
                "channel": channel_id,
                "limit": limit_per_page,
            }
            if oldest:
                kwargs["oldest"] = str(oldest)
            if cursor:
                kwargs["cursor"] = cursor

            resp = client.conversations_history(**kwargs)
        except SlackApiError as e:
            if e.response.get("error") == "not_in_channel":
                logger.warning("Bot not in channel %s, skipping", channel_id)
                return []
            logger.exception("Failed to fetch history for channel %s", channel_id)
            break

        all_raw.extend(resp.get("messages", []))

        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(_PAGE_DELAY_SECONDS)

    # Second pass: filter DM messages, add surrounding context for short ones.
    messages = []

    for i, msg in enumerate(all_raw):
        if msg.get("user") != decision_maker_id:
            continue
        if not msg.get("text"):
            continue
        if msg.get("subtype"):
            continue

        text = msg["text"]
        is_thread_reply = msg.get("thread_ts") and msg["thread_ts"] != msg["ts"]

        # Thread reply: fetch parent for Q&A context (existing logic)
        if is_thread_reply:
            try:
                thread = client.conversations_replies(
                    channel=channel_id, ts=msg["thread_ts"], limit=1,
                )
                parent = thread["messages"][0] if thread.get("messages") else None
                if parent and parent.get("user") != decision_maker_id and parent.get("text"):
                    text = f"[질문] {parent['text']}\n[답변] {msg['text']}"
            except Exception:
                pass

        # Short non-thread message: add preceding messages as context.
        # all_raw is newest-first, so i+1, i+2, ... are older (came before).
        elif len(msg["text"]) < _SHORT_MSG_THRESHOLD:
            preceding: list[str] = []
            for j in range(i + 1, min(i + 1 + _CONTEXT_WINDOW, len(all_raw))):
                prev = all_raw[j]
                if prev.get("text") and not prev.get("subtype"):
                    preceding.append(prev["text"])
            if preceding:
                preceding.reverse()  # chronological order (oldest first)
                context_text = "\n".join(preceding)
                text = f"[맥락]\n{context_text}\n[의사결정자 답변] {msg['text']}"

        if channel_name:
            text = f"[#{channel_name}] {text}"

        entry = {
            "text": text,
            "channel": channel_id,
            "ts": msg["ts"],
        }
        if is_thread_reply:
            entry["thread_ts"] = msg["thread_ts"]
        messages.append(entry)

    logger.info(
        "Fetched %d decision-maker messages from channel %s",
        len(messages), channel_id,
    )
    return messages


def fetch_all_workspace_history(
    client: WebClient,
    decision_maker_id: str,
    oldest: float = 0,
) -> tuple[list[dict], int]:
    """Fetch decision-maker messages from all channels the bot is in.

    Args:
        client: Slack WebClient with bot token.
        decision_maker_id: Slack user ID of the decision-maker.
        oldest: Unix timestamp — only fetch messages after this time.

    Returns:
        (messages, channels_processed) — messages in AI contract format, channel count.
    """
    channels = list_bot_channels(client)
    all_messages = []

    for ch in channels:
        logger.info("Fetching history from #%s (%s)", ch["name"], ch["id"])
        msgs = fetch_channel_history(
            client,
            channel_id=ch["id"],
            decision_maker_id=decision_maker_id,
            oldest=oldest,
            channel_name=ch["name"],
        )
        all_messages.extend(msgs)

    logger.info(
        "Total: %d decision-maker messages from %d channels",
        len(all_messages), len(channels),
    )
    return all_messages, len(channels)
