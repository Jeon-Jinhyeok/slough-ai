"""Ingestion orchestrator — fetches Slack history and passes to AI pipeline."""

import asyncio
import logging
import uuid

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from src.services.ai import ingest_messages
from src.services.db.connection import get_db
from src.services.db.ingestion_jobs import (
    create_ingestion_job,
    mark_job_completed,
    mark_job_failed,
    mark_job_running,
    update_ingestion_job,
)
from src.services.db.workspaces import get_workspace_by_team_id, update_workspace
from src.services.slack.conversations import (
    fetch_channel_history,
    join_channel,
    list_bot_channels,
)

logger = logging.getLogger(__name__)

# Process messages in batches to avoid memory issues with large histories
_BATCH_SIZE = 100


def run_ingestion(team_id: str, channel_ids: list[str] | None = None, incremental: bool = False) -> None:
    """Run the full ingestion pipeline for a workspace.

    1. Look up workspace and create a job record
    2. Resolve channels (user-selected or all bot channels)
    3. Fetch decision-maker messages from each channel
    4. Pass messages to AI pipeline in batches
    5. Mark job complete, extract persona, and notify decision-maker

    Args:
        team_id: Slack team ID of the workspace to ingest.
        channel_ids: Specific channel IDs to ingest. If None, all bot channels.
        incremental: If True, only fetch messages since the last successful
                     ingestion (avoids duplicates).
    """
    # 1. Look up workspace
    with get_db() as db:
        workspace = get_workspace_by_team_id(db, team_id)
        if workspace is None:
            logger.error("Workspace not found for team %s", team_id)
            return

        workspace_id = workspace.id
        bot_token = workspace.bot_token
        decision_maker_id = workspace.decision_maker_id

        # Determine oldest timestamp for incremental ingestion
        oldest = 0
        if incremental:
            from src.services.db.ingestion_jobs import get_latest_job as _get_latest
            latest = _get_latest(db, workspace_id)
            if latest and latest.status == "completed" and latest.completed_at:
                oldest = latest.completed_at.timestamp()
                logger.info(
                    "Incremental ingestion: fetching messages after %s",
                    latest.completed_at.isoformat(),
                )

        # Create job record
        job = create_ingestion_job(db, workspace_id=workspace_id)
        job_id = job.id

    client = WebClient(token=bot_token)

    # 2. Resolve channels
    try:
        if channel_ids:
            # User selected specific channels — get their names for logging
            channels = []
            for cid in channel_ids:
                try:
                    info = client.conversations_info(channel=cid)
                    channels.append({"id": cid, "name": info["channel"]["name"]})
                except Exception:
                    channels.append({"id": cid, "name": cid})
        else:
            channels = list_bot_channels(client)
    except Exception as e:
        with get_db() as db:
            mark_job_failed(db, job_id, f"Failed to resolve channels: {e}")
        return

    with get_db() as db:
        mark_job_running(db, job_id, total_channels=len(channels))

    if not channels:
        logger.warning("No channels found for team %s", team_id)
        with get_db() as db:
            mark_job_completed(db, job_id, total_messages=0, processed_messages=0)
        _notify_completion(client, decision_maker_id, 0, 0)
        return

    # 3. Fetch messages from each channel
    all_messages = []
    channels_processed = 0

    for ch in channels:
        logger.info("Ingesting #%s (%s)", ch["name"], ch["id"])

        # Auto-join the channel before fetching history
        if not join_channel(client, ch["id"]):
            logger.warning("Could not join #%s, skipping", ch["name"])
            channels_processed += 1
            continue

        try:
            msgs = fetch_channel_history(
                client,
                channel_id=ch["id"],
                decision_maker_id=decision_maker_id,
                oldest=oldest,
                channel_name=ch["name"],
            )
            all_messages.extend(msgs)
        except Exception:
            logger.exception("Error fetching channel %s, continuing", ch["id"])

        channels_processed += 1
        with get_db() as db:
            update_ingestion_job(db, job_id, processed_channels=channels_processed)

    total_messages = len(all_messages)
    logger.info("Collected %d messages from %d channels", total_messages, channels_processed)

    if not all_messages:
        with get_db() as db:
            mark_job_completed(db, job_id, total_messages=0, processed_messages=0)
        # Still re-extract persona even with 0 new messages (prompt may have changed)
        try:
            from src.services.ai.persona_extractor import extract_persona
            extract_persona(str(workspace_id))
            logger.info("Persona re-extracted (0 new messages) for workspace %s", workspace_id)
        except Exception:
            logger.exception("Persona extraction failed for workspace %s", workspace_id)
        _notify_completion(client, decision_maker_id, 0, 0)
        return

    # 4. Pass to AI pipeline in batches
    processed = 0
    try:
        for i in range(0, total_messages, _BATCH_SIZE):
            batch = all_messages[i : i + _BATCH_SIZE]

            asyncio.run(ingest_messages(
                workspace_id=str(workspace_id),
                messages=batch,
            ))

            processed += len(batch)
            with get_db() as db:
                update_ingestion_job(
                    db, job_id,
                    total_messages=total_messages,
                    processed_messages=processed,
                )

        logger.info("Ingestion complete: %d messages processed", processed)
    except Exception as e:
        logger.exception("Ingestion failed at message %d/%d", processed, total_messages)
        with get_db() as db:
            mark_job_failed(db, job_id, str(e))
        _notify_failure(client, decision_maker_id, str(e))
        return

    # 5. Mark complete and notify
    with get_db() as db:
        mark_job_completed(db, job_id, total_messages=total_messages, processed_messages=processed)
        update_workspace(db, workspace_id, onboarding_completed=True)

    # 6. Extract persona from ingested messages
    try:
        from src.services.ai.persona_extractor import extract_persona
        extract_persona(str(workspace_id))
        logger.info("Persona extracted after ingestion for workspace %s", workspace_id)
    except Exception:
        logger.exception("Persona extraction failed after ingestion for workspace %s", workspace_id)

    _notify_completion(client, decision_maker_id, total_messages, channels_processed)


def _notify_completion(
    client: WebClient,
    user_id: str,
    total_messages: int,
    channels_processed: int,
) -> None:
    """Send a DM to the decision-maker when ingestion is done."""
    logger.info("Sending completion DM to %s", user_id)
    try:
        dm = client.conversations_open(users=[user_id])
        channel_id = dm["channel"]["id"]
        client.chat_postMessage(
            channel=channel_id,
            text=(
                f"학습이 완료되었습니다!\n\n"
                f"- 분석한 채널: {channels_processed}개\n"
                f"- 학습한 메시지: {total_messages}개\n\n"
                f"이제 팀원들이 DM으로 질문을 보내면, "
                f"의사결정자님의 사고 방식을 반영한 답변을 제공합니다."
            ),
        )
        logger.info("Completion DM sent to %s", user_id)
    except Exception:
        logger.exception("Failed to send completion DM to %s", user_id)


def _notify_failure(client: WebClient, user_id: str, error: str) -> None:
    """Send a DM to the decision-maker when ingestion fails."""
    try:
        dm = client.conversations_open(users=[user_id])
        channel_id = dm["channel"]["id"]
        client.chat_postMessage(
            channel=channel_id,
            text=(
                f"학습 중 오류가 발생했습니다.\n\n"
                f"오류 내용: {error}\n\n"
                f"다시 시도하려면 관리자에게 문의해 주세요."
            ),
        )
    except SlackApiError:
        logger.exception("Failed to send failure DM to %s", user_id)
