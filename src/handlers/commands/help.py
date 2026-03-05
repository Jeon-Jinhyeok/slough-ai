"""Handler for /slough-help (English) and /slough-help-kr (Korean)."""

import logging

from src.services.db import get_db
from src.services.db.workspaces import get_workspace_by_team_id

logger = logging.getLogger(__name__)


def register(app):
    """Register both help command handlers."""

    @app.command("/slough-help")
    def handle_help_en(ack, command, respond):
        ack()
        role = _get_role(command)
        respond(blocks=_build_en(role), text="Slough.ai Help")

    @app.command("/slough-help-kr")
    def handle_help_kr(ack, command, respond):
        ack()
        role = _get_role(command)
        respond(blocks=_build_kr(role), text="Slough.ai 도움말")


def _get_role(command: dict) -> str:
    team_id = command.get("team_id", "")
    user_id = command.get("user_id", "")
    role = "employee"
    try:
        with get_db() as db:
            workspace = get_workspace_by_team_id(db, team_id)
            if workspace:
                if user_id == workspace.admin_id:
                    role = "admin"
                elif user_id == workspace.decision_maker_id:
                    role = "decision_maker"
    except Exception:
        logger.exception("DB error during help command")
    return role


def _build_en(role: str) -> list[dict]:
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Slough.ai Help"},
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "I learn from a decision-maker's Slack conversations "
                    "and provide answers that reflect their thinking.\n\n"
                    "Answers are for reference only and do not replace legal, "
                    "financial, or operational responsibility."
                ),
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*Everyone*\n"
                    "  *DM the bot* to ask a question\n"
                    "  The bot responds based on the decision-maker's thinking\n"
                    "  Responses are AI-generated and for reference only\n\n"
                    "  *After receiving an answer:*\n"
                    '  Click "검토 요청" to request the decision-maker to review\n'
                    "  You'll be notified when the decision-maker responds\n\n"
                    "  *Commands:*\n"
                    "  `/slough-help` — This help page\n"
                    "  `/slough-help-kr` — 한국어 도움말"
                ),
            },
        },
    ]

    if role in ("admin", "decision_maker"):
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*Decision-maker only*\n"
                    "  When an employee requests a review, you receive a DM with feedback buttons:\n"
                    '  "문제 없음" — Answer is correct, employee is notified\n'
                    '  "틀림" — Answer is wrong, employee is told to ask you directly\n'
                    '  "직접 수정" — Edit the answer yourself and send the corrected version\n'
                    '  "주의 필요" — Answer needs caution, employee is warned\n\n'
                    "  *Commands:*\n"
                    "  `/slough-ingest` — Learn new messages since last ingestion\n"
                    "  `/slough-ingest full` — Re-learn everything from scratch"
                ),
            },
        })

    if role == "admin":
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*Admin only*\n"
                    '  `/slough-rule add "text"` — Add a rule\n'
                    "  `/slough-rule list` — List rules\n"
                    "  `/slough-rule delete [ID]` — Delete a rule\n"
                    "  `/slough-ingest` — Learn new messages since last ingestion\n"
                    "  `/slough-ingest full` — Re-learn everything from scratch\n"
                    "  `/slough-stats` — This week's live stats"
                ),
            },
        })

    return blocks


def _build_kr(role: str) -> list[dict]:
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Slough.ai 도움말"},
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "저는 의사결정자의 Slack 대화를 학습하여, "
                    "의사결정자의 사고 방식을 반영한 답변을 제공합니다.\n\n"
                    "이 답변은 참고용이며, 법적·재무적·운영상 책임을 대체하지 않습니다."
                ),
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*모든 사용자*\n"
                    "  *봇에게 DM을 보내* 질문하세요\n"
                    "  의사결정자의 사고 방식을 반영한 답변을 제공합니다\n"
                    "  AI가 생성한 참고용 답변이며, 최종 판단은 아닙니다\n\n"
                    "  *답변을 받은 후:*\n"
                    '  "검토 요청" 버튼을 눌러 의사결정자에게 확인을 요청할 수 있습니다\n'
                    "  의사결정자가 응답하면 알림을 받게 됩니다\n\n"
                    "  *명령어:*\n"
                    "  `/slough-help` — English help\n"
                    "  `/slough-help-kr` — 이 도움말 보기"
                ),
            },
        },
    ]

    if role in ("admin", "decision_maker"):
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*의사결정자 전용*\n"
                    "  직원이 검토를 요청하면 피드백 버튼이 포함된 DM을 받습니다:\n"
                    '  "문제 없음" — 답변이 맞음, 직원에게 알림 전송\n'
                    '  "틀림" — 답변이 틀림, 직원에게 직접 문의하라고 안내\n'
                    '  "직접 수정" — 답변을 수정하여 직원에게 전달\n'
                    '  "주의 필요" — 답변에 주의가 필요함을 직원에게 알림\n\n'
                    "  *명령어:*\n"
                    "  `/slough-ingest` — 새 메시지 추가 학습\n"
                    "  `/slough-ingest full` — 전체 재학습 (처음부터 다시)"
                ),
            },
        })

    if role == "admin":
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*관리자 전용*\n"
                    '  `/slough-rule add "내용"` — 규칙 추가\n'
                    "  `/slough-rule list` — 규칙 목록\n"
                    "  `/slough-rule delete [ID]` — 규칙 삭제\n"
                    "  `/slough-ingest` — 새 메시지 추가 학습\n"
                    "  `/slough-ingest full` — 전체 재학습 (처음부터 다시)\n"
                    "  `/slough-stats` — 이번 주 실시간 현황"
                ),
            },
        })

    return blocks
