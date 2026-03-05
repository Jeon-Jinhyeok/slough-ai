"""LangGraph node functions for the RAG pipeline.

Each node takes the current ``AgentState`` and returns a partial dict
that gets merged back into the state.
"""

import logging
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import settings
from src.services.ai.memory import trim_and_summarize
from src.services.ai.persona import build_system_prompt
from src.services.ai.state import AgentState, streaming_callback
from src.services.ai.vector_store import search_similar
from src.services.redis_client import get_persona_profile
from src.utils.keywords import detect_high_risk_keywords
from src.utils.prohibited import check_prohibited

logger = logging.getLogger(__name__)

# Lazy-loaded LLM singleton
_llm: Optional[ChatOpenAI] = None


def _get_decision_maker_name(workspace_id: str) -> str:
    """Look up the decision-maker's display name, cached in Redis.

    Falls back to empty string if anything fails.
    """
    from src.services.redis_client import RedisManager

    cache_key = f"dm_name:{workspace_id}"
    cache = RedisManager.get_cache()
    cached = cache.get(cache_key)
    if cached:
        return cached

    try:
        import uuid as _uuid
        from src.services.db import get_db
        from src.services.db.models import Workspace
        from slack_sdk import WebClient

        with get_db() as db:
            ws = db.query(Workspace).filter(
                Workspace.id == _uuid.UUID(workspace_id)
            ).first()
            if not ws or not ws.decision_maker_id or not ws.bot_token:
                return ""

            client = WebClient(token=ws.bot_token)
            resp = client.users_info(user=ws.decision_maker_id)
            name = (
                resp["user"].get("real_name")
                or resp["user"].get("profile", {}).get("display_name")
                or resp["user"].get("name", "")
            )

        if name:
            cache.set(cache_key, name, ex=86400)  # cache 24h
        return name
    except Exception:
        logger.debug("Failed to look up decision-maker name for %s", workspace_id)
        return ""


def _get_llm() -> ChatOpenAI:
    """Return a singleton ChatOpenAI (GPT-4o) instance."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            api_key=settings.openai_api_key,
        )
    return _llm


def _to_openai_messages(msgs: list) -> list[dict]:
    """Convert LangChain BaseMessages to OpenAI-style dicts."""
    result = []
    for m in msgs:
        if isinstance(m, SystemMessage):
            result.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            result.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            result.append({"role": "assistant", "content": m.content})
    return result


# ── Node: check_rules ────────────────────────────────────────────────

def check_rules(state: AgentState) -> dict:
    """Match the question against active rules (keyword search).

    If a rule matches, the answer is set immediately and the pipeline
    can skip to END via the conditional edge.
    """
    question = state["question"]
    rules = state.get("rules", [])

    for rule in rules:
        rule_text = rule.get("rule_text", "")
        # Simple keyword containment — same logic as SloughAI
        if rule_text and rule_text.lower() in question.lower():
            return {
                "answer": f"📋 [규칙 적용]\n{rule_text}",
                "is_rule_matched": True,
            }

    return {"is_rule_matched": False}


# ── Node: check_safety ───────────────────────────────────────────────

def check_safety(state: AgentState) -> dict:
    """Run prohibited-domain and high-risk keyword checks."""
    question = state["question"]

    # Prohibited domain check
    prohibited = check_prohibited(question)
    if prohibited["is_prohibited"]:
        return {
            "is_safe": False,
            "is_prohibited": True,
            "is_high_risk": False,
        }

    # High-risk keyword check (answer anyway, but flag it)
    risk = detect_high_risk_keywords(question)
    return {
        "is_safe": True,
        "is_prohibited": False,
        "is_high_risk": risk["is_high_risk"],
    }


# ── Node: retrieve ───────────────────────────────────────────────────

def retrieve(state: AgentState) -> dict:
    """Retrieve similar documents from pgvector with similarity threshold."""
    workspace_id = state.get("workspace_id", "")
    question = state["question"]

    try:
        results = search_similar(
            workspace_id=workspace_id,
            query=question,
            k=8,
            threshold=0.3,
        )
        # Annotate each doc with relevance level for the LLM
        annotated = []
        for content, score in results:
            if score > 0.5:
                label = "[높은 관련성]"
            elif score >= 0.35:
                label = "[관련성 있음]"
            else:
                label = "[낮은 관련성]"
            annotated.append(f"{label}\n{content}")

        return {
            "context": annotated,
            "sources_used": len(results),
        }
    except Exception:
        logger.exception("Vector search failed")
        return {
            "context": [],
            "sources_used": 0,
        }


# ── Node: generate ───────────────────────────────────────────────────

async def generate(state: AgentState) -> dict:
    """Generate an answer using GPT-4o with persona prompt + conversation memory.

    Memory management:
      1. ``trim_and_summarize`` compresses old messages into a summary
      2. Recent 2 Q&A pairs are kept verbatim
      3. Token cost is bounded at ~400-700 regardless of conversation length
    """
    # Skip if rule already matched
    if state.get("is_rule_matched"):
        return {}

    rules = state.get("rules", [])
    context = state.get("context", [])
    workspace_id = state.get("workspace_id", "")

    persona = get_persona_profile(workspace_id) if workspace_id else ""

    # Look up decision-maker name for self-identity in prompt
    dm_name = ""
    if workspace_id:
        dm_name = _get_decision_maker_name(workspace_id)

    system_prompt = build_system_prompt(
        rules, context, persona=persona, decision_maker_name=dm_name,
    )

    # Trim conversation history for token efficiency
    raw_messages = state.get("messages", [])
    trimmed = await trim_and_summarize(raw_messages, max_recent_pairs=2)

    # Build final message list: system prompt + trimmed memory + current question
    messages = [
        {"role": "system", "content": system_prompt},
        *_to_openai_messages(trimmed),
    ]

    llm = _get_llm()

    # Check for streaming callback
    on_chunk = streaming_callback.get(None)

    try:
        if on_chunk:
            # Streaming mode — send tokens progressively
            answer_chunks: list[str] = []
            async for chunk in llm.astream(messages):
                if hasattr(chunk, "content") and chunk.content:
                    answer_chunks.append(chunk.content)
                    on_chunk("".join(answer_chunks))
            answer_text = "".join(answer_chunks)
        else:
            # Non-streaming mode (backward compatible)
            response = await llm.ainvoke(messages)
            answer_text = response.content
    except Exception:
        logger.exception("LLM generation failed")
        answer_text = "죄송합니다. 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

    return {
        "answer": answer_text,
        "messages": [AIMessage(content=answer_text)],
    }


# ── Node: refuse_answer ──────────────────────────────────────────────

def refuse_answer(state: AgentState) -> dict:
    """Refuse to answer prohibited-domain questions."""
    return {
        "answer": (
            "죄송합니다. 이 주제는 법적·재무적·운영상 판단이 필요한 영역으로, "
            "AI가 답변을 제공할 수 없습니다. 직접 의사결정자에게 문의해 주세요."
        ),
    }

