"""
AI Service — LangGraph RAG Pipeline
====================================
Implements the three interface functions consumed by Slack handlers:
  - ``generate_answer()``  — question → AI answer
  - ``ingest_messages()``  — messages → embed → store
  - ``process_feedback()`` — feedback → update KB

The RAG pipeline is built with LangGraph (StateGraph) and uses pgvector
for vector similarity search against the existing ``embeddings`` table.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import HumanMessage

from src.services.ai.graph import get_compiled_graph
from src.services.ai.memory import get_checkpointer
from src.services.ai.state import streaming_callback
from src.services.ai.vector_store import search_similar, store_embeddings

logger = logging.getLogger(__name__)


# ── Data classes (unchanged interface) ────────────────────────────────

@dataclass
class AnswerResult:
    answer: str
    is_high_risk: bool
    is_prohibited: bool
    sources_used: int


@dataclass
class IngestResult:
    chunks_created: int
    embeddings_stored: int


# ── 1. generate_answer ────────────────────────────────────────────────

async def generate_answer(
    question: str,
    workspace_id: str,
    asker_id: str,
    rules: list[dict],
) -> AnswerResult:
    """Generate an answer using the RAG pipeline with conversation memory.

    Memory management (3-layer hybrid):
      - AsyncPostgresSaver persists full state per thread_id
      - trim_and_summarize (in generate node) compresses old messages
      - Sliding window keeps the last 2 Q&A pairs verbatim

    Thread ID strategy: ``{workspace_id}:{asker_id}`` — one thread per user
    per workspace, so conversation context is maintained across DMs.
    """
    thread_id = f"{workspace_id}:{asker_id}"
    config = {"configurable": {"thread_id": thread_id}}

    inputs = {
        "question": question,
        "workspace_id": workspace_id,
        "rules": rules,
        "messages": [HumanMessage(content=question)],
    }

    try:
        async with get_checkpointer() as checkpointer:
            graph = get_compiled_graph(checkpointer=checkpointer)
            result = await graph.ainvoke(inputs, config=config)
    except Exception:
        logger.exception("RAG pipeline failed for workspace %s", workspace_id)
        return AnswerResult(
            answer="죄송합니다. 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            is_high_risk=False,
            is_prohibited=False,
            sources_used=0,
        )

    return AnswerResult(
        answer=result.get("answer", "답변을 생성할 수 없습니다."),
        is_high_risk=result.get("is_high_risk", False),
        is_prohibited=result.get("is_prohibited", False),
        sources_used=result.get("sources_used", 0),
    )


async def generate_answer_streaming(
    question: str,
    workspace_id: str,
    asker_id: str,
    rules: list[dict],
    on_chunk: callable = None,
) -> AnswerResult:
    """Generate answer with streaming support.

    Wraps ``generate_answer`` but sets a context-local callback so that
    the ``generate`` node streams tokens via ``on_chunk(text_so_far)``.
    """
    token = streaming_callback.set(on_chunk)
    try:
        return await generate_answer(question, workspace_id, asker_id, rules)
    finally:
        streaming_callback.reset(token)


# ── 2. ingest_messages ───────────────────────────────────────────────

_CHUNK_MAX_LENGTH = 2000  # characters per chunk (increased for contextualized messages)


def _chunk_messages(messages: list[dict]) -> list[dict]:
    """Split messages into chunks suitable for embedding.

    Groups sequential messages and splits long ones into manageable sizes.
    Each chunk keeps the original metadata (channel, ts, thread_ts).
    """
    chunks: list[dict] = []

    for msg in messages:
        text = msg.get("text", "").strip()
        if not text:
            continue

        base_meta = {
            "channel_id": msg.get("channel", ""),
            "message_ts": msg.get("ts", ""),
            "thread_ts": msg.get("thread_ts"),
        }

        # Split long messages into sub-chunks
        if len(text) <= _CHUNK_MAX_LENGTH:
            chunks.append({"content": text, **base_meta})
        else:
            for i in range(0, len(text), _CHUNK_MAX_LENGTH):
                sub = text[i : i + _CHUNK_MAX_LENGTH]
                chunks.append({"content": sub, **base_meta})

    return chunks


async def ingest_messages(
    workspace_id: str,
    messages: list[dict],
) -> IngestResult:
    """Ingest decision-maker messages: chunk → embed → store in pgvector.

    Args:
        workspace_id: UUID of the workspace.
        messages: [{\"text\": str, \"channel\": str, \"ts\": str, \"thread_ts\"?: str}]

    Returns:
        IngestResult with chunk and embedding counts.
    """
    chunks = _chunk_messages(messages)

    if not chunks:
        return IngestResult(chunks_created=0, embeddings_stored=0)

    try:
        stored = store_embeddings(workspace_id, chunks)
    except Exception:
        logger.exception(
            "Failed to ingest messages for workspace %s", workspace_id
        )
        return IngestResult(chunks_created=len(chunks), embeddings_stored=0)

    logger.info(
        "Ingested %d chunks (%d embeddings) for workspace %s",
        len(chunks),
        stored,
        workspace_id,
    )
    return IngestResult(chunks_created=len(chunks), embeddings_stored=stored)


# ── 3. process_feedback ──────────────────────────────────────────────

async def process_feedback(
    workspace_id: str,
    question_id: str,
    feedback_type: str,
    corrected_answer: Optional[str] = None,
) -> None:
    """Reflect decision-maker feedback into the knowledge base.

    - approved:  no action needed (answer was correct)
    - rejected:  no action needed (answer was wrong, but no correction given)
    - corrected: embed the corrected Q&A pair and store as a new embedding
    - caution:   no action needed (flagged for awareness)

    Only the ``corrected`` type adds new data to the knowledge base.
    """
    if feedback_type != "corrected" or not corrected_answer:
        logger.info(
            "Feedback '%s' for question %s — no KB update needed",
            feedback_type,
            question_id,
        )
        return

    # Build a Q&A content string for embedding
    # We don't have the original question text here, so we embed the
    # corrected answer with a reference to the question ID.
    content = f"[수정된 답변] {corrected_answer}"

    try:
        chunk = {
            "content": content,
            "channel_id": "",
            "message_ts": question_id,
            "thread_ts": None,
        }
        stored = store_embeddings(workspace_id, [chunk])
        logger.info(
            "Feedback correction stored: question=%s, embeddings=%d",
            question_id,
            stored,
        )
    except Exception:
        logger.exception(
            "Failed to store feedback correction for question %s",
            question_id,
        )
