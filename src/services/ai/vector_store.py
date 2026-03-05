"""pgvector search and storage using the existing Embedding ORM model."""

import logging
import uuid as uuid_mod

from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from src.services.db.connection import get_db
from src.services.db.models import Embedding
from src.services.ai.embeddings import embed_text, embed_texts

logger = logging.getLogger(__name__)


def search_similar(
    workspace_id: str,
    query: str,
    k: int = 5,
    threshold: float = 0.3,
) -> list[tuple[str, float]]:
    """Search for the top-k most similar embeddings with time-weighted scoring.

    Combines cosine similarity with a time decay factor so that recent
    messages are prioritized over older ones.  The time weight decays
    logarithmically — a 6-month-old message scores ~85% of a fresh one.

    Args:
        workspace_id: UUID string of the workspace.
        query: The question text to search for.
        k: Maximum number of results to return.
        threshold: Minimum cosine similarity (0-1). Documents below this
                   threshold are excluded even if they are in the top-k.

    Returns:
        List of (content, final_score) tuples from the most relevant embeddings.
    """
    query_embedding = embed_text(query)

    with get_db() as db:
        # Combined score = similarity * time_weight
        # time_weight = 1 / (1 + 0.1 * ln(age_days + 1))
        #   - 1 day old  → weight ≈ 0.94
        #   - 30 days    → weight ≈ 0.77
        #   - 180 days   → weight ≈ 0.66
        #   - 365 days   → weight ≈ 0.60
        results = db.execute(
            sa_text("""
                WITH scored AS (
                    SELECT
                        content,
                        1 - (embedding <=> :query_vec) AS similarity,
                        1.0 / (1.0 + 0.1 * LN(
                            GREATEST(EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0, 0) + 1
                        )) AS time_weight
                    FROM embeddings
                    WHERE workspace_id = :ws_id
                      AND 1 - (embedding <=> :query_vec) > :threshold
                )
                SELECT content, similarity, time_weight, similarity * time_weight AS final_score
                FROM scored
                ORDER BY final_score DESC
                LIMIT :k
            """),
            {
                "ws_id": uuid_mod.UUID(workspace_id),
                "query_vec": str(query_embedding),
                "k": k,
                "threshold": threshold,
            },
        ).fetchall()

    if results:
        logger.info(
            "Vector search: %d results (top=%.3f, min=%.3f, time_w=%.2f~%.2f)",
            len(results),
            results[0][1],       # top similarity
            results[-1][1],      # min similarity
            results[0][2],       # top time_weight
            results[-1][2],      # min time_weight
        )
    else:
        logger.info("Vector search: 0 results above threshold %.2f", threshold)

    return [(row[0], row[3]) for row in results]  # (content, final_score)


def store_embeddings(
    workspace_id: str,
    chunks: list[dict],
) -> int:
    """Embed and store message chunks into the embeddings table.

    Args:
        workspace_id: UUID string of the workspace.
        chunks: List of dicts with keys: "content", "channel_id", "message_ts",
                and optionally "thread_ts".

    Returns:
        Number of embeddings stored.
    """
    if not chunks:
        return 0

    texts = [c["content"] for c in chunks]
    vectors = embed_texts(texts)

    ws_uuid = uuid_mod.UUID(workspace_id)
    stored = 0

    with get_db() as db:
        for chunk, vector in zip(chunks, vectors):
            record = Embedding(
                workspace_id=ws_uuid,
                content=chunk["content"],
                embedding=vector,
                channel_id=chunk.get("channel_id"),
                message_ts=chunk.get("message_ts"),
                thread_ts=chunk.get("thread_ts"),
            )
            db.add(record)
            stored += 1
        db.flush()

    logger.info("Stored %d embeddings for workspace %s", stored, workspace_id)
    return stored
