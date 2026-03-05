"""Persona extractor — analyzes decision-maker messages to build a persona profile.

Fetches sample messages from the embeddings table, sends them to GPT-4o-mini
for analysis, and caches the resulting persona profile in Redis.
"""

import logging

from langchain_openai import ChatOpenAI

from src.config import settings
from src.services.ai.vector_store import search_similar
from src.services.redis_client import set_persona_profile

logger = logging.getLogger(__name__)

_SAMPLE_QUERIES = [
    "업무 방향", "결정", "팀", "의견", "진행", "좋아", "안 돼",
    "프로젝트", "일정", "우선순위", "승인", "검토", "피드백",
]

_PERSONA_ANALYSIS_PROMPT = """\
아래는 한 회사 의사결정자의 실제 Slack 발언 모음입니다.
이 발언들을 분석하여 다음 항목을 포함한 "페르소나 프로필"을 작성하세요:

1. **말투/어조**: 반말/존댓말, 격식 수준, 특유의 표현이나 말버릇 (실제 예시 포함)
2. **성격 특성**: 직설적/우회적, 긍정적/현실적, 유머 사용 여부
3. **의사결정 스타일**: 빠른 결단형/신중형, 데이터 중시/직감 중시
4. **핵심 관심 주제 및 프로젝트**: 자주 언급하는 구체적 프로젝트명, 기술, 비즈니스 주제를 나열하세요
5. **커뮤니케이션 패턴**: 답변 길이, 질문 방식, 피드백 방식
6. **절대 하지 않는 것**: 쓰지 않는 표현, 피하는 주제

7. **대표 발언 예시 (3-5개)**: 아래 상황별로 실제 발언을 원문 그대로 인용하세요:
   - 승인/동의할 때
   - 거절/반대할 때
   - 방향을 지시할 때
   - 피드백을 줄 때
   - (해당 상황이 없으면 생략)

[발언 모음]
{messages}

위 발언을 기반으로 이 의사결정자의 페르소나 프로필을 한국어로 작성하세요.
프로필은 AI가 이 사람처럼 대화하기 위한 가이드 역할을 합니다.
구체적인 주제, 프로젝트명, 실제 표현을 반드시 포함하여 작성하세요 (1000자 이내)."""


def extract_persona(workspace_id: str) -> str:
    """Extract persona profile from decision-maker messages in the vector DB.

    1. Sample diverse messages via multiple keyword searches
    2. Analyze with GPT-4o-mini
    3. Cache result in Redis

    Args:
        workspace_id: UUID string of the workspace.

    Returns:
        The generated persona profile string, or empty string on failure.
    """
    # 1. Sample diverse messages from vector DB
    all_samples: list[str] = []
    seen: set[str] = set()

    for query in _SAMPLE_QUERIES:
        try:
            results = search_similar(workspace_id=workspace_id, query=query, k=5)
            for content, _score in results:
                if content not in seen:
                    seen.add(content)
                    all_samples.append(content)
        except Exception:
            continue

    if not all_samples:
        logger.warning("No messages found for persona extraction (workspace %s)", workspace_id)
        return ""

    # Limit to 50 samples to stay within token limits
    samples = all_samples[:50]
    messages_text = "\n---\n".join(samples)

    # 2. Analyze with GPT-4o-mini
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=1200,
            api_key=settings.openai_api_key,
        )
        response = llm.invoke([
            {"role": "user", "content": _PERSONA_ANALYSIS_PROMPT.format(messages=messages_text)},
        ])
        persona_profile = response.content.strip()
    except Exception:
        logger.exception("Persona analysis LLM call failed (workspace %s)", workspace_id)
        return ""

    # 3. Cache in Redis
    try:
        set_persona_profile(workspace_id, persona_profile)
    except Exception:
        logger.exception("Failed to cache persona profile in Redis (workspace %s)", workspace_id)

    logger.info(
        "Persona extracted from %d messages for workspace %s",
        len(samples),
        workspace_id,
    )
    return persona_profile
