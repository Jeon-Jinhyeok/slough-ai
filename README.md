# Slough.ai (슬러프)

> Slack 대화로 의사결정자의 사고를 학습하는 AI

의사결정자의 Slack 대화 기록을 학습하여, 팀원들의 질문에 의사결정자의 사고 방식으로 답변하는 B2B SaaS Slack 봇입니다.

- **랜딩 페이지:** https://slough-lp.vercel.app
- **기술명세서:** https://bit.ly/slough_ai_tech_docs
- **Slack 앱 설치:** https://dylcsnb4u0bo.cloudfront.net/slack/install
- **데모 워크스페이스:** https://join.slack.com/t/isfet/shared_invite/zt-3qneskbtv-dHWGw2T0sEged3rSREYiAw

## 빠른 체험 가이드

### 1. 데모 워크스페이스에서 체험하기

이미 설치된 환경에서 바로 체험할 수 있습니다:

1. [데모 Slack 워크스페이스](https://join.slack.com/t/isfet/shared_invite/zt-3qneskbtv-dHWGw2T0sEged3rSREYiAw)에 가입
2. 앱 목록에서 **Slough.ai** 봇을 찾아 DM 보내기
3. 질문을 입력하면 의사결정자 스타일의 AI 답변을 받을 수 있습니다
4. 답변 하단의 "검토 요청" 버튼으로 의사결정자에게 확인을 요청할 수 있습니다

### 2. 내 워크스페이스에 설치하기

1. [설치 링크](https://dylcsnb4u0bo.cloudfront.net/slack/install)를 클릭
2. Slack OAuth 인증 화면에서 워크스페이스를 선택하고 **허용** 클릭
3. 온보딩 모달이 표시됩니다:
   - 학습 대상 의사결정자를 선택
   - 학습할 채널을 선택
   - 데이터 수집 동의 체크
4. 봇이 선택한 채널에서 의사결정자의 메시지를 백그라운드로 학습합니다
5. 학습 완료 DM을 받으면, 팀원들이 봇에게 DM으로 질문을 보낼 수 있습니다

> **참고:** 반드시 위 설치 링크를 통해 설치해야 합니다. Slack API 사이트의 "Install App" 버튼은 OAuth 플로우를 거치지 않아 정상 동작하지 않습니다.

---

## 프로젝트 개요

**문제:** 의사결정자에게 반복적인 질문이 몰리고, 답변 대기로 업무가 지연됨
**해결:** Slack 대화를 학습하여 의사결정자의 사고 패턴을 반영한 AI 답변을 즉시 제공

### 핵심 기능 (MVP)

| 기능 | 설명 |
|------|------|
| 온보딩 & 학습 | OAuth 설치 → 채널 선택 → 의사결정자 메시지만 선별 수집 → 페르소나 자동 추출 |
| Q&A | 팀원이 DM으로 질문 → AI가 의사결정자 스타일로 답변 |
| 증분 학습 | `/slough-ingest`로 새 메시지만 추가 학습 |
| 피드백 루프 | 검토 요청 → 의사결정자 피드백 (승인/수정/주의) |
| 안전장치 | AI 면책, 고위험 키워드 감지, 금지 도메인 차단 |
| 규칙 선언 | `/slough-rule`로 명시적 규칙 등록 (학습보다 우선) |
| 주간 리포트 | 매주 월요일 10시 활동 요약 DM |

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| 언어 | Python 3.12 |
| 웹 프레임워크 | FastAPI |
| Slack SDK | Slack Bolt for Python |
| AI 파이프라인 | LangGraph (RAG + 페르소나) |
| LLM | OpenAI GPT-4o |
| 임베딩 | OpenAI text-embedding-3-small (1536차원) |
| 벡터 검색 | pgvector (PostgreSQL 확장) |
| RDBMS | PostgreSQL (AWS RDS) |
| 캐시/큐 | Redis (AWS ElastiCache) + Celery |
| 배포 | AWS ECS Fargate (3-서비스) |
| CI/CD | GitHub Actions |
| CDN/HTTPS | AWS CloudFront + ALB |
| 패키지 관리 | uv |

---

## 아키텍처

```
┌─────────────┐     HTTPS (CloudFront)       ┌──────────────────────┐
│  Slack API  │ ──────────────────────────→   │  slough-app          │
│             │ ←──────────────────────────   │  (FastAPI + Bolt)    │
└─────────────┘     JSON response            │                      │
                                             │  - 이벤트/커맨드 처리  │
                                             │  - OAuth 설치         │
                                             │  - 헬스체크           │
                                             └──────┬───────────────┘
                                                    │ Celery task
                                                    ▼
┌──────────────┐                            ┌──────────────────────┐
│  PostgreSQL  │ ←─────────────────────────→ │  slough-worker       │
│  (RDS)       │                            │  (Celery worker)     │
│  + pgvector  │                            │                      │
└──────────────┘                            │  - 데이터 수집         │
                                            │  - AI 파이프라인 호출  │
┌──────────────┐                            │  - 페르소나 추출       │
│  Redis       │ ←── 작업 큐 + 캐시 ──→      └──────────────────────┘
│ (ElastiCache)│
└──────────────┘                            ┌──────────────────────┐
                                            │  slough-beat         │
                                            │  (Celery scheduler)  │
                                            │  - 주간 리포트 스케줄  │
                                            └──────────────────────┘
```

**운영 모드:**
- **프로덕션 (AWS):** HTTP 모드 — CloudFront → ALB → ECS Fargate
- **로컬 개발:** Socket Mode — WebSocket으로 연결 (공개 URL 불필요)

---

## AI 파이프라인 (LangGraph)

### 그래프 구조

```
질문 입력
    │
    ▼
┌─────────────┐
│ check_rules │ ──── 규칙 매칭? ──→ 규칙 적용 답변 → END
└──────┬──────┘
       │ No match
       ▼
┌──────────────┐
│ check_safety │ ──── 금지 도메인? ──→ refuse_answer → END
└──────┬───────┘
       │ Safe
       ▼
┌──────────────┐
│   retrieve   │ ──── pgvector 유사도 검색 (threshold=0.5, 시간 가중치)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   generate   │ ──── GPT-4o + 페르소나 프롬프트 + 3-layer 메모리
└──────────────┘
```

### 주요 특징

- **유사도 임계값 (0.5):** 관련 없는 문서 필터링
- **시간 가중치 검색:** 최신 메시지를 우선 반영 (`1 / (1 + 0.1 * ln(age_days + 1))`)
- **페르소나 자동 추출:** GPT-4o-mini가 50개 샘플 메시지 분석 → 말투/성격/의사결정 스타일 프로필 생성
- **CEO 이름 주입:** Slack에서 의사결정자 이름 조회 → "당신은 {name}의 AI 분신입니다"
- **3-layer 메모리:** PostgresSaver 체크포인트 + GPT-4o-mini 요약 + 슬라이딩 윈도우 (2 Q&A 쌍)
- **의사결정자 메시지만 수집:** 채널의 전체 대화가 아닌, 의사결정자가 작성한 메시지만 선별 수집
- **Q&A 쌍 보존:** 스레드 답변 시 부모 질문(다른 사용자)을 `[질문]...[답변]...` 형태로 함께 학습하여 맥락 보존
- **채널 맥락:** 메시지에 `[#channel-name]` 접두사 추가

---

## 폴더 구조

```
slough-ai/
├── entrypoint.py                  # SERVICE_TYPE 기반 프로세스 라우팅
├── requirements.txt
├── Dockerfile
│
├── .github/workflows/
│   └── deploy.yml                 # GitHub Actions 배포 워크플로우
│
├── infra/
│   └── cloudformation.yaml        # AWS 인프라 정의
│
├── src/
│   ├── app.py                     # Bolt 앱 초기화 + 핸들러 등록 + 진입점
│   ├── config.py                  # 환경변수 (pydantic-settings)
│   ├── web.py                     # FastAPI (헬스체크, OAuth, Bolt HTTP 핸들러)
│   ├── worker.py                  # Celery 앱 인스턴스 + beat 스케줄
│   │
│   ├── handlers/                  # Slack 이벤트/액션/커맨드 핸들러
│   │   ├── events/
│   │   │   ├── message.py         # DM 메시지 수신 → AI 답변
│   │   │   └── uninstall.py       # 앱 삭제 이벤트
│   │   ├── actions/
│   │   │   ├── feedback.py        # 피드백 버튼 (승인/틀림/수정/주의)
│   │   │   ├── onboarding.py      # 온보딩 모달 열기
│   │   │   └── review_request.py  # 검토 요청 버튼
│   │   ├── commands/
│   │   │   ├── rule.py            # /slough-rule (규칙 관리)
│   │   │   ├── ingest.py          # /slough-ingest (증분 학습)
│   │   │   ├── stats.py           # /slough-stats (통계)
│   │   │   └── help.py            # /slough-help
│   │   └── views/
│   │       ├── edit_answer.py     # 직접 수정 모달
│   │       └── onboarding.py      # 온보딩 모달 제출 처리
│   │
│   ├── services/
│   │   ├── ai/                    # AI 파이프라인 (LangGraph)
│   │   │   ├── __init__.py        # 인터페이스 (generate_answer, ingest_messages, process_feedback)
│   │   │   ├── graph.py           # LangGraph 그래프 정의
│   │   │   ├── nodes.py           # 그래프 노드 (check_rules, check_safety, retrieve, generate, refuse)
│   │   │   ├── state.py           # AgentState 정의
│   │   │   ├── persona.py         # 페르소나 시스템 프롬프트 구성
│   │   │   ├── persona_extractor.py # GPT-4o-mini 페르소나 자동 추출
│   │   │   ├── memory.py          # 3-layer 대화 메모리 관리
│   │   │   ├── vector_store.py    # pgvector 유사도 검색 + 시간 가중치
│   │   │   ├── embeddings.py      # OpenAI 임베딩 생성
│   │   │   └── chunking.py        # 메시지 청킹 로직
│   │   ├── slack/
│   │   │   ├── conversations.py   # Slack 대화 기록 가져오기 (Q&A 쌍 보존)
│   │   │   └── oauth.py           # OAuth 설치 플로우
│   │   ├── db/
│   │   │   ├── connection.py      # SQLAlchemy 엔진/세션
│   │   │   ├── models.py          # ORM 모델 (6개 테이블)
│   │   │   ├── workspaces.py      # 워크스페이스 CRUD
│   │   │   ├── rules.py           # 규칙 CRUD
│   │   │   ├── qa_history.py      # Q&A 기록 CRUD
│   │   │   ├── weekly_stats.py    # 주간 통계
│   │   │   └── ingestion_jobs.py  # 수집 작업 추적
│   │   ├── ingestion/
│   │   │   └── ingest.py          # 수집 오케스트레이터 (초기 + 증분)
│   │   └── redis_client.py        # Redis 멀티-DB 클라이언트 + 캐시 헬퍼
│   │
│   ├── tasks/                     # Celery 비동기 작업
│   │   ├── ingestion.py           # 백그라운드 데이터 수집
│   │   └── weekly_report.py       # 주간 리포트 발송
│   │
│   └── utils/
│       ├── blocks.py              # Block Kit 메시지 빌더
│       ├── keywords.py            # 고위험 키워드 감지
│       ├── prohibited.py          # 금지 도메인 체크
│       └── logger.py              # 로깅 설정
│
├── migrations/versions/           # Alembic DB 마이그레이션
├── tests/                         # 테스트
├── scripts/                       # 유틸리티 스크립트
└── docs/                          # 상세 문서
```

---

## 로컬 개발 환경 설정

```bash
# 1. uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 의존성 설치 (가상환경 자동 생성)
uv sync

# 3. 환경변수 설정
cp .env.example .env
# .env 파일에 Slack 토큰, DB URL 등 입력

# 4. Docker로 PostgreSQL + Redis 실행
docker-compose up -d

# 5. DB 테이블 생성
uv run alembic upgrade head

# 6. 앱 실행 (Socket Mode)
PYTHONPATH=. uv run python src/app.py

# 7. Celery 워커 (별도 터미널)
PYTHONPATH=. uv run celery -A src.worker worker --loglevel=info --concurrency=2

# 8. Celery Beat (별도 터미널, 주간 리포트 스케줄러)
PYTHONPATH=. uv run celery -A src.worker beat --loglevel=info
```

## 배포 (AWS ECS Fargate)

3개 서비스가 같은 Docker 이미지에서 `SERVICE_TYPE` 환경변수로 분기합니다:

| 서비스 | SERVICE_TYPE | 역할 |
|--------|-------------|------|
| slough-app | `web` | FastAPI + Bolt (HTTP 모드) |
| slough-worker | `worker` | Celery 작업 처리 |
| slough-beat | `beat` | Celery 스케줄러 |

### AWS 인프라 구성

| 서비스 | AWS 리소스 |
|--------|-----------|
| 컴퓨팅 | ECS Fargate (3 서비스) |
| 데이터베이스 | RDS PostgreSQL + pgvector |
| 캐시/큐 | ElastiCache Redis |
| 로드밸런서 | ALB |
| CDN/HTTPS | CloudFront |
| 로깅 | CloudWatch |
| CI/CD | GitHub Actions → ECR → ECS |

배포는 `main` 브랜치에 push하면 GitHub Actions가 자동 실행합니다.

## DB 스키마

| 테이블 | 설명 |
|--------|------|
| `workspaces` | 설치된 워크스페이스 (admin_id, decision_maker_id, 토큰) |
| `rules` | 의사결정자가 등록한 규칙 (/slough-rule) |
| `qa_history` | 질문/답변 기록 + 피드백 상태 |
| `weekly_stats` | 주간 통계 (자동 집계) |
| `ingestion_jobs` | 데이터 수집 진행 상태 |
| `embeddings` | 벡터 임베딩 (pgvector, vector(1536)) |

임베딩은 pgvector 확장을 통해 PostgreSQL에 직접 저장됩니다.

## 라이선스

Private - All rights reserved
