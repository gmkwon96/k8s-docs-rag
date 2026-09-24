# Kubernetes Docs RAG + Eval — 구현 계획

## Context
Kubernetes 공식 문서를 근거로 답하는 Q&A 서비스. 목적은 미국 테크/AI 기업 취업용 포트폴리오.
차별점은 챗봇이 아니라 **평가(eval) 파이프라인**이다. "이 시스템은 몇 % 정확한가, 이 변경이 좋아졌나 나빠졌나"를 숫자와 신뢰구간으로 답할 수 있어야 한다.

핵심 원칙:
1. **근거 없이 답하지 않는다.** 모든 문장은 검색된 문서 조각을 인용하고, 근거가 부족하면 답변을 거절한다.
2. **버전을 구분한다.** 같은 질문이라도 Kubernetes 버전에 따라 답이 달라질 수 있다. 버전 처리가 이 프로젝트의 핵심 기술 과제다.
3. **모든 설정 변경은 eval로 검증한다.** 실험 결과는 표로 남기고 README에 공개한다.

별도 repo 권장 (예: `k8s-docs-rag`). README, 실험 리포트는 영어로 작성.

## 기술 스택
- **언어**: Python 3.14, `uv` (AI 기업 면접 기준 표준)
- **API 서버**: FastAPI, SSE 스트리밍 응답
- **DB**: PostgreSQL 18 + `pgvector` (로컬은 Homebrew) (벡터 검색) + Postgres full-text search (키워드 검색). 검색 인프라를 DB 하나로 통일한 이유를 README에 설명
- **LLM**: Anthropic Python SDK (`anthropic`)
  - 답변 생성: `claude-opus-5` (env로 변경 가능, 실험에서 `claude-sonnet-5`와 effort 수준별 비교)
  - LLM 채점(judge): `claude-opus-5`, 모델·프롬프트 버전 고정 (채점 기준이 바뀌면 점수 비교가 무의미해짐)
  - Citations 기능: 검색된 조각을 `document` 블록으로 넣고 `citations: {enabled: true}` → 응답에서 문장별 인용 위치를 받음. (주의: citations는 structured output `output_config.format`과 동시 사용 불가)
  - Prompt caching: 고정된 system prompt 캐싱, contextual chunking 단계에서 문서 전체를 캐시 prefix로 두고 조각별 요약 생성
  - Message Batches API: 전체 eval 실행과 대량 전처리는 배치로 (비동기, 비용 50% 절감)
  - `stop_reason == "refusal"` 처리 + 서버 측 fallback 설정
- **임베딩 / reranker**: 외부 API 1종 + 오픈소스 1종을 비교 실험 대상으로 (예: Voyage AI 임베딩·rerank vs `sentence-transformers` 계열 오픈소스 모델). 착수 시점의 최신 모델을 확인해 선택
- **평가**: 자체 eval 러너 (pytest 스타일), 결과는 JSONL + 집계 표. 필요하면 Ragas 지표와 교차 확인
- **프론트엔드**: Next.js + TypeScript + Tailwind (간단한 단일 페이지 + eval 대시보드)
- **CI**: GitHub Actions
- **배포**: Postgres는 Neon 또는 Supabase(pgvector 지원), API는 Fly.io 또는 Render, 프론트는 Vercel

## 데이터
- 출처: `github.com/kubernetes/website` 저장소의 `content/en/docs/` (Hugo 마크다운)
- 라이선스: 문서는 CC BY 4.0 (착수 시 재확인). UI와 README에 출처 표기, 답변의 인용 링크는 kubernetes.io 원문으로 연결
- 버전: 최신 minor 버전 3개 (2026-09 기준 1.35, 1.36, 1.37). 현재 버전 문서는 `main`에 있고 `release-1.xx`는 다음 버전 출시 후에 만들어지므로 1.35/1.36은 `release-1.xx`, 1.37은 `main`. 수집 커밋은 `ingest/sources.lock.json`에 고정
- 전처리에서 풀어야 할 문제:
  - Hugo shortcode 정리: `{{< glossary_tooltip >}}`, `{{< note >}}`, `{{< tabs >}}` 등을 텍스트로 변환
  - **`{{< feature-state for_k8s_version="v1.xx" state="beta" >}}`** → 기능별 버전·안정성 메타데이터로 추출 (버전 질문 처리의 핵심 자산)
  - 코드 블록과 YAML 예시는 조각 경계에서 자르지 않음
  - 페이지 front matter의 제목과 URL 경로로 인용 링크 생성
  - (구현, `ingest/clean.py`) shortcode는 `layouts/shortcodes/` 템플릿 동작을 그대로 따라 markdown으로 변환, 라벨은 `i18n/en/en.toml`에서 읽음. 남은 shortcode 태그 0개가 통과 기준
  - 제외: `contribute/`, `doc-contributor-tools/`, `test.md`(문서 사이트 기여 안내), headless·`render: false` 페이지(include로만 쓰임), 본문이 빈 섹션 목차
  - 추가 레코드: feature gate마다 1건(단계 이력 + 설명, 링크는 feature-gates 페이지 `#<이름>`), 용어집 항목마다 1건(`glossary/?all=true#term-<id>`)
  - 인용 링크: 최신 버전은 `https://kubernetes.io`, 이전 버전은 `https://v1-35.docs.kubernetes.io` 형식
  - 주의: 1.36부터 API reference(`reference/kubernetes-api/`)가 하위 타입 정의를 페이지마다 인라인해서 코퍼스의 39%를 차지 (1.35는 19%). 청킹·중복 제거 단계에서 처리

### 청킹 (구현, `ingest/chunk.py`)
- 기본 전략 "heading": 코드 블록 밖 헤딩 단위 섹션. 최대 512 토큰, 64 토큰 미만은 같은 레코드의 다음 청크와 병합
- 큰 섹션은 블록(문단·목록·코드) 경계에서, 그래도 크면 항목 단위(들여쓴 설명 줄은 앞 항목에 붙임)로 자름. 코드 블록은 혼자 512 토큰을 넘을 때만 줄 단위로 자르고 다시 fence
- 토큰 수는 임베딩 모델 선택(E3) 전까지 `tiktoken` `o200k_base`로 계산
- 청크 메타데이터: `heading_path`(제목 › H2 › H3), 인용 앵커(Hugo blackfriday 규칙, 실제 사이트와 486/486 일치 확인), feature state(해당 섹션과 하위 섹션에 상속), `content_hash`
- 제목 경로를 임베딩 텍스트 앞에 붙이는 것(breadcrumb)은 임베딩 단계의 옵션으로 E1에서 비교

### 버전 간 중복 제거
대부분의 페이지는 버전 간 내용이 동일하다. 조각마다 content hash를 만들어 **한 번만 저장**하고, 해당 조각이 유효한 버전 목록(`versions: ["1.35","1.36","1.37"]`)을 메타데이터로 붙인다.
→ 인덱스 크기와 임베딩 비용 감소. 버전 필터링은 메타데이터 조건으로 처리. (면접 이야깃거리)
- (구현, `ingest/dedupe.py`) 3개 버전 청크 33,147개 → 고유 텍스트 14,025개(42%), 토큰 805만 → 327만(41%). 임베딩할 양 59% 감소
- 텍스트만 공유하고 메타데이터(URL, heading_path, feature state)는 출현(occurrence)마다 보존: 같은 텍스트라도 상위 섹션의 feature state가 버전마다 다른 경우가 146건 (예: 1.35 alpha → 1.36 beta)
- 중복 판정 기준은 임베딩할 문자열: `plain`(본문) 또는 `breadcrumb`(제목 경로 + 본문, 고유 16,804개). E1에서 비교

### DB 스키마 (구현, `db/schema.sql`, `ingest/load.py`)
- `contents`(고유 텍스트, `index_name`별), `occurrences`(버전별 URL·heading_path·feature state), `embeddings`(텍스트 × 모델)
- `index_name`(예: `heading-plain`)으로 청킹·임베딩 텍스트 변형을 한 DB에 나란히 적재 → E1 비교
- 버전 필터: `versions text[]` + GIN(`versions @> ARRAY['1.36']`). 키워드 검색: `tsv` 생성 컬럼 + GIN
- 모델마다 차원이 달라 HNSW는 모델별 부분 인덱스(`(embedding::vector(1024))`, `WHERE model = ...`)
- 재적재는 동기화 방식: 바뀌지 않은 텍스트는 id 유지 → 임베딩 보존

### 임베딩 (구현, `ingest/embed.py`, `rag/embedders.py`)
- 기준 모델: `voyage-4`, 1024차원, float. 고유 텍스트 14,025개 → 3,288,371 토큰 과금(무료 한도 2억 안), 60초
- 과금 안전장치: 모든 호출을 `data/usage/voyage.jsonl`에 기록, 누적 `VOYAGE_TOKEN_BUDGET`(2,000만) 초과가 예상되면 요청 전 거부. Batch API는 무료 토큰 대상이 아니므로 사용하지 않음
- 벡터 캐시 `data/embeddings/<model>.jsonl`(텍스트 hash 키): DB를 다시 만들어도 재과금 없음
- 검색 시 버전 필터와 함께 쓰려면 `SET hnsw.iterative_scan = relaxed_order` (pgvector 0.8)

## 전체 흐름
```
[질문] → 버전 파악 (질문에 명시된 버전 추출, 없으면 최신 버전 기본값)
      → 질문 재작성 (선택, 실험 대상)
      → 하이브리드 검색: 벡터 top-k + 키워드 top-k → RRF(Reciprocal Rank Fusion)로 결합, 버전 필터 적용
      → rerank → 상위 N개 조각
      → Claude 생성: 조각을 document 블록으로 전달, citations 활성화
            근거 부족 → "문서에서 찾을 수 없음" + 가장 가까운 관련 문서 링크
            버전별 차이 발견 → 버전별로 나눠서 답변
      → 응답: 답변 + 인용(원문 링크) + 검색된 조각 목록(디버그 패널)
      → 로그: 질문, 검색 결과 id, 점수, 지연 시간, 토큰 사용량
```

### 최소 RAG (구현, 마일스톤 2)
- `rag/query.py` 버전 파악(질문의 `1.xx`, 없으면 최신, 색인에 없으면 최신으로 답하고 알림) → `rag/retrieve.py` 벡터 검색(버전 필터, 해당 버전 URL) → `rag/generate.py` Sonnet 5(effort medium, max_tokens 2000), 청크마다 plain-text document + citations, URL·버전·feature state는 `context`
- 근거 부족 시 고정 문구 "I couldn't find this in the Kubernetes documentation."로 시작 → `found=False` (refusal accuracy 측정용)
- 인용은 URL 단위로 번호 부여, 질의 로그 `data/logs/queries.jsonl`
- 비용 상한: `rag/billing.py`가 요청 전 최악 비용(count_tokens 입력 + max_tokens 출력)을 $5 상한과 비교, SDK 자동 재시도 끔, 응답 유실 시 최악 비용으로 기록
- 검증 질문 10개: 질문당 $0.01~0.02, 2~8초. 버전별 feature gate 답변(1.35 alpha/disabled, 1.37 beta/enabled), 범위 밖 질문 거절, 존재하지 않는 필드 지적 확인
- 알려진 한계: 여러 버전 비교 질문은 첫 버전만 사용(E6), 인용된 코드 조각이 코드 블록 형식을 잃을 수 있음

## 평가 설계 (핵심 자산)

### 1. 정답 세트 (golden set)
목표 200~300문항. 각 문항 = `{question, reference_answer, evidence, version, category, answerable}`.

- (현황) dev 120 / test 80 = 200문항, 카테고리 비율 목표와 일치. 버전 짝 문항 12쌍. dev/test 배정은 `eval/assign_split.py`(카테고리 층화, 고정 시드, 짝 문항은 같은 split)로 사람이 고르지 않음
- 근거는 (페이지, 원문 문장) 쌍: 청크가 **같은 페이지**에서 나오고 문장을 포함해야 충족. "Feature state: ..." 같은 상용구가 최대 17개 페이지에 반복되므로 문장만으로는 오탐
- (변경) 정답을 `gold_chunk_ids` 대신 **근거 문장(evidence quote) + 섹션 URL**로 표시: 청킹을 바꿔도(E1) 라벨이 유효. 검색된 청크가 근거 문장을 포함하면 relevant
- 근거마다 **대체 근거(alternatives)**: 같은 사실을 다른 페이지가 말하면 그중 하나만 찾아도 충족. 검색이 놓친 문항의 상위 결과를 사람이 확인해 정당한 대체 근거를 추가(IR의 pooling 방식). 실험이 새로 찾은 대체 근거도 같은 방식으로 추가해야 설정 간 비교가 공정함
- `eval/golden.py`(스키마), `eval/validate.py`: 근거 문장이 해당 버전 문서의 해당 섹션에 실제로 있는지, dev/test 간 중복 질문(누수), 카테고리 비율을 검사. 현재 청킹에서 근거 문장이 청크 경계에 걸리면 경고

| 카테고리 | 비율 | 예시 형태 |
|---|---|---|
| 단순 사실 조회 | 25% | "기본 terminationGracePeriodSeconds 값은?" |
| How-to / 명령어 | 20% | "Deployment를 이전 리비전으로 롤백하는 kubectl 명령은?" |
| 버전 의존 | 20% | feature-state가 버전 사이에 바뀐 기능에 대한 질문 |
| 멀티홉 (여러 페이지 종합) | 15% | 두 개념을 연결해야 답할 수 있는 질문 |
| 답이 없는 질문 | 15% | 특정 클라우드 서비스 가격, Helm 차트 세부사항 등 문서 범위 밖 |
| 잘못된 전제 | 5% | 존재하지 않는 필드나 옵션을 전제로 한 질문 |

만드는 방법:
- **실제 질문 수집**: Stack Overflow `kubernetes` 태그, GitHub Issues, 커뮤니티 포럼에서 질문 문장을 가져와 정답은 직접 문서에서 작성. 합성 질문만 쓰면 실제 사용자 질문과 분포가 달라지는 문제를 방지
- **LLM 합성 + 사람 검수**: 문서 조각에서 Claude로 질문 후보를 생성한 뒤 직접 검수해서 채택/수정/폐기
- **dev / test 분리**: dev(60%)로만 튜닝하고, test(40%)는 마일스톤마다 한 번만 측정. test 세트에 과적합되지 않도록 README에 이 규칙을 명시

### 2. 지표
- (구현, `eval/metrics_retrieval.py`, `eval/run_retrieval.py`) 검색 지표는 LLM 없이 계산, 질문 임베딩은 캐시(재실행 비용 0). 결과: `eval/results/retrieval/<split>-<name>.{jsonl,summary.json}`
- **baseline (dev 49문항, voyage-4, heading-plain, k=20)**: Recall@5 0.765, Recall@10 0.929, Hit@5 0.857, MRR 0.716, nDCG@10 0.762. 약점: multihop Recall@5 0.389(두 번째 사실을 못 찾음), false premise(잘못된 전제가 검색을 엉뚱한 곳으로 유도)
| 단계 | 지표 | 측정 방법 |
|---|---|---|
| 검색 | Recall@k, MRR, nDCG@10 | gold_chunk_ids 대비, 코드로 계산 (LLM 불필요) |
| 생성 | Correctness | judge가 reference_answer와 비교, 0/1/2 루브릭 |
| 생성 | Faithfulness | 답변 문장을 claim으로 나누고 인용된 조각이 각 claim을 뒷받침하는지 판정 |
| 생성 | Citation precision | 인용된 조각 중 실제로 관련 있는 비율 |
| 생성 | Refusal accuracy | 답이 없는 질문 → 거절했나 / 답이 있는 질문 → 잘못 거절하지 않았나 (둘 다 측정) |
| 생성 | Version accuracy | 버전 의존 질문에서 올바른 버전 기준으로 답했나 |
| 운영 | p50/p95 지연 시간, 질문당 비용 | 요청 로그에서 집계 |

### (구현 현황, 마일스톤 3)
- 생성 eval `eval/run_generation.py`: Batches API(50% 할인), 제출 전 배치 전체 최악 비용을 예약하고 결과 후 정산, 재실행 시 같은 배치 재개. dev 120문항 답변 $1.08
- judge `eval/judge.py`: `claude-haiku-4-5`(답변 모델과 다른 모델, 비용 때문에 Opus 대신), structured outputs로 정확성 0/1/2 + 주장 단위 근거 충실도. 120문항 $0.20
- 보정 `eval/calibration.py`: 층화 50문항 사람 블라인드 채점 → 원점수 대비 kappa 0.31(점수 쏠림: 49/50이 2점), 불일치 5건 판정 후 rubric v2 → kappa 0.88. 같은 50문항으로 rubric을 고쳤으므로 낙관적 추정. 새 표본 30문항으로 재검증하면 편향 제거 가능
- 보고 `eval/report.py`: 모든 지표에 95% bootstrap CI, 설정 비교는 paired bootstrap(`--compare`)
- 누적 Claude 비용 $1.67 / $5

### 3. Judge 신뢰성 검증
- dev 세트 50문항은 직접 채점 → judge 채점과의 일치도(Cohen's kappa) 계산해 README에 공개
- 불일치 사례를 분석해 judge 프롬프트 개선 → 재측정
- judge 모델·프롬프트 버전을 결과 파일에 기록

### 4. 통계적 유의성
200~300문항에서는 1~2%p 차이가 노이즈일 수 있다. 모든 비교에 **bootstrap 95% 신뢰구간**을 붙이고, 같은 문항 쌍으로 비교(paired)한다.
→ "개선됐다"고 주장할 때 근거를 댈 수 있음. AI 기업 면접에서 특히 강한 포인트.

## 실험 계획
기준 설정(baseline)에서 한 번에 하나씩 바꿔 측정한다.

| # | 실험 | 비교 대상 |
|---|---|---|
| E1 | 청킹 전략 | 고정 길이 512토큰 vs 마크다운 헤딩 단위 vs 헤딩 단위 + 제목 경로(breadcrumb) prefix |
| E2 | Contextual chunking | 조각마다 "이 조각이 문서 전체에서 어떤 내용인지" 요약을 붙여 임베딩 (prompt caching으로 비용 절감) |
| E3 | 임베딩 모델 | 외부 API vs 오픈소스 |
| E4 | 검색 방식 | 벡터만 vs 키워드만 vs 하이브리드(RRF) |
| E5 | Reranker | 없음 vs 있음, rerank 전 후보 수 20/50/100 |
| E6 | 버전 처리 | 버전 필터 없음 vs 메타데이터 필터 vs 필터 + 버전별 답변 분리 |
| E7 | 질문 재작성 | 없음 vs 질문 재작성 vs HyDE |
| E8 | 생성 모델 / effort | `claude-opus-5` vs `claude-sonnet-5`, effort low/medium/high → 정확도 대비 비용·지연 시간 곡선 |

결과는 `experiments/results.md`에 표로 누적 (지표 + 신뢰구간 + 비용 + 커밋 해시).

### 실험 결과 요약
- **E4 (검색 방식)**: dev 102문항, paired bootstrap. 키워드만(Postgres `ts_rank`, OR 질의)은 모든 지표에서 유의하게 나쁨(Recall@10 −0.126). 하이브리드(RRF k=60, 후보 50+50)는 Recall@5 +0.034 [−0.010, +0.083]로 유의하지 않음, MRR −0.011. 카테고리별로 multihop Hit@5 0.889→1.000, false premise Recall@10 0.50→0.92 개선, howto는 하락. 기본값은 벡터 유지, E5에서 rerank와 결합해 재평가
- **E5 (Reranker)**: Voyage `rerank-3`(무료 토큰 대상; rerank-2.5는 무료 토큰 없음). 벡터 후보 50 → 상위 8: Recall@5 +0.059 [+0.010, +0.113], MRR +0.131 [+0.070, +0.194], nDCG@10 +0.112 [+0.061, +0.166] 모두 유의. 후보 20/50/100 차이 미미, 하이브리드 후보는 rerank 뒤 이점 없음. 호출당 0.28초, 질의당 약 1.4만 토큰. **새 기본값: 벡터 50 + rerank-3 → 8** (`rag.retrieve.DEFAULT`). 남은 약점: false premise(Recall@5 0.33)
- **E6 (버전 처리)**: 버전 필터(`versions @> [v]`)를 끄고 전 버전에서 검색(동일 텍스트는 질문 버전 occurrence 우선)하면 유의하게 나빠짐. 벡터 MRR −0.040 [−0.064, −0.019], rerank50 MRR −0.024 [−0.050, −0.004]. 하락은 version 카테고리에 집중(다른 버전 페이지가 상위 차지). **버전 필터 유지**
- **E7 (질의 변환)**: `claude-haiku-4-5`로 dev 질문을 한 번에 batch 변환(재작성 약 $0.01, HyDE 약 $0.04). rerank는 항상 원 질문으로 수행. 원 질문 대비: 재작성 nDCG@10 −0.024 [−0.053, −0.001]로 유의하게 나쁨(키워드 나열로 바뀌며 의도 손실: dev-006, 076, 120 등 7문항 하락, false premise Recall@5 −0.17. 프롬프트로 전제를 의심하라고 해도 전제를 그대로 옮김). HyDE는 Recall@10 +0.020 [−0.015, +0.059]로 유의하지 않음(4문항 개선, 1문항 하락). 원 질문 후보 ∪ 재작성 후보를 rerank하면 48문항에서 상위 8이 바뀌지만 근거 청크 순위는 그대로라 모든 지표 차이 0. 질의마다 LLM 호출(지연과 비용)을 더할 근거가 없으므로 **원 질문 유지**. false premise는 검색이 아니라 생성 단계의 과제로 남김
- **생성 eval (새 기본 검색)**: dev 120문항, Sonnet 5 medium, 검색 = 벡터 50 + rerank-3 → 8. baseline(벡터 8) 대비 correctness +0.008 [−0.033, +0.054], faithfulness +0.021 [−0.004, +0.050] (P=0.06), citation hit +0.029 모두 유의하지 않음. 잘못된 거절 1→0건, 답할 수 없는 질문 거절 18/18→17/18 (dev-052는 "문서가 최선을 정하지 않는다"고 답한 적절한 답변이라 judge 2점). correctness가 이미 0.94로 상한 근처라 검색 개선은 주로 faithfulness 쪽에 나타남. 비용 $1.17 + judge $0.22
- **E3 (오픈소스 임베딩)**: `Qwen/Qwen3-Embedding-0.6B`를 로컬(Apple MPS, fp16)에서 14,025개 임베딩(25.6분, $0; fp32는 16GB 메모리를 다 써서 swap으로 거의 멈춤). voyage-4 대비 벡터만: Recall@10 −0.034 [−0.088, +0.015], MRR −0.031 [−0.104, +0.044]; rerank-3 결합 시: Recall@10 −0.025 [−0.064, +0.010], MRR −0.010 [−0.037, +0.011]. 모두 유의하지 않지만 방향은 전부 음수. multihop이 가장 약함(Recall@5 0.546). **voyage-4 유지**, rerank가 임베딩 차이를 대부분 메움 → API 없이 돌려야 하는 환경이면 Qwen + rerank가 현실적 대안
- **E8 (답변 모델)**: 예산 때문에 dev 50문항(카테고리 비율 유지, seed 고정) subset에서 `claude-haiku-4-5`(thinking 없음) vs Sonnet 5 medium, 검색은 같은 기본값. correctness +0.010 [−0.040, +0.060], faithfulness +0.001 [−0.028, +0.027], citation hit −0.048 [−0.143, +0.048]로 차이가 유의하지 않음. 질문당 비용 $0.0031 vs $0.0100(약 1/3), 출력 토큰 180 vs 621. 한계: n=50이라 ±0.05 안쪽 차이는 구분 못 하고, judge도 Haiku라 자기 선호 편향을 배제할 수 없음. **기본값은 Sonnet 5 유지**(test 측정 설정), 비용이 중요한 배포에서는 Haiku가 합리적 대안. 비용 $0.15 + judge $0.07
- **test split 최종 측정 (1회)**: dev에서 고른 설정(벡터 50 + rerank-3 → 8, Sonnet 5 medium) 그대로. 검색(68문항): baseline 대비 MRR +0.130 [+0.068, +0.194], nDCG@10 +0.109, Recall@10 +0.049 모두 유의해 E5 결론이 held-out에서도 재현됨. 생성(80문항): correctness 0.988 [0.969, 1.000], faithfulness 0.997, citation hit 0.985, 답할 수 없는 질문 거절 100%, 잘못된 거절 0%. dev보다 높은 것은 dev가 pooling·오류 분석으로 어려운 문항이 드러난 쪽이기 때문으로 보이며, 판정 모델 편향(kappa 한계)도 감안해야 함. 비용 $0.81 + judge $0.15

## 디렉터리 구조
```
ingest/
  fetch.py              # kubernetes/website 브랜치별 문서 수집
  clean.py              # Hugo shortcode 정리, feature-state 메타데이터 추출
  chunk.py              # 청킹 전략들 (전략별 함수, 설정으로 선택)
  dedupe.py             # content hash 기반 버전 간 중복 제거
  embed.py              # 임베딩 생성 및 DB 적재
rag/
  retrieve.py           # 벡터 / 키워드 / 하이브리드(RRF), 버전 필터
  rerank.py
  query.py              # 버전 추출, 질문 재작성
  generate.py           # Claude 호출, citations, 거절 처리
  pipeline.py           # 설정(yaml) → 파이프라인 조립
  config/               # baseline.yaml, e1_heading.yaml ...
api/
  main.py               # FastAPI: POST /ask (SSE), GET /eval/results
eval/
  dataset/              # dev.jsonl, test.jsonl
  build_dataset.py      # 합성 질문 생성 + 검수 도구 (CLI)
  metrics_retrieval.py  # Recall@k, MRR, nDCG
  judge.py              # correctness / faithfulness judge (Batches API)
  run.py                # 설정 파일 하나로 eval 실행 → results/*.jsonl
  report.py             # 집계, bootstrap CI, 실험 비교 표 생성
  calibration/          # 사람 채점 결과, kappa 계산
web/                    # Next.js: 질문 UI + eval 대시보드
tests/                  # pytest (전처리, 청킹, RRF, 지표 계산 단위 테스트)
.github/workflows/      # ci.yml, eval.yml
```

## 마일스톤
1. **데이터 파이프라인**: 문서 수집, shortcode 정리, feature-state 추출, 헤딩 단위 청킹, 중복 제거, pgvector 적재
   - 완료 기준: 3개 버전 인덱싱, 중복 제거율 수치 확인, 전처리 단위 테스트 통과
2. **최소 RAG**: 벡터 검색 + Claude 생성 + citations, CLI로 질문 가능
   - 완료 기준: 질문 10개에 인용 포함 답변 생성
3. **eval 기반 구축** (가장 중요): 정답 세트 dev 120문항 이상, 검색 지표, judge, calibration, bootstrap CI
   - 완료 기준: baseline 점수표 확보, judge kappa 측정
4. **실험 E1~E8**: 하나씩 적용하고 결과 표 누적, 가장 좋은 조합을 새 기본값으로
   - 완료 기준: 실험 표 완성, test 세트로 최종 점수 1회 측정
5. **서비스화**: FastAPI SSE, Next.js UI (답변 + 인용 링크 + 버전 선택 + 검색 조각 디버그 패널), eval 대시보드 페이지
6. **CI & 배포 & 문서화**
   - PR마다: 단위 테스트 + dev 검색 eval (LLM 없음, Voyage 무료 토큰, 캐시), 결과를 PR 코멘트로 게시, 기준 대비 하락 시 실패. 생성 eval은 Claude 예산($5) 때문에 PR마다 돌리지 않고 수동 실행
   - 전체 eval: 수동 실행 또는 주 1회 (Batches API)
   - 배포, rate limit, 동일 질문 응답 캐시. 공개 데모는 실시간 Claude 호출 없이 미리 생성한 예시 답변을 보여주거나, 방문자 본인 API 키를 쓰는 방식 (운영자 비용 $0)
   - README (영어): 아키텍처 다이어그램, 실험 결과 표, 설계 결정과 트레이드오프, 한계점
   - 기술 블로그 글 1편: 실험 과정에서 발견한 가장 흥미로운 결과

## 리스크 & 대응
- **정답 세트 제작 시간** → 가장 큰 비용. 마일스톤 3에 충분한 시간 배정, 합성 + 검수로 속도 확보. 처음엔 120문항으로 시작해 점진 확대
- **judge 편향** (길고 자신감 있는 답변 선호 등) → 사람 채점과의 kappa 측정, 루브릭을 구체적으로
- **test 세트 과적합** → dev/test 분리, test는 마일스톤마다 1회만
- **문서 업데이트로 정답이 바뀜** → 수집 시점의 커밋 해시 고정, 정답 세트에 기준 커밋 기록
- **비용** → eval은 Batches API, 검색 지표는 LLM 없이 계산, 실험 중 생성 결과 캐싱(설정 hash + 질문 hash), 공개 데모에 rate limit
- **공개 데모 남용** → IP별 rate limit, 입력 길이 제한, 일일 예산 상한

## 면접에서 설명할 포인트
- 왜 벡터 DB 대신 Postgres 하나로 갔나 (운영 단순성 vs 대규모 성능)
- 버전 간 중복 제거와 버전 필터링 설계
- 하이브리드 검색이 어떤 유형의 질문에서 효과가 있었나 (예: 필드명·명령어처럼 정확한 토큰이 중요한 질문)
- judge를 어떻게 믿을 수 있게 만들었나 (kappa, 루브릭)
- 개선이 노이즈가 아니라는 근거 (paired bootstrap CI)
- 모델·effort 선택에 따른 비용/정확도/지연 시간 트레이드오프
- 규모가 커지면 무엇을 바꿀 것인가 (전용 벡터 DB, 인덱스 샤딩, 캐시 계층, 온라인 평가)
