# RAG Template Monorepo (Scaffold)

A **local-first** Retrieval-Augmented Generation (RAG) template organized as a monorepo.  
Focus: clear **contracts** and **separation of concerns**. Implementation is intentionally minimal so you can grow it step by step.

## Packages

```text
packages/
  rag-data/        # Ingest & chunk → SQLite catalog + FAISS index (artifacts/)
  rag-backend/     # Unified FastAPI backend: /, /health, /docs, /redoc, /api/generate (+ optional /api/retrieve)
  rag-frontend/    # Minimal Vite+React UI calling /api/generate
```

### Why this split?

- **rag-data** produces FAISS index that are **read-only** for runtime services.
- **rag-backend** orchestrates retrieval + prompt + LLM and returns **answer + citations** (and can expose `/api/retrieve` for internal tuning).
- **rag-frontend** shows the answer and references.

## Data

- RAG Reference Documents example: `data/chunks.jsonl`

```jsonl
{"chunk_text": "세종(한국 한자: 世宗, 중세 한국어: ·솅조ᇰ, 1397년 5월 15일 (음력 4월 10일) ~ 1450년 3월 30일 (음력 2월 17일))은 조선의 제4대 국왕(재위 : 1418년 9월 9일 ~ 1450년 3월 30일)으로, 태종과 원경왕후의 아들이다. 형인 양녕대군이 폐세자가 되자 세자에 책봉되었으며 태종의 양위를 받아 즉위하였다.", "chunk_index": 0, "title": "세종", "url": "<https://ko.wikipedia.org/wiki/%EC%84%B8%EC%A2%85>", "source_type": "Wikipedia"}
{"chunk_text": "세종은 과학 기술, 예술, 문화, 국방 등 여러 분야에서 다양한 업적을 남겼다. 백성들에게 농사에 관한 책을 펴내었지만 글을 몰라 이해하지 못하는 모습을 보고 누구나 쉽게 배울 수 있는 효율적이고 과학적인 문자 체계인 훈민정음(訓民正音)을 창제하였다. 훈민정음은 언문으로 불리며 왕실과 민간에서 사용되다가 20세기 주시경이 한글로 발전시켜 오늘날 대한민국의 공식 문자로서 널리 쓰이고 있다.\n과학 기술에도 두루 관심을 기울여 혼천의, 앙부일구{"id": "bb149fce-12ac-46d9-9460-85a1dc46144b", "chunk_text": "세종(한국 한자: 世宗, 중세 한국어: ·솅조ᇰ, 1397년 5월 15일 (음력 4월 10일) ~ 1450년 3월 30일 (음력 2월 17일))은 조선의 제4대 국왕(재위 : 1418년 9월 9일 ~ 1450년 3월 30일)으로, 태종과 원경왕후의 아들이다. 형인 양녕대군이 폐세자가 되자 세자에 책봉되었으며 태종의 양위를 받아 즉위하였다.", "chunk_index": 0, "title": "세종", "url": "https://ko.wikipedia.org/wiki/%EC%84%B8%EC%A2%85", "source_type": "Wikipedia"}
{"id": "2eeb9a3d-7e27-4757-8ac6-71352764118a", "chunk_text": "세종은 과학 기술, 예술, 문화, 국방 등 여러 분야에서 다양한 업적을 남겼다. 백성들에게 농사에 관한 책을 펴내었지만 글을 몰라 이해하지 못하는 모습을 보고 누구나 쉽게 배울 수 있는 효율적이고 과학적인 문자 체계인 훈민정음(訓民正音)을 창제하였다. 훈민정음은 언문으로 불리며 왕실과 민간에서 사용되다가 20세기 주시경이 한글로 발전시켜 오늘날 대한민국의 공식 문자로서 널리 쓰이고 있다.\n과학 기술에도 두루 관심을 기울여 혼천의, 앙부일구, 자격루, 측우기 등의 발명을 전폭적으로 지원했고 신분을 뛰어넘어 장영실, 최해산 등의 학자들을 후원하였다.\n국방에 있어서는 이종무를 파견하여 왜구를 토벌하고 대마도를 정벌하였으며 이징옥, 최윤덕, 김종서 등을 북방으로 보내 평안도와 함길도에 출몰하는 여진족을 국경 밖으로 몰아내고 4군 6진을 개척하여 압록강과 두만강 유역으로 국경을 확장하였고 백성들을 옮겨 살게 하는 사민정책(徙民政策)을 실시하여 국토의 균형 발전을 위해서도 노력하였다.", "chunk_index": 1, "title": "세종", "url": "https://ko.wikipedia.org/wiki/%EC%84%B8%EC%A2%85", "source_type": "Wikipedia"}
{"id": "4a49c61e-b4f3-4f7a-8003-63c0e7de26cb", "chunk_text": "정치면에서는 황희와 맹사성, 윤회, 김종서 등을 등용하여 정무를 주관하였는데 이 통치 체제는 일종의 내각중심 정치제도인 의정부서사제의 효시가 되었다. 이 밖에도 법전과 문물을 정비하였고 전분 6등법과 연분 9등법 등의 공법(貢法)을 제정하여 조세 제도의 확립에도 업적을 남겼다.\n오늘날 대한민국에서는 세종의 업적에 대한 존경의 의미를 담아 '세종대왕'(世宗大王)으로 부르기도 한다.", "chunk_index": 2, "title": "세종", "url": "https://ko.wikipedia.org/wiki/%EC%84%B8%EC%A2%85", "source_type": "Wikipedia"}
```

- data (output of `rag-data`):

  ```text
  data/
    wiki.jsonl             # output of collect_wiki.py
    chunks.jsonl           # output of build_chunks.py
    faiss_index/           # output of build_faiss.py
  ```

## APIs

### Base

- OpenAPI: `3.1.0`
- Title/Version: `rag-backend 0.1.0`:contentReference[oaicite:1]{index=1}

---

### `GET /` — Root

- 간단한 루트 응답
- **200 OK**

  ```json
  {}
  ```

---

### `GET /health` — Health Check

- 서버 상태 확인
- **200 OK**

  ```json
  {}
  ```

---

### `POST /api/generate` — Generate

사용자 질문에 대한 **LLM 응답 생성** (RAG 옵션 포함)

#### Request

```json
{
  "query": "세종대왕 업적 요약해줘",
  "use_rag": true,
  "max_tokens": 4096
}
```

- `query` *(string, required)*: 질문 텍스트
- `use_rag` *(boolean, default: true)*: RAG 사용 여부
- `max_tokens` *(integer, default: 4096)*: 최대 생성 토큰 수

#### Response 200

```json
{
  "response": "세종은 ...",
  "reference_documents": [
    {
      "chunk_text": "세종...",
      "chunk_index": 0,
      "title": "세종",
      "url": "https://ko.wikipedia.org/wiki/%EC%84%B8%EC%A2%85",
      "source_type": "Wikipedia",
      "score": 0.87
    }
  ],
  "elapsed_ms": 420
}
```

- `response` *(string, required)*: 생성된 답변
- `reference_documents` *(DocumentChunk\[] | null, default: \[])*: 참조 문서 목록
- `elapsed_ms` *(integer, required)*: 처리 시간(ms)

#### DocumentChunk Schema

- `chunk_text` *(string)*
- `chunk_index` *(integer)*
- `title` *(string)*
- `url` *(string)*
- `source_type` *(string)*
- `score` *(number)*
  (모두 required)

#### Errors

- `422` Validation Error

---

### `POST /api/retrieve` — Retrieve

질문에 대한 **관련 문서 검색**

#### Request

```json
{
  "query": "세종 업적",
  "candidate_k": 10,
  "top_k": 3
}
```

- `query` *(string, required)*: 검색 질의
- `candidate_k` *(integer, default: 10)*: 1차 후보 수
- `top_k` *(integer, default: 3)*: 최종 반환 수

#### Response 200

```json
{
  "chunks": [
    {
      "chunk_text": "세종은 과학 기술...",
      "chunk_index": 1,
      "title": "세종",
      "url": "https://ko.wikipedia.org/wiki/%EC%84%B8%EC%A2%85",
      "source_type": "Wikipedia",
      "score": 0.91
    }
  ],
  "elapsed_ms": 35
}
```

- `chunks` *(DocumentChunk\[], required)*: 검색된 청크들
- `elapsed_ms` *(integer, required)*: 처리 시간(ms)

#### Errors

- `422` Validation Error

## Setup (venv + editable installs)

```bash
# 0) (optional) create and activate a virtualenv
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# install backend dependent packages(FastAPI/uvicorn, etc.)
pip install -r packages/rag-backend/requirements.txt
```

## Quick Start (scaffold)

1) **Start Backend (stub)**

   ```bash
   cd packages/rag-backend
   python run_server.py
   # open http://localhost:8000/ , /health , /docs , /redoc
   # to expose /api/retrieve:
   # EXPOSE_RETRIEVE_ENDPOINT=true python run_server.py
   ```

2) **Start UI**

   ```bash
   cd packages/rag-frontend
   npm install
   npm run dev
   # open http://localhost:5173
   ```

---

## Contributors

- Sangbong Lee

**License**: MIT
