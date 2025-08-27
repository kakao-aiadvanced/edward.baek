# 🤖 RAG Assistant with LangGraph

AI 에이전트, 프롬프트 엔지니어링, LLM 보안에 특화된 RAG(Retrieval-Augmented Generation) 시스템입니다. LangGraph를 사용하여 지능적으로 문서 검색과 웹 검색 사이를 라우팅하며, 할루시네이션 검증 기능을 포함합니다.

## ✨ 주요 기능

- **🎯 지능형 라우팅**: 질문 내용에 따라 자동으로 문서 검색 또는 웹 검색 선택
- **📊 문서 품질 평가**: 검색된 문서의 관련성 자동 평가
- **🔍 할루시네이션 검증**: 생성된 답변의 신뢰성 검사 (최대 3회 재시도)
- **🌐 웹 검색 통합**: Tavily를 활용한 실시간 웹 검색
- **📚 출처 추적**: 답변에 사용된 문서와 웹 페이지 출처 표시
- **💬 Streamlit UI**: 사용자 친화적인 채팅 인터페이스

## 🏗️ 시스템 아키텍처

```
질문 입력 → 라우팅 → 문서검색/웹검색 → 문서 평가 → 답변 생성 
→ 할루시네이션 검증 → 답변 품질 확인 → 최종 답변 or 재시도
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. API 키 설정

`.env` 파일을 만들거나 Streamlit 사이드바에서 직접 입력:

```bash
cp .env.template .env
# .env 파일을 편집하여 API 키 입력
```

### 3. 실행 방법

#### 🖥️ Streamlit 웹 앱 (추천)

```bash
# 스크립트로 실행
./run_streamlit.sh

# 또는 직접 실행
streamlit run streamlit_app.py
```

브라우저에서 `http://localhost:8501` 접속

#### 💻 커맨드라인

```python
from rag_langgraph import RAGLangGraph

# 시스템 초기화
rag = RAGLangGraph()

# 질문하기
result = rag.run_query("LLM에 대한 공격 방법은 무엇인가요?")
print(result["generation"])
```

#### 🧪 테스트 실행

```bash
python test_rag.py
```

## 📱 Streamlit 앱 기능

### 🎛️ 사이드바 기능
- **🔑 API 키 관리**: OpenAI, Tavily API 키 입력
- **⚙️ 설정**: 최대 재시도 횟수 조정
- **🗑️ 채팅 기록 삭제**
- **ℹ️ 시스템 정보**: 지원 주제 및 기능 안내

### 💬 채팅 인터페이스
- **실시간 진행 상황**: 각 처리 단계별 진행률 표시
- **📚 참고 문헌**: 클릭 가능한 출처 링크
- **🔍 상세 정보**: 재시도 횟수, 검증 결과 등 메타데이터
- **채팅 히스토리**: 대화 내용 자동 저장

## 📁 파일 구조

```
day3/
├── streamlit_app.py          # Streamlit 웹 앱
├── rag_langgraph.py          # 메인 RAG 시스템
├── document_loader.py        # 문서 로더
├── test_rag.py              # 테스트 스크립트
├── run_streamlit.sh         # 실행 스크립트
├── requirements.txt         # 의존성
├── .env.template           # 환경변수 템플릿
└── README.md              # 이 파일
```

## 📚 지식 베이스

시스템에 포함된 문서:
- [AI Agent 가이드](https://lilianweng.github.io/posts/2023-06-23-agent/)
- [프롬프트 엔지니어링](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
- [LLM 적대적 공격](https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/)

## 🔑 필수 API 키

- **OpenAI API Key**: GPT 모델 및 임베딩용
  - [OpenAI API 키 발급](https://platform.openai.com/api-keys)
- **Tavily API Key**: 웹 검색용
  - [Tavily API 키 발급](https://app.tavily.com/)

## 💡 사용 팁

### 질문 예시
- "AI 에이전트의 메모리 유형은 무엇인가요?"
- "프롬프트 엔지니어링의 주요 기법들을 알려주세요"
- "LLM에 대한 적대적 공격 방법은?"
- "테슬라의 최신 뉴스는?" (웹 검색)

### 최적 사용법
1. **구체적인 질문**: 명확하고 구체적인 질문일수록 좋은 답변
2. **전문 분야**: AI/ML 관련 질문에 특화되어 있음
3. **출처 확인**: 답변 하단의 참고 문헌을 통해 원본 확인 가능
4. **재시도 활용**: 만족스럽지 않은 답변은 다시 질문해보세요

## 🛡️ 안전 기능

- **할루시네이션 감지**: 부정확한 정보 생성 방지
- **출처 검증**: 모든 답변에 참고 문헌 포함
- **재시도 제한**: 최대 3회 재시도로 무한 루프 방지
- **답변 불가 안내**: 신뢰할 수 없는 경우 솔직한 안내

## 🔧 문제 해결

### 일반적인 오류
1. **API 키 오류**: `.env` 파일 또는 사이드바에서 올바른 키 입력 확인
2. **모듈 오류**: `pip install -r requirements.txt`로 의존성 재설치
3. **포트 충돌**: 다른 포트 사용 `streamlit run streamlit_app.py --server.port 8502`

### 성능 최적화
- 첫 실행 시 문서 로딩으로 시간이 소요될 수 있습니다
- ChromaDB 벡터 저장소는 `./chroma_db`에 캐시됩니다
- 네트워크 연결이 웹 검색 성능에 영향을 줍니다