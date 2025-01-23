# rag-expense-explainer

## 개요
rag-expense-explainer는 LangChain을 활용한 Retrieval-Augmented Generation(RAG) 프로젝트로, 카드 사용 내역을 적합한 계정과목(예: 복리후생비, 사무용품비 등)으로 분류하고 그 이유를 한국어로 설명합니다. 문서 임베딩, 검색 기반 질의응답 및 자연어 생성을 결합하여 분류를 자동화합니다.

## 주요 기능
- **PDF 문서 처리**: PDF 파일에서 텍스트를 추출하고 벡터 데이터베이스로 변환합니다.
- **비용 분류**: 머신러닝 모델을 사용하여 카드 사용 내역의 계정과목을 예측합니다.
- **설명 생성**: 계정과목 분류의 이유를 한국어로 상세히 설명합니다.
- **사용자 맞춤 질의 지원**: 특정 카드 사용 내역이 특정 계정과목으로 분류된 이유를 질의.

## 기술 스택
- **프로그래밍 언어**: Python
- **프레임워크 및 라이브러리**:
  - LangChain: RAG 파이프라인 구축
  - OpenAI: 임베딩 및 자연어 생성
  - Chroma: 벡터 데이터베이스 관리
  - PyPDFLoader: PDF 텍스트 추출
- **개발 환경**:
  - Jupyter Notebook: 프로토타이핑 및 분석

## 설치
### 사전 준비
1. Python 3.8 이상 설치
2. 의존성 설치:

```bash
pip install langchain chromadb openai pypdf
```

3. OpenAI API 키를 환경 변수로 설정:

```bash
export OPENAI_API_KEY='your-openai-api-key'
```

## 작동 원리
1. **문서 로드 및 분할**:
   - 계정과목 정의가 포함된 PDF 문서(`계정과목해설.pdf`)를 로드하고 분할합니다.

   ```python
   from langchain.document_loaders import PyPDFLoader
   from langchain.text_splitter import CharacterTextSplitter

   loader = PyPDFLoader("계정과목해설.pdf")
   documents = loader.load()

   text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=6)
   texts = text_splitter.split_documents(documents)
   ```

2. **임베딩 및 벡터 스토어 생성**:
   - 텍스트 청크를 OpenAI 임베딩을 통해 벡터화하고, Chroma 벡터 데이터베이스에 저장합니다.

   ```python
   from langchain.embeddings.openai import OpenAIEmbeddings
   from langchain.vectorstores import Chroma

   embeddings = OpenAIEmbeddings()
   vector_store = Chroma.from_documents(texts, embeddings)
   retriever = vector_store.as_retriever(search_kwargs={"k": 3})
   ```

3. **질의 및 설명 생성**:
   - `RetrievalQAWithSourcesChain`을 사용해 관련 문서를 검색하고, 카드 사용 내역 분류 이유를 생성합니다.

   ```python
   from langchain.chains import RetrievalQAWithSourcesChain
   from langchain.chat_models import ChatOpenAI

   llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)
   chain = RetrievalQAWithSourcesChain.from_chain_type(
       llm=llm,
       chain_type="stuff",
       retriever=retriever,
       return_source_documents=True
   )

   query = "해당 지출이 왜 복리후생비로 분류되었는지 설명해주세요."
   result = chain(query)
   print(result['answer'])
   ```

4. **비용 분류 파이프라인**:
   - 카드 사용 내역 데이터를 처리하고, 계정과목 분류 및 분류 이유를 제공합니다.
   - 출력 예시는 다음과 같습니다:
     - 카드 사용 내역
     - 분류 결과와 확률
     - 분류 기준 설명 (한국어)

## 예제 출력
### 입력
카드 사용 내역: `"종로면선생 판교파미어스몰점" 60,000원 (창현님과 저녁식사)`

### 출력
**분류 결과**:
- 복리후생비 (99.81% 확률)
- 접대비 (0.12% 확률)
- 사무용품비 (0.03% 확률)

**설명 (한국어)**:
```
카드 사용 내역 "종로면선생 판교파미어스몰점"에서 발생한 비용은 복리후생비로 분류되었습니다. 복리후생비는 회사나 조직에서 직원의 복지를 위해 제공되는 지출로, 저녁식사 비용이 포함될 수 있습니다. 해당 내역은 복리후생비와 가장 높은 연관성을 보였습니다.
```

## 레포지토리 구조
```
RAGCardClassifier/
├── data/
│   └── 계정과목해설.pdf          # 소스 문서
├── notebooks/
│   └── granter_classification_reason.ipynb  # 프로토타이핑 노트북
├── src/
│   ├── document_processing.py    # PDF 처리 및 임베딩
│   ├── classification.py         # 분류 로직
│   └── explain_generation.py     # 설명 생성
├── tests/
│   └── test_classification.py    # 분류 유닛 테스트
└── README.md
```

## 실행 방법
1. 레포지토리 클론:
   ```bash
   git clone https://github.com/your-username/RAGCardClassifier.git
   cd RAGCardClassifier
   ```

2. Jupyter Notebook 실행:
   ```bash
   jupyter notebook notebooks/granter_classification_reason.ipynb
   ```

3. Python 스크립트 실행:
   ```bash
   python src/classification.py
   ```

## 향후 개선 사항
- **다국어 지원**: 한국어 외 영어 및 다른 언어 지원.
- **분류 정확도 개선**: 고급 미세 조정 모델 통합.
- **웹 인터페이스 개발**: 사용자 친화적 질의응답 웹 UI 추가.
- **회계 소프트웨어와 통합**: 기업 환경에 맞는 자동화 분류 제공.

## 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.

## 감사의 말
- LangChain 팀의 강력한 프레임워크.
- OpenAI의 GPT 모델 제공.
- 피드백 및 제안을 제공해 주신 분들께 감사드립니다.

