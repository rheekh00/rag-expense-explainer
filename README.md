# rag-expense-explainer

## 개요
rag-expense-explainer는 LangChain을 활용한 Retrieval-Augmented Generation(RAG) 프로젝트로, 카드 사용 내역을 적합한 계정과목(예: 복리후생비, 사무용품비 등)으로 분류하고 그 이유를 한국어로 설명합니다. 문서 임베딩, 검색 기반 질의응답 및 자연어 생성을 결합하여 분류를 자동화합니다.

계정과목은 기업의 모든 거래를 기록하기 위해 사용하는 분류 체계로, 복리후생비, 사무용품비 등과 같이 지출의 성격에 따라 정의됩니다. 본 repository는 이러한 계정과목 분류와 분류 근거 설명 생성을 자동화하여 효율성을 높이는 프로젝트입니다.

## 주요 기능
- **PDF 문서 처리**: PDF 파일에서 텍스트를 추출하고 벡터 데이터베이스로 변환합니다.
- **비용 분류**: 딥러닝 모델을 사용하여 카드 사용 내역의 계정과목을 예측합니다.
- **설명 생성**: 계정과목 분류의 이유를 한국어로 상세히 설명합니다.

## 기술 스택
- **프로그래밍 언어**: Python
- **프레임워크 및 라이브러리**:
  - LangChain: RAG 파이프라인 구축
  - OpenAI: 임베딩 및 자연어 생성
  - Chroma: 벡터 데이터베이스 관리
  - PyPDFLoader: PDF 텍스트 추출
- **개발 환경**:
  - Jupyter Notebook: 프로토타이핑 및 분석



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
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import RetrievalQAWithSourcesChain, LLMChain
    from langchain.prompts import PromptTemplate 
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    
    
    system_template="""Explain why the card usage was classified as the account subject in Korean.
    You should consider the card usage information and make explanation relating it with the definition, characteristics, and classification of the account subject.
    And also, include the 
    ----------
    {summaries}
    ----------
    
    The explanation should be detailed including the rational reason for the classification of the account subject, and the explanation of the account subject classification criteria.
    Answer kindly to a customer in Korean: {question}
    
    """
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    
    prompt = ChatPromptTemplate.from_messages(messages)
    
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import RetrievalQAWithSourcesChain
    
    chain_type_kwargs = {"prompt": prompt}
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)  # Modify model_name if you have access to GPT-4
    
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever = retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )
   ```

4. **비용 분류 파이프라인**:
   - 카드 사용 내역 데이터를 처리하고, 계정과목 분류 및 분류 이유를 제공합니다.
   - 출력 예시는 다음과 같습니다:
     - 카드 사용 내역
     - 분류 결과와 확률
     - 분류 기준 설명 (한국어)

## 예제 출력
### 입력 1
카드 사용 내역: `"종로면선생 판교파미어스몰점" 60,000원 (창현님과 저녁식사)`

### 출력 2
**분류 결과**:
- 복리후생비 (99.81% 확률)
- 접대비 (0.12% 확률)
- 사무용품비 (0.03% 확률)

**설명 (한국어)**:
```
카드 사용 내역 "종로면선생 판교파미어스몰점"에서 발생한 비용은 복리후생비로 분류되었습니다.
복리후생비는 회사나 조직에서 직원의 복지를 위해 제공되는 지출로, 저녁식사 비용이 포함될 수 있습니다.
해당 내역은 복리후생비와 가장 높은 연관성을 보였습니다.
```


### 입력 2
카드 사용 내역: `"이케아코리아 유한회사" 545000원 (책상 4개 구매)`

### 출력 2
**분류 결과**:
- 사무용품비 (88.60% 확률)
- 복리후생비 (5.10% 확률)
- 소모품비 (4.79% 확률)

**설명 (한국어)**:
```
이케아코리아 유한회사의 카드 사용 내역 "545,000원 (책상 4개 구매)"이 사무용품비(88.60% 확률), 복리후생비(5.10% 확률), 소모품비(4.79% 확률)라는 회계과목으로 분류된 이유를 설명드리겠습니다.

분류된 회계과목인 사무용품비는 사무용 제품의 구매비용에 해당합니다. 여기서 "책상 4개 구매"라는 내용을 보면, 이는 회사의 업무를 수행하는데 필요한 사무용품 구매로 해석할 수 있습니다. 따라서, 이 사용 내역은 제고나 재고로 관리되지 않고 일회성으로 사용된 것으로 판단되어 사무용품비로 분류되었습니다.

또한, 복리후생비는 직원들의 복리후생 및 간접적인 재화나 서비스에 소요되는 비용을 포함합니다. 하지만 "책상 4개 구매"는 직원들의 복리후생을 위한 구매로 해석하기에는 한계가 있습니다.

소모품비는 회사의 업무나 생산과정에서 소규모로 사용되거나 소모되는 물품의 구입비용을 포함합니다. "책상 4개 구매"는 1년 이상 사용할 수 있는 취득단가가 50만원 미만인 소액의 물품으로 분류된 것입니다.

따라서, 이카이코리아 유한회사의 카드 사용 내역이 사무용품비로 분류된 이유는 구매한 책상이 회사의 업무에 필요한 사무용품이라고 판단되어 분류되었습니다.
```

