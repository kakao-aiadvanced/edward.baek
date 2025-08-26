# -*- coding: utf-8 -*-
import os
import re
from typing import List, Dict
import logging
from pathlib import Path
import unicodedata

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """텍스트에서 surrogate characters와 문제가 있는 문자들을 제거합니다."""
    if not text:
        return ""
    
    try:
        # Surrogate characters 제거
        cleaned = ''.join(char for char in text if not (0xD800 <= ord(char) <= 0xDFFF))
        
        # Unicode 정규화
        cleaned = unicodedata.normalize('NFC', cleaned)
        
        # 제어 문자 제거 (필수 문자는 유지: \n, \r, \t)
        cleaned = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', cleaned)
        
        # 인코딩 테스트
        cleaned.encode('utf-8')
        
        return cleaned.strip()
    except Exception as e:
        logger.warning(f"텍스트 정리 중 오류: {e}")
        # 안전한 fallback: ASCII만 유지
        return re.sub(r'[^\x20-\x7E\n\r\t가-힣]', '', text).strip()


class RAGSystem:
    def __init__(self, openai_api_key: str, documents_path: str = "documents"):
        self.openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        self.vectorstore = None
        self.documents_path = Path(documents_path)
        self.documents_path.mkdir(exist_ok=True)
        
        self.relevance_threshold = 0.7
        self.max_hallucination_retries = 5
        
        self._setup_prompts()
    
    def _setup_prompts(self):
        self.answer_prompt = ChatPromptTemplate.from_template("""
        당신은 도움이 되는 AI 어시스턴트입니다. 제공된 컨텍스트를 바탕으로 사용자의 질문에 답변해주세요.
        
        컨텍스트:
        {context}
        
        질문: {question}
        
        답변할 때 다음 규칙을 따라주세요:
        1. 제공된 컨텍스트에서만 정보를 사용하세요
        2. 컨텍스트에 없는 정보는 추측하지 마세요
        3. 명확하고 정확한 답변을 제공하세요
        4. 컨텍스트가 불충분하면 그렇게 말씀해주세요
        
        답변:
        """)
        
        self.relevance_prompt = ChatPromptTemplate.from_template("""
        다음 검색된 문서가 사용자의 질문과 얼마나 관련이 있는지 0-10점으로 평가해주세요.
        
        질문: {question}
        
        문서 내용:
        {document}
        
        0-3점: 전혀 관련 없음
        4-6점: 약간 관련 있음
        7-8점: 관련성 높음
        9-10점: 매우 관련성 높음
        
        점수만 숫자로 답변해주세요 (예: 7):
        """)
        
        self.hallucination_check_prompt = ChatPromptTemplate.from_template("""
        다음 답변이 제공된 컨텍스트에 근거하여 작성되었는지 확인해주세요.
        
        컨텍스트:
        {context}
        
        답변:
        {answer}
        
        질문: {question}
        
        답변이 컨텍스트에 근거하여 작성되었으면 "YES", 그렇지 않거나 컨텍스트에 없는 정보를 포함했으면 "NO"로 답변해주세요.
        
        판단 결과:
        """)

    def load_documents(self, file_paths: List[str]) -> List[Document]:
        documents = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                
                # 텍스트 정리
                cleaned_content = clean_text(content)
                if not cleaned_content:
                    logger.warning(f"문서 내용이 비어있음: {file_path}")
                    continue
                    
                doc = Document(
                    page_content=cleaned_content,
                    metadata={"source": file_path}
                )
                documents.append(doc)
                logger.info(f"문서 로드됨: {file_path}")
                
            except Exception as e:
                logger.error(f"문서 로드 실패 {file_path}: {e}")
        
        return documents

    def create_vectorstore(self, documents: List[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        texts = text_splitter.split_documents(documents)
        logger.info(f"문서를 {len(texts)}개 청크로 분할")
        
        self.vectorstore = FAISS.from_documents(texts, self.embeddings)
        logger.info("벡터 저장소 생성 완료")

    def search_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        if self.vectorstore is None:
            raise ValueError("벡터 저장소가 초기화되지 않았습니다. create_vectorstore()를 먼저 실행하세요.")
        
        docs = self.vectorstore.similarity_search(query, k=k)
        logger.info(f"검색된 문서: {len(docs)}개")
        
        return docs

    def check_relevance(self, query: str, documents: List[Document]) -> List[Document]:
        relevant_docs = []
        
        for doc in documents:
            try:
                # 텍스트 정리
                clean_query = clean_text(query)
                clean_doc = clean_text(doc.page_content[:500])
                
                if not clean_query or not clean_doc:
                    logger.warning("관련성 평가를 위한 텍스트가 비어있음, 문서 포함")
                    relevant_docs.append(doc)
                    continue
                
                relevance_chain = self.relevance_prompt | self.llm | StrOutputParser()
                score_str = relevance_chain.invoke({
                    "question": clean_query,
                    "document": clean_doc
                })
                
                score = float(score_str.strip())
                
                if score >= (self.relevance_threshold * 10):
                    relevant_docs.append(doc)
                    logger.info(f"관련 문서 포함 (점수: {score}/10)")
                else:
                    logger.info(f"관련성 낮은 문서 제외 (점수: {score}/10)")
                    
            except (ValueError, Exception) as e:
                logger.warning(f"관련성 평가 실패, 문서 포함: {e}")
                relevant_docs.append(doc)
        
        return relevant_docs

    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        if not context_docs:
            return "죄송합니다. 질문과 관련된 문서를 찾을 수 없습니다."
        
        # 컨텍스트와 질문 텍스트 정리
        clean_query = clean_text(query)
        clean_context = clean_text("\n\n".join([doc.page_content for doc in context_docs]))
        
        if not clean_query or not clean_context:
            return "죄송합니다. 텍스트 처리 중 문제가 발생했습니다."
        
        answer_chain = self.answer_prompt | self.llm | StrOutputParser()
        answer = answer_chain.invoke({
            "context": clean_context,
            "question": clean_query
        })
        
        return clean_text(answer) if answer else "답변을 생성할 수 없습니다."

    def check_hallucination(self, query: str, answer: str, context_docs: List[Document]) -> bool:
        try:
            # 텍스트 정리
            clean_query = clean_text(query)
            clean_answer = clean_text(answer)
            clean_context = clean_text("\n\n".join([doc.page_content for doc in context_docs]))
            
            if not clean_query or not clean_answer or not clean_context:
                logger.warning("할루시네이션 체크를 위한 텍스트가 비어있음, 통과로 처리")
                return True
            
            hallucination_chain = self.hallucination_check_prompt | self.llm | StrOutputParser()
            result = hallucination_chain.invoke({
                "context": clean_context,
                "answer": clean_answer,
                "question": clean_query
            })
            
            is_valid = result.strip().upper() == "YES"
            logger.info(f"할루시네이션 체크 결과: {'통과' if is_valid else '실패'}")
            
            return is_valid
            
        except Exception as e:
            logger.warning(f"할루시네이션 체크 실패: {e}")
            return True

    def process_query(self, query: str) -> Dict[str, any]:
        try:
            logger.info(f"사용자 질의: {query}")
            
            # 1. 관련 문서 검색
            docs = self.search_relevant_documents(query)
            
            # 2. 관련성 확인
            relevant_docs = self.check_relevance(query, docs)
            
            if not relevant_docs:
                return {
                    "success": False,
                    "answer": "죄송합니다. 질문과 관련된 정보를 찾을 수 없습니다.",
                    "retries": 0
                }
            
            # 3. 답변 생성 및 할루시네이션 체크 (최대 5번 재시도)
            for retry in range(self.max_hallucination_retries):
                answer = self.generate_answer(query, relevant_docs)
                
                if self.check_hallucination(query, answer, relevant_docs):
                    logger.info(f"답변 생성 완료 (재시도 횟수: {retry})")
                    return {
                        "success": True,
                        "answer": answer,
                        "retries": retry,
                        "sources": [doc.metadata.get("source", "Unknown") for doc in relevant_docs]
                    }
                
                logger.warning(f"할루시네이션 감지, 재시도 {retry + 1}/{self.max_hallucination_retries}")
            
            return {
                "success": False,
                "answer": "죄송합니다. 정확한 답변을 생성할 수 없습니다. 다시 시도해주세요.",
                "retries": self.max_hallucination_retries
            }
            
        except Exception as e:
            logger.error(f"질의 처리 중 오류 발생: {e}")
            return {
                "success": False,
                "answer": f"오류가 발생했습니다: {str(e)}",
                "retries": 0
            }


def main():
    # OpenAI API 키 설정
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY 환경변수를 설정해주세요.")
        return
    
    # RAG 시스템 초기화
    rag = RAGSystem(api_key)
    
    # 샘플 문서 생성 (실제 사용시에는 실제 문서 파일들을 사용)
    sample_docs_dir = Path("sample_docs")
    sample_docs_dir.mkdir(exist_ok=True)
    
    sample_doc1 = sample_docs_dir / "doc1.txt"
    sample_doc2 = sample_docs_dir / "doc2.txt"
    
    if not sample_doc1.exists():
        with open(sample_doc1, 'w', encoding='utf-8') as f:
            f.write("""
            인공지능(AI)의 발전과 응용
            
            인공지능은 컴퓨터 시스템이 인간의 지능적인 행동을 모방할 수 있도록 하는 기술입니다.
            머신러닝, 딥러닝, 자연어처리 등의 기술이 포함됩니다.
            
            주요 응용 분야:
            1. 자율주행 자동차
            2. 의료 진단
            3. 금융 서비스
            4. 고객 서비스 챗봇
            5. 이미지 및 음성 인식
            
            AI는 우리의 일상생활을 크게 변화시키고 있으며, 앞으로도 지속적인 발전이 예상됩니다.
            """)
    
    if not sample_doc2.exists():
        with open(sample_doc2, 'w', encoding='utf-8') as f:
            f.write("""
            머신러닝의 기본 개념
            
            머신러닝은 데이터로부터 패턴을 학습하여 예측이나 결정을 내리는 AI의 한 분야입니다.
            
            주요 유형:
            1. 지도학습 (Supervised Learning)
               - 레이블이 있는 데이터로 학습
               - 분류, 회귀 문제 해결
            
            2. 비지도학습 (Unsupervised Learning)
               - 레이블이 없는 데이터로 학습
               - 클러스터링, 차원 축소
            
            3. 강화학습 (Reinforcement Learning)
               - 환경과의 상호작용을 통한 학습
               - 게임, 로봇 제어 등에 활용
            
            머신러닝 모델을 구축할 때는 데이터 전처리, 모델 선택, 하이퍼파라미터 튜닝이 중요합니다.
            """)
    
    # 문서 로드 및 벡터 저장소 생성
    documents = rag.load_documents([str(sample_doc1), str(sample_doc2)])
    rag.create_vectorstore(documents)
    
    print("RAG 시스템이 준비되었습니다!")
    print("질문을 입력하세요 (종료하려면 'quit' 입력):")
    
    while True:
        user_query = input("\n질문: ").strip()
        
        if user_query.lower() in ['quit', 'exit', '종료']:
            print("시스템을 종료합니다.")
            break
        
        if not user_query:
            continue
        
        result = rag.process_query(user_query)
        
        print(f"\n답변: {result['answer']}")
        
        if result['success']:
            print(f"재시도 횟수: {result['retries']}")
            print(f"참고 문서: {', '.join(result['sources'])}")
        
        print("-" * 50)


if __name__ == "__main__":
    main()