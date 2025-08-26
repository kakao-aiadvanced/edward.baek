# -*- coding: utf-8 -*-
import asyncio
import os
import re
from typing import List, Dict
import logging
from pathlib import Path
import unicodedata

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

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

    async def load_docs_from_web(self) -> List[Document]:
        urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]
        loader = WebBaseLoader()
        datas = await loader.fetch_all(urls)
        docs = []
        for doc in datas:
            docs.append(Document(doc))

        return docs

    
    def _setup_prompts(self):
        # 1. RAG 답변 생성 프롬프트 - Hub에서 로드
        try:
            self.answer_prompt = hub.pull("rlm/rag-prompt")
            logger.info("Hub에서 rlm/rag-prompt 로드 성공")
        except Exception as e:
            logger.warning(f"Hub에서 rlm/rag-prompt 로드 실패, 기본 프롬프트 사용: {e}")
            self.answer_prompt = ChatPromptTemplate.from_template("""
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            
            Question: {question}
            Context: {context}
            Answer:
            """)
        
        # 2. 관련성 평가 프롬프트 - 영어로 통일하여 일관성 향상
        self.relevance_prompt = ChatPromptTemplate.from_template("""
        Rate how relevant the following document is to the user's question on a scale of 0-10.
        
        Question: {question}
        
        Document content:
        {document}
        
        0-3: Not relevant at all
        4-6: Slightly relevant  
        7-8: Highly relevant
        9-10: Very highly relevant
        
        Please respond with only a number (e.g., 7):
        """)
        logger.info("관련성 평가 프롬프트 설정 완료")
        
        # 3. 할루시네이션 체크 프롬프트 - 영어로 통일하여 일관성 향상
        self.hallucination_check_prompt = ChatPromptTemplate.from_template("""
        Please verify if the following answer is based on the provided context. 
        
        Context:
        {context}
        
        Answer:
        {answer}
        
        Question: {question}
        
        If the answer is based on the context, respond with "YES". If it contains information not in the context or makes assumptions, respond with "NO".
        
        Verification result:
        """)
        logger.info("할루시네이션 체크 프롬프트 설정 완료")

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


async def main():
    # OpenAI API 키 설정
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY 환경변수를 설정해주세요.")
        return
    
    # RAG 시스템 초기화
    rag = RAGSystem(api_key)

    web_docs = await rag.load_docs_from_web()
    rag.create_vectorstore(web_docs)
    
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
    asyncio.run(main())