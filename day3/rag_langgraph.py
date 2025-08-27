import os
from typing import Dict, Any, List

import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from langgraph.graph.state import CompiledStateGraph
from tavily import TavilyClient
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from document_loader import load_and_create_vector_store

dotenv.load_dotenv()

# Pydantic models for structured output
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    
    datasource: str = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

class GradeDocuments(BaseModel):
    """Grade the relevance of retrieved documents to a user question."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Grade the answer based on whether it answers the question and is free from hallucinations."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Grade whether the answer contains hallucinations."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

# State
class GraphState(TypedDict):
    """Represents the state of our graph."""

    question: str
    generation: str
    web_search: str
    documents: List[str]
    retry_count: int
    hallucination_free: bool
    answers_question: bool
    source_references: List[Dict[str, str]]

class RAGLangGraph:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        self.vectorstore = load_and_create_vector_store()
        self.retriever = self.vectorstore.as_retriever()
        self.MAX_RETRIES = 3  # Maximum 1 retry attempt

    def route_question(self, state: GraphState) -> str:
        """Route question to web search or RAG."""
        
        print("---ROUTE QUESTION---")
        question = state["question"]
        source = self.question_router.invoke({"question": question})
        
        print(f"---ROUTING DECISION: {source.datasource}---")
        
        if source.datasource == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "web_search"
        elif source.datasource == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"
        else:
            # Default to vectorstore if unexpected response
            print(f"---UNEXPECTED DATASOURCE: {source.datasource}, DEFAULTING TO WEB SEARCH---")
            return "web_search"

    def retrieve(self, state: GraphState) -> Dict[str, Any]:
        """Retrieve documents from vectorstore."""
        
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = self.retriever.invoke(question)
        
        # Extract source references from document metadata
        source_references = []
        for doc in documents:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                source_references.append({
                    'url': doc.metadata['source'],
                    'title': doc.metadata.get('title', doc.metadata['source'].split('/')[-1]),
                    'type': 'vectorstore'
                })
        
        print(f"---FOUND {len(documents)} VECTORSTORE SOURCES---")
        
        return {
            "documents": documents, 
            "question": question,
            "source_references": source_references
        }

    def grade_documents(self, state: GraphState) -> Dict[str, Any]:
        """Determines whether the retrieved documents are relevant to the question."""

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        source_references = state.get("source_references", [])

        # Score each doc
        filtered_docs = []
        filtered_sources = []
        web_search = "No"
        
        for i, d in enumerate(documents):
            score = self.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
                # Keep corresponding source reference
                if i < len(source_references):
                    filtered_sources.append(source_references[i])
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
                
        return {
            "documents": filtered_docs, 
            "question": question, 
            "web_search": web_search,
            "source_references": filtered_sources
        }

    def web_search(self, state: GraphState) -> Dict[str, Any]:
        """Web search based on the question."""

        print("---WEB SEARCH---")
        question = state["question"]
        retry_count = state.get("retry_count", 0)

        # Web search
        docs = self.tavily_client.search(query=question)
        
        # Create document objects with source metadata
        web_documents = []
        source_references = []
        
        for result in docs["results"]:
            # Create document with metadata
            web_doc = type('Document', (), {
                'page_content': result["content"],
                'metadata': {
                    'source': result["url"],
                    'title': result.get("title", ""),
                    'score': result.get("score", 0)
                }
            })()
            web_documents.append(web_doc)
            
            # Add to source references
            source_references.append({
                'url': result["url"],
                'title': result.get("title", "Web Search Result"),
                'type': 'web_search'
            })
        
        print(f"---FOUND {len(web_documents)} WEB SOURCES---")
        
        # Increment retry count when doing web search as fallback
        return {
            "documents": web_documents, 
            "question": question, 
            "retry_count": retry_count + 1,
            "source_references": source_references
        }

    def generate(self, state: GraphState) -> Dict[str, Any]:
        """Generate answer using RAG on retrieved documents."""

        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        retry_count = state.get("retry_count", 0)
        source_references = state.get("source_references", [])

        # Format sources for the prompt
        sources_text = "\n".join([
            f"- {ref['title']}: {ref['url']}" 
            for ref in source_references
        ]) if source_references else "No specific sources available"

        # RAG generation
        generation = self.rag_chain.invoke({
            "context": documents, 
            "question": question,
            "sources": sources_text
        })
        
        return {
            "documents": documents, 
            "question": question, 
            "generation": generation, 
            "retry_count": retry_count,
            "source_references": source_references
        }

    def check_hallucinations(self, state: GraphState) -> Dict[str, Any]:
        """Check if the generation contains hallucinations."""
        
        print("---CHECK HALLUCINATIONS---")
        documents = state["documents"]
        generation = state["generation"]
        retry_count = state.get("retry_count", 0)
        source_references = state.get("source_references", [])
        
        score = self.hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score.binary_score
        
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            return {
                "documents": documents, 
                "question": state["question"], 
                "generation": generation, 
                "retry_count": retry_count, 
                "hallucination_free": True,
                "source_references": source_references
            }
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
            return {
                "documents": documents, 
                "question": state["question"], 
                "generation": generation, 
                "retry_count": retry_count, 
                "hallucination_free": False,
                "source_references": source_references
            }
    
    def check_answer_quality(self, state: GraphState) -> Dict[str, Any]:
        """Check if the generation answers the question properly."""
        
        print("---GRADE GENERATION vs QUESTION---")
        question = state["question"]
        generation = state["generation"]
        retry_count = state.get("retry_count", 0)
        source_references = state.get("source_references", [])
        
        score = self.answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return {
                "documents": state["documents"], 
                "question": question, 
                "generation": generation, 
                "retry_count": retry_count, 
                "answers_question": True,
                "source_references": source_references
            }
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return {
                "documents": state["documents"], 
                "question": question, 
                "generation": generation, 
                "retry_count": retry_count, 
                "answers_question": False,
                "source_references": source_references
            }
    
    def cannot_answer(self, state: GraphState) -> Dict[str, Any]:
        """Generate a message when unable to provide a reliable answer."""
        
        print("---CANNOT PROVIDE RELIABLE ANSWER---")
        question = state["question"]
        retry_count = state.get("retry_count", 0)
        source_references = state.get("source_references", [])
        
        # Generate a polite "cannot answer" message
        cannot_answer_message = f"""죄송합니다. 제공된 문서와 검색 결과를 바탕으로 "{question}"에 대한 신뢰할 수 있는 답변을 제공할 수 없습니다.

이는 다음과 같은 이유 때문일 수 있습니다:
- 검색된 문서에서 관련 정보를 찾을 수 없음
- 생성된 답변이 문서 내용과 일치하지 않음 (할루시네이션 감지)
- 질문이 현재 사용 가능한 지식 범위를 벗어남

더 구체적인 질문을 하시거나, 다른 방식으로 질문해 주시면 도움이 될 수 있습니다."""
        
        return {
            "documents": state["documents"],
            "question": question,
            "generation": cannot_answer_message,
            "retry_count": retry_count,
            "source_references": source_references,
            "answers_question": False,
            "hallucination_free": True  # This is our final safe answer
        }

    def decide_to_generate(self, state: GraphState) -> str:
        """Determines whether to generate an answer, or add web search."""

        print("---ASSESS GRADED DOCUMENTS---")
        question = state["question"]
        web_search = state.get("web_search", "No")
        filtered_documents = state["documents"]

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
            )
            return "websearch"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"

    def decide_hallucination_action(self, state: GraphState) -> str:
        """Decide what to do after hallucination check."""
        
        hallucination_free = state.get("hallucination_free", False)
        retry_count = state.get("retry_count", 0)
        
        if hallucination_free:
            return "check_answer"
        else:
            # Check if we can retry
            if retry_count >= self.MAX_RETRIES:
                print(f"---HALLUCINATION DETECTED, MAXIMUM RETRIES REACHED ({retry_count})---")
                return "cannot_answer"
            else:
                print(f"---HALLUCINATION DETECTED, RETRY GENERATION (Retry {retry_count + 1}/{self.MAX_RETRIES})---")
                return "retry_generation"
    
    def decide_final_action(self, state: GraphState) -> str:
        """Determines whether to re-generate, or web search, or finish."""

        retry_count = state.get("retry_count", 0)
        answers_question = state.get("answers_question", False)

        # Check if we've exceeded maximum retries (1 attempt)
        if retry_count >= self.MAX_RETRIES:
            print(f"---MAXIMUM RETRIES REACHED ({retry_count}), CANNOT PROVIDE RELIABLE ANSWER---")
            return "cannot_answer"

        if answers_question:
            print("---DECISION: GENERATION IS ACCEPTABLE---")
            return "finish"
        else:
            print(f"---DECISION: GENERATION NEEDS IMPROVEMENT (Retry {retry_count + 1}/{self.MAX_RETRIES})---")
            return "websearch"
    
    def decide_retry_action(self, state: GraphState) -> str:
        """Decide what to do when retrying generation."""
        
        retry_count = state.get("retry_count", 0)
        
        # Check if we've exceeded maximum retries (1 attempt)
        if retry_count >= self.MAX_RETRIES:
            print(f"---MAXIMUM RETRIES REACHED ({retry_count}), CANNOT PROVIDE RELIABLE ANSWER---")
            return "cannot_answer"
        
        print(f"---RETRY GENERATION (Retry {retry_count + 1}/{self.MAX_RETRIES})---")
        return "websearch"

    def setup_chains(self):
        """Setup all the chains for routing, grading, and generation."""
        
        # Router
        system = """
        You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks on LLMs.
        Use the vectorstore for questions on these topics. Otherwise, use web-search.
        """
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )
        structured_llm_router = self.llm.with_structured_output(RouteQuery)
        self.question_router = route_prompt | structured_llm_router

        # Retrieval Grader
        system = """You are a grader assessing relevance of a retrieved document to a user question. 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        self.retrieval_grader = grade_prompt | structured_llm_grader

        # Hallucination Grader
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )
        structured_llm_grader = self.llm.with_structured_output(GradeHallucinations)
        self.hallucination_grader = hallucination_prompt | structured_llm_grader

        # Answer Grader
        system = """You are a grader assessing whether an answer addresses / resolves a question 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )
        structured_llm_grader = self.llm.with_structured_output(GradeAnswer)
        self.answer_grader = answer_prompt | structured_llm_grader

        # RAG Chain
        prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        
        After your answer, include a "References:" section listing the sources you used.

        Question: {question} 
        Context: {context} 
        Sources: {sources}
        Answer: """)
        self.rag_chain = prompt | self.llm | StrOutputParser()

    def build_graph(self) ->  CompiledStateGraph[GraphState, Any, Any, Any]:
        """Build the graph workflow."""
        
        self.setup_chains()
        
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("websearch", self.web_search)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("check_hallucinations", self.check_hallucinations)
        workflow.add_node("check_answer_quality", self.check_answer_quality)
        workflow.add_node("cannot_answer", self.cannot_answer)

        # Build graph
        workflow.set_conditional_entry_point(
            self.route_question,
            {
                "web_search": "websearch",
                "vectorstore": "retrieve",
            },
        )

        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",
            },
        )
        workflow.add_edge("websearch", "generate")
        workflow.add_edge("generate", "check_hallucinations")
        
        workflow.add_conditional_edges(
            "check_hallucinations",
            self.decide_hallucination_action,
            {
                "check_answer": "check_answer_quality",
                "retry_generation": "websearch",
                "cannot_answer": "cannot_answer",
            },
        )
        
        workflow.add_conditional_edges(
            "check_answer_quality",
            self.decide_final_action,
            {
                "finish": END,
                "websearch": "websearch",
                "cannot_answer": "cannot_answer",
            },
        )
        
        workflow.add_edge("cannot_answer", END)

        # Compile
        app = workflow.compile()
        return app

    def run_query(self, question: str) -> Dict[str, Any]:
        """Run a query through the graph."""
        app = self.build_graph()

        inputs = {"question": question, "retry_count": 0, "source_references": []}

        # for output in app.stream(inputs):
        #     for key, value in output.items():
        #         print(f"Node '{key}':")
        #         print("---")

        result = app.invoke(inputs)
        
        return result

if __name__ == "__main__":
    # Example usage
    rag = RAGLangGraph()
    result = rag.run_query("llm attack은 어떤식으로 이뤄지는지 알려줘")
    print("\nFinal Answer:")
    print(result.get("generation", "No answer generated"))