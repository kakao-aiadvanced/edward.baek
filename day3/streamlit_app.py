import streamlit as st
import os
import sys
from typing import Dict, Any, List
import time
from datetime import datetime

# Import our RAG system
from rag_langgraph import RAGLangGraph

# Configure Streamlit page
st.set_page_config(
    page_title="🤖 RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitRAGApp:
    def __init__(self):
        self.setup_session_state()
        self.setup_sidebar()
        
    def setup_session_state(self):
        """Initialize session state variables."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
        if 'api_keys_set' not in st.session_state:
            st.session_state.api_keys_set = False

    def setup_sidebar(self):
        """Setup sidebar with API keys and settings."""
        with st.sidebar:
            st.header("⚙️ 설정")
            
            # API Keys section
            st.subheader("🔑 API 키")
            
            openai_key = st.text_input(
                "OpenAI API Key", 
                type="password",
                value=os.environ.get("OPENAI_API_KEY", ""),
                help="OpenAI API 키를 입력하세요"
            )
            
            tavily_key = st.text_input(
                "Tavily API Key", 
                type="password", 
                value=os.environ.get("TAVILY_API_KEY", ""),
                help="웹 검색을 위한 Tavily API 키를 입력하세요"
            )
            
            # Set environment variables
            if openai_key and tavily_key:
                os.environ["OPENAI_API_KEY"] = openai_key
                os.environ["TAVILY_API_KEY"] = tavily_key
                st.session_state.api_keys_set = True
                
                if st.session_state.rag_system is None:
                    with st.spinner("🔧 RAG 시스템 초기화 중..."):
                        try:
                            st.session_state.rag_system = RAGLangGraph()
                            st.success("✅ RAG 시스템 준비 완료!")
                        except Exception as e:
                            st.error(f"❌ 초기화 실패: {str(e)}")
                            
            else:
                st.warning("🔑 API 키를 모두 입력해주세요")
                st.session_state.api_keys_set = False
            
            # Settings section
            st.subheader("🎛️ 시스템 설정")
            
            if st.session_state.rag_system:
                max_retries = st.slider(
                    "최대 재시도 횟수", 
                    min_value=1, 
                    max_value=5, 
                    value=st.session_state.rag_system.MAX_RETRIES,
                    help="할루시네이션 감지 시 최대 재시도 횟수"
                )
                st.session_state.rag_system.MAX_RETRIES = max_retries
            
            # Clear chat button
            if st.button("🗑️ 채팅 기록 삭제", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
                
            # System info
            st.subheader("ℹ️ 시스템 정보")
            st.caption("📚 지원 주제: AI 에이전트, 프롬프트 엔지니어링, LLM 공격")
            st.caption("🔍 검색: 벡터스토어 + 웹검색")
            st.caption("✅ 할루시네이션 검증 포함")

    def display_message(self, message: Dict[str, Any]):
        """Display a single message in the chat."""
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # Display the answer
                st.markdown(message["content"])
                
                # Display sources if available
                if "sources" in message and message["sources"]:
                    st.markdown("---")
                    st.markdown("### 📚 참고 문헌")
                    
                    for i, source in enumerate(message["sources"], 1):
                        source_type = "🌐 웹검색" if source["type"] == "web_search" else "📖 문서"
                        st.markdown(f"**{i}. {source_type}**: [{source['title']}]({source['url']})")
                
                # Display metadata if available
                if "metadata" in message:
                    with st.expander("🔍 상세 정보"):
                        metadata = message["metadata"]
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("재시도 횟수", metadata.get("retry_count", 0))
                        with col2:
                            st.metric("할루시네이션 검증", "✅" if metadata.get("hallucination_free", False) else "❌")
                        with col3:
                            st.metric("질문 답변", "✅" if metadata.get("answers_question", False) else "❌")
            else:
                st.markdown(message["content"])

    def process_query_with_progress(self, question: str) -> Dict[str, Any]:
        """Process query with real-time progress updates."""
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Initialize progress tracking
        progress_steps = {
            "__start__": "🎯 질문 라우팅 중...",
            "retrieve": "📚 문서 검색 중...",
            "websearch": "🌐 웹 검색 중...",
            "grade_documents": "📊 문서 관련성 평가 중...",
            "generate": "✨ 답변 생성 중...",
            "check_hallucinations": "🔍 할루시네이션 검증 중...",
            "check_answer_quality": "✅ 답변 품질 확인 중...",
            "cannot_answer": "❌ 답변 불가 처리 중..."
        }
        
        progress_bar = progress_placeholder.progress(0)
        current_step = 0
        total_steps = len(progress_steps)
        
        try:
            app = st.session_state.rag_system.build_graph()
            inputs = {"question": question, "retry_count": 0, "source_references": []}
            
            # Stream the execution
            for i, output in enumerate(app.stream(inputs)):
                for key, value in output.items():
                    if key in progress_steps:
                        current_step = min(current_step + 1, total_steps)
                        progress = current_step / total_steps
                        progress_bar.progress(progress)
                        status_placeholder.info(f"🔄 {progress_steps[key]}")
                        time.sleep(0.3)  # Small delay for visual effect
            
            # Final result
            result = app.invoke(inputs)
            progress_bar.progress(1.0)
            status_placeholder.success("✅ 완료!")
            
            # Clear progress indicators after a moment
            time.sleep(1)
            progress_placeholder.empty()
            status_placeholder.empty()
            
            return result
            
        except Exception as e:
            progress_placeholder.empty()
            status_placeholder.error(f"❌ 오류 발생: {str(e)}")
            raise e

    def run(self):
        """Main app runner."""
        # Header
        st.title("🤖 RAG Assistant")
        st.markdown("*AI 에이전트, 프롬프트 엔지니어링, LLM 보안에 대한 질문을 해보세요!*")
        
        # Check if system is ready
        if not st.session_state.api_keys_set:
            st.warning("👈 사이드바에서 API 키를 입력해주세요")
            st.stop()
        
        if st.session_state.rag_system is None:
            st.error("RAG 시스템이 초기화되지 않았습니다. API 키를 확인해주세요.")
            st.stop()
        
        # Display chat messages
        for message in st.session_state.messages:
            self.display_message(message)
        
        # Chat input
        if prompt := st.chat_input("질문을 입력하세요... (예: 'LLM에 대한 공격 방법은 무엇인가요?')"):
            # Add user message
            user_message = {"role": "user", "content": prompt}
            st.session_state.messages.append(user_message)
            self.display_message(user_message)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                try:
                    # Process query with progress tracking
                    result = self.process_query_with_progress(prompt)
                    
                    # Extract response and metadata
                    answer = result.get("generation", "답변을 생성할 수 없습니다.")
                    sources = result.get("source_references", [])
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display sources
                    if sources:
                        st.markdown("---")
                        st.markdown("### 📚 참고 문헌")
                        
                        for i, source in enumerate(sources, 1):
                            source_type = "🌐 웹검색" if source["type"] == "web_search" else "📖 문서"
                            st.markdown(f"**{i}. {source_type}**: [{source['title']}]({source['url']})")
                    
                    # Display metadata
                    with st.expander("🔍 상세 정보"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("재시도 횟수", result.get("retry_count", 0))
                        with col2:
                            hallucination_free = result.get("hallucination_free", False)
                            st.metric("할루시네이션 검증", "✅ 통과" if hallucination_free else "❌ 감지")
                        with col3:
                            answers_question = result.get("answers_question", False)
                            st.metric("질문 답변", "✅ 적절함" if answers_question else "❌ 부적절함")
                    
                    # Save assistant message
                    assistant_message = {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "metadata": {
                            "retry_count": result.get("retry_count", 0),
                            "hallucination_free": result.get("hallucination_free", False),
                            "answers_question": result.get("answers_question", False),
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    st.session_state.messages.append(assistant_message)
                    
                except Exception as e:
                    error_msg = f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}"
                    st.error(error_msg)
                    
                    # Save error message
                    error_message = {
                        "role": "assistant",
                        "content": error_msg,
                        "sources": [],
                        "metadata": {"error": True, "timestamp": datetime.now().isoformat()}
                    }
                    st.session_state.messages.append(error_message)

def main():
    """Main function to run the Streamlit app."""
    app = StreamlitRAGApp()
    app.run()

if __name__ == "__main__":
    main()