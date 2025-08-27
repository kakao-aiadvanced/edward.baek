#!/bin/bash

echo "🚀 RAG Assistant Streamlit App 시작..."

# 가상환경 활성화
source venv/bin/activate

# 환경변수 로드 (.env 파일이 있는 경우)
if [ -f .env ]; then
    echo "📋 환경변수 로드 중..."
    export $(cat .env | xargs)
fi

# Streamlit 앱 실행
echo "🌐 http://localhost:8501 에서 앱을 확인하세요"
streamlit run streamlit_app.py --server.port 8501 --server.address localhost