#!/bin/bash
# Annotation System 실행 스크립트

cd "$(dirname "$0")"

# 가상환경 활성화
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install Flask>=2.3.0
else
    source venv/bin/activate
fi

# 서버 실행
echo "Starting annotation server..."
python annotation_app.py

