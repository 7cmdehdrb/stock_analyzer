"""
WSGI config for PythonAnywhere deployment
"""
import sys
import os

# PythonAnywhere 경로 설정 (배포 시 수정 필요)
# 예: /home/yourusername/stock_analyzer
project_home = '/home/yourusername/stock_analyzer'

if project_home not in sys.path:
    sys.path.insert(0, project_home)

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv(os.path.join(project_home, '.env'))

# Flask 앱 임포트
from app import app as application

# WSGI application
if __name__ == "__main__":
    application.run()
