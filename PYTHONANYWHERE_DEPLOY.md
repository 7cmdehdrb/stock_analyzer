# PythonAnywhere 배포 가이드

## 📋 사전 준비

1. **PythonAnywhere 계정 가입**
   - https://www.pythonanywhere.com 접속
   - 무료 계정 생성 (Beginner Account)

2. **GitHub 레포지토리 준비**
   - 코드가 GitHub에 푸시되어 있어야 함
   - 레포지토리: https://github.com/7cmdehdrb/stock_analyzer

---

## 🚀 배포 단계

### 1. PythonAnywhere 대시보드 접속

1. https://www.pythonanywhere.com 로그인
2. **Consoles** 탭 클릭
3. **Bash** 콘솔 시작

---

### 2. 프로젝트 클론

```bash
# 홈 디렉토리로 이동
cd ~

# GitHub에서 프로젝트 클론
git clone https://github.com/7cmdehdrb/stock_analyzer.git

# 프로젝트 디렉토리로 이동
cd stock_analyzer
```

---

### 3. 가상환경 생성 및 패키지 설치

```bash
# Python 3.10 가상환경 생성
python3.10 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
```

**⚠️ 주의사항:**
- 무료 계정은 CPU 시간 제한이 있어 설치가 오래 걸릴 수 있습니다
- 설치 중 타임아웃 발생 시: 콘솔을 새로 시작하고 다시 시도

---

### 4. 환경변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# nano 에디터로 .env 파일 편집
nano .env
```

**필수 설정:**
```
OPENAI_API_KEY=your_actual_openai_api_key
SECRET_KEY=random_secret_key_here
FLASK_DEBUG=False
```

- `Ctrl + O` : 저장
- `Enter` : 파일명 확인
- `Ctrl + X` : 종료

---

### 5. 데이터베이스 초기화

```bash
# Python 인터프리터 실행
python

# 다음 명령어 입력
>>> from app import app, db
>>> with app.app_context():
...     db.create_all()
>>> exit()
```

---

### 6. Web App 설정

1. **Web** 탭으로 이동
2. **Add a new web app** 클릭
3. **Manual configuration** 선택
4. **Python 3.10** 선택

---

### 7. WSGI 설정

1. **Web** 탭의 **Code** 섹션에서 **WSGI configuration file** 링크 클릭
2. 파일 내용을 모두 삭제하고 다음으로 교체:

```python
import sys
import os

# 프로젝트 경로 (본인의 username으로 수정)
project_home = '/home/yourusername/stock_analyzer'

if project_home not in sys.path:
    sys.path.insert(0, project_home)

# 가상환경 활성화
activate_this = f'{project_home}/venv/bin/activate_this.py'
# Python 3.10+에서는 activate_this.py가 없을 수 있음
# 대신 직접 sys.path 수정
sys.path.insert(0, f'{project_home}/venv/lib/python3.10/site-packages')

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv(os.path.join(project_home, '.env'))

# Flask 앱 임포트
from app import app as application
```

**⚠️ 중요:** `yourusername`을 본인의 PythonAnywhere 사용자명으로 변경!

3. **Save** 클릭

---

### 8. 가상환경 경로 설정

1. **Web** 탭의 **Virtualenv** 섹션
2. **Enter path to a virtualenv** 입력:
   ```
   /home/yourusername/stock_analyzer/venv
   ```
3. 체크 표시 클릭

---

### 9. 정적 파일 설정

**Web** 탭의 **Static files** 섹션에서:

| URL           | Directory                                    |
|---------------|----------------------------------------------|
| /static/      | /home/yourusername/stock_analyzer/static     |

**Add** 버튼으로 추가

---

### 10. 배포 완료

1. **Web** 탭 상단의 **Reload yourusername.pythonanywhere.com** 버튼 클릭
2. 웹사이트 링크 클릭하여 접속 확인

---

## 🔧 문제 해결

### 1. 500 Internal Server Error

**에러 로그 확인:**
```bash
# 에러 로그 보기
cat /var/log/yourusername.pythonanywhere.com.error.log

# 실시간 에러 로그
tail -f /var/log/yourusername.pythonanywhere.com.error.log
```

**일반적인 원인:**
- `.env` 파일이 없거나 환경변수가 잘못됨
- 패키지 설치가 완료되지 않음
- WSGI 파일 경로가 잘못됨

---

### 2. Import Error

```bash
# 가상환경에서 패키지 재설치
source ~/stock_analyzer/venv/bin/activate
pip install -r ~/stock_analyzer/requirements.txt
```

---

### 3. 데이터베이스 권한 오류

```bash
# instance 디렉토리 생성 및 권한 설정
mkdir -p ~/stock_analyzer/instance
chmod 755 ~/stock_analyzer/instance
```

---

## 🔄 코드 업데이트 방법

```bash
# 프로젝트 디렉토리로 이동
cd ~/stock_analyzer

# 최신 코드 가져오기
git pull origin main

# 가상환경 활성화
source venv/bin/activate

# 패키지 업데이트 (requirements.txt 변경 시)
pip install -r requirements.txt

# Web App 재시작
# Web 탭에서 Reload 버튼 클릭
```

---

## 📊 무료 계정 제한사항

- **CPU 시간**: 하루 100초 제한
- **저장공간**: 512MB
- **웹 트래픽**: 제한 없음
- **Always-on tasks**: 사용 불가 (자동 작업 X)
- **SSH 접속**: 불가 (Web 콘솔만 사용)

**업그레이드 옵션:**
- Hacker Plan ($5/월): CPU 제한 완화, 2GB 저장공간
- 더 많은 기능 필요 시 상위 플랜 고려

---

## 📝 추가 설정

### 커스텀 도메인 (유료 플랜)

1. **Web** 탭 → **Custom domain** 섹션
2. 도메인 추가 및 DNS 설정

### HTTPS 설정

PythonAnywhere는 기본적으로 HTTPS를 제공합니다:
- `https://yourusername.pythonanywhere.com`

---

## 🆘 도움말

- **PythonAnywhere 문서**: https://help.pythonanywhere.com/
- **Flask 배포 가이드**: https://help.pythonanywhere.com/pages/Flask/
- **포럼**: https://www.pythonanywhere.com/forums/

---

## ✅ 배포 체크리스트

- [ ] GitHub에 코드 푸시
- [ ] PythonAnywhere 계정 생성
- [ ] 프로젝트 클론
- [ ] 가상환경 생성 및 패키지 설치
- [ ] `.env` 파일 설정
- [ ] 데이터베이스 초기화
- [ ] Web App 생성
- [ ] WSGI 파일 설정
- [ ] 가상환경 경로 설정
- [ ] 정적 파일 경로 설정
- [ ] Reload 및 테스트
- [ ] 에러 로그 확인

---

**배포 성공을 기원합니다! 🚀**
