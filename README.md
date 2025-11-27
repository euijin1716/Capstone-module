# 🎙️ AI-Powered Meeting Assistant

**실시간 회의 음성 인식(STT), AI 투표 감지, 회의록 자동 요약 통합 시스템**

이 프로젝트는 LiveKit 기반의 화상 회의 솔루션에서 작동하는 AI 어시스턴트입니다. 실시간 음성을 텍스트로 변환하고, 회의 중 투표가 필요한 순간을 자동으로 감지하며, 회의 종료 후 전문적인 회의록을 자동 생성합니다.

---

## 📋 목차

- [주요 기능](#주요-기능)
- [프로젝트 구조](#프로젝트-구조)
- [기술 스택](#기술-스택)
- [설치 및 실행](#설치-및-실행)
- [환경 변수 설정](#환경-변수-설정)
- [사용 방법](#사용-방법)
- [라이선스](#라이선스)

---

## 🌟 주요 기능

### 1️⃣ STT 모듈 (실시간 음성 인식)

- **고품질 한국어 STT**: Faster-Whisper (large-v3-turbo) 모델 사용
- **실시간 처리**: LiveKit Agents와 통합되어 지연 시간 최소화
- **AI 투표 감지**: 
  - 1단계: Zero-shot Classification으로 의사결정 발화 필터링
  - 2단계: Google Gemini 2.0 Flash로 투표 주제 및 선택지 추출
  - 프론트엔드에 `VOTE_CREATED` 이벤트 자동 송출
- **회의록 자동 백업**: 
  - AWS S3에 실시간 로그 저장 (5분 간격)
  - JSON 형식으로 메타데이터, 참가자, 발화 내용 구조화
- **중간 요약(Recap) 지원**: 
  - 늦게 참여한 사용자를 위한 실시간 요약 생성
  - LiveKit DataChannel을 통한 요약 결과 전달

### 2️⃣ Summarize 모듈 (회의록 자동 요약)

- **구조화된 회의록 생성**: 
  - 회의 주제, 도메인 자동 분류
  - 논의 내용을 세부 주제별로 구분 (정보 공유, 의사결정, 문제 해결 등)
  - 결정 사항 및 실행 항목 자동 추출
- **실시간 Recap 생성**: 
  - 회의 중간에 참여한 사용자를 위한 요약
  - 현재 주제, 지금까지의 흐름, 주요 결정 사항 포함
- **다양한 회의 유형 지원**:
  - 정보 공유, 의사결정, 운영 보고, 문제 해결
  - 계획 수립, 팀 빌딩, 브레인스토밍, 회고
- **Google Gemini API 활용**: 고품질 자연어 처리 및 요약

---

## 📁 프로젝트 구조

```
Capstone-module/
├── STT/                      # 실시간 음성 인식 모듈
│   ├── main.py              # LiveKit Agent 메인 엔트리포인트
│   ├── whisper_plugin.py    # Faster-Whisper STT 구현
│   ├── logger.py            # 회의록 로깅 시스템
│   ├── S3_upload.py         # AWS S3 업로드 관리
│   ├── requirements.txt     # STT 모듈 의존성
│   ├── .env                 # 환경 변수 (Git 제외)
│   └── README.md            # STT 상세 문서
│
├── Summarize/               # 회의록 요약 모듈
│   ├── S3_Summarization.py  # 전체 회의록 요약 스크립트
│   ├── S3_Recap.py          # 중간 요약(Recap) 스크립트
│   ├── prompts.py           # Gemini API 프롬프트 템플릿
│   └── gemini_api_test.py   # 로컬 테스트용 스크립트
│
├── requirements.txt         # 전체 프로젝트 통합 의존성
└── README.md                # 프로젝트 전체 문서 (본 파일)
```

---

## 🛠️ 기술 스택

### Core
- **Python** 3.9+
- **LiveKit** - 실시간 화상 회의 플랫폼
- **LiveKit Agents** - AI Worker 프레임워크

### AI/ML
- **Faster-Whisper** (large-v3-turbo) - 음성 인식
- **Hugging Face Transformers** - Zero-shot Classification
- **Google Gemini 2.0 Flash / 2.5 Pro** - 의사결정 감지 및 회의록 요약
- **PyTorch** - 딥러닝 프레임워크
- **Silero VAD** - 음성 활동 감지

### Infrastructure
- **AWS S3** - 회의록 클라우드 저장소
- **Boto3** - AWS SDK
- **Tenacity** - API 재시도 로직

---

## 🚀 설치 및 실행

### 1. 사전 준비

- Python 3.9 이상 설치
- CUDA 지원 GPU 권장 (STT 성능 향상)
- LiveKit Server (Cloud 또는 Self-hosted)
- API 키 발급:
  - [Google AI Studio](https://aistudio.google.com/) - Gemini API 키
  - [AWS Console](https://aws.amazon.com/) - S3 Access Key
  - [LiveKit Cloud](https://cloud.livekit.io/) - LiveKit API 키

### 2. 저장소 클론 및 패키지 설치

```bash
git clone https://github.com/your-username/Capstone-module.git
cd Capstone-module

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# 통합 의존성 설치
pip install -r requirements.txt
```

### 3. 환경 변수 설정

각 모듈의 디렉터리에 `.env` 파일을 생성하세요.

#### STT/.env
```ini
# LiveKit 설정
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# Google Gemini API (투표 감지용)
GOOGLE_API_KEY=your_gemini_api_key

# AWS S3 설정
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_BUCKET_NAME=your_bucket_name
AWS_REGION=ap-northeast-2
```

#### Summarize 환경 변수
Summarize 모듈은 AWS 자격 증명을 시스템 환경 변수에서 읽습니다:

```bash
export GEMINI_API_KEY=your_gemini_api_key
export AWS_ACCESS_KEY_ID=your_aws_access_key
export AWS_SECRET_ACCESS_KEY=your_aws_secret_key
```

### 4. 실행

#### STT 모듈 실행

```bash
cd STT

# 개발 모드
python main.py dev

# 프로덕션 모드
python main.py start
```

#### Summarize 모듈 실행

**전체 회의록 요약:**
```bash
cd Summarize

# 단일 파일 처리
python S3_Summarization.py --file_ids room001_20231121_143000

# 여러 파일 동시 처리
python S3_Summarization.py --file_ids file1 file2 file3
```

**중간 요약(Recap) 생성:**
```bash
# 전체 내용 요약
python S3_Recap.py --file_id room001_20231121_143000

# 특정 발화 ID까지만 요약 (회의 중간 시점 시뮬레이션)
python S3_Recap.py --file_id room001_20231121_143000 --end_id 50
```

---

## ⚙️ 환경 변수 설정

### 필수 환경 변수

| 변수명 | 설명 | 관련 모듈 |
|--------|------|-----------|
| `LIVEKIT_URL` | LiveKit 서버 WebSocket URL | STT |
| `LIVEKIT_API_KEY` | LiveKit API 키 | STT |
| `LIVEKIT_API_SECRET` | LiveKit API 시크릿 | STT |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | Google Gemini API 키 | STT, Summarize |
| `AWS_ACCESS_KEY_ID` | AWS 액세스 키 | STT, Summarize |
| `AWS_SECRET_ACCESS_KEY` | AWS 시크릿 키 | STT, Summarize |
| `AWS_BUCKET_NAME` | S3 버킷 이름 | STT |
| `AWS_REGION` | AWS 리전 (예: ap-northeast-2) | STT |

---

## 📖 사용 방법

### 1. STT 에이전트 연결

1. LiveKit Room 생성
2. STT Agent 실행 (`python main.py dev`)
3. 클라이언트에서 Room 참여
4. Agent가 자동으로 Room에 조인하여 STT 시작

### 2. 실시간 기능

- **음성 인식**: 참가자 발화가 실시간으로 텍스트 변환되어 DataChannel로 전송
- **투표 감지**: 의사결정 발화 감지 시 `VOTE_CREATED` 이벤트 자동 발생
- **Recap 요청**: 클라이언트에서 `Request_Recap` 메시지 전송 시 현재까지의 요약 생성

### 3. 회의록 생성

회의 종료 후 S3에 저장된 JSON 파일을 사용하여:

```bash
# 전체 회의록 생성
python Summarize/S3_Summarization.py --file_ids [파일명]
```

생성된 회의록은 `s3://[bucket]/meeting_logs/[파일명]_final.json`에 저장됩니다.

---

## 📊 S3 데이터 구조

### 입력 데이터 (STT 모듈에서 생성)

```json
{
  "metadata": {
    "room_name": "room001",
    "start_time": "2023-11-21T14:30:00",
    "end_time": "2023-11-21T15:30:00"
  },
  "participants": [
    {
      "USER_ID": "user123",
      "name": "홍길동",
      "identity": "user123"
    }
  ],
  "utterances": [
    {
      "id": "1",
      "USER_ID": "user123",
      "content": "안녕하세요",
      "timestamp": "2023-11-21T14:30:05"
    }
  ]
}
```

### 출력 데이터 (Summarize 모듈에서 생성)

- **전체 요약**: `meeting_logs/[파일명]_final.json`
- **중간 요약**: `Recap/[파일명]_recap.json`

---

## 🔧 개발자 가이드

### 커스텀 프롬프트 수정

`Summarize/prompts.py`에서 회의 유형별 프롬프트를 수정할 수 있습니다:

- `STRUCTURE_PROMPT`: 회의 구조 분석
- `TYPE_PROMPTS`: 8가지 회의 유형별 상세 프롬프트
- `CONSOLIDATION_PROMPT`: 최종 통합 요약
- `RECAP_PROMPT`: 중간 요약 생성

### STT 모델 변경

`STT/whisper_plugin.py`에서 Whisper 모델 크기 및 설정 변경 가능:

```python
WhisperSTT(
    model_size_or_path="large-v3-turbo",  # 모델 변경
    device="cuda",                         # CPU/CUDA 선택
    compute_type="float16"                 # 정밀도 조정
)
```

---

## 📄 라이선스

This project is licensed under the MIT License.

---

## 🤝 기여

Issues 및 Pull Requests를 환영합니다!

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 문의

프로젝트 관련 문의사항은 Issue를 통해 남겨주세요.
