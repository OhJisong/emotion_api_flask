# Flask 화면 기능에 AI 감정 분석 목적 Flask API 개발 프로젝트

## 프로젝트 개요

Colab에서 학습한 AI 감정 분석 모델 (Transformer 기반)을
Ubuntu 버치에서 Flask API 방식으로 버전을 추발 통해 리엔지보드했습니다.

* 특정 문장을 드림으로 전달처리 (POST /predict)
* 출력은 AI 감정 분석 결과가 JSON으로 나가는 API
* 검증 보수: PyTorch, Transformers, Flask, Python 3.12

## 프로젝트 구조

```
emotion_api_flask/
├── app.py                    # Flask API 서버 실행 코드
├── saved_emotion_model/     # 학습된 모델 및 토크나이저 (Colab에서 내보내었다)
│   ├── config.json              ← 모델 설정값 (레이어 수, hidden size 등)
│   ├── pytorch_model.bin        ← 학습된 모델 가중치 (PyTorch binary)
│   ├── tokenizer_config.json    ← 토크나이저 설정
│   ├── vocab.txt                ← 토크나이저 단어 사전
│   ├── special_tokens_map.json  ← [CLS], [SEP] 등의 토큰 정의
│   └── training_args.bin        ← 학습시 사용한 파리머터 (있을 수 도 있음)
├── requirements.txt         # 의존성 패키지 목록 (freeze)
├── requirements_simple.txt  # 간단 설치용 (pip install -r)
├── .gitignore               # 기본 venv, __pycache__ 무시 파일
└── README.md                # 프로젝트 설명서
```

## 필요 패키지

### requirements.txt (freeze)

```
flask==3.0.2
torch==2.3.0
transformers==4.52.3
typing-extensions==4.12.0
```

### requirements\_simple.txt

```
flask
torch
transformers
```

## 가상 다음 프로젝트와의 연계 경로

* Colab에서 학습된 감정 분류 AI 모델을 가져와 Flask API 서버에 연결
* 향후 로그인 기반 챗봇 시스템(웹 기반 PHP+MariaDB)과 이 Flask API 연동 예정
* 사용자가 입력한 문장을 PHP 서버에서 Flask로 전송 → 감정 분석 결과를 받아 DB에 저장 및 음악 추천

## 같이 관련된 프로젝트들

* Colab 학습 코드 레포: [OhJisong/emotion-model-colab](https://github.com/OhJisong/emotion-model-colab)
* 챗봇 통합 시스템 레포: OhJisong/VM_Linux_chatbot-emotion_project

###  모델 다운로드

> 모델 파일은 GitHub의 업로드 제한(100MB)을 초과하여, 외부 링크를 통해 제공합니다.

 [saved_emotion_model.zip 다운로드 (Google Drive)](https://drive.google.com/drive/folders/1UiAKVImfwa704ZWMl6vpILQBX_xRExCM?usp=drive_link)

압축을 풀고 `saved_emotion_model/` 폴더를 프로젝트 루트에 배치해 주세요.

