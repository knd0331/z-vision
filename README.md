# Z-Vision

Z-Image-Turbo 기반 AI 이미지 생성 웹 애플리케이션

## 지원 백엔드

| 백엔드 | 플랫폼 | 속도 | 설명 |
|--------|--------|------|------|
| **MLX** | Apple Silicon (M1/M2/M3/M4) | 빠름 | 4스텝 ~20초 |
| **CUDA** | NVIDIA GPU | 가장 빠름 | RTX 3090 기준 ~5초 |
| **MPS** | Apple Metal | 보통 | PyTorch Metal 백엔드 |
| **CPU** | 모든 플랫폼 | 느림 | 폴백 옵션 |

## 요구사항

- Python 3.10+
- 최소 16GB RAM (Apple Silicon) 또는 8GB VRAM (CUDA)

## 설치

### Apple Silicon (M1/M2/M3/M4) - 권장

```bash
cd z-vision
python -m venv venv
source venv/bin/activate

# 공통 패키지 + MLX
pip install pillow gradio huggingface-hub
pip install mflux
```

### NVIDIA GPU (CUDA)

```bash
cd z-vision
python -m venv venv
source venv/bin/activate

# 전체 패키지 설치
pip install -r requirements.txt
```

### CPU / Intel Mac

```bash
pip install -r requirements.txt
```

## 실행

```bash
python app.py
```

브라우저에서 http://localhost:7860 접속

## 기능

- 텍스트 프롬프트로 이미지 생성
- 한국어/영어 프롬프트 지원
- 자동 백엔드 감지 (MLX → CUDA → MPS → CPU)
- 백엔드별 최적화된 기본 설정
- 이미지 크기 조절 (512x512 ~ 2048x2048)
- 시드 값으로 재현 가능한 결과
- 자동 이미지 저장 (outputs 폴더)

## 백엔드별 권장 설정

| 백엔드 | 스텝 수 | 이미지 크기 |
|--------|---------|-------------|
| MLX | 4 | 512x512 |
| CUDA | 6-8 | 1024x1024 |
| MPS | 6 | 512x512 |
| CPU | 4 | 512x512 |

## 모델 정보

- **모델**: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- **아키텍처**: S3-DiT (6B 파라미터)
- **특징**: Turbo 고속 생성, 포토리얼리스틱 품질

## 파일 구조

```
z-vision/
├── app.py              # 메인 앱 (통합 버전)
├── backup/
│   ├── app_mlx.py      # MLX 전용 버전
│   └── app_diffusers.py # PyTorch 전용 버전
├── outputs/            # 생성된 이미지
└── requirements.txt
```
