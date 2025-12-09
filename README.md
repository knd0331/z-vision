# BarunVision

Z-Image-Turbo 기반 AI 이미지 생성 웹 애플리케이션

## 요구사항

- Python 3.10+
- CUDA GPU (권장) 또는 Apple Silicon MPS
- 최소 16GB VRAM (GPU) 또는 32GB RAM (CPU)

## 설치

```bash
cd BarunVision

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
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
- 이미지 크기 조절 (512x512 ~ 1536x1536)
- 시드 값으로 재현 가능한 결과
- 자동 이미지 저장 (outputs 폴더)

## 사용 팁

- **스텝 수**: 8-12 권장 (높을수록 품질 향상, 속도 감소)
- **시드**: -1은 랜덤, 특정 숫자로 동일 결과 재현 가능
- 첫 실행 시 모델 다운로드에 시간이 소요됩니다 (~15GB)

## 모델 정보

- **모델**: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- **아키텍처**: S3-DiT (6B 파라미터)
- **특징**: 8스텝 고속 생성, 포토리얼리스틱 품질
