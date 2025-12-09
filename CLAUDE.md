# BarunVision Project Guidelines

## Project Overview

Z-Image-Turbo 기반 AI 이미지 생성 Gradio 웹 애플리케이션

## Tech Stack

- **Python**: 3.10+
- **ML Framework**: PyTorch, Diffusers
- **Model**: Tongyi-MAI/Z-Image-Turbo (S3-DiT 6B)
- **UI**: Gradio 4.x

## Project Structure

```
BarunVision/
├── app.py              # 메인 Gradio 웹 애플리케이션
├── requirements.txt    # Python 의존성
├── outputs/            # 생성된 이미지 저장
└── README.md           # 사용자 가이드
```

## Key Implementation Details

### Model Configuration

- `torch_dtype`: bfloat16 (메모리 최적화)
- `guidance_scale`: 0.0 (Turbo 모델 필수)
- `num_inference_steps`: 9 (8 NFEs)

### Device Support

1. **CUDA GPU** - 최적 성능
2. **Apple MPS** - M1/M2/M3 지원
3. **CPU** - enable_model_cpu_offload() 사용

## Development Commands

```bash
# 가상환경 설정
python -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 앱 실행
python app.py

# 공유 링크 생성 시 app.py에서 share=True 설정
```

## Code Conventions

- 함수/변수: snake_case
- 클래스: PascalCase
- 한국어 주석 허용
- Type hints 권장

## Common Tasks

### 새 기능 추가 시
1. `app.py`의 `create_ui()` 함수에 Gradio 컴포넌트 추가
2. 필요 시 별도 처리 함수 작성
3. 이벤트 핸들러 연결

### 모델 설정 변경 시
- `get_pipeline()` 함수에서 파이프라인 옵션 수정
- Flash Attention 활성화: `pipe.transformer.set_attention_backend("flash")`
- 컴파일 최적화: `pipe.transformer.compile()`

## Dependencies

핵심 패키지:
- `diffusers>=0.31.0` - ZImagePipeline 포함
- `gradio>=4.0.0` - 웹 UI
- `torch>=2.0.0` - GPU 지원

## Resources

- [Z-Image-Turbo Model](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [Gradio Documentation](https://www.gradio.app/docs)
