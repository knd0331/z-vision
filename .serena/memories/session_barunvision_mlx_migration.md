# Z-Vision Session - Multi-Backend Support Complete

## Date: 2025-12-09

## Summary
Z-Image-Turbo 이미지 생성 앱을 PyTorch/diffusers에서 MLX/MFLUX로 성공적으로 마이그레이션 완료.

## Key Accomplishments

### 1. MLX Migration
- **Before**: PyTorch + diffusers → MPS에서 ~7분/이미지 (9스텝)
- **After**: MLX + MFLUX → ~17-34초/이미지 (4스텝)
- 성능 개선: **10-20배 빠름**

### 2. Code Changes
- Main app: `/Users/mideum/projects/BarunVision/app.py` (MLX version)
- Backup: `/Users/mideum/projects/BarunVision/backup/app_diffusers.py` (PyTorch version)

### 3. Key Implementation Details
```python
from mflux.models.z_image.variants.turbo.z_image_turbo import ZImageTurbo
_model = ZImageTurbo(quantize=8)  # 8-bit quantization

# GeneratedImage에서 PIL.Image 추출 필수
result = model.generate_image(...)
image = result.image  # PIL.Image
gen_time = result.generation_time  # float seconds
```

### 4. Issues Resolved
- CUDA warning on MPS: diffusers library bug (GitHub issue registered by user)
- GeneratedImage type error: Extract `result.image` for Gradio compatibility
- Memory pressure: 24GB RAM system with ~12GB compressor usage

## Technical Notes

### MFLUX Features
- LoRA 지원: `lora_paths=["path.safetensors"], lora_scales=[0.8]`
- 파인튜닝: 미지원 (inference only)
- Quantization: 4-bit, 8-bit 지원

### Performance Optimization
- 메모리 압박 시 속도 저하 (17초 → 34초)
- 권장: 다른 앱 종료 후 사용
- 4스텝 권장 (Turbo 최적화)

### Prompt Tips
- 영어 상세 프롬프트 권장
- "single", "one" 명시로 중복 객체 방지
- 스타일 지정: photorealistic, anime, painting 등

## Files Structure
```
BarunVision/
├── app.py              # MLX version (current)
├── backup/
│   └── app_diffusers.py  # PyTorch backup
├── outputs/            # Generated images
└── venv/               # Python 3.13 + mflux
```

## Dependencies
- mflux==0.13.3
- mlx==0.30.0
- gradio>=4.0.0
