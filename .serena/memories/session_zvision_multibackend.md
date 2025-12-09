# Z-Vision Session - Multi-Backend Support

## Date: 2025-12-09

## Summary
Z-Image-Turbo 이미지 생성 앱의 멀티 백엔드 지원 완료.
자동 백엔드 감지로 MLX/CUDA/MPS/CPU 모두 지원.

## Key Accomplishments

### 1. Multi-Backend Architecture
- **Auto-detection**: MLX → CUDA → MPS → CPU 순서로 최적 백엔드 선택
- **Backend-specific settings**: 각 백엔드에 맞는 기본값 자동 적용
- **UI integration**: 헤더에 현재 백엔드 상태 표시

### 2. Performance Comparison
| Backend | Steps | Size | Time |
|---------|-------|------|------|
| MLX | 4 | 512x512 | ~20초 |
| CUDA | 6 | 1024x1024 | ~5초 |
| MPS | 6 | 512x512 | ~7분 |
| CPU | 4 | 512x512 | 느림 |

### 3. Code Structure
```
z-vision/
├── app.py              # 통합 멀티 백엔드 버전
├── backup/
│   ├── app_mlx.py      # MLX 전용 버전
│   └── app_diffusers.py # PyTorch 전용 버전
├── outputs/
└── requirements.txt
```

### 4. Key Implementation
```python
def detect_backend():
    # 1. Apple Silicon + MLX 우선
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx.core
            return "mlx"
        except ImportError:
            pass
    
    # 2. PyTorch backends
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

## GitHub Repository
- URL: https://github.com/knd0331/z-vision
- Commits:
  1. Initial commit: MLX version
  2. feat: Unified multi-backend support

## Dependencies
- mflux>=0.13.0 (MLX backend, Apple Silicon only)
- torch>=2.0.0 (CUDA/MPS/CPU backend)
- diffusers>=0.31.0
- gradio>=4.0.0

## Technical Notes

### Backend Detection Priority
1. Apple Silicon + MLX installed → MLX (fastest on Mac)
2. NVIDIA GPU → CUDA (fastest overall)
3. Apple Metal → MPS (PyTorch Metal)
4. Fallback → CPU

### Memory Considerations
- MLX: Uses unified memory, ~6-8GB with 8-bit quantization
- CUDA: VRAM dependent, bfloat16
- MPS: Requires attention_slicing for stability
- CPU: Slowest, use with attention_slicing
