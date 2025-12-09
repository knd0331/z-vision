# BarunVision 세션 기록 - 2025-12-09

## 주요 작업

### 1. Diffusers 라이브러리 버그 발견 및 이슈 등록
- **문제**: CUDA 없는 환경(Apple Silicon MPS)에서 import 시 경고 발생
- **원인**: `transformer_kandinsky.py:168`에서 `@torch.autocast(device_type="cuda")` 하드코딩
- **조치**: huggingface/diffusers GitHub에 이슈 등록 완료

### 2. MPS 성능 최적화 적용
- **enable_attention_slicing()**: MPS/CPU에서만 활성화, CUDA는 미적용 (성능 저하 방지)
- **기본값 변경**:
  - 너비/높이: 1024 → 512
  - 스텝 수: 9 → 6
- **목적**: MPS에서 스텝당 ~45초 소요되어 속도 개선 필요

### 3. 모델 정보
- **모델**: Tongyi-MAI/Z-Image-Turbo (6B 파라미터)
- **다운로드 위치**: `~/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo`
- **용량**: 약 31GB

## 코드 변경 사항

### get_pipeline() 함수
```python
elif torch.backends.mps.is_available():
    _pipeline.to("mps")
    _pipeline.enable_attention_slicing()  # 추가
    print("✅ Apple MPS에서 실행 (attention slicing 활성화)")
else:
    _pipeline.to("cpu")
    _pipeline.enable_attention_slicing()  # 추가
```

### create_ui() 함수
- width/height value: 1024 → 512
- num_steps value: 9 → 6

## 성능 비교
| 환경 | 스텝당 시간 | 9스텝 총 시간 |
|------|-----------|-------------|
| H800 GPU (공식) | ~0.1초 | 1초 미만 |
| CUDA GPU (일반) | ~1-3초 | 10-30초 |
| Apple MPS | ~45-50초 | 7-8분 |

## 참고 문서
- [Diffusers MPS 최적화](https://huggingface.co/docs/diffusers/optimization/mps)
- [Z-Image-Turbo 모델](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
