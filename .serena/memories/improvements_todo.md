# BarunVision 개선점 목록

## ✅ 완료된 항목
| # | 항목 | 완료일 | 구현 방법 |
|---|------|--------|----------|
| 1 | Progress Bar (Diffusers) | 2025-12-09 | `callback_on_step_end` + `gr.Progress()` |
| 2 | 생성 취소 버튼 | 2025-12-09 | `_cancel_requested` + `pipeline._interrupt` |
| 3 | MPS attention slicing | 2025-12-09 | `enable_attention_slicing()` |
| 4 | 기본값 최적화 | 2025-12-09 | 512x512, 6 steps |
| 5 | **Image-to-Image 기능** | 2025-12-16 | `ZImageImg2ImgPipeline` + Tabs UI + MLX 비활성화 |

## 🎉 오픈소스 기여
- **PR #12815**: Flux2ImageProcessor AttributeError 수정 (리뷰 대기 중)
- **이슈 #12809**: Kandinsky5 CUDA 하드코딩 문제 제기 (PR #12814에서 해결됨)

## ❌ 남은 개선점
| # | 항목 | 우선순위 | 설명 |
|---|------|----------|------|
| 1 | **메모리 해제 기능** | 🔴 높음 | 모델 언로드 버튼 + gc.collect() + torch.cuda.empty_cache() |
| 2 | 생성 중 버튼 상태 변경 | 중간 | `gr.update(interactive=False)`로 생성 중 버튼 비활성화 |
| 3 | 예상 시간(ETA) 표시 | 중간 | 첫 스텝 시간 측정 → 남은 시간 계산 표시 |
| 4 | MLX Progress Bar | 낮음 | MFLUX 콜백 지원 여부 확인 필요 |
| 5 | MLX 취소 기능 | 낮음 | MFLUX 중단 메커니즘 확인 필요 |

## 구현 노트

### 1. 메모리 해제 기능 (우선 구현)
```python
def unload_model():
    global _model, _pipeline
    
    if _pipeline is not None:
        del _pipeline
        _pipeline = None
    
    if _model is not None:
        del _model
        _model = None
    
    import gc
    gc.collect()
    
    # GPU 메모리 해제
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except:
        pass
    
    return "✅ 모델 언로드 완료 - 메모리 해제됨"
```

UI 추가:
- "🗑️ 모델 언로드" 버튼
- 현재 메모리 사용량 표시 (선택사항)