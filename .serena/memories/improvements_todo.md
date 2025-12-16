# BarunVision 개선점 목록

## ✅ 완료된 항목
| # | 항목 | 완료일 | 구현 방법 |
|---|------|--------|----------|
| 1 | Progress Bar (Diffusers) | 2025-12-09 | `callback_on_step_end` + `gr.Progress()` |
| 2 | 생성 취소 버튼 | 2025-12-09 | `_cancel_requested` + `pipeline._interrupt` |
| 3 | MPS attention slicing | 2025-12-09 | `enable_attention_slicing()` |
| 4 | 기본값 최적화 | 2025-12-09 | 512x512, 6 steps |
| 5 | **Image-to-Image 기능** | 2025-12-16 | `ZImageImg2ImgPipeline` + Tabs UI + MLX 비활성화 |
| 6 | **메모리 해제 기능** | 2025-12-16 | `unload_model()` + 🗑️ 버튼 + gc.collect() + GPU cache clear |

## 🎉 오픈소스 기여
- **PR #12815**: Flux2ImageProcessor AttributeError 수정 (리뷰 대기 중)
- **이슈 #12809**: Kandinsky5 CUDA 하드코딩 문제 제기 (PR #12814에서 해결됨)

## ❌ 남은 개선점
| # | 항목 | 우선순위 | 설명 |
|---|------|----------|------|
| 1 | 생성 중 버튼 상태 변경 | 중간 | `gr.update(interactive=False)`로 생성 중 버튼 비활성화 |
| 2 | 예상 시간(ETA) 표시 | 중간 | 첫 스텝 시간 측정 → 남은 시간 계산 표시 |
| 3 | MLX Progress Bar | 낮음 | MFLUX 콜백 지원 여부 확인 필요 |
| 4 | MLX 취소 기능 | 낮음 | MFLUX 중단 메커니즘 확인 필요 |

## 구현 노트

### 메모리 해제 기능 (완료)
- `unload_model()` 함수: app.py:302-351
- MLX 모델, T2I/I2I 파이프라인 모두 해제
- `gc.collect()` + `torch.cuda.empty_cache()` / `torch.mps.empty_cache()`
- UI: 탭 아래, 이벤트 핸들러 전에 버튼 배치
