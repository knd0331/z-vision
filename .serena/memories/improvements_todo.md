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
| 7 | **버튼 토글 UX** | 2025-12-16 | Generator 패턴 + visibility 토글 (생성↔취소) |
| 8 | **Multi-LoRA 지원** | 2025-12-16 | MLX/Diffusers 모두 다중 LoRA 적용 지원 |
| 9 | **Real-ESRGAN 업스케일** | 2025-12-16 | py-real-esrgan 4x 업스케일 (1024→4096) |

## 🎉 오픈소스 기여
- **PR #12815**: Flux2ImageProcessor AttributeError 수정 (리뷰 대기 중)
- **이슈 #12809**: Kandinsky5 CUDA 하드코딩 문제 제기 (PR #12814에서 해결됨)

## ✅ 추가 완료 (2025-12-19)
| # | 항목 | 구현 방법 |
|---|------|----------|
| 10 | **OOM 예외 처리** | `is_oom_error()` + `get_oom_message()` 헬퍼 함수 |
| 11 | **ETA 표시** | `make_progress_callback()`에 시간 측정 로직 추가 |
| 12 | **LoRA UI 동적 선택** | `scan_loras()` + Accordion UI + 3개 슬롯 |

## ❌ 남은 개선점
| # | 항목 | 우선순위 | 설명 |
|---|------|----------|------|
| 1 | MLX Progress Bar | 낮음 | MFLUX 콜백 미지원 (라이브러리 업데이트 대기) |
| 2 | MLX 취소 기능 | 낮음 | MFLUX 중단 메커니즘 미지원 |



## 구현 노트

### 버튼 토글 UX (완료)
- Generator 패턴: `generate_image()` → yield로 즉시 UI 업데이트
- `gr.update(visible=True/False)`로 버튼 visibility 토글
- 대기 중: 생성 버튼만 표시
- 생성 중: 취소 버튼만 표시
- outputs에 버튼 포함: `[output_image, status, generate_btn, cancel_btn]`

### 메모리 해제 기능 (완료)
- `unload_model()` 함수: app.py:302-351
- MLX 모델, T2I/I2I 파이프라인 모두 해제
- `gc.collect()` + `torch.cuda.empty_cache()` / `torch.mps.empty_cache()`
