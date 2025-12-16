# BarunVision 세션 기록 - 2025-12-16

## 주요 작업: Image-to-Image 기능 추가

### 구현 내용
1. **ZImageImg2ImgPipeline 통합**
   - Diffusers의 `ZImageImg2ImgPipeline` 사용
   - Z-Image-Turbo 모델로 img2img 지원 확인

2. **새로 추가된 함수**
   - `get_img2img_pipeline(device)`: Img2Img 파이프라인 로딩
   - `generate_img2img()`: Img2Img 생성 로직
   - `generate_image_i2i()`: 통합 Img2Img 생성 함수

3. **UI 변경**
   - `gr.Tabs`로 T2I/I2I 탭 분리
   - Image-to-Image 탭 추가:
     - 입력 이미지 업로드 (`gr.Image`)
     - Strength 슬라이더 (0.1~1.0)
     - 생성 버튼 (MLX에서는 disabled)
   - MLX 경고 박스 추가 (노란색 배경, 갈색 텍스트)

### 백엔드 지원
| 백엔드 | Text-to-Image | Image-to-Image |
|--------|---------------|----------------|
| MLX | ✅ | ❌ (MFLUX 미지원) |
| CUDA | ✅ | ✅ |
| MPS | ✅ | ✅ |
| CPU | ✅ | ✅ |

### 코드 위치
- 파이프라인 함수: `app.py:223-287`
- 통합 생성 함수: `app.py:371-436`
- UI (Tabs): `app.py:470-693`
- CSS (warning-box): `app.py:721-722`

### 기술 노트
- `ZImageImg2ImgPipeline`은 Diffusers 0.31.0+ 에 포함
- `strength` 파라미터: 0.1(원본 유지) ~ 1.0(완전 변형)
- MLX/MFLUX는 img2img 미지원 → UI에서 버튼 비활성화
- CSS `!important` 필요 (Gradio 스타일 오버라이드)

### 학습된 내용
1. Z-Image-Turbo는 text-to-image + img2img 모두 지원
2. Z-Image-Edit는 아직 미출시 (별도 editing 모델)
3. Diffusers AutoPipeline vs 명시적 Pipeline 차이 이해

---

## 추가 작업: Real-ESRGAN 업스케일링 (2025-12-16)

### 구현 내용
1. **Spandrel + HuggingFace 통합**
   - 패키지: `spandrel>=0.4.0` (최신 huggingface_hub 호환)
   - 모델: `ai-forever/Real-ESRGAN` (HuggingFace에서 자동 다운로드)
   - 4x 업스케일링 (1024→4096)
   - MPS/CUDA/CPU 자동 감지

2. **새로 추가된 함수**
   - `get_upscaler()`: Real-ESRGAN 업스케일러 로딩 (lazy loading, spandrel 사용)
   - `upscale_image(image)`: 이미지 업스케일 실행 (PIL→Tensor→PIL 변환)

3. **UI 변경**
   - T2I 탭: "🔍 4x 업스케일 (Real-ESRGAN)" 체크박스
   - I2I 탭: 동일한 체크박스

### 코드 위치
- 업스케일러 함수: `app.py:67-125`
- T2I 체크박스: `app.py:828`
- I2I 체크박스: `app.py:935`

### 기술 노트
- `py-real-esrgan`: `cached_download` 오류 (huggingface_hub 최신 버전 호환 X)
- `realesrgan`: Python 3.13 빌드 실패
- **해결책**: `spandrel` + `hf_hub_download()` 조합
- Tensor 변환: PIL → numpy (0-1) → torch (BCHW) → model → numpy → PIL

### 문제 해결
- **cached_download 오류**: `py-real-esrgan` → `spandrel`로 변경
- **Python 3.13 호환**: `realesrgan` 빌드 실패 → `spandrel` 사용
- **체크박스 안 보임**: 서버 재시작 필요 (이전 코드로 실행 중이었음)