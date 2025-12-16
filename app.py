"""
Z-Vision - Z-Image-Turbo 기반 이미지 생성 웹 UI (통합 버전)

지원 백엔드:
    - MLX (Apple Silicon 최적화)
    - CUDA (NVIDIA GPU)
    - MPS (Apple Metal - PyTorch)
    - CPU (폴백)

사용법:
    python app.py

브라우저에서 http://localhost:7860 접속
"""

import os
import platform
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image

# 출력 디렉토리 설정
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# LoRA 설정
LORA_DIR = Path("loras")
LORA_DIR.mkdir(exist_ok=True)

# 다중 LoRA 설정: [(파일명, 강도), ...]
# 예: [("zy_CinematicShot_zit", 1.0), ("anime_style", 0.5)]
# 빈 리스트 = LoRA 미사용
LORA_CONFIG = [
    ("zy_CinematicShot_zit", 1.0),
    # ("another_lora", 0.5),  # 주석 해제하여 추가
]


def get_configured_loras() -> list[tuple[str, float]]:
    """설정된 LoRA 목록 반환. [(경로, 강도), ...] 형태."""
    result = []
    for name, scale in LORA_CONFIG:
        path = LORA_DIR / f"{name}.safetensors"
        if path.exists():
            result.append((str(path), scale))
        else:
            print(f"⚠️ LoRA 파일 없음: {path}")
    return result


# 전역 변수
_backend = None
_model = None
_pipeline = None
_img2img_pipeline = None  # Image-to-Image 파이프라인
_is_generating = False
_cancel_requested = False

# LoRA 상태 추적
_mlx_lora_configs = None  # MLX 모델에 적용된 LoRA 설정 (재생성 필요 확인용)
_loaded_adapters = []  # Diffusers T2I에 로드된 어댑터 이름들
_loaded_i2i_adapters = []  # Diffusers I2I에 로드된 어댑터 이름들

# 업스케일러
_upscaler = None
_upscaler_device = None


def get_upscaler():
    """Real-ESRGAN 업스케일러 로드 (lazy loading, spandrel 사용)."""
    global _upscaler, _upscaler_device

    if _upscaler is None:
        import torch
        from spandrel import ImageModelDescriptor, ModelLoader
        from huggingface_hub import hf_hub_download

        # 디바이스 선택 (MPS > CUDA > CPU)
        if torch.backends.mps.is_available():
            _upscaler_device = torch.device("mps")
        elif torch.cuda.is_available():
            _upscaler_device = torch.device("cuda")
        else:
            _upscaler_device = torch.device("cpu")

        print(f"🔍 업스케일러 로딩 중... (Real-ESRGAN 4x, {_upscaler_device})")

        # HuggingFace에서 모델 다운로드
        model_path = hf_hub_download(
            repo_id="ai-forever/Real-ESRGAN",
            filename="RealESRGAN_x4.pth",
        )

        # Spandrel로 모델 로드
        _upscaler = ModelLoader().load_from_file(model_path)
        assert isinstance(_upscaler, ImageModelDescriptor)
        _upscaler.model.to(_upscaler_device).eval()
        print("✅ 업스케일러 로딩 완료!")

    return _upscaler


def upscale_image(image: Image.Image) -> Image.Image:
    """이미지 업스케일 (Real-ESRGAN 4x)."""
    import torch
    import numpy as np

    upscaler = get_upscaler()

    # PIL → Tensor (BCHW, 0-1 범위)
    img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(_upscaler_device)

    # 업스케일 실행
    with torch.no_grad():
        output = upscaler.model(img_tensor)

    # Tensor → PIL
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = (output * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(output)


def detect_backend() -> str:
    """사용 가능한 최적의 백엔드를 감지."""
    # 1. Apple Silicon에서 MLX 우선 체크
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx.core  # noqa: F401
            from mflux.models.z_image.variants.turbo.z_image_turbo import ZImageTurbo  # noqa: F401
            return "mlx"
        except ImportError:
            pass  # MLX 없으면 PyTorch로 폴백

    # 2. PyTorch 기반 백엔드 체크
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except ImportError:
        pass

    # 3. 아무것도 없으면 None
    return None


def get_backend_info(backend: str) -> dict:
    """백엔드별 설정 정보 반환."""
    configs = {
        "mlx": {
            "name": "MLX (Apple Silicon)",
            "emoji": "🍎",
            "default_steps": 4,
            "default_size": 512,
            "max_size": 1536,
            "step_info": "4 권장 (Turbo 최적화)",
        },
        "cuda": {
            "name": "CUDA (NVIDIA GPU)",
            "emoji": "🎮",
            "default_steps": 6,
            "default_size": 1024,
            "max_size": 2048,
            "step_info": "6-8 권장",
        },
        "mps": {
            "name": "MPS (Apple Metal)",
            "emoji": "🍏",
            "default_steps": 6,
            "default_size": 512,
            "max_size": 1024,
            "step_info": "6 권장 (낮을수록 빠름)",
        },
        "cpu": {
            "name": "CPU",
            "emoji": "💻",
            "default_steps": 4,
            "default_size": 512,
            "max_size": 768,
            "step_info": "4 권장 (느림 주의)",
        },
    }
    return configs.get(backend, configs["cpu"])


# ============================================================
# MLX Backend
# ============================================================

def get_mlx_model(lora_configs: list[tuple[str, float]] | None = None):
    """MFLUX ZImageTurbo 모델 로드 (lazy loading, 다중 LoRA 지원)."""
    global _model, _mlx_lora_configs

    # LoRA 설정 변경 시 모델 재생성 필요
    if _model is not None and _mlx_lora_configs != lora_configs:
        old_names = [Path(p).name for p, _ in (_mlx_lora_configs or [])]
        new_names = [Path(p).name for p, _ in (lora_configs or [])]
        print(f"🔄 LoRA 변경 감지: {old_names} → {new_names}")
        del _model
        _model = None
        _mlx_lora_configs = None
        import gc
        gc.collect()

    if _model is None:
        from mflux.models.z_image.variants.turbo.z_image_turbo import ZImageTurbo

        if lora_configs:
            lora_paths = [path for path, _ in lora_configs]
            lora_scales = [scale for _, scale in lora_configs]
            lora_names = [Path(p).name for p in lora_paths]
            print(f"🚀 Z-Image-Turbo 모델 로딩 중 (MLX + LoRA: {lora_names})...")
            _model = ZImageTurbo(
                quantize=8,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
            )
            _mlx_lora_configs = lora_configs
            print(f"✅ 모델 로딩 완료! (MLX + 8-bit 양자화 + {len(lora_configs)} LoRA)")
        else:
            print("🚀 Z-Image-Turbo 모델 로딩 중 (MLX)...")
            _model = ZImageTurbo(quantize=8)
            _mlx_lora_configs = None
            print("✅ 모델 로딩 완료! (MLX + 8-bit 양자화)")

    return _model


def generate_mlx(
    prompt: str, width: int, height: int, num_steps: int, seed: int,
    lora_configs: list[tuple[str, float]] | None = None,
) -> tuple[Image.Image, float]:
    """MLX 백엔드로 이미지 생성 (다중 LoRA 지원)."""
    import random
    model = get_mlx_model(lora_configs)

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    result = model.generate_image(
        seed=seed,
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_steps,
    )

    return result.image, result.generation_time, seed


# ============================================================
# PyTorch/Diffusers Backend
# ============================================================

def get_diffusers_pipeline(device: str):
    """Diffusers ZImagePipeline 로드 (lazy loading)."""
    global _pipeline
    if _pipeline is None:
        print(f"🚀 Z-Image-Turbo 모델 로딩 중 (PyTorch/{device.upper()})...")
        import torch
        from diffusers import ZImagePipeline

        _pipeline = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        if device == "cuda":
            _pipeline.to("cuda")
            print("✅ CUDA GPU에서 실행")
        elif device == "mps":
            _pipeline.to("mps")
            _pipeline.enable_attention_slicing()
            print("✅ Apple MPS에서 실행 (attention slicing 활성화)")
        else:
            _pipeline.to("cpu")
            _pipeline.enable_attention_slicing()
            print("⚠️ CPU에서 실행 (느릴 수 있음)")

        print("✅ 모델 로딩 완료!")

    return _pipeline


def make_progress_callback(num_steps: int, progress_fn=None):
    """Diffusers용 진행률 콜백 생성."""
    def callback(pipeline, step, timestep, callback_kwargs):
        global _cancel_requested
        
        # 진행률 업데이트
        if progress_fn is not None:
            progress_fn((step + 1) / num_steps, desc=f"Step {step + 1}/{num_steps}")
        
        # 취소 요청 확인
        if _cancel_requested:
            pipeline._interrupt = True
        
        return callback_kwargs
    
    return callback


def generate_diffusers(
    prompt: str, width: int, height: int, num_steps: int, seed: int, device: str,
    progress_fn=None, lora_configs: list[tuple[str, float]] | None = None,
) -> tuple[Image.Image, float]:
    """PyTorch/Diffusers 백엔드로 이미지 생성 (다중 LoRA 지원)."""
    global _loaded_adapters
    import torch
    pipe = get_diffusers_pipeline(device)

    # 현재 요청된 어댑터 이름과 가중치 계산
    requested_adapters = []
    adapter_weights = []
    if lora_configs:
        for path, scale in lora_configs:
            adapter_name = Path(path).stem  # 파일명 (확장자 제외)
            requested_adapters.append(adapter_name)
            adapter_weights.append(scale)

    # 새로운 LoRA 로드 (아직 로드되지 않은 것만)
    for path, scale in (lora_configs or []):
        adapter_name = Path(path).stem
        if adapter_name not in _loaded_adapters:
            print(f"🎨 LoRA 로딩: {adapter_name}")
            try:
                pipe.load_lora_weights(path, adapter_name=adapter_name)
                _loaded_adapters.append(adapter_name)
            except Exception as e:
                print(f"⚠️ LoRA 로드 실패: {adapter_name} - {e}")

    # 어댑터 설정 (활성화)
    if requested_adapters:
        print(f"🎨 LoRA 적용: {list(zip(requested_adapters, adapter_weights))}")
        pipe.set_adapters(requested_adapters, adapter_weights=adapter_weights)
    elif _loaded_adapters:
        # LoRA 설정이 비어있으면 모든 어댑터 비활성화
        pipe.set_adapters([])

    if seed == -1:
        seed = torch.randint(0, 2**32, (1,)).item()

    generator = torch.Generator(device if device != "cpu" else "cpu").manual_seed(int(seed))

    # 진행률 콜백 생성
    callback = make_progress_callback(num_steps, progress_fn)

    start_time = time.time()
    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_steps,
        guidance_scale=0.0,  # Turbo 모델은 0.0 필수
        generator=generator,
        callback_on_step_end=callback,
    )
    gen_time = time.time() - start_time

    return result.images[0], gen_time, seed


# ============================================================
# Image-to-Image Backend (Diffusers Only)
# ============================================================

def get_img2img_pipeline(device: str):
    """Diffusers ZImageImg2ImgPipeline 로드 (lazy loading)."""
    global _img2img_pipeline
    if _img2img_pipeline is None:
        print(f"🚀 Z-Image-Turbo Img2Img 모델 로딩 중 (PyTorch/{device.upper()})...")
        import torch
        from diffusers import ZImageImg2ImgPipeline

        _img2img_pipeline = ZImageImg2ImgPipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        if device == "cuda":
            _img2img_pipeline.to("cuda")
            print("✅ CUDA GPU에서 실행 (Img2Img)")
        elif device == "mps":
            _img2img_pipeline.to("mps")
            _img2img_pipeline.enable_attention_slicing()
            print("✅ Apple MPS에서 실행 (Img2Img, attention slicing 활성화)")
        else:
            _img2img_pipeline.to("cpu")
            _img2img_pipeline.enable_attention_slicing()
            print("⚠️ CPU에서 실행 (Img2Img, 느릴 수 있음)")

        print("✅ Img2Img 모델 로딩 완료!")

    return _img2img_pipeline


def generate_img2img(
    prompt: str,
    init_image: Image.Image,
    strength: float,
    num_steps: int,
    seed: int,
    device: str,
    progress_fn=None,
    lora_configs: list[tuple[str, float]] | None = None,
) -> tuple[Image.Image, float, int]:
    """PyTorch/Diffusers 백엔드로 Image-to-Image 생성 (다중 LoRA 지원)."""
    global _loaded_i2i_adapters
    import torch
    pipe = get_img2img_pipeline(device)

    # 현재 요청된 어댑터 이름과 가중치 계산
    requested_adapters = []
    adapter_weights = []
    if lora_configs:
        for path, scale in lora_configs:
            adapter_name = Path(path).stem  # 파일명 (확장자 제외)
            requested_adapters.append(adapter_name)
            adapter_weights.append(scale)

    # 새로운 LoRA 로드 (아직 로드되지 않은 것만)
    for path, scale in (lora_configs or []):
        adapter_name = Path(path).stem
        if adapter_name not in _loaded_i2i_adapters:
            print(f"🎨 LoRA 로딩 (I2I): {adapter_name}")
            try:
                pipe.load_lora_weights(path, adapter_name=adapter_name)
                _loaded_i2i_adapters.append(adapter_name)
            except Exception as e:
                print(f"⚠️ LoRA 로드 실패 (I2I): {adapter_name} - {e}")

    # 어댑터 설정 (활성화)
    if requested_adapters:
        print(f"🎨 LoRA 적용 (I2I): {list(zip(requested_adapters, adapter_weights))}")
        pipe.set_adapters(requested_adapters, adapter_weights=adapter_weights)
    elif _loaded_i2i_adapters:
        # LoRA 설정이 비어있으면 모든 어댑터 비활성화
        pipe.set_adapters([])

    if seed == -1:
        seed = torch.randint(0, 2**32, (1,)).item()

    generator = torch.Generator(device if device != "cpu" else "cpu").manual_seed(int(seed))

    # 진행률 콜백 생성
    callback = make_progress_callback(num_steps, progress_fn)

    start_time = time.time()
    result = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        num_inference_steps=num_steps,
        guidance_scale=0.0,  # Turbo 모델은 0.0 필수
        generator=generator,
        callback_on_step_end=callback,
    )
    gen_time = time.time() - start_time

    return result.images[0], gen_time, seed


# ============================================================
# Unified Generation
# ============================================================

def cancel_generation():
    """생성 취소 요청. 버튼 상태도 함께 반환."""
    global _cancel_requested, _is_generating
    if _is_generating:
        _cancel_requested = True
        # 취소 요청 시 버튼 상태 복원 (생성 버튼 표시, 취소 버튼 숨김)
        return "⏹️ 취소 요청됨... 현재 스텝 완료 후 중단됩니다.", gr.update(visible=True), gr.update(visible=False)
    return "생성 중이 아닙니다.", gr.update(visible=True), gr.update(visible=False)


def unload_model():
    """모델 언로드 및 메모리 해제."""
    global _model, _pipeline, _img2img_pipeline, _upscaler, _upscaler_device, _is_generating

    if _is_generating:
        return "⚠️ 생성 중에는 언로드할 수 없습니다. 먼저 취소해주세요."

    unloaded = []

    # MLX 모델 해제
    if _model is not None:
        del _model
        _model = None
        unloaded.append("MLX")

    # Diffusers T2I 파이프라인 해제
    if _pipeline is not None:
        del _pipeline
        _pipeline = None
        unloaded.append("T2I Pipeline")

    # Diffusers I2I 파이프라인 해제
    if _img2img_pipeline is not None:
        del _img2img_pipeline
        _img2img_pipeline = None
        unloaded.append("I2I Pipeline")

    # 업스케일러 해제
    if _upscaler is not None:
        del _upscaler
        _upscaler = None
        _upscaler_device = None
        unloaded.append("Upscaler")

    if not unloaded:
        return "ℹ️ 언로드할 모델이 없습니다."
    
    # 가비지 컬렉션
    import gc
    gc.collect()
    
    # GPU 메모리 캐시 정리
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass
    
    return f"✅ 모델 언로드 완료: {', '.join(unloaded)}\n💾 메모리가 해제되었습니다."


# ============================================================
# Unified Generation
# ============================================================

def generate_image(
    prompt: str,
    width: int,
    height: int,
    num_steps: int,
    seed: int,
    save_image: bool,
    upscale: bool = False,
    progress=gr.Progress(),
):
    """통합 이미지 생성 함수 (Generator 버전 - 버튼 토글, LoRA, 업스케일 지원)."""
    global _backend, _is_generating, _cancel_requested

    # 버튼 상태: (생성 버튼 visible, 취소 버튼 visible)
    BTN_GENERATE = (gr.update(visible=True), gr.update(visible=False))
    BTN_CANCEL = (gr.update(visible=False), gr.update(visible=True))

    if not prompt.strip():
        yield None, "❌ 프롬프트를 입력해주세요.", *BTN_GENERATE
        return

    if _backend is None:
        yield None, "❌ 사용 가능한 백엔드가 없습니다. PyTorch 또는 MLX를 설치해주세요.", *BTN_GENERATE
        return

    if _is_generating:
        yield None, "⚠️ 이미 생성 중입니다. 완료될 때까지 기다리거나 취소해주세요.", *BTN_GENERATE
        return

    # 전역 LoRA 설정 사용
    lora_configs = get_configured_loras()
    lora_names = [Path(p).stem for p, _ in lora_configs] if lora_configs else []

    try:
        _is_generating = True
        _cancel_requested = False

        # 즉시 버튼 상태 변경 (취소 버튼으로)
        lora_info = f" + LoRA: {lora_names}" if lora_configs else ""
        yield None, f"🚀 생성 시작...{lora_info}", *BTN_CANCEL

        print(f"🎨 이미지 생성 중... (backend: {_backend}{lora_info})")
        progress(0, desc="생성 시작...")

        if _backend == "mlx":
            # MLX는 콜백 미지원, 단순 진행률 표시
            progress(0.1, desc="MLX 생성 중... (진행률 표시 미지원)")
            image, gen_time, used_seed = generate_mlx(
                prompt, width, height, num_steps, seed,
                lora_configs=lora_configs,
            )
        else:
            # Diffusers는 콜백으로 진행률 표시
            image, gen_time, used_seed = generate_diffusers(
                prompt, width, height, num_steps, seed, _backend,
                progress_fn=progress, lora_configs=lora_configs,
            )

        # 취소 확인
        if _cancel_requested:
            _is_generating = False
            _cancel_requested = False
            yield None, "⏹️ 생성이 취소되었습니다.", *BTN_GENERATE
            return

        # 업스케일 적용
        original_size = image.size if image else None
        if upscale and image is not None:
            progress(0.9, desc="🔍 업스케일 중...")
            print("🔍 업스케일 중... (Real-ESRGAN 4x)")
            image = upscale_image(image)
            print(f"✅ 업스케일 완료: {original_size} → {image.size}")

        # 상태 메시지
        backend_info = get_backend_info(_backend)
        status = f"✅ 생성 완료! ({backend_info['emoji']} {_backend.upper()}, seed: {used_seed}, {gen_time:.1f}초)"
        if upscale and original_size:
            status += f"\n🔍 업스케일: {original_size[0]}x{original_size[1]} → {image.size[0]}x{image.size[1]}"

        # 이미지 저장
        if save_image and image is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "_upscaled" if upscale else ""
            filename = OUTPUT_DIR / f"zvision_{timestamp}_{used_seed}{suffix}.png"
            image.save(filename)
            status += f"\n💾 저장됨: {filename}"

        progress(1.0, desc="완료!")
        yield image, status, *BTN_GENERATE

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield None, f"❌ 오류 발생: {str(e)}", *BTN_GENERATE
    finally:
        _is_generating = False
        _cancel_requested = False


def generate_image_i2i(
    prompt: str,
    init_image: Image.Image,
    strength: float,
    num_steps: int,
    seed: int,
    save_image: bool,
    upscale: bool = False,
    progress=gr.Progress(),
):
    """통합 Image-to-Image 생성 함수 (Generator 버전 - 버튼 토글, LoRA, 업스케일 지원)."""
    global _backend, _is_generating, _cancel_requested

    # 버튼 상태: (생성 버튼 visible, 취소 버튼 visible)
    BTN_GENERATE = (gr.update(visible=True), gr.update(visible=False))
    BTN_CANCEL = (gr.update(visible=False), gr.update(visible=True))

    if not prompt.strip():
        yield None, "❌ 프롬프트를 입력해주세요.", *BTN_GENERATE
        return

    if init_image is None:
        yield None, "❌ 입력 이미지를 업로드해주세요.", *BTN_GENERATE
        return

    if _backend is None:
        yield None, "❌ 사용 가능한 백엔드가 없습니다.", *BTN_GENERATE
        return

    if _backend == "mlx":
        yield None, "❌ MLX 백엔드는 Image-to-Image를 지원하지 않습니다. PyTorch 백엔드를 사용해주세요.", *BTN_GENERATE
        return

    if _is_generating:
        yield None, "⚠️ 이미 생성 중입니다. 완료될 때까지 기다리거나 취소해주세요.", *BTN_GENERATE
        return

    # 전역 LoRA 설정 사용
    lora_configs = get_configured_loras()
    lora_names = [Path(p).stem for p, _ in lora_configs] if lora_configs else []

    try:
        _is_generating = True
        _cancel_requested = False

        # 즉시 버튼 상태 변경 (취소 버튼으로)
        lora_info = f" + LoRA: {lora_names}" if lora_configs else ""
        yield None, f"🚀 Img2Img 생성 시작...{lora_info}", *BTN_CANCEL

        print(f"🖼️ Image-to-Image 생성 중... (backend: {_backend}, strength: {strength}{lora_info})")
        progress(0, desc="Img2Img 생성 시작...")

        image, gen_time, used_seed = generate_img2img(
            prompt, init_image, strength, num_steps, seed, _backend,
            progress_fn=progress, lora_configs=lora_configs,
        )

        # 취소 확인
        if _cancel_requested:
            _is_generating = False
            _cancel_requested = False
            yield None, "⏹️ 생성이 취소되었습니다.", *BTN_GENERATE
            return

        # 업스케일 적용
        original_size = image.size if image else None
        if upscale and image is not None:
            progress(0.9, desc="🔍 업스케일 중...")
            print("🔍 업스케일 중... (Real-ESRGAN 4x)")
            image = upscale_image(image)
            print(f"✅ 업스케일 완료: {original_size} → {image.size}")

        # 상태 메시지
        backend_info = get_backend_info(_backend)
        status = f"✅ Img2Img 완료! ({backend_info['emoji']} {_backend.upper()}, strength: {strength}, seed: {used_seed}, {gen_time:.1f}초)"
        if upscale and original_size:
            status += f"\n🔍 업스케일: {original_size[0]}x{original_size[1]} → {image.size[0]}x{image.size[1]}"

        # 이미지 저장
        if save_image and image is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "_upscaled" if upscale else ""
            filename = OUTPUT_DIR / f"zvision_i2i_{timestamp}_{used_seed}{suffix}.png"
            image.save(filename)
            status += f"\n💾 저장됨: {filename}"

        progress(1.0, desc="완료!")
        yield image, status, *BTN_GENERATE

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield None, f"❌ 오류 발생: {str(e)}", *BTN_GENERATE
    finally:
        _is_generating = False
        _cancel_requested = False


# ============================================================
# Gradio UI
# ============================================================

def create_ui():
    """Gradio UI 생성."""
    global _backend
    _backend = detect_backend()
    backend_info = get_backend_info(_backend) if _backend else {"name": "None", "emoji": "❌"}

    # Image-to-Image 지원 여부 (MLX는 미지원)
    i2i_supported = _backend is not None and _backend != "mlx"

    with gr.Blocks() as app:

        gr.HTML(f"""
        <div class="title">
            <h1>🎨 Z-Vision</h1>
            <p>Z-Image-Turbo AI 이미지 생성기</p>
            <p class="backend-info">{backend_info['emoji']} Backend: <strong>{backend_info['name'] if _backend else '없음'}</strong></p>
        </div>
        """)

        if _backend is None:
            gr.HTML("""
            <div class="error-box">
                <p>⚠️ 사용 가능한 백엔드가 없습니다!</p>
                <p>PyTorch 또는 MLX를 설치해주세요.</p>
            </div>
            """)

        with gr.Tabs():
            # ============================================================
            # Tab 1: Text-to-Image
            # ============================================================
            with gr.TabItem("🎨 Text-to-Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # 입력 섹션
                        prompt_t2i = gr.Textbox(
                            label="프롬프트",
                            placeholder="생성하고 싶은 이미지를 설명해주세요...",
                            lines=4,
                            max_lines=8,
                        )

                        with gr.Row():
                            width_t2i = gr.Slider(
                                label="너비",
                                minimum=512,
                                maximum=backend_info.get("max_size", 1536) if _backend else 1024,
                                value=backend_info.get("default_size", 512) if _backend else 512,
                                step=64,
                            )
                            height_t2i = gr.Slider(
                                label="높이",
                                minimum=512,
                                maximum=backend_info.get("max_size", 1536) if _backend else 1024,
                                value=backend_info.get("default_size", 512) if _backend else 512,
                                step=64,
                            )

                        with gr.Row():
                            num_steps_t2i = gr.Slider(
                                label="스텝 수",
                                minimum=2,
                                maximum=10,
                                value=backend_info.get("default_steps", 4) if _backend else 4,
                                step=1,
                                info=backend_info.get("step_info", "") if _backend else "",
                            )
                            seed_t2i = gr.Number(
                                label="시드",
                                value=-1,
                                precision=0,
                                info="-1 = 랜덤",
                            )

                        save_image_t2i = gr.Checkbox(
                            label="이미지 자동 저장",
                            value=True,
                            info=f"저장 위치: {OUTPUT_DIR.absolute()}",
                        )

                        upscale_t2i = gr.Checkbox(
                            label="🔍 4x 업스케일 (Real-ESRGAN)",
                            value=False,
                            info="생성 후 4배 확대 (예: 1024→4096)",
                        )

                        with gr.Row():
                            generate_btn_t2i = gr.Button(
                                "🎨 이미지 생성",
                                variant="primary",
                                size="lg",
                                interactive=_backend is not None,
                                visible=True,
                            )
                            cancel_btn_t2i = gr.Button(
                                "⏹️ 취소",
                                variant="stop",
                                size="lg",
                                visible=False,
                            )

                    with gr.Column(scale=1):
                        # 출력 섹션
                        output_image_t2i = gr.Image(
                            label="생성된 이미지",
                            type="pil",
                            height=512,
                        )
                        status_t2i = gr.Textbox(
                            label="상태",
                            interactive=False,
                        )

                # 예제
                gr.Examples(
                    examples=[
                        ["A majestic mountain landscape at sunset with snow-capped peaks and a crystal clear lake reflection"],
                        ["귀여운 하얀 고양이가 창가에서 낮잠을 자고 있는 모습, 따뜻한 햇살"],
                        ["Cyberpunk city street at night, neon lights, rain reflections, cinematic atmosphere"],
                        ["한복을 입은 여성이 벚꽃 나무 아래 서 있는 동양화 스타일"],
                        ["Delicious Korean bibimbap in a stone pot, food photography, top view"],
                    ],
                    inputs=[prompt_t2i],
                    label="예제 프롬프트",
                )

            # ============================================================
            # Tab 2: Image-to-Image
            # ============================================================
            with gr.TabItem("🖼️ Image-to-Image"):
                # MLX 경고 메시지
                if _backend == "mlx":
                    gr.HTML("""
                    <div class="warning-box">
                        <p>⚠️ MLX 백엔드는 Image-to-Image를 지원하지 않습니다.</p>
                        <p>PyTorch 백엔드 (CUDA/MPS/CPU)를 사용해주세요.</p>
                    </div>
                    """)

                with gr.Row():
                    with gr.Column(scale=1):
                        # 입력 이미지
                        init_image = gr.Image(
                            label="입력 이미지",
                            type="pil",
                            height=256,
                        )

                        # 프롬프트
                        prompt_i2i = gr.Textbox(
                            label="프롬프트",
                            placeholder="이미지를 어떻게 변형할지 설명해주세요...",
                            lines=3,
                            max_lines=6,
                        )

                        # Strength 슬라이더
                        strength = gr.Slider(
                            label="변형 강도 (Strength)",
                            minimum=0.1,
                            maximum=1.0,
                            value=0.6,
                            step=0.05,
                            info="0.1 = 원본 유지, 1.0 = 완전 변형",
                        )

                        with gr.Row():
                            num_steps_i2i = gr.Slider(
                                label="스텝 수",
                                minimum=2,
                                maximum=10,
                                value=backend_info.get("default_steps", 6) if _backend else 6,
                                step=1,
                            )
                            seed_i2i = gr.Number(
                                label="시드",
                                value=-1,
                                precision=0,
                                info="-1 = 랜덤",
                            )

                        save_image_i2i = gr.Checkbox(
                            label="이미지 자동 저장",
                            value=True,
                            info=f"저장 위치: {OUTPUT_DIR.absolute()}",
                        )

                        upscale_i2i = gr.Checkbox(
                            label="🔍 4x 업스케일 (Real-ESRGAN)",
                            value=False,
                            info="생성 후 4배 확대 (예: 1024→4096)",
                        )

                        with gr.Row():
                            generate_btn_i2i = gr.Button(
                                "🖼️ 이미지 변형" if i2i_supported else "🖼️ 이미지 변형 (MLX 미지원)",
                                variant="primary",
                                size="lg",
                                interactive=i2i_supported,
                                visible=True,
                            )
                            cancel_btn_i2i = gr.Button(
                                "⏹️ 취소",
                                variant="stop",
                                size="lg",
                                visible=False,
                            )

                    with gr.Column(scale=1):
                        # 출력 섹션
                        output_image_i2i = gr.Image(
                            label="변형된 이미지",
                            type="pil",
                            height=512,
                        )
                        status_i2i = gr.Textbox(
                            label="상태",
                            interactive=False,
                        )

        # ============================================================
        # 유틸리티 섹션 (모델 언로드)
        # ============================================================
        gr.HTML("<hr style='margin: 1.5rem 0; opacity: 0.3;'>")
        
        with gr.Row():
            with gr.Column(scale=1):
                unload_btn = gr.Button(
                    "🗑️ 모델 언로드",
                    variant="secondary",
                    size="sm",
                )
            with gr.Column(scale=3):
                unload_status = gr.Textbox(
                    label="",
                    placeholder="모델 언로드 시 메모리가 해제됩니다. 다음 생성 시 모델이 다시 로드됩니다.",
                    interactive=False,
                    show_label=False,
                )

        # ============================================================
        # 이벤트 연결
        # ============================================================

        # Text-to-Image 이벤트
        generate_btn_t2i.click(
            fn=generate_image,
            inputs=[prompt_t2i, width_t2i, height_t2i, num_steps_t2i, seed_t2i, save_image_t2i, upscale_t2i],
            outputs=[output_image_t2i, status_t2i, generate_btn_t2i, cancel_btn_t2i],
        )

        prompt_t2i.submit(
            fn=generate_image,
            inputs=[prompt_t2i, width_t2i, height_t2i, num_steps_t2i, seed_t2i, save_image_t2i, upscale_t2i],
            outputs=[output_image_t2i, status_t2i, generate_btn_t2i, cancel_btn_t2i],
        )

        cancel_btn_t2i.click(
            fn=cancel_generation,
            inputs=[],
            outputs=[status_t2i, generate_btn_t2i, cancel_btn_t2i],
        )

        # Image-to-Image 이벤트
        generate_btn_i2i.click(
            fn=generate_image_i2i,
            inputs=[prompt_i2i, init_image, strength, num_steps_i2i, seed_i2i, save_image_i2i, upscale_i2i],
            outputs=[output_image_i2i, status_i2i, generate_btn_i2i, cancel_btn_i2i],
        )

        cancel_btn_i2i.click(
            fn=cancel_generation,
            inputs=[],
            outputs=[status_i2i, generate_btn_i2i, cancel_btn_i2i],
        )

        # 모델 언로드 이벤트
        unload_btn.click(
            fn=unload_model,
            inputs=[],
            outputs=[unload_status],
        )

        gr.HTML("""
        <div class="footer">
            <p>Powered by <a href="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo" target="_blank">Z-Image-Turbo</a> |
            <a href="https://github.com/filipstrand/mflux" target="_blank">MFLUX</a> (MLX) |
            <a href="https://huggingface.co/docs/diffusers" target="_blank">Diffusers</a> (PyTorch)</p>
        </div>
        """)

    return app


if __name__ == "__main__":
    print("=" * 50)
    print("🎨 Z-Vision 시작")
    print("=" * 50)

    backend = detect_backend()
    if backend:
        info = get_backend_info(backend)
        print(f"{info['emoji']} 감지된 백엔드: {info['name']}")
    else:
        print("⚠️ 사용 가능한 백엔드 없음")

    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css="""
        .title { text-align: center; margin-bottom: 1rem; }
        .title .backend-info { font-size: 0.9em; color: #666; margin-top: 0.5rem; }
        .footer { text-align: center; margin-top: 1rem; opacity: 0.7; }
        .error-box { background: #fee; border: 1px solid #fcc; padding: 1rem; border-radius: 8px; margin: 1rem 0; text-align: center; }
        .warning-box { background: #fff3cd !important; border: 1px solid #ffc107 !important; padding: 1rem; border-radius: 8px; margin: 1rem 0; text-align: center; color: #856404 !important; }
        .warning-box p { color: #856404 !important; }
        """,
        head="<title>Z-Vision</title>",
    )
