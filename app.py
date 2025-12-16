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

# 전역 변수
_backend = None
_model = None
_pipeline = None
_img2img_pipeline = None  # Image-to-Image 파이프라인
_is_generating = False
_cancel_requested = False


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

def get_mlx_model():
    """MFLUX ZImageTurbo 모델 로드 (lazy loading)."""
    global _model
    if _model is None:
        print("🚀 Z-Image-Turbo 모델 로딩 중 (MLX)...")
        from mflux.models.z_image.variants.turbo.z_image_turbo import ZImageTurbo
        _model = ZImageTurbo(quantize=8)
        print("✅ 모델 로딩 완료! (MLX + 8-bit 양자화)")
    return _model


def generate_mlx(prompt: str, width: int, height: int, num_steps: int, seed: int) -> tuple[Image.Image, float]:
    """MLX 백엔드로 이미지 생성."""
    import random
    model = get_mlx_model()

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


def generate_diffusers(prompt: str, width: int, height: int, num_steps: int, seed: int, device: str, progress_fn=None) -> tuple[Image.Image, float]:
    """PyTorch/Diffusers 백엔드로 이미지 생성."""
    import torch
    pipe = get_diffusers_pipeline(device)

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
) -> tuple[Image.Image, float, int]:
    """PyTorch/Diffusers 백엔드로 Image-to-Image 생성."""
    import torch
    pipe = get_img2img_pipeline(device)

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
    """생성 취소 요청."""
    global _cancel_requested, _is_generating
    if _is_generating:
        _cancel_requested = True
        return "⏹️ 취소 요청됨... 현재 스텝 완료 후 중단됩니다."
    return "생성 중이 아닙니다."


def generate_image(
    prompt: str,
    width: int,
    height: int,
    num_steps: int,
    seed: int,
    save_image: bool,
    progress=gr.Progress(),
) -> tuple[Image.Image, str]:
    """통합 이미지 생성 함수."""
    global _backend, _is_generating, _cancel_requested

    if not prompt.strip():
        return None, "❌ 프롬프트를 입력해주세요."

    if _backend is None:
        return None, "❌ 사용 가능한 백엔드가 없습니다. PyTorch 또는 MLX를 설치해주세요."

    if _is_generating:
        return None, "⚠️ 이미 생성 중입니다. 완료될 때까지 기다리거나 취소해주세요."

    try:
        _is_generating = True
        _cancel_requested = False
        
        print(f"🎨 이미지 생성 중... (backend: {_backend})")
        progress(0, desc="생성 시작...")

        if _backend == "mlx":
            # MLX는 콜백 미지원, 단순 진행률 표시
            progress(0.1, desc="MLX 생성 중... (진행률 표시 미지원)")
            image, gen_time, used_seed = generate_mlx(prompt, width, height, num_steps, seed)
        else:
            # Diffusers는 콜백으로 진행률 표시
            image, gen_time, used_seed = generate_diffusers(
                prompt, width, height, num_steps, seed, _backend,
                progress_fn=progress
            )

        # 취소 확인
        if _cancel_requested:
            _is_generating = False
            _cancel_requested = False
            return None, "⏹️ 생성이 취소되었습니다."

        # 상태 메시지
        backend_info = get_backend_info(_backend)
        status = f"✅ 생성 완료! ({backend_info['emoji']} {_backend.upper()}, seed: {used_seed}, {gen_time:.1f}초)"

        # 이미지 저장
        if save_image and image is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = OUTPUT_DIR / f"zvision_{timestamp}_{used_seed}.png"
            image.save(filename)
            status += f"\n💾 저장됨: {filename}"

        progress(1.0, desc="완료!")
        return image, status

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ 오류 발생: {str(e)}"
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
    progress=gr.Progress(),
) -> tuple[Image.Image, str]:
    """통합 Image-to-Image 생성 함수."""
    global _backend, _is_generating, _cancel_requested

    if not prompt.strip():
        return None, "❌ 프롬프트를 입력해주세요."

    if init_image is None:
        return None, "❌ 입력 이미지를 업로드해주세요."

    if _backend is None:
        return None, "❌ 사용 가능한 백엔드가 없습니다."

    if _backend == "mlx":
        return None, "❌ MLX 백엔드는 Image-to-Image를 지원하지 않습니다. PyTorch 백엔드를 사용해주세요."

    if _is_generating:
        return None, "⚠️ 이미 생성 중입니다. 완료될 때까지 기다리거나 취소해주세요."

    try:
        _is_generating = True
        _cancel_requested = False

        print(f"🖼️ Image-to-Image 생성 중... (backend: {_backend}, strength: {strength})")
        progress(0, desc="Img2Img 생성 시작...")

        image, gen_time, used_seed = generate_img2img(
            prompt, init_image, strength, num_steps, seed, _backend,
            progress_fn=progress
        )

        # 취소 확인
        if _cancel_requested:
            _is_generating = False
            _cancel_requested = False
            return None, "⏹️ 생성이 취소되었습니다."

        # 상태 메시지
        backend_info = get_backend_info(_backend)
        status = f"✅ Img2Img 완료! ({backend_info['emoji']} {_backend.upper()}, strength: {strength}, seed: {used_seed}, {gen_time:.1f}초)"

        # 이미지 저장
        if save_image and image is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = OUTPUT_DIR / f"zvision_i2i_{timestamp}_{used_seed}.png"
            image.save(filename)
            status += f"\n💾 저장됨: {filename}"

        progress(1.0, desc="완료!")
        return image, status

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ 오류 발생: {str(e)}"
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

                        with gr.Row():
                            generate_btn_t2i = gr.Button(
                                "🎨 이미지 생성",
                                variant="primary",
                                size="lg",
                                interactive=_backend is not None,
                                scale=3,
                            )
                            cancel_btn_t2i = gr.Button(
                                "⏹️ 취소",
                                variant="stop",
                                size="lg",
                                scale=1,
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

                        with gr.Row():
                            generate_btn_i2i = gr.Button(
                                "🖼️ 이미지 변형" if i2i_supported else "🖼️ 이미지 변형 (MLX 미지원)",
                                variant="primary",
                                size="lg",
                                interactive=i2i_supported,
                                scale=3,
                            )
                            cancel_btn_i2i = gr.Button(
                                "⏹️ 취소",
                                variant="stop",
                                size="lg",
                                scale=1,
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
        # 이벤트 연결
        # ============================================================

        # Text-to-Image 이벤트
        generate_btn_t2i.click(
            fn=generate_image,
            inputs=[prompt_t2i, width_t2i, height_t2i, num_steps_t2i, seed_t2i, save_image_t2i],
            outputs=[output_image_t2i, status_t2i],
        )

        prompt_t2i.submit(
            fn=generate_image,
            inputs=[prompt_t2i, width_t2i, height_t2i, num_steps_t2i, seed_t2i, save_image_t2i],
            outputs=[output_image_t2i, status_t2i],
        )

        cancel_btn_t2i.click(
            fn=cancel_generation,
            inputs=[],
            outputs=[status_t2i],
        )

        # Image-to-Image 이벤트
        generate_btn_i2i.click(
            fn=generate_image_i2i,
            inputs=[prompt_i2i, init_image, strength, num_steps_i2i, seed_i2i, save_image_i2i],
            outputs=[output_image_i2i, status_i2i],
        )

        cancel_btn_i2i.click(
            fn=cancel_generation,
            inputs=[],
            outputs=[status_i2i],
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
