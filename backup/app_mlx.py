"""
BarunVision - Z-Image-Turbo 기반 이미지 생성 웹 UI (MLX 버전)

사용법:
    python app.py

브라우저에서 http://localhost:7860 접속
"""

import os
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image

# 출력 디렉토리 설정
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# 전역 모델 (lazy loading)
_model = None


def get_model():
    """MFLUX ZImageTurbo 모델을 로드하거나 캐시된 인스턴스 반환."""
    global _model
    if _model is None:
        print("🚀 Z-Image-Turbo 모델 로딩 중 (MLX)...")

        from mflux.models.z_image.variants.turbo.z_image_turbo import ZImageTurbo

        # 8-bit 양자화로 메모리 최적화 및 속도 향상
        _model = ZImageTurbo(quantize=8)

        print("✅ 모델 로딩 완료! (MLX + 8-bit 양자화)")

    return _model


def generate_image(
    prompt: str,
    width: int,
    height: int,
    num_steps: int,
    seed: int,
    save_image: bool,
) -> tuple[Image.Image, str]:
    """
    프롬프트로 이미지 생성.

    Args:
        prompt: 이미지 설명 텍스트
        width: 이미지 너비
        height: 이미지 높이
        num_steps: 추론 스텝 수 (4-6 권장)
        seed: 랜덤 시드 (-1이면 랜덤)
        save_image: 이미지 저장 여부

    Returns:
        생성된 이미지와 상태 메시지
    """
    if not prompt.strip():
        return None, "❌ 프롬프트를 입력해주세요."

    try:
        model = get_model()

        # 시드 설정
        if seed == -1:
            import random
            seed = random.randint(0, 2**32 - 1)

        print(f"🎨 이미지 생성 중... (seed: {seed})")

        # 이미지 생성 (MLX)
        result = model.generate_image(
            seed=seed,
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
        )

        # GeneratedImage에서 PIL.Image 추출
        image = result.image
        gen_time = result.generation_time

        # 이미지 저장
        status = f"✅ 생성 완료! (seed: {seed}, {gen_time:.1f}초)"
        if save_image:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = OUTPUT_DIR / f"barunvision_{timestamp}_{seed}.png"
            image.save(filename)
            status += f"\n💾 저장됨: {filename}"

        return image, status

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ 오류 발생: {str(e)}"


def create_ui():
    """Gradio UI 생성."""

    with gr.Blocks() as app:

        gr.HTML("""
        <div class="title">
            <h1>🎨 BarunVision</h1>
            <p>Z-Image-Turbo (MLX) - Apple Silicon 최적화</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # 입력 섹션
                prompt = gr.Textbox(
                    label="프롬프트",
                    placeholder="생성하고 싶은 이미지를 설명해주세요...",
                    lines=4,
                    max_lines=8,
                )

                with gr.Row():
                    width = gr.Slider(
                        label="너비",
                        minimum=512,
                        maximum=1536,
                        value=512,
                        step=64,
                    )
                    height = gr.Slider(
                        label="높이",
                        minimum=512,
                        maximum=1536,
                        value=512,
                        step=64,
                    )

                with gr.Row():
                    num_steps = gr.Slider(
                        label="스텝 수",
                        minimum=2,
                        maximum=10,
                        value=4,
                        step=1,
                        info="4 권장 (Turbo 모델 최적화)",
                    )
                    seed = gr.Number(
                        label="시드",
                        value=-1,
                        precision=0,
                        info="-1 = 랜덤",
                    )

                save_image = gr.Checkbox(
                    label="이미지 자동 저장",
                    value=True,
                    info=f"저장 위치: {OUTPUT_DIR.absolute()}",
                )

                generate_btn = gr.Button(
                    "🎨 이미지 생성",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                # 출력 섹션
                output_image = gr.Image(
                    label="생성된 이미지",
                    type="pil",
                    height=512,
                )
                status = gr.Textbox(
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
            inputs=[prompt],
            label="예제 프롬프트",
        )

        # 이벤트 연결
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, width, height, num_steps, seed, save_image],
            outputs=[output_image, status],
        )

        # Enter 키로 생성
        prompt.submit(
            fn=generate_image,
            inputs=[prompt, width, height, num_steps, seed, save_image],
            outputs=[output_image, status],
        )

        gr.HTML("""
        <div class="footer">
            <p>Powered by <a href="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo" target="_blank">Z-Image-Turbo</a> + <a href="https://github.com/filipstrand/mflux" target="_blank">MFLUX</a> | BarunVision</p>
        </div>
        """)

    return app


if __name__ == "__main__":
    print("=" * 50)
    print("🎨 BarunVision 시작 (MLX 버전)")
    print("=" * 50)

    # 첫 요청 시 모델이 로드됨 (lazy loading)
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 공유 링크 생성하려면 True로 변경
        theme=gr.themes.Soft(),
        css="""
        .title { text-align: center; margin-bottom: 1rem; }
        .footer { text-align: center; margin-top: 1rem; opacity: 0.7; }
        """,
        head="<title>BarunVision</title>",
    )
