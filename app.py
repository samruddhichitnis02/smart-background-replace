# app.py
# Modernized Gradio UI for realistic background replacement using inpainting

import gradio as gr
from pipeline import load_inpaint_pipe, bg_replace_realistic

# Load the pipeline once at startup
pipe = load_inpaint_pipe()

def run_pipeline(image, prompt, steps, seed, feather, add_shadow, harmonize, harmonize_strength, guidance_scale):
    if image is None:
        return None, None, None
    return bg_replace_realistic(
        pipe=pipe,
        image=image,
        prompt=prompt,
        steps=steps,
        seed=seed,
        feather=feather,
        add_shadow=add_shadow,
        harmonize=harmonize,
        harmonize_strength=harmonize_strength,
        guidance_scale=guidance_scale,
    )

CSS = """
#container {
    max-width: 1200px;
    margin: auto;
    padding: 20px;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.header {
    text-align: center;
    margin-bottom: 30px;
}
.header h1 {
    font-size: 2.5em;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 10px;
}
.header p {
    font-size: 1.1em;
    color: #a0aec0;
}
.generate-btn {
    background: linear-gradient(135deg, #4299e1 0%, #667eea 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
}
.generate-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(66, 153, 225, 0.4);
}
.advanced-accordion {
    border: 1px solid #2d3748;
    border-radius: 8px;
    margin-top: 20px;
}
.output-group {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
"""

with gr.Blocks() as demo:
    with gr.Column(elem_id="container"):
        with gr.Column(elem_classes="header"):
            gr.Markdown("# Background Swap")
            gr.Markdown("Transform your photos with AI-powered inpainting. Professional results, simplified.")

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(type="pil", label="Upload Photo", height=400)
                prompt = gr.Textbox(
                    label="Describe your new background",
                    placeholder="e.g., a lush minimalist garden, soft evening daylight, cinematic lighting...",
                    lines=3
                )
                
                with gr.Accordion("Advanced Settings", open=False, elem_classes="advanced-accordion"):
                    with gr.Row():
                        steps = gr.Slider(15, 60, value=30, step=1, label="Inpaint Steps")
                        guidance_scale = gr.Slider(1.0, 20.0, value=9.0, step=0.5, label="Guidance Scale")
                    
                    with gr.Row():
                        seed = gr.Number(value=0, label="Seed (0=random)")
                        feather = gr.Slider(0, 30, value=10, step=1, label="Edge Smoothing")
                    
                    with gr.Row():
                        add_shadow = gr.Checkbox(value=True, label="Add Contact Shadow")
                        harmonize = gr.Checkbox(value=True, label="Color Harmonization")
                    
                    harmonize_strength = gr.Slider(0.0, 1.0, value=0.35, step=0.05, label="Harmonization Intensity")

                generate_btn = gr.Button("Generate New Background", elem_classes="generate-btn")

            with gr.Column(scale=1):
                final_output = gr.Image(type="pil", label="Final Professional Result", height=400)
                
                with gr.Accordion("Debugging & Intermediate Steps", open=False):
                    with gr.Row():
                        fg_cutout = gr.Image(type="pil", label="Foreground Segment")
                        mask_preview = gr.Image(type="pil", label="Background Mask")

        generate_btn.click(
            fn=run_pipeline,
            inputs=[input_img, prompt, steps, seed, feather, add_shadow, harmonize, harmonize_strength, guidance_scale],
            outputs=[fg_cutout, mask_preview, final_output]
        )

if __name__ == "__main__":
    demo.launch(
        share=True, 
        css=CSS, 
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
    )
