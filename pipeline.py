# pipeline.py
from __future__ import annotations

import gc
import os
import numpy as np
from PIL import Image, ImageFilter
from rembg import remove

import torch
from diffusers import StableDiffusionInpaintPipeline

INPAINT_MODEL_ID = "runwayml/stable-diffusion-inpainting"

# -----------------------------
# GPU cleanup
# -----------------------------
def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()

# -----------------------------
# Load pipeline
# -----------------------------
def load_inpaint_pipe():
    free_gpu()  # free GPU BEFORE loading, not before function is defined
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        INPAINT_MODEL_ID,
        torch_dtype=torch.float16,
        # no use_safetensors — this model uses .bin files
    )
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe

# -----------------------------
# Masking / Matting
# -----------------------------
def remove_bg_rgba(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    fg = remove(rgba)
    return fg.convert("RGBA")


def alpha_to_bg_mask(fg_rgba: Image.Image, feather: int = 10, threshold: int = 8) -> Image.Image:
    alpha = np.array(fg_rgba.split()[-1])

    subject = (alpha > 30).astype(np.uint8) * 255

    subject_img = Image.fromarray(subject, mode="L")
    subject_img = subject_img.filter(ImageFilter.MaxFilter(size=5))
    subject = np.array(subject_img)

    bg = 255 - subject
    mask = Image.fromarray(bg, mode="L")

    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))

    return mask


# -----------------------------
# Optional realism helpers
# -----------------------------
def add_contact_shadow(
    base_rgb: Image.Image,
    fg_rgba: Image.Image,
    offset=(0, 20),
    squash_y: float = 0.55,
    blur: int = 18,
    opacity: int = 110,
) -> Image.Image:
    base = base_rgb.convert("RGBA")
    fg = fg_rgba.convert("RGBA")

    alpha = fg.split()[-1]
    shadow = Image.new("RGBA", fg.size, (0, 0, 0, 0))
    shadow.putalpha(alpha)

    sh_a = np.array(shadow.split()[-1]).astype(np.float32)
    sh_a = (sh_a * (opacity / 255.0)).clip(0, 255).astype(np.uint8)
    shadow.putalpha(Image.fromarray(sh_a, mode="L"))

    w, h = shadow.size
    new_h = max(1, int(h * squash_y))
    shadow = shadow.resize((w, new_h), Image.BICUBIC)

    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas.paste(shadow, (offset[0], offset[1] + (h - new_h)), shadow)
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=blur))

    out = Image.alpha_composite(base, canvas)
    return out.convert("RGB")


def simple_color_harmonize(fg_rgba: Image.Image, target_rgb: Image.Image, strength: float = 0.35) -> Image.Image:
    strength = float(np.clip(strength, 0.0, 1.0))

    fg = fg_rgba.convert("RGBA")
    bg = target_rgb.convert("RGB").resize(fg.size, Image.LANCZOS)

    fg_np = np.array(fg).astype(np.float32)
    bg_np = np.array(bg).astype(np.float32)

    alpha = fg_np[..., 3:4] / 255.0
    if alpha.max() <= 0.0:
        return fg_rgba

    fg_rgb = fg_np[..., :3]
    fg_sum = (fg_rgb * alpha).sum(axis=(0, 1))
    fg_count = alpha.sum(axis=(0, 1)) + 1e-6
    fg_mean = fg_sum / fg_count

    bg_mean = bg_np.reshape(-1, 3).mean(axis=0)

    shift = (bg_mean - fg_mean) * strength
    fg_rgb_shifted = np.clip(fg_rgb + shift, 0, 255)

    out_np = fg_np.copy()
    out_np[..., :3] = fg_rgb_shifted
    return Image.fromarray(out_np.astype(np.uint8), mode="RGBA")


# -----------------------------
# Inpainting generation
# -----------------------------
def inpaint_background(
    pipe,
    original_rgb: Image.Image,
    mask_bg: Image.Image,
    prompt: str,
    steps: int = 20,   #reduced from 30 for speed
    seed: int = 0,
    guidance: float = 7.5,
) -> Image.Image:

    gen = None
    if seed and seed > 0:
        gen = torch.Generator(device="cuda").manual_seed(int(seed))

    full_prompt = (
        f"{prompt}, photorealistic DSLR photo, natural daylight, "
        f"Canon EOS R5, f/2.8 bokeh background, realistic shadows, high detail"
    )
    neg_prompt = (
        "cartoon, CGI, painting, illustration, oversaturated, plastic, "
        "watermark, text, blur on subject, unrealistic, low quality, artifacts"
    )

    original_size = original_rgb.size
    w, h = original_rgb.size

    # Resize to nearest multiple of 8
    new_w = (w // 8) * 8
    new_h = (h // 8) * 8
    img_resized = original_rgb.convert("RGB").resize((new_w, new_h), Image.LANCZOS)
    mask_resized = mask_bg.convert("L").resize((new_w, new_h), Image.NEAREST)

    out = pipe(
        prompt=full_prompt,
        negative_prompt=neg_prompt,
        image=img_resized,
        mask_image=mask_resized,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=gen,
        #removed height/width — not needed for SD 1.5, only SDXL
    ).images[0]

    return out.resize(original_size, Image.LANCZOS)


def bg_replace_realistic(
    pipe: StableDiffusionInpaintPipeline,
    image: Image.Image,
    prompt: str,
    steps: int = 20,
    seed: int = 0,
    feather: int = 10,
    add_shadow: bool = True,
    harmonize: bool = True,
    harmonize_strength: float = 0.35,
    guidance_scale: float = 7.5,
):
    if image is None:
        return None, None, None

    prompt = (prompt or "").strip()
    if not prompt:
        return None, None, None

    fg_rgba = remove_bg_rgba(image)
    mask_bg = alpha_to_bg_mask(fg_rgba, feather=feather)

    inpainted = inpaint_background(
        pipe=pipe,
        original_rgb=image,
        mask_bg=mask_bg,
        prompt=prompt,
        steps=steps,
        seed=seed,
        guidance=guidance_scale,
    )

    inpainted = inpainted.resize(image.size, Image.LANCZOS)

    if add_shadow:
        inpainted = add_contact_shadow(inpainted, fg_rgba)

    fg_for_comp = fg_rgba
    if harmonize:
        fg_for_comp = simple_color_harmonize(fg_rgba, inpainted, strength=harmonize_strength)

    bg_rgba = inpainted.convert("RGBA")
    bg_rgba.paste(fg_for_comp.convert("RGBA"), (0, 0), fg_for_comp.convert("RGBA"))
    final = bg_rgba.convert("RGB")

    return fg_rgba, mask_bg, final