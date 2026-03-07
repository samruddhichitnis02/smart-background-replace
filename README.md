# 🎨 Background Swap — AI-Powered Background Replacement

> Transform your photos with AI-powered inpainting. Professional results, simplified.

![Background Swap Demo](demo-banner.png)

---

## 🧠 Overview

**Background Swap** is an AI application that uses **Stable Diffusion inpainting** to seamlessly replace the background of any photo using a natural language text prompt. Upload a photo, describe your ideal background, and get a photorealistic result in seconds — no Photoshop skills required.

---

## ✨ Features

- 📸 Upload any photo and auto-detect the subject using segmentation
- ✍️ Describe a new background in plain text (e.g., *"cozy coffee shop interior, warm yellow lights, wooden tables, soft bokeh"*)
- 🖼️ Receive a **photorealistic, seamlessly composited** result via Stable Diffusion inpainting
- 🔍 Debugging & Intermediate Steps panel for inspecting the pipeline
- 🌐 Clean Gradio web interface — no coding required to use

---

## ⚙️ How It Works — The Stable Diffusion Pipeline

```
Input Photo
    │
    ▼
Subject Segmentation (Mask Generation)
    │
    ▼
Background Mask Created (inverted subject mask)
    │
    ▼
Stable Diffusion Inpainting Model
 ├─ Conditioned on: text prompt + original image + mask
 └─ Generates new background pixels within masked region
    │
    ▼
Post-processing & Blending
    │
    ▼
Final Professional Result
```

### 🔬 Model Details

| Component | Details |
|---|---|
| **Base Model** | Stable Diffusion v1.5 / v2 Inpainting |
| **Inpainting Pipeline** | `StableDiffusionInpaintPipeline` (HuggingFace Diffusers) |
| **Segmentation** | SAM / rembg / SegFormer for subject masking |
| **Scheduler** | DDIM / PNDM (configurable) |
| **Inference Steps** | 20–50 steps (quality vs. speed tradeoff) |
| **Guidance Scale** | 7.5 (default) |
| **UI Framework** | Gradio |

---

## 🖥️ GPU Requirements

> ⚠️ **A CUDA-compatible NVIDIA GPU is strongly recommended.** Stable Diffusion inpainting is computationally intensive and will be extremely slow (or fail) on CPU.

### Minimum Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| **GPU** | NVIDIA GTX 1060 6GB | NVIDIA RTX 3080 / A100 |
| **VRAM** | 6 GB | 10–16 GB+ |
| **RAM** | 16 GB | 32 GB |
| **CUDA Version** | 11.7+ | 12.x |
| **Storage** | 10 GB (model weights) | 20 GB+ |

### GPU Memory Optimization Tips

```python
# Enable attention slicing to reduce VRAM usage
pipe.enable_attention_slicing()

# Use half-precision (fp16) for ~50% VRAM reduction
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cuda")

# Enable xFormers for memory-efficient attention (optional)
pipe.enable_xformers_memory_efficient_attention()

# Offload to CPU if VRAM is limited
pipe.enable_sequential_cpu_offload()
```

### Running Without a GPU (Not Recommended)

CPU inference is supported but will take **5–30 minutes per image** depending on step count. Set:

```python
device = "cpu"
torch_dtype = torch.float32  # fp16 not supported on CPU
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/samruddhichitnis02/smart-background-replace.git
cd smart-background-replace
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install CUDA-enabled PyTorch

Visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the right command for your CUDA version. Example for CUDA 11.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Download Model Weights

Model weights are downloaded automatically on first run via HuggingFace Hub (~5–7 GB):

```
runwayml/stable-diffusion-inpainting
```

You can also pre-download manually:

```python
from diffusers import StableDiffusionInpaintPipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
```

---

## ▶️ Usage

### Launch the Gradio App

```bash
python app.py
```

Then open your browser at: `http://localhost:7860`

### Using the App

1. **Upload Photo** — drag and drop or click to select your image
2. **Describe your new background** — enter a detailed text prompt, e.g.:
   - `"cozy coffee shop interior, warm yellow lights, wooden tables, soft bokeh background, autumn afternoon atmosphere, photorealistic DSLR photo"`
   - `"outdoor stone garden table, natural earthy tones, soft dappled sunlight through trees, lush green garden background, shallow depth of field, photorealistic"`
3. Click **Generate** — watch the intermediate steps in the debug panel
4. **Download** your result using the download icon

### Prompt Tips for Best Results

- Include **lighting style** (e.g., *"soft bokeh", "golden hour", "studio lighting"*)
- Mention **photography style** (e.g., *"DSLR photo", "photorealistic", "shallow depth of field"*)
- Describe **atmosphere** for cohesive blending (e.g., *"warm tones", "natural light"*)
- Avoid conflicting descriptions that don't match the subject

---

## 📁 Project Structure

```
smart-background-replace/
├── app.py                  # Gradio interface & main entry point
├── pipeline.py             # Stable Diffusion inpainting pipeline
├── segmentation.py         # Subject mask generation
├── utils.py                # Image preprocessing & blending helpers
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 📦 Key Dependencies

```txt
torch>=2.0.0
diffusers>=0.21.0
transformers>=4.30.0
accelerate>=0.20.0
gradio>=3.40.0
Pillow>=9.0.0
numpy>=1.24.0
rembg                       # Background removal / segmentation
xformers                    # Optional: memory-efficient attention
```

---

## ☁️ Cloud GPU Options

If you don't have a local GPU, you can run this project on:

| Platform | Free Tier | Notes |
|---|---|---|
| **Google Colab** | T4 GPU (free) | Great for testing |
| **Kaggle Notebooks** | P100 GPU (free) | 30 hrs/week |
| **RunPod** | Pay-per-use | RTX 3090 / A100 available |
| **HuggingFace Spaces** | A10G (paid) | Deploy as a public Space |
| **AWS / GCP / Azure** | Pay-per-use | Production deployments |

---

## 🙏 Acknowledgements

- [Stability AI](https://stability.ai/) — Stable Diffusion
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers) — Inpainting pipeline
- [Gradio](https://gradio.app/) — Web UI framework
- [rembg](https://github.com/danielgatis/rembg) — Background segmentation

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">Made with ❤️ by <a href="https://github.com/samruddhichitnis02">Samruddhi Chitnis</a></p>
