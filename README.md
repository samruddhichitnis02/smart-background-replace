# 🎨 Background Swap — AI-Powered Background Replacement

> Transform your photos with AI-powered inpainting. Professional results, simplified.

## 🖼️ Demo

| Original | Result |
|---|---|
| ![Original Photo](demo/image__1_.png) | ![Background Swapped](demo/image__2_.png) |

---

## 🧠 Overview

**Background Swap** is an AI application that uses **Stable Diffusion Inpainting** to seamlessly replace the background of any photo using a natural language text prompt. Upload a photo, describe your ideal background, and get a photorealistic result in seconds — no Photoshop skills required.

Built with:
- 🔵 **Stable Diffusion Inpainting** — generative AI backbone
- 🟢 **rembg** — foreground subject segmentation
- 🟠 **Gradio** — clean, browser-based UI

---

## ⚙️ How It Works — The Stable Diffusion Pipeline

```
Input Photo
    │
    ▼
Subject Segmentation via rembg (Foreground Mask)
    │
    ▼
Background Mask Generated (inverted subject mask)
    │
    ▼
Stable Diffusion Inpainting Model
 ├─ Input: original image + background mask + text prompt
 └─ Generates new photorealistic background pixels
    │
    ▼
Post-processing & Compositing
    │
    ▼
Final Professional Result
```

### 🔬 Model Details

| Component | Details |
|---|---|
| **Base Model** | `runwayml/stable-diffusion-inpainting` |
| **Pipeline** | `StableDiffusionInpaintPipeline` (HuggingFace Diffusers) |
| **Segmentation** | `rembg` — foreground subject isolation |
| **Scheduler** | DDIM / PNDM |
| **Inference Steps** | 20–50 steps |
| **Guidance Scale** | 7.5 (default) |
| **UI Framework** | Gradio |

---

## 🖥️ GPU — Tested on Our Own Hardware

This project was **developed and tested entirely on a local GPU machine**. Below is the actual `nvidia-smi` output captured during a live inference run:

```
nvidia-smi  (Fri Mar 6 10:44:14 2026)
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 591.59       Driver Version: 591.59       CUDA Version: 13.1    |
+-------------------------------+----------------------+----------------------+
| GPU  Name            WDDM    |  Bus-Id              |  Memory-Usage        |
|  0   NVIDIA GeForce RTX 3060 |  00000000:01:00.0    |  3297MiB / 6144MiB  |
+-----------------------------------------------------------------------------+
| Processes:  C:\Python311\python.exe                                         |
+-----------------------------------------------------------------------------+
```

> 📸 *Real inference captured — ~3.3 GB of 6 GB VRAM actively used during Stable Diffusion inpainting*

### ✅ Verified Hardware Configuration

| Spec | Value |
|---|---|
| **GPU** | NVIDIA GeForce RTX 3060 |
| **VRAM Total** | 6 GB (6144 MiB) |
| **VRAM Used During Inference** | ~3.3 GB (3297 MiB) |
| **Driver Version** | 591.59 |
| **CUDA Version** | 13.1 |
| **OS** | Windows 11 |
| **Python** | 3.11 |

### 🔧 GPU Memory Optimizations for 6 GB VRAM

Since we ran this on an RTX 3060 (6 GB), the following optimizations keep inference within VRAM limits:

```python
# fp16 half-precision — cuts VRAM usage roughly in half
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cuda")

# Attention slicing — reduces peak VRAM at slight speed cost
pipe.enable_attention_slicing()

# Optional: xFormers for further memory efficiency
pipe.enable_xformers_memory_efficient_attention()
```

### Minimum vs. Recommended

| Resource | Minimum | This Project Used |
|---|---|---|
| **GPU** | NVIDIA GTX 1060 6GB | RTX 3060 6GB ✅ |
| **VRAM** | 6 GB | 6 GB ✅ |
| **CUDA** | 11.7+ | 13.1 ✅ |
| **Driver** | 450+ | 591.59 ✅ |
| **OS** | Windows / Linux | Windows 11 ✅ |

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

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> 💡 Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the exact command matching your CUDA version.

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify GPU is Detected

```python
import torch
print(torch.cuda.is_available())       # Should print: True
print(torch.cuda.get_device_name(0))   # Should print: NVIDIA GeForce RTX 3060
```

---

## ▶️ Usage

### Launch the Gradio App

```bash
python app.py
```

Open your browser at: `http://localhost:7860`

### Using the App

1. **Upload Photo** — drag and drop or click to select your image
2. **Describe your new background** — enter a detailed text prompt, e.g.:
   - `"cozy coffee shop interior, warm yellow lights, wooden tables, soft bokeh background, autumn afternoon atmosphere, photorealistic DSLR photo"`
   - `"outdoor stone garden table, natural earthy tones, soft dappled sunlight through trees, lush green garden background, shallow depth of field, photorealistic"`
3. Click **Generate** and inspect intermediate steps in the debug panel
4. **Download** your result using the download icon

### Prompt Tips for Best Results

- Include **lighting** — *"soft bokeh", "golden hour", "warm studio lighting"*
- Specify **photography style** — *"DSLR photo", "photorealistic", "shallow depth of field"*
- Describe **atmosphere** — *"autumn afternoon", "warm tones", "natural light"*

---

## 📁 Project Structure

```
smart-background-replace/
├── demo/                   # Demo images (before & after)
├── app.py                  # Gradio interface & entry point
├── pipeline.py             # Stable Diffusion inpainting pipeline
├── .gitignore
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
rembg
xformers          # optional but recommended for 6GB VRAM cards
```

---

## ☁️ No Local GPU? Cloud Options

| Platform | GPU | Cost |
|---|---|---|
| **Google Colab** | T4 (15 GB VRAM) | Free |
| **Kaggle Notebooks** | P100 (16 GB VRAM) | Free (30 hrs/week) |
| **HuggingFace Spaces** | A10G | Paid |
| **RunPod** | RTX 3090 / A100 | Pay-per-use |

---

## 🙏 Acknowledgements

- [Stability AI](https://stability.ai/) — Stable Diffusion
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers) — Inpainting pipeline
- [rembg](https://github.com/danielgatis/rembg) — Foreground segmentation
- [Gradio](https://gradio.app/) — Web UI

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">Made with ❤️ by <a href="https://github.com/samruddhichitnis02">Samruddhi Chitnis</a> & <a href="https://github.com/AnushkaKhadatkar">Anushka Khadatkar</a></p>
