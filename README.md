
# Scientific Image Synthesis: Benchmarking, Methodologies, and Downstream Utility

While synthetic data has proven effective for improving scientific reasoning in the text domain, multimodal reasoning remains constrained by the difficulty of synthesizing scientifically rigorous images. We conduct a systematic study of scientific image synthesis, analyzing both pixel-based generation and programmatic synthesis. We propose **ImgCoder**, a logic-driven framework that follows an explicit *"Understand → Plan → Code"* workflow to improve structural precision . To rigorously assess scientific correctness, we introduce **SciGenBench**, which evaluates generated images based on information utility and logical validity. Finally, we demonstrate that fine-tuning Large Multimodal Models (LMMs) on rigorously verified synthetic scientific images yields consistent reasoning gains. 

---

## 🌟 Key Contributions

* **ImgCoder Framework**: A logic-driven programmatic framework decoupling reasoning from rendering to achieve state-of-the-art structural precision.


* **SciGenBench**: A large-scale benchmark with **1.4K problems** across **5 domains** (Math, Physics, Chemistry, Biology, Universal) and **25 image types**, utilizing a hybrid evaluation protocol (LMM-as-Judge + Inverse Quiz Validation) .


* **Systematic Analysis**: We reveal a "Precision-Expressiveness" trade-off between pixel-based and code-based paradigms and categorize 5 systematic failure modes.


* **Downstream Utility**: We prove that high-fidelity synthetic data scales multimodal reasoning capabilities, with performance following a log-linear growth trend.

---

## 📊 SciGenBench: The Benchmark

SciGenBench evaluates scientific image generation on two core dimensions: **Information Utility** (via Inverse Quiz Validation) and **Logical Correctness** (via LMM-as-Judge).

### Taxonomy

The benchmark covers 5 major subjects and 25 fine-grained image types:

* 🧮 **Math**: Geometry (Plane/Solid), Analytic, Set & Probability.
* ⚛️ **Physics**: Mechanics, Fields, Optics, Circuits, Thermodynamics, etc.
* 🧪 **Chemistry**: Molecular Structures, Crystal Structures, Reaction Schemes.
* 🧬 **Biology**: Cell Diagrams, Genetics, Ecological, Molecular Processes.
* 📈 **Universal**: Plots, Charts, Graphs, Tables.

### Main Results (Leaderboard)

| Model Type | Model Name |  (%) | Judge Score (C&F) | Judge Score (L&P) |
| --- | --- | --- | --- | --- |
| **Closed Source** | **Nanobanana-Pro** | **73.41** | **1.59** | **1.87** |
|  | GPT-Image-1.5 | 63.52 | 0.98 | 1.70 |
| **ImgCoder** | **Gemini-3-Pro-ImgCoder** | **77.87** | **1.93** | **1.82** |
|  | Qwen3-ImgCoder | 56.38 | 1.21 | 1.30 |
| **Open Source** | Qwen-Image | 38.86 | 0.78 | 0.24 |
|  | HunyuanImage-3.0 | 30.79 | 0.39 | 0.70 |

> Note:  denotes Inverse Validation Rate. C&F = Correctness & Fidelity, L&P = Layout & Precision. See paper for full details.
> 
> 

---

## 🚀 ImgCoder: Logic-Driven Synthesis

Unlike pixel-based models that generate images end-to-end, **ImgCoder** adopts a programmatic paradigm:

1. **Understand**: Parses the scientific problem.
2. **Plan**: Explicitly plans image content, layout, labels, and drawing constraints .


3. **Code**: Generates executable code (Python/Matplotlib/TikZ) to render the diagram deterministically.

This approach eliminates hallucinations in structure-heavy tasks (e.g., coordinate systems, circuit topologies).

---

## 📈 Downstream Utility & Scaling

We explored whether synthetic data improves downstream LMM reasoning. The answer is **Yes**.

### Training Performance

We fine-tuned Qwen3-VL-8B using synthetic data from different sources.

* **Result:** Models trained on higher-quality data (e.g., `Nanobanana-Pro`, `Gemini-ImgCoder`) significantly outperform baselines.
* **Scaling:** Performance scales log-linearly with data size without saturation, similar to text-domain scaling laws.

(Figure: Training Reward Curves and Downstream Accuracy. Higher quality generators (Nano-Banana-Pro) yield higher rewards and test accuracy. )

### Evaluation on Benchmarks (GEO3K & MathVision)

| Model Variant | GEO3K | MV | Average |
| --- | --- | --- | --- |
| **Nanobanana-Pro** | **70.7** | 46.1 | **58.4** |
| Nanobanana-Pro (Filt) | 68.7 | **47.7** | 58.2 |
| Gemini-ImgCoder | 69.1 | 46.9 | 58.0 |
| Qwen-Image (Filt) | 68.6 | 47.0 | 57.8 |
| Qwen-Image | 68.2 | 45.9 | 57.1 |
| *Baseline* | *61.9* | *39.0* | *54.5* |

---

## 📂 Project Structure

```bash
.
├── data/                 # SciGenBench data and taxonomy
├── infer/                # Source code for image generation
├── evaluation/           # Evaluation scripts
└── training/             # VeRL-based training scripts

```

---

## 🛠️ Usage

### 1. Installation

```bash
git clone https://github.com/SciGenBench/SciGenBench.git
cd SciGenBench
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
# Set API Key
export OPENAI_API_KEY="your-api-key-here"

# Optional: Set font directory (for Chinese/special characters)
export FONT_DIR="/path/to/fonts"
```

### 3. Quick Start (Unified Runner)

We provide a unified runner script that supports both image generation and evaluation:

```bash
# List available models
python run.py --list-models --dataset scigen
python run.py --list-models --dataset seephys

# Generate images only
python run.py --dataset scigen --model gemini-3-pro-imgcoder --mode generate

# Evaluate only (LLM-as-Judge)
python run.py --dataset scigen --model gemini-3-pro-imgcoder --mode eval --metric judge

# Evaluate only (Inverse Quiz Validation)
python run.py --dataset scigen --model gemini-3-pro-imgcoder --mode eval --metric quiz

# Generate and evaluate all metrics
python run.py --dataset scigen --model gemini-3-pro-imgcoder --mode all

# With verbose output
python run.py --dataset scigen --model gemini-3-pro-imgcoder --mode all -v
```

### 4. Evaluation Metrics

#### 4.1 LLM-as-Judge

Evaluates generated images on 5 dimensions (0-2 scale):
- **Correctness & Fidelity**: Accuracy and completeness of elements
- **Layout & Precision**: Visual arrangement and technical precision
- **Readability & Occlusion**: Clarity and absence of overlapping elements
- **Scientific Plausibility**: Conformity to scientific principles
- **Expressiveness & Richness**: Contextual completeness

#### 4.2 Inverse Quiz Validation

Validates information completeness through reverse verification:
- Generates questions based on images
- Verifies if models can extract key information from images

### 5. Supported Models

#### SciGen Dataset
- `gemini-3-pro-imgcoder` - Gemini 3 Pro + ImgCoder
- `gemini-3-flash-imgcoder` - Gemini 3 Flash + ImgCoder
- `qwen3-imgcoder` - Qwen3 + ImgCoder
- `gpt-image1` - GPT Image 1
- `gpt-image1_5` - GPT Image 1.5
- `gpt-image1-mini` - GPT Image 1 Mini
- `nanobanana` - NanoBanana
- `nanobananapro` - NanoBanana Pro
- `qwen-image-plus` - Qwen Image Plus
- `hunyuan` - Hunyuan
- `flux2` - Flux 2
- `seedream` - SeeDream

#### SeePhys Dataset
- `gemini-3-pro-imgcoder`
- `gemini-3-flash-imgcoder`
- `qwen3-imgcoder`
- `gpt-image1`
- `gpt-image1_5`
- `nanobanana`
- `nanopro` - Nano Pro (SeePhys specific)
- `qwen-image-plus`
- `hunyuan`
- `flux2`
- `seedream`

### 6. Output Results

#### Image Output
- Location: `images/{dataset}/{model}/images/`
- Format: PNG or JPG
- ImgCoder Code: `images/{dataset}/{model}/codes/` (ImgCoder models only)

#### Evaluation Results

**LLM-as-Judge**
- Location: `results/{dataset}/llm_as_judge/{model}_quality_scores.csv`
- Contains: 5-dimensional scores (0-2 scale)

**Inverse Quiz Validation**
- Location: `results/{dataset}/quiz/{model}_detailed_evaluation.csv`
- Contains: Per-question correctness, overall accuracy, perfect image rate

The script automatically displays a results summary after running the full pipeline.

### 7. Advanced Usage

#### Skip Results Summary

```bash
python run.py --dataset scigen --model gemini-3-pro-imgcoder --mode all --skip-summary
```

#### Evaluate Only (No Generation)

```bash
python run.py --dataset scigen --model gemini-3-pro-imgcoder --mode eval --metric all
```

#### Batch Processing

```bash
# Example: Batch generation and evaluation
for model in gemini-3-pro-imgcoder qwen3-imgcoder nanobanana; do
    python run.py --dataset scigen --model $model --mode all
done
```

### 8. Manual Usage (Advanced)

#### Run ImgCoder Generation

```bash
cd src/infer/scigen
python gemini-3-pro-imgcoder.py
```

#### Run Evaluation

```bash
# LLM-as-Judge
cd src/eval
python llm_as_judge.py --dataset scigen

# Inverse Quiz Validation
python quiz.py  # for scigen
python quiz_seephys.py  # for seephys
```
---

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{anonymous2025scientific,
  title={Scientific Image Synthesis: Benchmarking, Methodologies, and Downstream Utility},
  author={Anonymous Authors},
  journal={Under Review},
  year={2025}
}

```
