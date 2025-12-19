# CS454 Project
## Overview

1. **Prompt Generation** (`prompt_generator/`): Generate n-wise CT prompt sets (2-wise, 3-wise, 4-wise)
2. **Image Generation** (`diffusion_runner/`): Generate images using diffusion models (SDXL, etc.)
3. **Automated Evaluation** (`evaluator/`): VQA and CLIP-based attribute verification
4. **Statistical Analysis** (`evaluator/`): Chi-square and KS tests for bias detection
5. **Human Annotation** (`annotations/`): Web-based annotation UI with IAA calculation
6. **Results & Visualization** (`results/`): Intersectional metrics and charts

---

## Quick Start

### 1. Environment Setup

- **Python**: 3.x
- **PyTorch**: Install separately based on your CUDA version (see [official guide](https://pytorch.org/get-started/locally/))

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2. Prompt Generation (Optional)

Pre-generated prompts are already included in `prompt_generator/prompt/`.

```bash
cd prompt_generator

# Generate 2-wise prompts
python ct_prompt_generator.py prompt_config/ct_config_2wise.jsonc prompt/ct_prompts_2wise.json

# Generate 3-wise prompts
python ct_prompt_generator.py prompt_config/ct_config_3wise.jsonc prompt/ct_prompts_3wise.json

# Generate 4-wise prompts
python ct_prompt_generator.py prompt_config/ct_config_4wise.jsonc prompt/ct_prompts_4wise.json
```

See `prompt_generator/README.md` for configuration details.

### 3. Image Generation

Generate images using diffusion models. Requires GPU with sufficient VRAM.

```bash
python diffusion_runner/run.py \
  --prompt_path ./prompt_generator/prompt/ \
  --prompt_filename ct_prompts_2wise.json \
  --model sdxl \
  --batch_size 10
```

**Available models**: `sdxl`, `sdxl-turbo`, `sdxl-lightning`, `pixart-sigma`

Output: Images are saved to `./outputs/` as `{case_id}_{index}.jpg`.

### 4. VQA Evaluation

Evaluate generated images using VQA models.

```bash
python evaluator/run_vqa_evaluation.py -n 2
python evaluator/run_vqa_evaluation.py -n 4
```

- **Input**: Images from `outputs_{n}wise/`, prompts from `prompt_generator/prompt/ct_prompts_{n}wise.json`
- **Output**: `evaluator/vqa_results/scores_{n}wise.json`

### 5. CLIP Evaluation (Optional)

```bash
python evaluator/run_clip_evaluation.py -n 2
```

- **Output**: `evaluator/clip_results/scores_{n}wise.json`

## Human Annotation (Optional)

### Running the Annotation Server

The annotation app uses the following paths by default:

- Prompts: `prompt_generator/prompt/ct_prompts_2wise.json`
- VQA Results: `evaluator/vqa_results/scores_2wise.json`
- Images: `outputs_2wise/`

```bash
bash annotations/run_annotation.sh
```

Server runs on `0.0.0.0:5003`. For remote access, use SSH tunneling:

```bash
ssh -L 5003:localhost:5003 user@remote-host
```

### Annotation Analysis (Optional)

```bash
# Annotation vs VQA accuracy (filtered attributes)
python annotations/calculate_filtered_accuracy.py

# Inter-Annotator Agreement (IAA)
python annotations/calculate_iaa.py
```

---

## Results & Analysis

### `results/` Directory

- **Intersectional Metrics**: `intersectional_metrics_*.{json,csv,png}`
- **Radar Charts**: `radar_charts/*.png`
- **Analysis Notebook**: `result_analyzer.ipynb`

The notebook processes VQA results and computes intersectional bias metrics (Flip, Gap).

---

## Directory Structure

```text
cs454_ct_diffusion_bias/
├── prompt_generator/           # CT prompt generation
│   ├── prompt_config/          # Configuration files (ct_config_{n}wise.jsonc)
│   ├── prompt/                 # Generated prompts (ct_prompts_{n}wise.json)
│   └── README.md
├── diffusion_runner/           # Image generation scripts
│   └── run.py
├── outputs/                    # Generated images (or outputs_{n}wise/)
├── evaluator/                  # Evaluation & statistical tests
│   ├── vqa_results/            # VQA evaluation outputs
│   ├── clip_results/           # CLIP evaluation outputs
│   ├── chi_results/            # Chi-square test results
│   ├── ks_results_*/           # KS test results
│   ├── run_vqa_evaluation.py
│   ├── run_clip_evaluation.py
│   ├── run_chi_test.py
│   ├── run_ks_test.py
│   └── README_*.md
├── annotations/                # Human annotation system
│   ├── annotation_app.py       # Flask web server
│   ├── calculate_iaa.py        # Inter-Annotator Agreement
│   ├── calculate_filtered_accuracy.py
│   └── annotations_*.json      # Annotation data
├── results/                    # Analysis outputs & visualizations
│   ├── result_analyzer.ipynb
│   ├── intersectional_metrics_*.{json,csv,png}
│   └── radar_charts/
├── requirements.txt
└── README.md
```