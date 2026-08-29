# LoRA Evaluation Pipeline — User Guide
*Krea 2 · Z-Image Turbo · Wan 2.2*

This guide covers the full pipeline for evaluating character LoRAs: from training
checkpoints through objective scoring, checkpoint selection, and weight optimization.
It assumes you are already familiar with ComfyUI and creating LoRA files with AI-Toolkit.

The pipeline replaces visual inspection of sample images with objective, reproducible
metrics that make it possible to compare checkpoints and settings systematically.

---

## Table of Contents

1. [How It Works](#1-how-it-works)
2. [Scoring Metrics](#2-scoring-metrics)
3. [One-Time Setup](#3-one-time-setup)
4. [Folder Structure](#4-folder-structure)
5. [File Naming Conventions](#5-file-naming-conventions)
6. [Script Configuration](#6-script-configuration)
7. [Krea 2 Pipeline](#7-krea-2-pipeline)
8. [Z-Image Turbo Pipeline](#8-z-image-turbo-pipeline)
9. [Wan 2.2 Pipeline](#9-wan-22-pipeline)
10. [Reading the Results](#10-reading-the-results)
11. [Key Findings](#11-key-findings)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. How It Works

The pipeline takes a set of LoRA checkpoint files from AI-Toolkit and scores each one
objectively, producing an interactive comparison chart that shows how each checkpoint
performs across three metrics. It then repeats the process across a range of LoRA
strengths to find the optimal production weight for the best checkpoint.

```
Train in AI-Toolkit → download checkpoint files
  → generate baseline images (no LoRA, generated once)
  → generate evaluation images (one set per checkpoint)
  → score all checkpoints → interactive comparison chart
  → identify best checkpoint
  → generate weight test images (one set per strength value)
  → score weight results → weight comparison chart
  → production-ready LoRA file with known optimal strength
```

### Scripts at a glance

| Script | Folder | Purpose |
|---|---|---|
| `checkpoint_queue_{model}.py` | `scripts\{model}\` | Generate baseline + checkpoint images via ComfyUI |
| `weight_queue_{model}.py` | `scripts\{model}\` | Generate baseline + weight test images for one checkpoint |
| `lora_evaluation_{model}.json` | `scripts\{model}\` | Workflow json file used by the queue scripts, do not edit |
| `prompts_{model}.txt` | `scripts\{model}\` | 30 prompts engineered to test the model |
| `prompts_{model}_short.txt` | `scripts\{model}\` | Condensed prompts list of the "critical eight" prompts |
| `lora_validator.py` | `scripts\shared\` | Score image pairs: CLIP, LPIPS, ArcFace |
| `build_comparison.py` | `scripts\shared\` | Build comparison CSV and HTML chart |
| `regenerate_summaries.py` | `scripts\shared\` | Rebuild summary text files from existing CSVs |
| `run_checkpoints.bat` | `scripts\shared\` | Batch runner — scores all checkpoint folders |
| `run_weights.bat` | `scripts\shared\` | Batch runner — scores all weight folders |
| `setenv.bat` | `scripts\shared\` | Python environment — sourced by all bat files |

Each model has its own queue scripts and ComfyUI workflow in `scripts\{model}\`.
The shared scripts work identically across all models.

### Prompt files

Each model folder includes a full prompt list (`prompts_{model}.txt`) and a shorter
"Critical Eight" list (`prompts_{model}_short.txt`). The prompts are specifically
engineered for each model's text encoder style and are designed to test the LoRA
across a range of angles, expressions, poses, and scene conditions that reveal how
well the character identity has been learned. Use the short list for large checkpoint
ranges to reduce generation time, and the full list for final evaluation of the top
candidates. Do not edit the prompts unless you have a specific reason — changing
them breaks comparability between runs.

---

## 2. Scoring Metrics

Three metrics are computed for each image, each measuring a different aspect of how
well the LoRA is working.

### CLIP — Semantic Similarity
Compares generated images to the training dataset using OpenCLIP ViT-B/32 embeddings.
Measures how semantically similar the generated output is to the training data overall —
including clothing, scene, lighting, and general character appearance.

Key values in the chart:
- **clip_lora_vs_dataset** — how close LoRA images are to the dataset (higher = better)
- **clip_base_vs_dataset** — how close baseline (no-LoRA) images are to the dataset
  (your reference floor — the LoRA should be above this)
- **clip_delta** — the difference between the two (positive = LoRA pulling toward dataset)
- **clip_delta_positive_pct** — what percentage of images improved over baseline

### LPIPS — Perceptual Distance
Measures how visually different the LoRA images are from the no-LoRA baseline, using
a VGG-based perceptual model. Higher scores mean the LoRA is making larger visual
changes. Useful for detecting overtraining — if LPIPS rises sharply while CLIP drops,
the model is changing images dramatically but in the wrong direction.

### ArcFace — Face Identity
Uses InsightFace buffalo_l to extract face embeddings and compare them against a mean
face embedding computed from the training dataset. Measures face identity specifically,
independent of clothing, scene, or lighting. ArcFace often peaks at different checkpoints
than CLIP — both metrics together give a more complete picture than either alone.

> **License note:** InsightFace buffalo_l models are licensed for non-commercial
> research use only. See https://github.com/deepinsight/insightface

---

## 3. One-Time Setup

### Python virtual environment

The pipeline scripts require a dedicated Python virtual environment separate from
ComfyUI's embedded Python. This keeps dependencies isolated and prevents conflicts.

```
python -m venv G:\output\lora_eval\scripts\venv
G:\output\lora_eval\scripts\venv\Scripts\pip install ^
    insightface onnxruntime open-clip-torch lpips ^
    torch torchvision pandas tqdm
```

Adjust the venv path to match your setup. The venv path needs to match `PYTHON_DIR`
in `setenv.bat` (see below).

### setenv.bat

Edit `PYTHON_DIR` in `setenv.bat` to point to your venv:

```bat
set PYTHON_DIR=G:\output\lora_eval\scripts\venv\Scripts
```

All bat files source `setenv.bat` automatically. This is the only path you need to
configure system-wide.

### InsightFace models

The ArcFace scoring module uses InsightFace buffalo_l. On the first run the buffalo_l
model pack (~300 MB) downloads automatically to `~/.insightface` and is cached there
for all subsequent runs. This is separate from the venv — the venv itself is around
1.2 GB once all dependencies are installed. ArcFace runs on CPU — no GPU needed.

### Enabling and disabling scoring modules

Open `lora_validator.py` and set the three switches at the top:

```python
SCORE_CLIP = True
SCORE_LPIPS = True
SCORE_ARCFACE = True
```

All three are enabled by default. Disabling any module removes it from scoring and
from the comparison chart. CLIP alone is sufficient for a fast sweep; add ArcFace
for a complete face identity analysis on the best candidates.

---

## 4. Folder Structure

```
models\loras\
  krea\
    {character}\
      {character}_krea_{step}.safetensors             ← Krea 2 and ZIT
  z-image\
    {character}\
      {character}_zit_{step}.safetensors
  wan\
    {character}\
      {character}_wan_{step}_high_noise.safetensors
      {character}_wan_{step}_low_noise.safetensors

output\lora_eval\
  docs\
    LoRA_Pipeline_Guide.md         ← this file
    aitoolkit_settings.md
  scripts\
    shared\
      lora_validator.py
      build_comparison.py
      regenerate_summaries.py
      run_checkpoints.bat
      run_weights.bat
      setenv.bat
    krea\
      checkpoint_queue_krea.py
      weight_queue_krea.py
      lora_evaluation_krea.json
      prompts_krea.txt
      prompts_krea_short.txt
    zit\
      checkpoint_queue_zit.py
      weight_queue_zit.py
      lora_evaluation_zit.json
      prompts_zit.txt
      prompts_zit_short.txt
    wan\
      checkpoint_queue_wan.py
      weight_queue_wan.py
      lora_evaluation_wan.json
      prompts_wan.txt
      prompts_wan_short.txt
  {model}\
    {character}\
      dataset\       ← training images (required for scoring)
      baseline\      ← no-LoRA reference images (generated once, reused)
      checkpoint\
        1000\        ← evaluation images for checkpoint at 1000 steps
        1100\
        ...
      weight{step}\
        0.60\        ← weight test images at strength 0.60
        0.65\
        ...
      results\
        checkpoint_comparison.csv
        checkpoint_comparison.html
        weight{step}_comparison.html
```

---

## 5. File Naming Conventions

LoRA files must follow these naming patterns for the scripts to find them:

| Model | Convention | Example |
|---|---|---|
| Krea 2 | `{character}_krea_{step}.safetensors` | `k41t1yn_krea_5000.safetensors` |
| Z-Image Turbo | `{character}_zit_{step}.safetensors` | `k41t1yn_zit_3500.safetensors` |
| Wan 2.2 | `{character}_wan_{step}_high_noise.safetensors` | `k41t1yn_wan_1900_high_noise.safetensors` |

AI-Toolkit produces files named `{TrainingName}_000005000.safetensors`. Remove the
leading zeros to match the expected format: `k41t1yn_krea_5000.safetensors`.

The Wan 2.2 pipeline always passes the `_high_noise` file as the checkpoint argument.
The `_low_noise` counterpart is loaded automatically.

Tip: Renaming many files at once is easy with Bulk Rename Utlity. Remove the leading zeros from all the 
safetensor files at the same time. Free to use, download at: https://www.bulkrenameutility.co.uk/Download.php 

---

## 6. Script Configuration

Each queue script has a configuration block near the top. Edit these before running:

```python
DEFAULT_CHARACTER = "k41t1yn"     # character name — must match folder and LoRA prefix
DEFAULT_TRIGGER2  = ""            # second trigger word if used, empty if not
CHECKPOINT_LORA_STRENGTH = 1.0   # LoRA strength for checkpoint evaluation images
```

The `run_checkpoints.bat` and `run_weights.bat` batch files have their own variables:

```bat
set MODEL=krea          :: krea, zit, or wan
set CHARACTER=k41t1yn
set CHECKPOINT_STEP=5000
```

---

## 7. Krea 2 Pipeline

Krea 2 is the recommended model for character still image work. It produces the
highest quality outputs and learns fine character details other models miss. 

### Evaluation workflow setup

The workflow `lora_evaluation_krea.json` will need to be edited to include the path to 
your models. Best option is a text editor like Notepap++. The second option is within 
ComfyIU; move the workflow to your workflows folder and set the models and save the file, 
then move it back to the scripts/krea folder. Note: There are nodes in the workflow 
that may not be connected to anything, the scripts use these nodes as is. Do not change 
anything other than the models.
The workflow requires these ComfyUI models:
- `diffusion_models\krea\` — krea2_turbo_fp8_scaled.safetensors
- `text_encoders\krea\` — qwen3vl_4b_fp8_scaled.safetensors
- `vae\qwen` — qwen_image_vae.safetensors

### Running the checkpoint evaluation

:: 1. Place LoRA files in models\loras\krea\{character}\
:: 2. Edit CONFIGURATION in checkpoint_queue_krea.py -  MODEL=krea, CHARACTER={character}
:: 3. Generate baseline and checkpoint images:
python scripts\krea\checkpoint_queue_krea.py - usage and arguments are in the docstring
:: 4. Score all checkpoints and build chart:
:: Edit run_checkpoints.bat: MODEL=krea, CHARACTER={character}
run_checkpoints.bat

The script auto-detects all checkpoint files and generates evaluation images for each.
Use `--dry-run` to verify paths before committing to a full run.

### Running the weight test

:: 1. Edit weight_queue_krea.py - MODEL=krea, CHARACTER={character}, 
DEFAULT_CHECKPOINT=checkpoint file name to be tested, Set Weight Range to be tested
:: 2. Generate baseline and checkpoint images:
python scripts\krea\weight_queue_krea.py - usage and arguments are in the docstring
:: 3. Score weight results and build chart:
:: Edit run_weights.bat: MODEL=krea, CHARACTER={character}, CHECKPOINT_STEP={step}
run_weights.bat

---

## 8. Z-Image Turbo Pipeline

Z-Image Turbo is the recommended model for fast local iteration. Training runs on
a 16GB card in a few hours. Image quality is lower than Krea 2 but the speed makes
it ideal for testing settings before committing to a full run.

### Evaluation workflow setup

The workflow `lora_evaluation_zit.json` will need to be edited to include the path to 
your models. Best option is a text editor like Notepad++. The second option is within 
ComfyIU; move the workflow to your workflows folder, set the models and save the file, 
then move it back to the scripts/zit folder. Note: There are nodes in the workflow 
that may not be connected to anything, the scripts use these nodes as is. Do not change 
anything other than the models.
The workflow requires these ComfyUI models:
- `diffusion_models\z-image\` — z_image_turbo_bf16.safetensors
- `text_encoders\krea\` — Qwen_3_4b.safetensors
- `vae\` — ae.safetensors

### Running the checkpoint evaluation

:: 1. Place LoRA files in models\loras\zit\{character}\
:: 2. Edit CONFIGURATION in checkpoint_queue_zit.py -  MODEL=zit, CHARACTER={character}
:: 3. Generate baseline and checkpoint images:
python scripts\zit\checkpoint_queue_zit.py - usage and arguments are in the docstring
:: 4. Score all checkpoints and build chart:
:: Edit run_checkpoints.bat: MODEL=zit, CHARACTER={character}
run_checkpoints.bat

The script auto-detects all checkpoint files and generates evaluation images for each.
Use `--dry-run` to verify paths before committing to a full run.

### Running the weight test

:: 1. Edit CONFIGURATION in weight_queue_zit.py - MODEL=zit, CHARACTER={character}, 
DEFAULT_CHECKPOINT=checkpoint file name to be tested
:: 2. Generate baseline and checkpoint images:
python scripts\zit\weight_queue_zit.py - usage and arguments are in the docstring
:: 3. Score weight results and build chart:
:: Edit run_weights.bat: MODEL=zit, CHARACTER={character}, CHECKPOINT_STEP={step}
run_weights.bat

---

## 9. Wan 2.2 Pipeline

Wan 2.2 is the recommended model for video generation. The pipeline generates single
still frames from short video clips for evaluation — the trained LoRA applies directly
to full video generation at inference.

Wan 2.2 trains paired `_high_noise` and `_low_noise` LoRA files — both are required.

### Evaluation workflow setup

The workflow `lora_evaluation_wan.json` will need to be edited to include the path to 
your models. Best option is a text editor like Notepad++. The second option is within 
ComfyIU; move the workflow to your workflows folder, set the models and save the file, 
then move it back to the scripts/wan folder. Note: There are nodes in the workflow 
that may not be connected to anything, the scripts use these nodes as is. Do not change 
anything other than the models.
The workflow requires these ComfyUI models:
- `diffusion_models\wan\` — wan2.2_t2v_high_noise_14b_fp8_scaled.safetensors and 
wan2.2_t2v_low_noise_14b_fp8_scaled.safetensors
- `loras\wan\` — wan2.2_t2v_lightx2v_4steps_lora_a1.1_high_noise.safetensors and 
wan2.2_t2v_lightx2v_4steps_lora_a1.1_low_noise.safetensors
- `text_encoders\qwen\` — umt_xxl_fp8_e4m3fn_scaled.safetensors
- `vae\wan\` — wan2.1_vae.safetensors

The workflow generates a 17-frame video clip per prompt, saves one frame, discards
the rest. This gives better image quality than generating a single frame directly.

### Running the checkpoint evaluation

:: 1. Place LoRA files in models\loras\wan\{character}\
:: 2. Edit CONFIGURATION in checkpoint_queue_wan.py — CHARACTER={character}
Note: Always pass the _high_noise file — _low_noise is loaded automatically
:: 3. Generate baseline and checkpoint images:
python scripts\wan\checkpoint_queue_wan.py - usage and arguments are in the docstring
:: 4. Score all checkpoints and build chart:
:: Edit run_checkpoints.bat: MODEL=wan, CHARACTER={character}
run_checkpoints.bat

The script auto-detects all checkpoint files and generates evaluation images for each.
Use `--dry-run` to verify paths before committing to a full run.

### Running the weight test

:: 1. Edit CONFIGURATION in weight_queue_wan.py - MODEL=wan, CHARACTER={character}, 
DEFAULT_CHECKPOINT=checkpoint file name to be tested
:: 2. Generate baseline and checkpoint images:
python scripts\wan\weight_queue_wan.py - usage and arguments are in the docstring
:: 3. Score weight results and build chart:
:: Edit run_weights.bat: MODEL=wan, CHARACTER={character}, CHECKPOINT_STEP={step}
run_weights.bat

---

## 10. Reading the Results

The comparison chart (`checkpoint_comparison.html`) shows five panels:

**CLIP Score** — absolute similarity to training dataset. Watch the trend shape more
than the absolute numbers — different characters and models produce different ranges.

**CLIP Delta** — improvement over the no-LoRA baseline. Positive = LoRA pulling toward
the training data. This is the primary metric for finding the best checkpoint.

**Positive Delta %** — percentage of evaluation images that improved over baseline.
100% means every single generated image is closer to the training data than without
the LoRA.

**LPIPS** — visual change from baseline. Rising LPIPS with rising CLIP delta = healthy
learning. Rising LPIPS with falling CLIP delta = overtraining.

**ArcFace** — face identity vs dataset. Often peaks at a different step than CLIP.
Use CLIP to find the best overall checkpoint and ArcFace to confirm face identity,
or choose between the two based on your use case.

**All Metrics Normalised** — all metrics scaled 0–1 for trend comparison. Use this
to see the overall shape of the learning curve at a glance.

### Identifying the best checkpoint

Look for the step where CLIP delta peaks and positive delta % is highest. The peak
is often followed by a visible decline — that's overtraining. If the curve is still
climbing at the end of the run, the checkpoint range needs extending.

### Identifying the optimal weight

The weight comparison chart shows scores at each strength value. Look for where CLIP
delta levels off or the curve starts flattening. Most Krea 2 character LoRAs work
well at 0.80–1.00. ZIT and Wan may require slightly different ranges.

---

## 11. Key Findings

These findings come from systematic testing across multiple characters and models.

**Dataset quality matters more than dataset size.** Removing 13 low-quality images
from a 53-image dataset produced a smoother learning curve, earlier stable plateau,
and lower baseline CLIP score — all positive outcomes.

**Trigger word uniqueness has a large measurable impact.** Testing the same character
with trigger word `c4ri` (collides with "car", "carl") vs `c4ri4nn` (unique) showed
the longer trigger reaching 0.60 CLIP at 500 steps vs 4100 steps with the short
trigger — the same endpoint but eight times faster. Use trigger words with number
substitutions that don't resemble common English words.

**CLIP and ArcFace peak at different checkpoints.** CLIP measures overall semantic
similarity and tends to peak earlier. ArcFace measures face identity specifically and
continues improving after CLIP peaks. Keep both metrics to choose the right checkpoint
for your use case.

**DOP (Differential Output Preservation) improves training efficiency** but does not
solve multi-character identity separation at inference. With DOP enabled, k41t1yn
Krea 2 v5 peaked at 5000 steps vs 11100 for v4 without DOP — roughly half the steps
needed for the same or better scores. However, combining two DOP-trained LoRAs still
produces identity blending in multi-character scenes.

**Krea 2 Raw learns fine details other models miss.** In one test the model learned
a nose ring from just two images in the dataset — it appeared in 80% of generated
outputs. Remove or edit accessories from dataset images that shouldn't be part of
the character identity.

**Each model has a characteristic overtraining signature.** ZIT produces visible
artifacts and distorted anatomy when overtrained. Krea 2 degrades gracefully —
scores decline but outputs remain visually coherent, making the evaluation pipeline
essential for identifying the true peak. Wan 2.2 shows a sharp cliff-edge drop.

---

## 12. Troubleshooting

**"No matching files" error when starting the queue**
The LoRA filename doesn't match the expected pattern. Check that filenames follow
`{character}_{model}_{step}.safetensors` and that the character name in the filename
matches `DEFAULT_CHARACTER` in the script exactly.

**HTTP 400 error from ComfyUI**
The workflow JSON is being rejected. Check the ComfyUI console for which node is
failing and what value it received. Common causes: wrong model filename in the
LoraLoader node, sampler name not matching ComfyUI's list, seed node receiving a
string instead of integer.

**ArcFace "no faces detected" on many images**
Prompts producing images without visible faces (hands-only shots, rear views,
distance shots) will log `arcface_face_detected = False`. This is expected — check
the prompt descriptions and verify that most prompts do produce a visible face.

**InsightFace not installed error**
Run the pip install command from the venv Scripts folder:
```
G:\output\lora_eval\scripts\venv\Scripts\pip install insightface onnxruntime
```

**Images generating but all going to wrong character folder**
The workflow JSON has the character name hardcoded in a path node. The queue script
patches this node at runtime — verify `NODE_CHAR_NAME` in the script points to the
correct node ID in the workflow, and that both the baseline and checkpoint patch
blocks include `NODE_CHAR_NAME`.

**Baseline images look like the character (LoRA not zeroed)**
The `patch_lora_strength` function sets LoRA strength to 0.0 for baseline images.
If the LoRA node has only 2 widgets (`LoraLoaderModelOnly`) instead of 3 (`LoraLoader`),
the strength may not be getting zeroed. Check that `patch_lora_strength` handles
both cases — see the function in the script.

---

