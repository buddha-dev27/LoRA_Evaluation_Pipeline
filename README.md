# LoRA Evaluation Pipeline

The pipeline evaluates a set of LoRA checkpoint files from AI-Toolkit (or other sources) and scores each one objectively, producing an interactive comparison chart that shows how each checkpoint performs across three metrics. It then repeats the process across a range of LoRA strengths to find the optimal production weight for the best checkpoint.

## Supported Models

- **Krea 2** — recommended for character still image work
- **MiniMax H3** — recommended for character video work *(pipeline in testing)*
- **Wan 2.2** — video generation
- **Z-Image Turbo** — fast local iteration

Each model has its own ComfyUI evaluation workflow and a set of 30 prompts specifically engineered for that model's text encoder. A shorter Critical Eight prompt list is included for faster sweeps across large checkpoint ranges.

## How It Works

The pipeline generates three sets of images for each character:

1. **Baseline images** — generated once with the LoRA at zero strength. These represent what the model produces without any character influence and serve as the reference floor for all scoring.
2. **Checkpoint images** — one set per checkpoint, generated with the LoRA at full strength. The pipeline auto-detects all checkpoint files in the character's LoRA folder and queues them in order.
3. **Weight test images** — one set per strength value for the best checkpoint, used to identify the optimal production weight.

Each image is then scored against the training dataset using three metrics:

- **CLIP** — semantic similarity between generated images and the training dataset. Measures how well the overall character appearance, clothing, and scene context match the training data.
- **LPIPS** — perceptual distance between the LoRA image and the baseline. Measures how much visual change the LoRA is making. Rising LPIPS alongside rising CLIP delta indicates healthy learning; rising LPIPS with falling CLIP delta indicates overtraining.
- **ArcFace** — face identity similarity between generated images and the dataset's mean face embedding. Measures character face accuracy independently of clothing and scene. ArcFace often peaks at a different checkpoint than CLIP, providing a more complete picture of identity learning.

## Results

All scoring results are written to a standalone interactive HTML file that can be opened in any browser — no server or dependencies required. The chart includes:

- CLIP score vs baseline across all checkpoints
- CLIP delta bar chart (positive = improvement over baseline)
- Positive delta percentage across checkpoints
- LPIPS perceptual distance curve
- ArcFace face identity curve
- All metrics normalised to 0–1 for trend comparison
- Summary statistics highlighting best checkpoint for each metric

The weight test produces an equivalent chart across strength values rather than checkpoints.

## Requirements

- **ComfyUI** with the relevant model files installed
- **Python 3.10+** with a dedicated virtual environment
- **AI-Toolkit** for training (evaluation pipeline works with any LoRA source)
- Python packages: `insightface`, `onnxruntime`, `open-clip-torch`, `lpips`, `torch`, `torchvision`, `pandas`, `tqdm`

> **Note:** InsightFace buffalo_l models (~300 MB) download automatically to `~/.insightface` on first run. These models are licensed for non-commercial research use only.

## Documentation

Full setup instructions, folder structure, script configuration, and per-model workflow documentation are in `docs/LoRA_Eval_Pipeline_Guide.md`. AI-Toolkit training settings for each supported model are in `docs/aitoolkit_settings.md`.

## Key Findings

Developed through systematic testing across multiple characters and models:

- **Dataset quality matters more than dataset size** — curated, consistently sized, and carefully captioned images outperform larger but noisier datasets across all metrics
- **CLIP and ArcFace peak at different checkpoints** — using both metrics together identifies the true best checkpoint more reliably than either alone
- **LoRA weight is the primary production control** — character identity is encoded in the weight matrices and activates based on strength rather than trigger word alone
- **Overtraining has a distinct signature** — LPIPS continues rising while CLIP delta flattens or declines; the evaluation charts make this immediately visible
