"""
LoRA Validation Scorer
======================
Scores LoRA-generated images against:
  1. A dataset mean CLIP embedding  (semantic faithfulness)      [SCORE_CLIP]
  2. A baseline (no-LoRA) generated image via LPIPS              [SCORE_LPIPS]
  3. A dataset mean ArcFace face embedding (face identity)       [SCORE_ARCFACE]

Enable or disable each scoring module via the switches in the
CONFIGURATION block below.

ArcFace scoring requires the insightface package and the buffalo_l
model pack. InsightFace models are licensed for non-commercial
research use only. See https://github.com/deepinsight/insightface

Folder structure
-------------------------------------
  {character}/
    baseline/         no-LoRA images, generated once, shared by all tests
    checkpoint/
      1000/           LoRA images for checkpoint 1000 steps
      1250/
      ...
    weight/
      0.50/           LoRA images at strength 0.50
      0.55/
      ...
    dataset/          training images used to create the LoRA

Requirements
------------
    pip install open-clip-torch lpips Pillow torch torchvision pandas tqdm
    pip install insightface onnxruntime-gpu   # for SCORE_ARCFACE
"""

import warnings as _w
_w.filterwarnings("ignore", category=FutureWarning)
_w.filterwarnings("ignore", message=".*pretrained.*deprecated.*", category=UserWarning)
_w.filterwarnings("ignore", message=".*weights.*deprecated.*", category=UserWarning)

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — enable or disable scoring modules
# ══════════════════════════════════════════════════════════════════════════════

SCORE_CLIP     = True   # CLIP cosine similarity vs dataset mean
SCORE_LPIPS    = True   # LPIPS perceptual distance from baseline
SCORE_ARCFACE  = True   # ArcFace face identity cosine vs dataset mean faces
                        # Requires: pip install insightface onnxruntime-gpu
                        # Models: buffalo_l (auto-downloaded on first run ~300MB)
                        # License: non-commercial research use only
                        # https://github.com/deepinsight/insightface

# ══════════════════════════════════════════════════════════════════════════════


def _require(pkg, install_name=None):
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        name = install_name or pkg
        sys.exit(f"Missing package: {name}\n  Install with:  pip install {name}")


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SEED_IN_NAME_RE = re.compile(r'^(\d+)_(\d+)_')


def load_images_from_folder(folder: Path) -> list[Path]:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        sys.exit(f"No images found in: {folder}")
    return paths


def open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


# ── CLIP ──────────────────────────────────────────────────────────────────────

def build_clip(device: str):
    import warnings
    open_clip = _require("open_clip", "open-clip-torch")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
    model.eval().to(device)
    return model, preprocess


@torch.no_grad()
def embed_images_clip(paths: list[Path], model, preprocess, device: str,
                      desc="Embedding") -> torch.Tensor:
    vecs = []
    for p in tqdm(paths, desc=desc, unit="img"):
        img = preprocess(open_rgb(p)).unsqueeze(0).to(device)
        feat = model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        vecs.append(feat.cpu())
    return torch.cat(vecs, dim=0)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum())


# ── LPIPS ─────────────────────────────────────────────────────────────────────

def build_lpips(device: str):
    import warnings
    lpips_lib = _require("lpips")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        fn = lpips_lib.LPIPS(net="vgg").to(device)
    fn.eval()
    return fn


def preprocess_lpips(path: Path, device: str) -> torch.Tensor:
    import torchvision.transforms.functional as TF
    img = open_rgb(path).resize((256, 256), Image.LANCZOS)
    t = TF.to_tensor(img).unsqueeze(0).to(device)
    return t * 2.0 - 1.0


@torch.no_grad()
def lpips_score(fn, path_a: Path, path_b: Path, device: str) -> float:
    a = preprocess_lpips(path_a, device)
    b = preprocess_lpips(path_b, device)
    return float(fn(a, b).squeeze())


# ── ARCFACE ───────────────────────────────────────────────────────────────────

def build_arcface():
    """
    Load InsightFace FaceAnalysis with buffalo_l model pack.
    Models auto-download to ~/.insightface on first run (~300MB).
    License: non-commercial research use only.
    """
    try:
        import insightface
        from insightface.app import FaceAnalysis
    except ImportError:
        sys.exit(
            "InsightFace not installed.\n"
            "  pip install insightface onnxruntime\n"
            "Or disable ArcFace scoring: set SCORE_ARCFACE = False"
        )
    import warnings, io, contextlib, os
    # Suppress onnxruntime CUDA provider error messages
    os.environ["ORT_LOGGING_LEVEL"] = "3"  # ERROR only — suppresses CUDA DLL warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # Suppress InsightFace's verbose model loading output
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
            app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))
    return app


def get_face_embedding(app, path: Path) -> tuple[np.ndarray | None, bool]:
    """
    Detect largest face in image and return its ArcFace embedding.
    Returns (embedding, face_detected).
    embedding is None if no face found.
    """
    import cv2
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        return None, False
    faces = app.get(img_bgr)
    if not faces:
        return None, False
    # Use the largest face by bounding box area
    largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    emb = getattr(largest, "normed_embedding", None)
    if emb is None:
        raw = getattr(largest, "embedding", None)
        if raw is not None:
            norm = np.linalg.norm(raw)
            emb = raw / norm if norm > 1e-12 else raw
    return emb, emb is not None


def embed_dataset_faces(paths: list[Path], app) -> np.ndarray | None:
    """
    Compute mean ArcFace embedding across all dataset images with detectable faces.
    Returns normalized mean embedding or None if no faces detected.
    """
    embeddings = []
    no_face = 0
    for p in tqdm(paths, desc="  ArcFace dataset", unit="img"):
        emb, detected = get_face_embedding(app, p)
        if detected and emb is not None:
            embeddings.append(emb)
        else:
            no_face += 1
    if no_face:
        print(f"  [ArcFace] {no_face}/{len(paths)} dataset images had no detectable face")
    if not embeddings:
        print("  [ArcFace] WARNING: No faces detected in dataset — ArcFace scoring disabled")
        return None
    mean_emb = np.mean(np.stack(embeddings), axis=0)
    norm = np.linalg.norm(mean_emb)
    return mean_emb / norm if norm > 1e-12 else mean_emb


def arcface_cosine(emb: np.ndarray, reference: np.ndarray) -> float:
    return float(np.dot(emb, reference))


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_paired_folders(lora_dir, base_dir, dataset_paths, device,
                         clip_model=None, clip_prep=None, dataset_mean_clip=None,
                         lpips_fn=None,
                         arcface_app=None, dataset_mean_face=None):
    lora_paths = load_images_from_folder(Path(lora_dir))
    base_paths = load_images_from_folder(Path(base_dir))

    lora_by_stem = {p.stem: p for p in lora_paths}
    base_by_stem = {p.stem: p for p in base_paths}
    paired_stems = sorted(lora_by_stem.keys() & base_by_stem.keys())

    if not paired_stems:
        print(f"  [WARNING] No matched filenames in {lora_dir} vs {base_dir} — skipping")
        return []

    unmatched = (lora_by_stem.keys() | base_by_stem.keys()) - set(paired_stems)
    if unmatched:
        print(f"  [WARNING] {len(unmatched)} unmatched file(s) skipped")

    print(f"  Paired images: {len(paired_stems)}")

    ordered_lora = [lora_by_stem[s] for s in paired_stems]
    ordered_base = [base_by_stem[s] for s in paired_stems]

    # CLIP embeddings
    if SCORE_CLIP:
        lora_embs = embed_images_clip(ordered_lora, clip_model, clip_prep,
                                      device, desc="  CLIP lora ")
        base_embs = embed_images_clip(ordered_base, clip_model, clip_prep,
                                      device, desc="  CLIP base ")

    rows = []
    lpips_desc = "  LPIPS     " if SCORE_CLIP else "  Scoring   "
    iter_stems = tqdm(paired_stems, desc=lpips_desc, unit="pair") if SCORE_LPIPS else paired_stems

    for i, stem in enumerate(iter_stems if SCORE_LPIPS else tqdm(paired_stems,
                             desc="  Scoring   ", unit="pair")):
        lp = lora_by_stem[stem]
        bp = base_by_stem[stem]
        row = {"image": stem}

        # CLIP scores
        if SCORE_CLIP:
            clip_lora_vs_dataset = cosine_sim(lora_embs[i], dataset_mean_clip)
            clip_base_vs_dataset = cosine_sim(base_embs[i], dataset_mean_clip)
            clip_delta           = clip_lora_vs_dataset - clip_base_vs_dataset
            clip_lora_vs_base    = cosine_sim(lora_embs[i], base_embs[i])
            row.update({
                "clip_lora_vs_dataset": round(clip_lora_vs_dataset, 4),
                "clip_base_vs_dataset": round(clip_base_vs_dataset, 4),
                "clip_delta":           round(clip_delta,            4),
                "clip_lora_vs_base":    round(clip_lora_vs_base,    4),
            })

        # LPIPS score
        if SCORE_LPIPS:
            lp_score = lpips_score(lpips_fn, lp, bp, device)
            row["lpips_lora_vs_base"] = round(lp_score, 4)

        # ArcFace score
        if SCORE_ARCFACE and arcface_app is not None and dataset_mean_face is not None:
            lora_emb, lora_face_detected = get_face_embedding(arcface_app, lp)
            if lora_face_detected and lora_emb is not None:
                af_score = arcface_cosine(lora_emb, dataset_mean_face)
                row["arcface_lora_vs_dataset"] = round(af_score, 4)
                row["arcface_face_detected"]   = True
            else:
                row["arcface_lora_vs_dataset"] = None
                row["arcface_face_detected"]   = False

        row.update({"lora_path": str(lp), "base_path": str(bp)})
        rows.append(row)

    return rows


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, dataset_paths, summary_path: Path, checkpoint_label: str):
    sep = "─" * 60
    n = len(df)

    lines = [
        f"LoRA Validation Summary — {checkpoint_label}",
        sep,
        f"Images scored        : {n}",
        f"Dataset images       : {len(dataset_paths)}",
        f"Scoring modules      : "
        + ", ".join(m for m, on in [("CLIP", SCORE_CLIP),
                                     ("LPIPS", SCORE_LPIPS),
                                     ("ArcFace", SCORE_ARCFACE)] if on),
        "",
    ]

    if SCORE_CLIP and "clip_lora_vs_dataset" in df.columns:
        lines += [
            "── CLIP: LoRA vs dataset mean embedding ──────────────────",
            f"  Mean   : {df['clip_lora_vs_dataset'].mean():.4f}",
            f"  Median : {df['clip_lora_vs_dataset'].median():.4f}",
            f"  Min    : {df['clip_lora_vs_dataset'].min():.4f}  ({df.loc[df['clip_lora_vs_dataset'].idxmin(), 'image']})",
            f"  Max    : {df['clip_lora_vs_dataset'].max():.4f}  ({df.loc[df['clip_lora_vs_dataset'].idxmax(), 'image']})",
            "",
            "── CLIP delta (LoRA − baseline) — higher = LoRA doing more ──",
            f"  Mean   : {df['clip_delta'].mean():.4f}",
            f"  Median : {df['clip_delta'].median():.4f}",
            f"  Positive delta: {(df['clip_delta'] > 0).sum()} / {n}  ({(df['clip_delta'] > 0).mean()*100:.1f}%)",
            "",
        ]

    if SCORE_LPIPS and "lpips_lora_vs_base" in df.columns:
        lines += [
            "── LPIPS LoRA vs baseline — higher = more visual change ──",
            f"  Mean   : {df['lpips_lora_vs_base'].mean():.4f}",
            f"  Median : {df['lpips_lora_vs_base'].median():.4f}",
            "",
        ]

    if SCORE_ARCFACE and "arcface_lora_vs_dataset" in df.columns:
        af_scored = df["arcface_lora_vs_dataset"].dropna()
        no_face   = int(df["arcface_face_detected"].eq(False).sum()) if "arcface_face_detected" in df.columns else 0
        lines += [
            "── ArcFace: face identity vs dataset mean ─────────────────",
            f"  Faces detected : {len(af_scored)} / {n}  ({no_face} images had no detectable face)",
            f"  Mean   : {af_scored.mean():.4f}" if len(af_scored) else "  Mean   : n/a",
            f"  Median : {af_scored.median():.4f}" if len(af_scored) else "  Median : n/a",
            f"  Min    : {af_scored.min():.4f}" if len(af_scored) else "  Min    : n/a",
            f"  Max    : {af_scored.max():.4f}" if len(af_scored) else "  Max    : n/a",
            "",
        ]

    lines += [
        sep,
        "Score guide",
        sep,
    ]
    if SCORE_CLIP:
        lines += [
            "clip_lora_vs_dataset : cosine similarity to dataset mean (higher = closer to training data)",
            "clip_base_vs_dataset : same for baseline — your reference floor",
            "clip_delta           : positive = LoRA pulling toward training data (good)",
            "                       near zero = LoRA has little effect",
            "                       negative  = LoRA drifting away (check training)",
        ]
    if SCORE_LPIPS:
        lines.append("lpips_lora_vs_base   : perceptual distance from baseline (higher = more visual change)")
    if SCORE_ARCFACE:
        lines += [
            "arcface_lora_vs_dataset : ArcFace face identity cosine vs dataset mean face",
            "                          higher = generated face is more similar to the character",
            "                          None = no face detected in that image",
        ]

    # Top/bottom 5 by primary metric
    primary = None
    if SCORE_CLIP and "clip_lora_vs_dataset" in df.columns:
        primary = "clip_lora_vs_dataset"
    elif SCORE_ARCFACE and "arcface_lora_vs_dataset" in df.columns:
        primary = "arcface_lora_vs_dataset"

    if primary:
        cols = ["image"]
        if SCORE_CLIP and "clip_lora_vs_dataset" in df.columns:
            cols += ["clip_lora_vs_dataset", "clip_delta"]
        if SCORE_LPIPS and "lpips_lora_vs_base" in df.columns:
            cols.append("lpips_lora_vs_base")
        if SCORE_ARCFACE and "arcface_lora_vs_dataset" in df.columns:
            cols.append("arcface_lora_vs_dataset")

        df_sorted = df.dropna(subset=[primary])
        lines += ["", f"Top 5 images by {primary}:"]
        lines.append(df_sorted.nlargest(5, primary)[cols].to_string(index=False))
        lines += ["", f"Bottom 5 images by {primary}:"]
        lines.append(df_sorted.nsmallest(5, primary)[cols].to_string(index=False))

    text = "\n".join(lines)
    print("\n" + text)
    summary_path.write_text(text, encoding="utf-8")
    print(f"\nSummary saved → {summary_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def score(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Scoring: "
          + ", ".join(m for m, on in [("CLIP", SCORE_CLIP),
                                       ("LPIPS", SCORE_LPIPS),
                                       ("ArcFace", SCORE_ARCFACE)] if on))

    dataset_dir   = Path(args.dataset)
    dataset_paths = load_images_from_folder(dataset_dir)
    print(f"Dataset images: {len(dataset_paths)}")

    # Load enabled modules
    clip_model = clip_prep = dataset_mean_clip = None
    lpips_fn = None
    arcface_app = dataset_mean_face = None

    if SCORE_CLIP:
        print("\nLoading CLIP model (ViT-B/32)...")
        clip_model, clip_prep = build_clip(device)
        print("Embedding dataset images (CLIP)...")
        dataset_embs  = embed_images_clip(dataset_paths, clip_model, clip_prep,
                                          device, desc="  dataset")
        dataset_mean_clip = dataset_embs.mean(dim=0)
        dataset_mean_clip = dataset_mean_clip / dataset_mean_clip.norm()

    if SCORE_LPIPS:
        print("\nLoading LPIPS model (VGG)...")
        lpips_fn = build_lpips(device)

    if SCORE_ARCFACE:
        import pathlib
        buffalo_path = pathlib.Path.home() / ".insightface" / "models" / "buffalo_l"
        first_run = not buffalo_path.exists()
        print("\nLoading ArcFace model (buffalo_l)...")
        if first_run:
            print("  Downloading buffalo_l models (~300MB) to ~/.insightface...")
            print("  License: non-commercial research use only — insightface.ai")
        arcface_app = build_arcface()
        print("Embedding dataset images (ArcFace)...")
        dataset_mean_face = embed_dataset_faces(dataset_paths, arcface_app)
        if dataset_mean_face is None:
            print("  ArcFace disabled — no faces found in dataset")

    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_label = out_csv.stem.replace("_scores", "")

    all_rows = []

    if args.baseline:
        baseline_dir = Path(args.baseline)
    elif args.base:
        baseline_dir = Path(args.base)
    else:
        sys.exit("Provide --baseline or --base")

    lora_dir = Path(args.lora)

    sub_dirs = sorted(d for d in lora_dir.iterdir()
                      if d.is_dir() and (d / "lora").exists()) if lora_dir.is_dir() else []

    if sub_dirs:
        print(f"\nDetected legacy layout B ({len(sub_dirs)} seed set subfolders)")
        for sd in sub_dirs:
            bl = sd / "base" if (sd / "base").exists() else baseline_dir
            print(f"\n  Seed set: {sd.name}")
            rows = score_paired_folders(
                sd / "lora", bl, dataset_paths, device,
                clip_model, clip_prep, dataset_mean_clip,
                lpips_fn,
                arcface_app, dataset_mean_face,
            )
            all_rows.extend(rows)
    else:
        print(f"\nLoRA folder   : {lora_dir}")
        print(f"Baseline folder: {baseline_dir}")
        rows = score_paired_folders(
            lora_dir, baseline_dir, dataset_paths, device,
            clip_model, clip_prep, dataset_mean_clip,
            lpips_fn,
            arcface_app, dataset_mean_face,
        )
        all_rows.extend(rows)

    if not all_rows:
        sys.exit("No images were scored. Check your folder paths.")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print(f"\nScores saved → {out_csv}")

    summary_path = out_csv.with_name(out_csv.stem + "_summary.txt")
    print_summary(df, dataset_paths, summary_path, checkpoint_label)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score LoRA checkpoint images against training dataset and baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python lora_validator.py ^
      --dataset  output\\lora_eval\\zit\\k41t1yn\\dataset ^
      --baseline output\\lora_eval\\zit\\k41t1yn\\baseline ^
      --lora     output\\lora_eval\\zit\\k41t1yn\\checkpoint\\3500 ^
      --output   output\\lora_eval\\zit\\k41t1yn\\results\\checkpoint_3500_scores.csv
        """
    )
    parser.add_argument("--dataset",    required=True)
    parser.add_argument("--baseline",   default=None)
    parser.add_argument("--lora",       required=True)
    parser.add_argument("--base",       default=None,
                        help="Legacy: single base folder. Use --baseline instead.")
    parser.add_argument("--output",     default="scores.csv")
    args = parser.parse_args()

    if args.baseline is None and args.base is None:
        parser.error("Provide --baseline or --base")

    score(args)


if __name__ == "__main__":
    main()
