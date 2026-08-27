"""
regenerate_summaries.py
Regenerates all summary txt files from existing score CSVs.

Works for both checkpoint and weight test results.

Usage:
    python regenerate_summaries.py --character k41t1yn
    python regenerate_summaries.py --character k41t1yn --mode checkpoint
    python regenerate_summaries.py --character k41t1yn --mode weight
    python regenerate_summaries.py --character k41t1yn --mode both
"""

import argparse
from pathlib import Path
import pandas as pd

# ── CONFIGURATION ─────────────────────────────────────────────────────────
ROOT_DIR          = r"G:\output\lora_eval\zit"
DEFAULT_CHARACTER = "k41t1yn"
DEFAULT_MODE      = "both"   # checkpoint, weight, or both
# ──────────────────────────────────────────────────────────────────────────

SEP = "-" * 60


def make_summary(df, label):
    """Build summary text lines from a scores dataframe."""
    n = len(df)

    lines = [
        f"LoRA Validation Summary -- {label}",
        SEP,
        f"Images scored        : {n}",
        "",
        "CLIP: LoRA vs dataset mean embedding",
        f"  Mean   : {df['clip_lora_vs_dataset'].mean():.4f}",
        f"  Median : {df['clip_lora_vs_dataset'].median():.4f}",
        f"  Min    : {df['clip_lora_vs_dataset'].min():.4f}"
        f"  ({df.loc[df['clip_lora_vs_dataset'].idxmin(), 'image']})",
        f"  Max    : {df['clip_lora_vs_dataset'].max():.4f}"
        f"  ({df.loc[df['clip_lora_vs_dataset'].idxmax(), 'image']})",
        "",
        "CLIP delta (LoRA - baseline)",
        f"  Mean   : {df['clip_delta'].mean():.4f}",
        f"  Median : {df['clip_delta'].median():.4f}",
        f"  Positive delta: {(df['clip_delta'] > 0).sum()} / {n}"
        f"  ({(df['clip_delta'] > 0).mean()*100:.1f}%)",
        "",
        "LPIPS LoRA vs baseline",
        f"  Mean   : {df['lpips_lora_vs_base'].mean():.4f}",
        f"  Median : {df['lpips_lora_vs_base'].median():.4f}",
        "", SEP, "Score guide", SEP,
        "clip_lora_vs_dataset : cosine similarity to dataset mean (higher = better)",
        "clip_base_vs_dataset : baseline reference floor (no LoRA)",
        "clip_delta           : positive = LoRA pulling toward training data",
        "lpips_lora_vs_base   : perceptual distance from baseline",
        "",
        "Top 5 by clip_lora_vs_dataset:",
    ]
    cols = ["image", "clip_lora_vs_dataset", "clip_delta", "lpips_lora_vs_base"]
    lines.append(df.nlargest(5, "clip_lora_vs_dataset")[cols].to_string(index=False))
    lines += ["", "Bottom 5 by clip_lora_vs_dataset:"]
    lines.append(df.nsmallest(5, "clip_lora_vs_dataset")[cols].to_string(index=False))

    return "\n".join(lines)


def regen_csvs(results_dir, prefix, label_fn):
    """
    Regenerate summaries for all CSVs matching prefix_*_scores.csv.
    label_fn(stem) returns the human-readable label for the summary header.
    """
    pattern = f"{prefix}_*_scores.csv"
    csvs    = sorted(results_dir.glob(pattern))
    if not csvs:
        print(f"  No CSVs found matching {pattern} in {results_dir}")
        return

    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        # Extract the key part e.g. "3500" from "checkpoint_3500_scores.csv"
        key   = csv_path.stem.replace(f"{prefix}_", "").replace("_scores", "")
        label = label_fn(key)
        text  = make_summary(df, label)
        out   = csv_path.with_name(csv_path.stem + "_summary.txt")
        out.write_text(text, encoding="utf-8")
        print(f"  [DONE] {out.name}")


def main():
    parser = argparse.ArgumentParser(description="Regenerate summary txt files from score CSVs")
    parser.add_argument("--character", default=DEFAULT_CHARACTER)
    parser.add_argument("--mode",      default=DEFAULT_MODE,
                        choices=["checkpoint", "weight", "both"],
                        help="Which results to regenerate (default: both)")
    parser.add_argument("--root",      default=ROOT_DIR,
                        help="Root output folder")
    args = parser.parse_args()

    results_dir = Path(args.root) / args.character / "results"
    if not results_dir.exists():
        print(f"ERROR: Results folder not found: {results_dir}")
        return

    print(f"Regenerating summaries for: {args.character}")
    print(f"Results folder: {results_dir}")
    print(f"Mode: {args.mode}")
    print()

    if args.mode in ("checkpoint", "both"):
        print("Checkpoint summaries:")
        regen_csvs(results_dir, "checkpoint",
                   lambda key: f"{key} steps")

    if args.mode in ("weight", "both"):
        print("Weight summaries:")
        regen_csvs(results_dir, "weight",
                   lambda key: f"strength {key}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
