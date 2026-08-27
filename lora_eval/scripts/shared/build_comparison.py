#!/usr/bin/env python3
"""
build_comparison.py
Builds a cross-checkpoint comparison CSV and interactive HTML chart from
individual score CSVs. Called automatically by run_checkpoints.bat and
run_weights.bat.

Usage:
    python build_comparison.py --character k41t1yn --mode checkpoint
    python build_comparison.py --character k41t1yn --mode weight
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

# ── CONFIGURATION ─────────────────────────────────────────────────────────
ROOT_DIR          = r"G:\output\lora_eval\zit"
DEFAULT_CHARACTER = "k41t1yn"
DEFAULT_MODE      = "checkpoint"
# ──────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--character", default=DEFAULT_CHARACTER)
    p.add_argument("--mode",      default=DEFAULT_MODE,
                   choices=["checkpoint", "weight"])
    p.add_argument("--root",      default=ROOT_DIR)
    p.add_argument("--lora",      default=None,
                   help="LoRA filename for weight chart title (weight mode only)")
    p.add_argument("--checkpoint-step", default=None,
                   help="Checkpoint step for weight folder naming e.g. 5000")
    return p.parse_args()


def build(results_dir, prefix, sort_key):
    """
    Read all {prefix}_*_scores.csv files and build a comparison CSV.
    sort_key: function to extract sort value from filename stem
    """
    pattern = f"{prefix}_*_scores.csv"
    csvs    = sorted(results_dir.glob(pattern), key=lambda p: sort_key(p.stem))

    if not csvs:
        print(f"No CSVs found matching {pattern} in {results_dir}")
        sys.exit(1)

    rows = []
    for csv_path in csvs:
        df  = pd.read_csv(csv_path)
        key = csv_path.stem.replace(f"{prefix}_", "").replace("_scores", "")

        row = {
            "steps" if prefix == "checkpoint" else "strength": key,
            "seed_sets":   df["seed_set"].nunique() if "seed_set" in df.columns else 1,
            "image_count": len(df),
            "clip_lora_vs_dataset_mean":   round(df["clip_lora_vs_dataset"].mean(),   4),
            "clip_lora_vs_dataset_median": round(df["clip_lora_vs_dataset"].median(), 4),
            "clip_base_vs_dataset_mean":   round(df["clip_base_vs_dataset"].mean(),   4),
            "clip_delta_mean":             round(df["clip_delta"].mean(),             4),
            "clip_delta_positive_pct":     round((df["clip_delta"] > 0).mean() * 100, 1),
            "lpips_lora_vs_base_mean":     round(df["lpips_lora_vs_base"].mean(),     4),
            "lpips_lora_vs_base_median":   round(df["lpips_lora_vs_base"].median(),   4),
            "arcface_lora_vs_dataset_mean":   round(df["arcface_lora_vs_dataset"].dropna().mean(),   4) if "arcface_lora_vs_dataset" in df.columns else None,
            "arcface_lora_vs_dataset_median": round(df["arcface_lora_vs_dataset"].dropna().median(), 4) if "arcface_lora_vs_dataset" in df.columns else None,
            "arcface_face_detected_pct":      round(df["arcface_face_detected"].mean() * 100, 1)        if "arcface_face_detected" in df.columns else None,
        }
        rows.append(row)
        print(f"  [{key}] images={len(df)}"
              f"  clip={row['clip_lora_vs_dataset_mean']:.4f}"
              f"  delta={row['clip_delta_mean']:.4f}"
              f"  lpips={row['lpips_lora_vs_base_mean']:.4f}")

    out = pd.DataFrame(rows)
    out_path = results_dir / f"{prefix}_comparison.csv"
    out.to_csv(out_path, index=False)
    print(f"\nComparison saved -> {out_path}")
    return out


def build_chart(df, results_dir, prefix, character, lora_name=None):
    """
    Generate an interactive HTML chart from the comparison dataframe.
    Saved as {prefix}_comparison.html alongside the CSV.
    """
    is_checkpoint = (prefix == "checkpoint")
    x_key   = "steps" if is_checkpoint else "strength"
    x_label = "Training Steps" if is_checkpoint else "LoRA Strength"
    lora_suffix = f" for {lora_name}" if (lora_name and not is_checkpoint) else ""
    title   = (f"{character.capitalize()} — LoRA Checkpoint Evaluation"
               if is_checkpoint else
               f"{character.capitalize()} — LoRA Weight Evaluation{lora_suffix}")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    subtitle_detail = (f"{len(df)} checkpoints · {df[x_key].iloc[0]}–{df[x_key].iloc[-1]} steps"
                       if is_checkpoint else
                       f"{len(df)} strength values · {df[x_key].iloc[0]}–{df[x_key].iloc[-1]}")

    x_vals    = df[x_key].tolist()
    clip      = df["clip_lora_vs_dataset_mean"].tolist()
    delta     = df["clip_delta_mean"].tolist()
    pct       = df["clip_delta_positive_pct"].tolist()
    lpips     = df["lpips_lora_vs_base_mean"].tolist()
    baseline  = round(df["clip_base_vs_dataset_mean"].mean(), 4)
    has_arcface = "arcface_lora_vs_dataset_mean" in df.columns and df["arcface_lora_vs_dataset_mean"].notna().any()
    arcface   = df["arcface_lora_vs_dataset_mean"].tolist() if has_arcface else None
    img_count = int(df["image_count"].iloc[0]) if "image_count" in df.columns else "—"

    best_clip_idx   = df["clip_lora_vs_dataset_mean"].idxmax()
    best_delta_idx  = df["clip_delta_mean"].idxmax()
    best_pct_idx    = df["clip_delta_positive_pct"].idxmax()
    lowest_lpips_idx = df["lpips_lora_vs_base_mean"].idxmin()

    best_clip_x    = df[x_key].iloc[best_clip_idx]
    best_clip_v    = df["clip_lora_vs_dataset_mean"].iloc[best_clip_idx]
    best_delta_x   = df[x_key].iloc[best_delta_idx]
    best_delta_v   = df["clip_delta_mean"].iloc[best_delta_idx]
    best_pct_x     = df[x_key].iloc[best_pct_idx]
    best_pct_v     = df["clip_delta_positive_pct"].iloc[best_pct_idx]
    low_lpips_x    = df[x_key].iloc[lowest_lpips_idx]
    low_lpips_v    = df["lpips_lora_vs_base_mean"].iloc[lowest_lpips_idx]
    # ArcFace best
    best_arcface_x = best_arcface_v = None
    if has_arcface:
        af_clean = df["arcface_lora_vs_dataset_mean"].dropna()
        if not af_clean.empty:
            best_arcface_idx = af_clean.idxmax()
            best_arcface_x   = df[x_key].iloc[best_arcface_idx]
            best_arcface_v   = df["arcface_lora_vs_dataset_mean"].iloc[best_arcface_idx]

    early_neg = [row[x_key] for _, row in df.iterrows() if row["clip_delta_mean"] < 0]
    early_note = (f"Early negative delta at {early_neg[0]} {x_label.split()[1].lower()}"
                  if early_neg else "No negative delta checkpoints")

    subtitle_imgs = f"{img_count} images per {x_key}"

    arcface_hide = "" if has_arcface else ' style="display:none"'
    best_arcface_x_str = str(best_arcface_x) if best_arcface_x is not None else "n/a"
    best_arcface_v_str = f"{best_arcface_v:.4f}" if best_arcface_v is not None else "n/a"
    arcface_js = f"const arcface = {arcface};" if has_arcface else "const arcface = null;"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; background: #0f1117; color: #e0e0e0; margin: 0; padding: 20px; }}
  h1 {{ color: #a0c4ff; margin-bottom: 4px; font-size: 1.4rem; }}
  .subtitle {{ color: #888; font-size: 0.85rem; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .card {{ background: #1a1d27; border-radius: 10px; padding: 20px; border: 1px solid #2a2d3a; }}
  .card.wide {{ grid-column: 1 / -1; }}
  .card h2 {{ margin: 0 0 4px 0; font-size: 0.95rem; color: #c0c8e0; }}
  .card p {{ margin: 0 0 14px 0; font-size: 0.75rem; color: #666; }}
  .stats {{ display: flex; gap: 16px; margin-top: 20px; flex-wrap: wrap; }}
  .stat {{ background: #12141d; border-radius: 6px; padding: 10px 14px; border: 1px solid #2a2d3a; }}
  .stat .label {{ font-size: 0.7rem; color: #888; margin-bottom: 2px; }}
  .stat .value {{ font-size: 1.1rem; font-weight: 600; color: #a0c4ff; }}
  .stat .sub {{ font-size: 0.7rem; color: #666; }}
</style>
</head>
<body>

<h1>{title}</h1>
<p class="timestamp" style="color:#666; font-size:0.78rem; margin-bottom:2px;">Generated {generated_at}</p>
<p class="subtitle">{subtitle_detail} · {subtitle_imgs} · baseline CLIP: {baseline}</p>

<div class="grid">

  <div class="card wide">
    <h2>CLIP Score vs Dataset Mean</h2>
    <p>Higher = generated images are semantically closer to the training dataset. Orange line = baseline (no LoRA).</p>
    <canvas id="clipChart" height="80"></canvas>
  </div>

  <div class="card">
    <h2>CLIP Delta (LoRA − Baseline)</h2>
    <p>Positive = LoRA is pulling output toward training data. Negative = drifting away.</p>
    <canvas id="deltaChart" height="120"></canvas>
  </div>

  <div class="card">
    <h2>Positive Delta %</h2>
    <p>Percentage of image pairs where the LoRA improved CLIP similarity over baseline.</p>
    <canvas id="pctChart" height="120"></canvas>
  </div>

  <div class="card">
    <h2>LPIPS — Perceptual Distance from Baseline</h2>
    <p>How much the LoRA visually changes the image. Higher = more change.</p>
    <canvas id="lpipsChart" height="120"></canvas>
  </div>

  <div class="card{arcface_hide}">
    <h2>ArcFace — Face Identity vs Dataset</h2>
    <p>Face identity cosine similarity between generated images and dataset mean face embedding. Higher = stronger character identity match.</p>
    <canvas id="arcfaceChart" height="120"></canvas>
  </div>

  <div class="card wide">
    <h2>All Metrics Normalised</h2>
    <p>CLIP, delta, LPIPS, and ArcFace scaled 0–1 for visual comparison of trends.</p>
    <canvas id="normChart" height="80"></canvas>
  </div>

</div>

<div class="stats">
  <div class="stat">
    <div class="label">Best CLIP Score</div>
    <div class="value">{best_clip_x} {x_key}</div>
    <div class="sub">{best_clip_v:.4f}</div>
  </div>
  <div class="stat">
    <div class="label">Best CLIP Delta</div>
    <div class="value">{best_delta_x} {x_key}</div>
    <div class="sub">+{best_delta_v:.4f}</div>
  </div>
  <div class="stat">
    <div class="label">Best Delta %</div>
    <div class="value">{best_pct_x} {x_key}</div>
    <div class="sub">{best_pct_v:.1f}%</div>
  </div>
  <div class="stat">
    <div class="label">Lowest LPIPS</div>
    <div class="value">{low_lpips_x} {x_key}</div>
    <div class="sub">{low_lpips_v:.4f} (least visual change)</div>
  </div>
  <div class="stat{arcface_hide}">
    <div class="label">Best ArcFace</div>
    <div class="value">{best_arcface_x_str} {x_key}</div>
    <div class="sub">{best_arcface_v_str}</div>
  </div>
  <div class="stat">
    <div class="label">Baseline CLIP</div>
    <div class="value" style="color:#ff9944">{baseline}</div>
    <div class="sub">no LoRA reference</div>
  </div>
  <div class="stat">
    <div class="label">Note</div>
    <div class="value" style="color:#ff6666; font-size:0.85rem">{early_note}</div>
    <div class="sub">&nbsp;</div>
  </div>
</div>

<script>
const xVals = {json.dumps(x_vals)};
const clip  = {json.dumps(clip)};
const delta = {json.dumps(delta)};
const pct   = {json.dumps(pct)};
const lpips = {json.dumps(lpips)};
const baseline = {baseline};
const baselineData = xVals.map(() => baseline);

{arcface_js}
const cfg = {{
  responsive: true,
  interaction: {{ mode: 'index', intersect: false }},
  plugins: {{
    legend: {{ labels: {{ color: '#aaa', font: {{ size: 11 }} }} }},
    tooltip: {{ backgroundColor: '#1a1d27', titleColor: '#fff', bodyColor: '#ccc',
                borderColor: '#333', borderWidth: 1 }}
  }},
  scales: {{
    x: {{ ticks: {{ color: '#888', font: {{ size: 10 }} }}, grid: {{ color: '#1e2130' }} }},
    y: {{ ticks: {{ color: '#888', font: {{ size: 10 }} }}, grid: {{ color: '#1e2130' }} }}
  }}
}};

new Chart(document.getElementById('clipChart'), {{
  type: 'line',
  data: {{
    labels: xVals,
    datasets: [
      {{ label: 'CLIP vs Dataset', data: clip, borderColor: '#5b9cff',
         backgroundColor: 'rgba(91,156,255,0.08)', borderWidth: 2, pointRadius: 3, tension: 0.3 }},
      {{ label: 'Baseline (no LoRA)', data: baselineData, borderColor: '#ff9944',
         borderWidth: 1.5, borderDash: [6,4], pointRadius: 0 }}
    ]
  }},
  options: {{ ...cfg, scales: {{ ...cfg.scales,
    y: {{ ...cfg.scales.y, min: Math.min(...clip, baseline) - 0.01,
          max: Math.max(...clip, baseline) + 0.01 }} }} }}
}});

new Chart(document.getElementById('deltaChart'), {{
  type: 'bar',
  data: {{
    labels: xVals,
    datasets: [{{
      label: 'CLIP Delta',
      data: delta,
      backgroundColor: delta.map(v => v < 0 ? 'rgba(255,80,80,0.7)' : 'rgba(80,200,140,0.7)'),
      borderWidth: 0
    }}]
  }},
  options: {{ ...cfg }}
}});

new Chart(document.getElementById('pctChart'), {{
  type: 'line',
  data: {{
    labels: xVals,
    datasets: [{{
      label: 'Positive Delta %', data: pct, borderColor: '#c084fc',
      backgroundColor: 'rgba(192,132,252,0.08)', borderWidth: 2,
      pointRadius: 3, tension: 0.3, fill: true
    }}]
  }},
  options: {{ ...cfg, scales: {{ ...cfg.scales,
    y: {{ ...cfg.scales.y, min: Math.max(0, Math.min(...pct) - 5), max: 100 }} }} }}
}});

new Chart(document.getElementById('lpipsChart'), {{
  type: 'line',
  data: {{
    labels: xVals,
    datasets: [{{
      label: 'LPIPS (visual change)', data: lpips, borderColor: '#fbbf24',
      backgroundColor: 'rgba(251,191,36,0.08)', borderWidth: 2,
      pointRadius: 3, tension: 0.3, fill: true
    }}]
  }},
  options: {{ ...cfg, scales: {{ ...cfg.scales,
    y: {{ ...cfg.scales.y, min: Math.min(...lpips) - 0.02,
          max: Math.max(...lpips) + 0.02 }} }} }}
}});

function norm(arr) {{
  const clean = arr.filter(v => v !== null && !isNaN(v));
  const mn = Math.min(...clean), mx = Math.max(...clean);
  return arr.map(v => v === null ? null : +((v - mn) / (mx - mn)).toFixed(4));
}}
new Chart(document.getElementById('normChart'), {{
  type: 'line',
  data: {{
    labels: xVals,
    datasets: [
      {{ label: 'CLIP (norm)',    data: norm(clip),  borderColor: '#5b9cff', borderWidth: 2, pointRadius: 2, tension: 0.3 }},
      {{ label: 'Delta (norm)',   data: norm(delta), borderColor: '#80e0a0', borderWidth: 2, pointRadius: 2, tension: 0.3 }},
      {{ label: 'LPIPS (norm)',   data: norm(lpips), borderColor: '#fbbf24', borderWidth: 2, pointRadius: 2, tension: 0.3, borderDash: [4,3] }},
      ...(arcface !== null ? [{{ label: 'ArcFace (norm)', data: norm(arcface.map(v => v === null ? null : v)), borderColor: '#f472b6', borderWidth: 2, pointRadius: 2, tension: 0.3, borderDash: [2,2] }}] : [])
    ]
  }},
  options: {{ ...cfg, scales: {{ ...cfg.scales, y: {{ ...cfg.scales.y, min: 0, max: 1 }} }} }}
}});

if (arcface !== null) {{
  new Chart(document.getElementById('arcfaceChart'), {{
    type: 'line',
    data: {{
      labels: xVals,
      datasets: [{{
        label: 'ArcFace Identity', data: arcface, borderColor: '#f472b6',
        backgroundColor: 'rgba(244,114,182,0.08)', borderWidth: 2,
        pointRadius: 3, tension: 0.3, fill: true
      }}]
    }},
    options: {{ ...cfg, scales: {{ ...cfg.scales,
      y: {{ ...cfg.scales.y, min: Math.min(...arcface.filter(v=>v!==null)) - 0.02,
            max: Math.max(...arcface.filter(v=>v!==null)) + 0.02 }} }} }}
  }});
}}
</script>
</body>
</html>"""

    out_path = results_dir / f"{prefix}_comparison.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Chart saved      -> {out_path}")


def main():
    args        = parse_args()
    results_dir = Path(args.root) / args.character / "results"

    if not results_dir.exists():
        print(f"ERROR: Results folder not found: {results_dir}")
        sys.exit(1)

    print(f"Building {args.mode} comparison for {args.character}...")
    print(f"Results folder: {results_dir}\n")

    if args.mode == "checkpoint":
        # Sort numerically by step number
        def sort_key(stem):
            try: return int(stem.replace("checkpoint_", "").replace("_scores", ""))
            except: return 0
        df = build(results_dir, "checkpoint", sort_key)
        build_chart(df, results_dir, "checkpoint", args.character)

    elif args.mode == "weight":
        step = args.checkpoint_step or ""
        prefix = f"weight{step}"
        # Sort numerically by strength value
        def sort_key(stem):
            try:
                s = stem.replace(f"{prefix}_", "").replace("weight_", "").replace("_scores", "")
                return float(s)
            except: return 0.0
        df = build(results_dir, prefix, sort_key)
        build_chart(df, results_dir, prefix, args.character, lora_name=args.lora)


if __name__ == "__main__":
    main()
