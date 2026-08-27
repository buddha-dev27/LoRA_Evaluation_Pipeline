#!/usr/bin/env python3
"""
weight_queue_krea.py
LoRA Weight Testing — Image Generation Script

Tests a single LoRA checkpoint at multiple strength values to find the
optimal weight for production use. For each strength value, generates
LoRA images using the same prompts and seeds as the checkpoint evaluation.
Baseline images are reused from the existing baseline folder — they are
not regenerated.

Edit the variables below before running this script. For more information
see README_LoRA_Evaluation_Pipeline.md in the lora_eval/docs folder.

For each strength value:
  For each prompt in prompts_krea.txt:
    For each seed multiplier (default: 1000, 2000, 3000):
      Queue one LoRA image at the specified strength
      Save to: lora_eval/krea/{character}/weight/{strength}/

The baseline folder (lora_eval/krea/{character}/baseline/) must already exist.
Run checkpoint_queue_krea.py --baseline-only first if baseline images are missing.

REQUIREMENTS
------------
  - Python 3.x (no external packages required)
  - ComfyUI running and accessible at COMFY_HOST:COMFY_PORT
  - lora_evaluation_krea.json  (workflow, in same folder as this script)
  - prompts_krea.txt                   (one prompt per line, # = comment)
  - Baseline images already generated in lora_eval/krea/{character}/baseline/
    Run: python checkpoint_queue_krea.py --character {character} --baseline-only
  - LoRA checkpoint file: {character}_krea_{step}.safetensors

OUTPUT STRUCTURE
----------------
  ComfyUI output /
    lora_eval /
      zit /
        {character} /
          baseline /    (existing, reused — not modified)
          weight /
            0.50 /      01_1000_00001_.png ... 30_3000_00001_.png  (90 files)
            0.55 /
            0.60 /
            ...
          results /
            weight_0.50_scores.csv
            weight_0.50_scores_summary.txt
            weight_comparison.csv

CONFIGURATION (edit top of script)
-----------------------------------
  COMFY_HOST          ComfyUI host                    default: 127.0.0.1
  COMFY_PORT          ComfyUI port                    default: 8188
  WORKFLOW_FILE       Workflow JSON filename
  PROMPTS_FILE        Prompts text file
  LORAS_ROOT          Full local path to loras folder
  LORA_SUBROOT        Subfolder within loras root     e.g. z-image
  LORA_MODEL_TAG      Short model name in filenames   e.g. zit
  SEED_MULTIPLIERS    List of seed multipliers        default: [1000, 2000, 3000]
  OUTPUT_ROOT         ComfyUI-relative output root    e.g. lora_eval/zit
  DEFAULT_CHARACTER   Default character name
  DEFAULT_TRIGGER2    Second trigger word, empty if not used
  DEFAULT_CHECKPOINT  Default checkpoint filename (without path)
  LOG_ROOT            Full local path for log output

USAGE
-----
  # Dry run — preview all planned images without queuing anything (always do this first)
  python weight_queue_krea.py --dry-run

  # Full sweep — uses DEFAULT_CHARACTER and DEFAULT_CHECKPOINT set in variables
  python weight_queue_krea.py

  # Specify a different checkpoint
  python weight_queue_krea.py --checkpoint k41t1yn_krea_5000.safetensors

  # Different character — overrides DEFAULT_CHARACTER
  python weight_queue_krea.py --character k41t1yn

  # Custom strength range
  python weight_queue_krea.py --start 0.7 --end 0.9 --step 0.05

  # Second trigger word
  python weight_queue_krea.py --trigger2 redhead

  # Resume after interruption — N is the last completed image number from the log
  python weight_queue_krea.py --resume 15

ARGUMENTS
---------
  --character     Character name (trigger word 1, folder name)
  --checkpoint    LoRA filename e.g. k41t1yn_zit_1500.safetensors
  --trigger2      Second trigger word (empty = not used)
  --start         Lowest strength value to test   (default: 0.50)
  --end           Highest strength value to test  (default: 1.00)
  --step          Increment between values        (default: 0.05)
  --multipliers   Seed multipliers  (default: 1000 2000 3000)
  --dry-run       Print all planned runs without queuing anything
  --resume N      Skip first N images for the first strength level
  --host          ComfyUI host
  --port          ComfyUI port
  --workflow      Workflow filename override
  --prompts       Prompts filename override
  --check-interval  Seconds between queue polls  (default: 3.0)
"""

import json
import urllib.request
import urllib.error
import time
import argparse
import sys
import uuid
import copy
import re
from pathlib import Path
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit this section
# ══════════════════════════════════════════════════════════════════════════════

COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188

WORKFLOW_FILE = "lora_evaluation_krea.json"
PROMPTS_FILE  = "prompts_krea.txt"

LORAS_ROOT     = Path(r"G:\AI\models\loras")
LORA_SUBROOT   = "krea"
LORA_MODEL_TAG = "krea"

SEED_MULTIPLIERS   = [1000, 2000, 3000]
OUTPUT_ROOT        = "lora_eval/krea"
DEFAULT_CHARACTER  = "k41t1yn"
DEFAULT_TRIGGER2   = ""
DEFAULT_CHECKPOINT = "k41t1yn_krea_2700.safetensors"

# Weight range defaults
DEFAULT_START = 0.50
DEFAULT_END   = 1.00
DEFAULT_STEP  = 0.05

LOG_ROOT = Path(r"G:\output\lora_eval\krea")

# ── Node IDs (workflow) ───────────────────────────────────────────────
NODE_SEED_MULT   = 219
NODE_PROMPT_ID   = 220
NODE_PROMPT_TEXT = 221
NODE_TRIGGER1    = 222
NODE_TRIGGER2    = 223
NODE_PATH_LORA   = 225
NODE_PATH_BASE   = 225
NODE_PATH_ROOT   = 224
NODE_PATH_ROOT_B = 224
NODE_LORA_LOADER = 217    # LoraLoader
NODE_FILENAME    = 226    # easy string: filename prefix
NODE_CHAR_NAME   = 232    # easy string: character name for output path e.g. "k41t1yn/"

UUID_CLASS_MAP = {}

FALLBACK_NODES = {"ComfyUI-Krea2T-Enhancer", "KSampler", "LoraLoader", "LoraLoaderModelOnly"}

# ══════════════════════════════════════════════════════════════════════════════


# ── Logging ───────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w", encoding="utf-8", buffering=1)
        self._print(f"Log started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._print(f"Log file: {log_path}\n")

    def _print(self, msg="", end="\n"):
        print(msg, end=end, flush=True)
        self._file.write(msg + end)

    def __call__(self, msg="", end="\n"):
        self._print(msg, end=end)

    def close(self):
        self._file.close()


log = None


# ── Strength range ────────────────────────────────────────────────────────

def build_strengths(start, end, step):
    """
    Build list of strength values from start to end inclusive.
    Uses integer arithmetic to avoid floating point rounding.
    e.g. start=0.5, end=1.0, step=0.05 -> [0.50, 0.55, 0.60, ..., 1.00]
    """
    scale  = round(1 / step)
    i_start = round(start * scale)
    i_end   = round(end   * scale)
    i_step  = round(step  * scale)
    values  = []
    i = i_start
    while i <= i_end:
        values.append(round(i / scale, 10))
        i += i_step
    return values


def strength_str(val):
    """Format strength as folder name e.g. 0.5 -> '0.50', 1.0 -> '1.00'"""
    return f"{val:.2f}"


# ── Prompts ───────────────────────────────────────────────────────────────

def load_prompts(path):
    try:
        with open(path, encoding="utf-8") as f:
            lines = [l.strip() for l in f
                     if l.strip() and not l.strip().startswith("#")]
    except FileNotFoundError:
        log(f"ERROR: Prompts file not found: {path}")
        sys.exit(1)
    return [(f"{i:02d}", text) for i, text in enumerate(lines, 1)]


# ── ComfyUI API ───────────────────────────────────────────────────────────

def comfy_get(host, port, endpoint):
    url = f"http://{host}:{port}/{endpoint}"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read().decode())
    except urllib.error.URLError as e:
        log(f"\nERROR: Could not reach ComfyUI at {url}\n  {e}")
        sys.exit(1)


def comfy_post(host, port, endpoint, data):
    url = f"http://{host}:{port}/{endpoint}"
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        reason = e.read().decode("utf-8", errors="replace")
        log(f"\nERROR posting to {url}")
        log(f"  HTTP {e.code}: {e.reason}")
        log(f"  Response: {reason[:500]}")
        sys.exit(1)
    except urllib.error.URLError as e:
        log(f"\nERROR posting to {url}\n  {e}")
        sys.exit(1)


def queue_prompt(host, port, workflow_api, client_id):
    result = comfy_post(host, port, "prompt",
                        {"prompt": workflow_api, "client_id": client_id})
    pid = result.get("prompt_id")
    if not pid:
        log(f"  WARNING: unexpected queue response: {result}")
    return pid


def wait_for_completion(host, port, prompt_ids, check_interval):
    remaining = set(filter(None, prompt_ids))
    while remaining:
        q = comfy_get(host, port, "queue")
        active = (
            {item[1] for item in q.get("queue_running", [])} |
            {item[1] for item in q.get("queue_pending", [])}
        )
        remaining &= active
        if remaining:
            log(".", end="")
            time.sleep(check_interval)
    log(" done.")


# ── Node definitions ──────────────────────────────────────────────────────

NODE_DEFS = {}

CONNECTION_TYPES = {
    "MODEL", "CLIP", "VAE", "LATENT", "CONDITIONING", "IMAGE",
    "MASK", "CONTROL_NET", "STYLE_MODEL", "CLIP_VISION",
    "CLIP_VISION_OUTPUT", "BOOLEAN", "SIGMAS", "SAMPLER",
    "GUIDER", "NOISE", "FLOW_CONTROL",
}

HIDDEN_WIDGETS = {
    "KSampler":         [(1, "control_after_generate")],
    "KSamplerAdvanced": [(1, "add_noise"), (2, "noise_seed"),
                         (3, "control_after_generate")],
}


def fetch_node_defs(host, port):
    global NODE_DEFS
    try:
        url = f"http://{host}:{port}/object_info"
        with urllib.request.urlopen(url, timeout=10) as r:
            NODE_DEFS = json.loads(r.read().decode())
        log(f"  Node definitions loaded: {len(NODE_DEFS)} node types")
    except Exception as e:
        log(f"  WARNING: could not fetch node definitions: {e}")


def get_widget_names(node_type):
    info = NODE_DEFS.get(node_type)
    if not info:
        return None
    names = []
    for category in ("required", "optional"):
        for name, spec in info.get("input", {}).get(category, {}).items():
            if not isinstance(spec, list) or not spec:
                continue
            if isinstance(spec[0], str) and spec[0] in CONNECTION_TYPES:
                continue
            names.append(name)
    for idx, name in sorted(HIDDEN_WIDGETS.get(node_type, []),
                             key=lambda x: x[0]):
        names.insert(idx, name)
    return names if names else None


# ── Workflow helpers ──────────────────────────────────────────────────────

def load_workflow(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def patch_ui_workflow(wf_ui, patches):
    wf = copy.deepcopy(wf_ui)
    node_map = {n["id"]: n for n in wf["nodes"]}
    for node_id, value in patches.items():
        n = node_map.get(node_id)
        if not n:
            log(f"  WARNING: node {node_id} not found for patching")
            continue
        wv = n.get("widgets_values", [])
        if isinstance(value, list):
            n["widgets_values"] = value
        elif wv:
            wv[0] = value
            n["widgets_values"] = wv
        else:
            n["widgets_values"] = [value]
    return wf


def patch_lora_strength(wf_ui, node_id, strength):
    """Patch LoraLoader strength_model and strength_clip widgets."""
    wf = wf_ui  # already a deep copy from patch_ui_workflow
    node_map = {n["id"]: n for n in wf["nodes"]}
    n = node_map.get(node_id)
    if not n:
        log(f"  WARNING: LoraLoader node {node_id} not found")
        return
    wv = n.get("widgets_values", [])
    # widgets_values: [lora_name, strength_model, strength_clip]
    if len(wv) >= 3:
        wv[1] = strength
        wv[2] = strength


def ui_to_api(wf_ui):
    link_map = {}
    for lk in wf_ui.get("links", []):
        link_map[lk[0]] = [str(lk[1]), lk[2]]

    api = {}
    for node in wf_ui.get("nodes", []):
        nid       = str(node["id"])
        node_type = node["type"]
        if node_type in UUID_CLASS_MAP:
            node_type = UUID_CLASS_MAP[node_type]

        wv     = node.get("widgets_values", [])
        inputs = {}
        widget_names = get_widget_names(node_type)

        if node_type in FALLBACK_NODES:
            widget_names = None  # force fallback path for known problematic nodes

        if widget_names is not None:
            wv_map = {name: wv[i] for i, name in enumerate(widget_names)
                      if i < len(wv)}
            for inp in (node.get("inputs") or []):
                link_id = inp.get("link")
                if link_id is not None and link_id in link_map:
                    inputs[inp["name"]] = link_map[link_id]
                elif inp["name"] in wv_map:
                    val = wv_map[inp["name"]]
                    if inp["name"] == "operation" and isinstance(val, (int, float)):
                        op_map = {0:"add",1:"subtract",2:"multiply",
                                  3:"divide",4:"modulo",5:"power"}
                        val = op_map.get(int(val), "multiply")
                    inputs[inp["name"]] = val
            # Add any widget-only inputs not in inputs[] array
            # (e.g. LoraLoader: lora_name, strength_model, strength_clip)
            for name in widget_names:
                if name not in inputs and name in wv_map:
                    inputs[name] = wv_map[name]
            if not node.get("inputs") and wv:
                inputs["value"] = wv[0]
        else:
            wv_idx = 0
            for inp in (node.get("inputs") or []):
                link_id    = inp.get("link")
                has_widget = inp.get("widget") is not None
                linked     = link_id is not None and link_id in link_map
                if linked:
                    inputs[inp["name"]] = link_map[link_id]
                    # Do NOT increment wv_idx when linked — widget value
                    # slots in widgets_values only exist for unlinked inputs
                elif has_widget:
                    if wv_idx < len(wv):
                        val = wv[wv_idx]
                        if inp["name"] == "operation" and isinstance(val, (int, float)):
                            op_map = {0:"add",1:"subtract",2:"multiply",
                                      3:"divide",4:"modulo",5:"power"}
                            val = op_map.get(int(val), "multiply")
                        inputs[inp["name"]] = val
                    wv_idx += 1
            remaining = wv[wv_idx:]
            if remaining and not inputs:
                inputs["value"] = remaining[0]

        api[nid] = {
            "class_type": node_type,
            "inputs":     inputs,
            "_meta":      {"title": node.get("title", node_type)},
        }
    return api


def clean_api(api):
    return {
        nid: {k: v for k, v in node.items() if not k.startswith("_")}
        for nid, node in api.items()
    }


def lora_comfy_path(character, filename):
    """Build ComfyUI-relative LoRA path from character and filename."""
    return f"{LORA_SUBROOT}\\{character}\\{filename}"


# ── Baseline check ────────────────────────────────────────────────────────

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

def check_baseline(character, expected_count):
    """
    Check if baseline images exist and are complete.
    Returns (exists: bool, count: int)
    """
    folder = LOG_ROOT / character / "baseline"
    if not folder.exists():
        return False, 0
    count = len([f for f in folder.iterdir()
                 if f.suffix.lower() in IMAGE_EXTS])
    return count >= expected_count, count


# ── Args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="LoRA weight testing — image generation script")
    p.add_argument("--character",      default=DEFAULT_CHARACTER)
    p.add_argument("--checkpoint",     default=DEFAULT_CHECKPOINT,
                   help="LoRA filename e.g. k41t1yn_krea_5000.safetensors")
    p.add_argument("--trigger2",       default=DEFAULT_TRIGGER2)
    p.add_argument("--start",          type=float, default=DEFAULT_START,
                   help="Starting strength value (default: 0.50)")
    p.add_argument("--end",            type=float, default=DEFAULT_END,
                   help="Ending strength value inclusive (default: 1.00)")
    p.add_argument("--step",           type=float, default=DEFAULT_STEP,
                   help="Strength increment (default: 0.05)")
    p.add_argument("--multipliers",    type=int, nargs="+",
                   default=SEED_MULTIPLIERS)
    p.add_argument("--dry-run",        action="store_true")
    p.add_argument("--resume",         type=int, default=0)
    p.add_argument("--host",           default=COMFY_HOST)
    p.add_argument("--port",           type=int, default=COMFY_PORT)
    p.add_argument("--workflow",       default=WORKFLOW_FILE)
    p.add_argument("--prompts",        default=PROMPTS_FILE)
    p.add_argument("--check-interval", type=float, default=3.0)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    global log
    args      = parse_args()
    character = args.character.lower()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode      = "dryrun" if args.dry_run else "run"
    _ckpt_m = re.search(r'_(\d+)\.safetensors$', args.checkpoint)
    _ckpt_s = _ckpt_m.group(1) if _ckpt_m else ""
    log_path  = (LOG_ROOT / character / "results" /
                 f"{timestamp}_{character}_{_ckpt_s}_weight_{mode}.log")
    log = Logger(log_path)

    strengths = build_strengths(args.start, args.end, args.step)
    lora_path = lora_comfy_path(character, args.checkpoint)

    log("=" * 62)
    log("  LoRA Weight Testing — Image Generation Script")
    log("=" * 62)
    log(f"  Character  : {character}")
    log(f"  Checkpoint : {args.checkpoint}")
    # Extract step number from checkpoint filename for folder naming
    _m = re.search(r'_(\d+)\.safetensors$', args.checkpoint)
    ckpt_step = _m.group(1) if _m else ""
    log(f"  LoRA path  : {lora_path}")
    log(f"  Strengths  : {[strength_str(s) for s in strengths]}")
    log(f"  Multipliers: {args.multipliers}")
    log(f"  Workflow   : {args.workflow}")
    log(f"  Prompts    : {args.prompts}")
    if args.dry_run:
        log("  MODE       : DRY RUN — nothing will be queued")
    log()

    prompts  = load_prompts(args.prompts)
    expected = len(prompts) * len(args.multipliers)
    log(f"  Prompts loaded    : {len(prompts)}")
    log(f"  Images per weight : {expected}")
    log(f"  Total LoRA images : {expected * len(strengths)}")
    log()

    if not args.dry_run:
        info = comfy_get(args.host, args.port, "system_stats")
        ver  = info.get("system", {}).get("comfyui_version", "unknown")
        log(f"Connected to ComfyUI {ver} at {args.host}:{args.port}")
        fetch_node_defs(args.host, args.port)
        log()

    # ── Baseline check ────────────────────────────────────────────────────
    bl_ok, bl_count = check_baseline(character, expected)
    if bl_ok:
        log(f"  Baseline OK — {bl_count} images found, reusing.")
    else:
        log(f"  Baseline missing or incomplete ({bl_count}/{expected}) — generating now.")
        if not args.dry_run:
            baseline_start = time.time()
            log(f"\n{'─'*62}")
            log(f"  Generating {expected} baseline images (no LoRA)")
            log(f"{'─'*62}")
            bl_num = 0
            for prompt_id, prompt_text in prompts:
                pid_int = int(prompt_id)
                for multiplier in args.multipliers:
                    seed   = pid_int * multiplier
                    bl_num += 1
                    log(f"  [{bl_num}/{expected}] prompt={prompt_id} mult={multiplier:>5} seed={seed:>6}")
                    log(f"    baseline → {OUTPUT_ROOT}/{character}/baseline/{prompt_id}_{seed}")
                    patches = {
                        NODE_SEED_MULT:   str(multiplier),
                        NODE_PROMPT_ID:   prompt_id,
                        NODE_PROMPT_TEXT: prompt_text,
                        NODE_TRIGGER1:    character,
                        NODE_TRIGGER2:    args.trigger2,
                        NODE_PATH_ROOT:   f"{OUTPUT_ROOT}/",
                        NODE_PATH_LORA:   "/baseline/",
                    NODE_LORA_LOADER: lora_comfy_path(character, args.checkpoint),
                    NODE_CHAR_NAME:   f"{character}/",
                        NODE_FILENAME:    f"{prompt_id}_{seed}",
                    }
                    wf  = patch_ui_workflow(wf_ui_src, patches)
                    patch_lora_strength(wf, NODE_LORA_LOADER, 0.0)
                    patch_lora_strength(wf, NODE_LORA_LOADER_LN, 0.0)
                    api = clean_api(ui_to_api(wf))
                    pid = queue_prompt(args.host, args.port, api, client_id)
                    log(f"    queued {pid[:8] if pid else 'ERROR'}", end="")
                    wait_for_completion(args.host, args.port, [pid], args.check_interval)
            bl_elapsed = time.time() - baseline_start
            bl_mins, bl_secs = divmod(int(bl_elapsed), 60)
            log(f"  ✓ Baseline complete — {expected} images  ({bl_mins}m {bl_secs:02d}s)")
    log()

    wf_ui_src = load_workflow(args.workflow)
    client_id = str(uuid.uuid4())
    img_num   = 0
    total     = expected * len(strengths)

    for strength in strengths:
        s_str = strength_str(strength)

        log(f"\n{'─'*62}")
        log(f"  Strength : {s_str}  ({args.checkpoint})")
        log(f"{'─'*62}")

        combo_idx = 0

        for prompt_id, prompt_text in prompts:
            pid_int = int(prompt_id)

            for multiplier in args.multipliers:
                seed      = pid_int * multiplier
                combo_idx += 1
                img_num   += 1

                if combo_idx <= args.resume:
                    log(f"  [SKIP] prompt {prompt_id} "
                        f"mult {multiplier:>5} seed {seed}")
                    continue

                log(f"\n  [{img_num}/{total}] "
                    f"strength={s_str} | "
                    f"prompt={prompt_id} | "
                    f"mult={multiplier:>5} | seed={seed:>6}")
                log(f"    lora → {OUTPUT_ROOT}/{character}/weight{ckpt_step}/{s_str}/{prompt_id}_{seed}")

                if args.dry_run:
                    continue

                patches = {
                    NODE_SEED_MULT:   str(multiplier),
                    NODE_PROMPT_ID:   prompt_id,
                    NODE_PROMPT_TEXT: prompt_text,
                    NODE_TRIGGER1:    character,
                    NODE_CHAR_NAME:   f"{character}/",
                    NODE_TRIGGER2:    args.trigger2,
                    NODE_PATH_ROOT:   f"{OUTPUT_ROOT}/",
                    NODE_PATH_LORA:   f"/weight{ckpt_step}/{s_str}/",
                    NODE_LORA_LOADER: lora_path,
                    NODE_FILENAME:    f"{prompt_id}_{seed}",
                }

                wf = patch_ui_workflow(wf_ui_src, patches)
                patch_lora_strength(wf, NODE_LORA_LOADER, strength)
                api = clean_api(ui_to_api(wf))

                pid = queue_prompt(args.host, args.port, api, client_id)
                log(f"    queued {pid[:8] if pid else 'ERROR'}", end="")
                wait_for_completion(
                    args.host, args.port, [pid], args.check_interval)

        log(f"\n  ✓ Strength {s_str} complete — {expected} images")

    log(f"\n{'='*62}")
    log(f"  All done. {img_num} LoRA images generated.")
    log(f"  Baseline images : {LOG_ROOT / character / 'baseline'}")
    log(f"  Weight images   : {LOG_ROOT / character / f'weight{ckpt_step}'}")
    log(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'='*62}\n")
    log.close()


if __name__ == "__main__":
    main()
