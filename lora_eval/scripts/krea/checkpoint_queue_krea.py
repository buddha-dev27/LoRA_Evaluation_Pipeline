#!/usr/bin/env python3
"""
checkpoint_queue_krea.py
LoRA Checkpoint Evaluation — Image Generation Script

Automates the image generation phase of LoRA checkpoint evaluation.
For each checkpoint file found in the character's LoRA folder, and for
each prompt and seed multiplier, it queues images to ComfyUI:
  1. Baseline images — no LoRA, generated once and reused across all checkpoints
  2. LoRA images     — same settings with LoRA applied, one set per checkpoint

Both share the same prompt, seed, and sampler settings.
The baseline is the reference; the LoRA image shows the checkpoint's effect.
Images are later scored by lora_validator.py via run_checkpoints.bat.

Edit the variables below before running this script. For more information
see README_LoRA_Evaluation_Pipeline.md in the lora_eval/docs folder.

REQUIREMENTS
------------
  - Python 3.x (no external packages required)
  - ComfyUI running and accessible at COMFY_HOST:COMFY_PORT
  - Workflow file: lora_evaluation_krea.json (in same folder as this script)
  - Prompts file:  prompts_krea.txt (one prompt per line, # lines are comments)
  - LoRA files named: {character}_krea_{step}.safetensors
    in folder: LORAS_ROOT / LORA_SUBROOT / {character} /

OUTPUT STRUCTURE
----------------
  ComfyUI output /
    lora_eval /
      krea /
        {character} /
          baseline /   01_1000_00001_.png ... 30_3000_00001_.png  (90 files, generated once)
          checkpoint /
            {step} /   01_1000_00001_.png ... 30_3000_00001_.png  (90 lora files)
            ...

CONFIGURATION (edit top of script)
-----------------------------------
  COMFY_HOST        ComfyUI host (default: 127.0.0.1)
  COMFY_PORT        ComfyUI port (default: 8188)
  WORKFLOW_FILE     Workflow JSON filename
  PROMPTS_FILE      Prompts text file
  LORAS_ROOT        Full local path to loras folder e.g. G:\AI\models\loras
  LORA_SUBROOT      Subfolder within loras root e.g. z-image
  LORA_MODEL_TAG    Short model name used in LoRA filenames e.g. zit
  SEED_MULTIPLIERS  List of seed multipliers e.g. [1000, 2000, 3000]
  OUTPUT_ROOT       ComfyUI-relative output root e.g. lora_eval/zit
  DEFAULT_CHARACTER        Default character name (trigger word and folder name)
  DEFAULT_TRIGGER2         Second trigger word if used, empty string if not
  CHECKPOINT_LORA_STRENGTH LoRA strength for checkpoint image generation (default 1.0).
                           1.0 recommended — maximises score spread between checkpoints.
                           Lower values compress differences; use weight testing phase
                           to find the optimal production weight.
  COMFY_OUTPUT_ROOT Full local path to ComfyUI output folder root
  LOG_ROOT          Full local path to log output root folder

USAGE
-----
  # Dry run — preview all planned images without queuing anything (always do this first)
  python checkpoint_queue_krea.py --dry-run

  # Full sweep — scans LoRA folder automatically, uses set variables
  python checkpoint_queue_krea.py

  # Different character — overrides DEFAULT_CHARACTER
  python checkpoint_queue_krea.py --character k41t1yn

  # Specific checkpoints only
  python checkpoint_queue_krea.py --checkpoints 1400 1500 1600

  # Range of checkpoints (start end step)
  python checkpoint_queue_krea.py --range 1400 2600 100

  # Use the short prompt list
  python checkpoint_queue_krea.py --prompts prompts_krea_short.txt

  # Second trigger word
  python checkpoint_queue_krea.py --trigger2 redhead

  # Baseline images only — use if needed only for weight evaluation
  python checkpoint_queue_krea.py --baseline-only

  # Resume after interruption — N is the last completed pair number from the log
  python checkpoint_queue_krea.py --checkpoints 2500 2600 2700 --resume 47

ARGUMENTS
---------
  --character     Character name — used as trigger word 1, LoRA subfolder,
                  and output subfolder (default: DEFAULT_CHARACTER)
  --trigger2      Second trigger word, if the LoRA was trained with two
                  (default: DEFAULT_TRIGGER2, empty = not used)
  --checkpoints   Override checkpoint list with specific step numbers.
                  Default: scan the character's LoRA folder automatically.
  --range START END [STEP]
                  Generate a range of checkpoints from START to END inclusive.
                  Optional STEP defaults to 100. Overrides --checkpoints.
  --multipliers   Seed multipliers to use (default: 1000 2000 3000)
  --dry-run       Print all planned runs without queuing anything
  --baseline-only Generate baseline images only and exit. Use when prompts
                  have changed and baseline needs to be regenerated without
                  rerunning all checkpoint images.
  --resume N      Skip first N prompt×seed combos for the first checkpoint.
                  Use after an interrupted run. Check log for last pair number.
  --host          ComfyUI host (default: 127.0.0.1)
  --port          ComfyUI port (default: 8188)
  --workflow      Workflow filename (default: WORKFLOW_FILE)
  --prompts       Prompts filename (default: PROMPTS_FILE)
  --check-interval  Seconds between queue status polls (default: 3.0)
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

# Full local path to the loras folder that ComfyUI uses
LORAS_ROOT     = Path(r"G:\AI\models\loras")
LORA_SUBROOT   = "krea"
LORA_MODEL_TAG = "krea"

SEED_MULTIPLIERS  = [1000, 2000, 3000]
OUTPUT_ROOT       = "lora_eval/krea"
DEFAULT_CHARACTER = "k41t1yn"
DEFAULT_TRIGGER2  = ""          # second trigger word, leave empty if not used

# LoRA strength used when generating checkpoint evaluation images.
# 1.0 is recommended — maximum signal makes differences between checkpoints most
# visible and produces the widest scoring spread, making it easier to identify the
# best checkpoint. Lower values (e.g. 0.8) shift evaluation closer to production
# conditions but compress score differences. The weight testing phase is the right
# place to find the optimal production weight for your chosen checkpoint.
CHECKPOINT_LORA_STRENGTH = 1.0

# Full local path to ComfyUI output folder root
# This is where ComfyUI saves generated images
COMFY_OUTPUT_ROOT = Path(r"G:\output\lora_eval\krea")

# Log file location
LOG_ROOT = Path(r"G:\output\lora_eval\krea")

# ── Node IDs ────────────────────────────────────────────────────────────────
NODE_SEED_MULT   = 219    # easy string: seed multiplier e.g. "1000"
NODE_PROMPT_ID   = 220    # easy string: prompt ID e.g. "01"
NODE_PROMPT_TEXT = 221    # easy string: prompt text
NODE_TRIGGER1    = 222    # easy string: trigger word 1
NODE_TRIGGER2    = 223    # easy string: trigger word 2
NODE_PATH_LORA   = 225    # easy string: lora subfolder e.g. "/checkpoint/1000/"
NODE_PATH_BASE   = 225    # easy string: base subfolder "/baseline/"
NODE_PATH_ROOT   = 224    # easy string: root folder "lora_eval/krea/"
NODE_PATH_ROOT_B = 224    # easy string: root folder (base branch)
NODE_LORA_LOADER = 217    # LoraLoader — lora_name patched per checkpoint
NODE_FILENAME    = 226    # easy string: filename prefix e.g. "01_1000"
NODE_CHAR_NAME   = 232    # easy string: character name for output path e.g. "k41t1yn/"

UUID_CLASS_MAP = {}

# Node types whose widget values should always use the fallback path
# (bypasses NODE_DEFS lookup). Add node types here if their object_info
# widget ordering causes incorrect API payload construction.
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


# ── Checkpoint discovery ──────────────────────────────────────────────────

def scan_checkpoints(character):
    folder = LORAS_ROOT / LORA_SUBROOT / character
    if not folder.exists():
        log(f"ERROR: LoRA folder not found: {folder}")
        sys.exit(1)

    pattern = re.compile(
        rf"^{re.escape(character)}_{re.escape(LORA_MODEL_TAG)}_(\d+)\.safetensors$",
        re.IGNORECASE
    )

    found = []
    for f in folder.iterdir():
        m = pattern.match(f.name)
        if m:
            found.append((int(m.group(1)), f.stem))

    if not found:
        log(f"ERROR: No matching files in {folder}")
        log(f"  Expected: {character}_{LORA_MODEL_TAG}_NNNN.safetensors")
        log(f"\n  Files found:")
        for f in sorted(folder.iterdir()):
            log(f"    {f.name}")
        sys.exit(1)

    found.sort(key=lambda x: x[0])
    return found


def report_checkpoints(checkpoints):
    steps = [s for s, _ in checkpoints]
    log(f"  Checkpoints found : {len(steps)}")
    log(f"  Range             : {steps[0]} – {steps[-1]}")
    log(f"  Steps             : {steps}")
    if len(steps) > 1:
        diffs = [steps[i+1] - steps[i] for i in range(len(steps)-1)]
        interval = max(set(diffs), key=diffs.count)
        full = set(range(steps[0], steps[-1] + 1, interval))
        missing = sorted(full - set(steps))
        if missing:
            log(f"  Gaps (missing)    : {missing}")
        else:
            log(f"  No gaps detected")


def lora_comfy_path(character, stem):
    return f"{LORA_SUBROOT}\\{character}\\{stem}.safetensors"


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


# ── Workflow patching ─────────────────────────────────────────────────────

def load_workflow(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def patch_ui_workflow(wf_ui, patches):
    """
    Patch widget values directly in the UI-format workflow.
    patches = {node_id (int): new_widgets_values_list_or_first_value}
    Returns a deep copy with patches applied.
    """
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


def fetch_node_defs(host, port):
    """
    Fetch /object_info from ComfyUI to get the correct widget ordering
    for every node type. This is the only reliable way to map
    widgets_values to input names, since hidden widgets (like
    KSampler's control_after_generate) exist in widgets_values but
    have no entry in the UI node's inputs[] array.
    """
    global NODE_DEFS
    try:
        url = f"http://{host}:{port}/object_info"
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read().decode())
        NODE_DEFS = data
        log(f"  Node definitions loaded: {len(NODE_DEFS)} node types")
    except Exception as e:
        log(f"  WARNING: could not fetch node definitions: {e}")
        log(f"  Widget ordering may be incorrect for some nodes.")


# Connection types that appear in inputs[] but NOT in widgets_values
CONNECTION_TYPES = {
    "MODEL", "CLIP", "VAE", "LATENT", "CONDITIONING", "IMAGE",
    "MASK", "CONTROL_NET", "STYLE_MODEL", "CLIP_VISION",
    "CLIP_VISION_OUTPUT", "BOOLEAN", "SIGMAS", "SAMPLER",
    "GUIDER", "NOISE", "FLOW_CONTROL",
}

# Hidden widgets that exist in widgets_values but are NOT returned by
# object_info and have no entry in the node's inputs[] array.
# These must be explicitly skipped when mapping widgets_values to input names.
# Format: { node_type: [list of (index, name) for hidden widgets] }
HIDDEN_WIDGETS = {
    # Note: when seed is provided via link (not widget), the
    # control_after_generate hidden widget shifts to index 0
    # The workflow patches seed via link so widgets_values starts
    # at steps. control_after_generate is excluded from widgets_values.
    "KSampler":         [],  # seed linked — no hidden widgets in payload
    "KSamplerAdvanced": [(1, "add_noise"), (2, "noise_seed"), (3, "control_after_generate")],
}


def get_widget_names(node_type):
    """
    Return ordered list of widget input names for a node type,
    including hidden widgets, in the order they appear in widgets_values.
    Filters out connection-type inputs (MODEL, CLIP, VAE etc.) which do
    NOT consume a slot in widgets_values.
    Inserts hidden widgets (from HIDDEN_WIDGETS) at their correct positions.
    Returns None if node type not found in NODE_DEFS.
    """
    info = NODE_DEFS.get(node_type)
    if not info:
        return None

    names = []
    for category in ("required", "optional"):
        for name, spec in info.get("input", {}).get(category, {}).items():
            if not isinstance(spec, list) or not spec:
                continue
            input_type = spec[0]
            if isinstance(input_type, str) and input_type in CONNECTION_TYPES:
                continue
            names.append(name)

    # Insert hidden widgets at their correct positions
    for idx, name in sorted(HIDDEN_WIDGETS.get(node_type, []), key=lambda x: x[0]):
        names.insert(idx, name)

    return names if names else None


def patch_lora_strength(wf, node_id, strength):
    """Patch LoraLoader strength_model and strength_clip widgets."""
    node_map = {n["id"]: n for n in wf["nodes"]}
    n = node_map.get(node_id)
    if not n:
        return
    wv = n.get("widgets_values", [])
    # widgets_values: [lora_name, strength_model, strength_clip]
    if len(wv) >= 3:
        wv[1] = strength
        wv[2] = strength


def ui_to_api(wf_ui):
    """
    Convert UI-format workflow to API format.
    Uses NODE_DEFS from /object_info for correct widget ordering.
    """
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

        # Get the authoritative widget name order from object_info
        widget_names = get_widget_names(node_type)

        if node_type in FALLBACK_NODES:
            widget_names = None  # force fallback path for known problematic nodes

        if widget_names is not None:
            # Use object_info ordering — most reliable
            # widgets_values maps 1:1 to widget_names in order
            wv_map = {}
            for i, name in enumerate(widget_names):
                if i < len(wv):
                    wv_map[name] = wv[i]

            # Now assign inputs: links take priority, widgets fill the rest
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

            # Add widget-only inputs not declared in inputs[] array
            # e.g. LoraLoader: lora_name, strength_model, strength_clip
            for name in widget_names:
                if name not in inputs and name in wv_map:
                    inputs[name] = wv_map[name]

            # Nodes with no inputs[] (easy string, easy int) — use first widget value
            if not node.get("inputs") and wv:
                inputs["value"] = wv[0]

        else:
            # Fallback: use inputs[] widget flags to determine ordering
            # This may be wrong for nodes with hidden widgets
            wv_idx = 0
            for inp in (node.get("inputs") or []):
                link_id    = inp.get("link")
                has_widget = inp.get("widget") is not None
                linked     = link_id is not None and link_id in link_map

                if linked:
                    inputs[inp["name"]] = link_map[link_id]
                    # Do NOT increment wv_idx when linked — the widget value
                    # slot in widgets_values is only present for unlinked inputs
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
    """Strip internal keys and empty extra widgets before sending."""
    out = {}
    for nid, node in api.items():
        cleaned_inputs = {
            k: v for k, v in node["inputs"].items()
            if not k.startswith("_")
        }
        out[nid] = {
            "class_type": node["class_type"],
            "inputs":     cleaned_inputs,
            "_meta":      node["_meta"],
        }
    return out


# ── Main ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--character",      default=DEFAULT_CHARACTER)
    p.add_argument("--host",           default=COMFY_HOST)
    p.add_argument("--port",           type=int, default=COMFY_PORT)
    p.add_argument("--workflow",       default=WORKFLOW_FILE)
    p.add_argument("--prompts",        default=PROMPTS_FILE)
    p.add_argument("--multipliers",    type=int, nargs="+", default=SEED_MULTIPLIERS)
    p.add_argument("--trigger2",       default=DEFAULT_TRIGGER2,
                   help="Second trigger word (leave empty if not used)")
    p.add_argument("--checkpoints",    type=int, nargs="+", default=None)
    p.add_argument("--range",          type=int, nargs="+", metavar="N",
                   help="Range of checkpoints: START END [STEP]. "
                        "e.g. --range 1400 2600 100 generates 1400,1500,...,2600")
    p.add_argument("--dry-run",        action="store_true")
    p.add_argument("--baseline-only",  action="store_true",
                   help="Generate baseline images only, skip checkpoint LoRA images")
    p.add_argument("--resume",         type=int, default=0)
    p.add_argument("--check-interval", type=float, default=3.0)
    return p.parse_args()


def main():
    global log
    args      = parse_args()
    character = args.character.lower()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode      = "dryrun" if args.dry_run else "run"
    log_path  = LOG_ROOT / character / "results" / f"{timestamp}_{character}_{mode}.log"
    log       = Logger(log_path)

    run_start = time.time()
    log("=" * 62)
    log("  LoRA Evaluation Queue Script")
    log("=" * 62)
    log(f"  Character  : {character}")
    log(f"  Multipliers: {args.multipliers}")
    log(f"  Workflow   : {args.workflow}")
    log(f"  LoRA strength : {CHECKPOINT_LORA_STRENGTH}")
    log(f"  Prompts    : {args.prompts}")
    if args.dry_run:
        log("  MODE       : DRY RUN — nothing will be queued")
    log()

    # Checkpoints — always scan first to get real stems (preserves leading zeros)
    all_checkpoints = {s: stem for s, stem in scan_checkpoints(character)}

    if args.checkpoints:
        checkpoints = []
        for s in sorted(args.checkpoints):
            stem = all_checkpoints.get(s, f"{character}_{LORA_MODEL_TAG}_{s}")
            checkpoints.append((s, stem))
        log(f"  Checkpoints: {len(checkpoints)} (manual override)")
    elif args.range:
        r = args.range
        if len(r) < 2:
            log("ERROR: --range requires at least START END")
            sys.exit(1)
        start, end = r[0], r[1]
        step = r[2] if len(r) >= 3 else 100
        steps = list(range(start, end + 1, step))
        checkpoints = []
        for s in steps:
            stem = all_checkpoints.get(s, f"{character}_{LORA_MODEL_TAG}_{s}")
            checkpoints.append((s, stem))
        log(f"  Checkpoints: {len(checkpoints)} (range {start}–{end} step {step})")
    else:
        checkpoints = scan_checkpoints(character)
        report_checkpoints(checkpoints)
    log()

    # Prompts
    prompts = load_prompts(args.prompts)
    log(f"  Prompts loaded: {len(prompts)}")
    log()

    total_pairs = len(checkpoints) * len(prompts) * len(args.multipliers)
    log(f"  Total pairs planned : {total_pairs}")
    log(f"  Total images        : {total_pairs * 2}")
    log()

    if not args.dry_run:
        info = comfy_get(args.host, args.port, "system_stats")
        ver  = info.get("system", {}).get("comfyui_version", "unknown")
        log(f"Connected to ComfyUI {ver} at {args.host}:{args.port}")
        fetch_node_defs(args.host, args.port)
        log()

    # Load the UI-format workflow once
    wf_ui_src  = load_workflow(args.workflow)
    client_id  = str(uuid.uuid4())
    pair_num   = 0

    # ── BASELINE IMAGES ──────────────────────────────────────────────────
    expected_baseline = len(prompts) * len(args.multipliers)
    baseline_local    = COMFY_OUTPUT_ROOT / character / "baseline"
    baseline_ok       = False

    if not args.dry_run:
        if baseline_local.exists():
            existing = len([f for f in baseline_local.iterdir()
                            if f.suffix.lower() in {".png",".jpg",".jpeg",".webp"}])
            if existing >= expected_baseline:
                log(f"  Baseline exists ({existing} images) — skipping generation.")
                baseline_ok = True
            else:
                log(f"  Baseline incomplete ({existing}/{expected_baseline}) — regenerating.")
        else:
            log(f"  Baseline folder not found — will generate {expected_baseline} images.")

    if not baseline_ok:
        baseline_start = time.time()
        log(f"\n{'─'*62}")
        log(f"  Generating {expected_baseline} baseline images (no LoRA)")
        log(f"{'─'*62}")
        # Use first checkpoint's lora file for LoraLoader validation only
        # (base branch bypasses LoraLoader — any valid file works)
        first_stem = checkpoints[0][1]
        bl_num = 0
        for prompt_id, prompt_text in prompts:
            pid_int = int(prompt_id)
            for multiplier in args.multipliers:
                seed   = pid_int * multiplier
                bl_num += 1
                log(f"  [{bl_num}/{expected_baseline}] "
                    f"prompt={prompt_id} mult={multiplier:>5} seed={seed:>6}")
                log(f"    baseline → {OUTPUT_ROOT}/{character}/baseline/{prompt_id}_{seed}")
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
                    NODE_PATH_LORA:   "/baseline/",
                    NODE_LORA_LOADER: lora_comfy_path(character, first_stem),
                    NODE_FILENAME:    f"{prompt_id}_{seed}",
                }
                wf  = patch_ui_workflow(wf_ui_src, patches)
                patch_lora_strength(wf, NODE_LORA_LOADER, 0.0)
                api = clean_api(ui_to_api(wf))
                pid = queue_prompt(args.host, args.port, api, client_id)
                log(f"    queued {pid[:8] if pid else 'ERROR'}", end="")
                wait_for_completion(args.host, args.port, [pid], args.check_interval)
        bl_elapsed = time.time() - baseline_start
        bl_mins, bl_secs = divmod(int(bl_elapsed), 60)
        log(f"  ✓ Baseline complete — {expected_baseline} images  ({bl_mins}m {bl_secs:02d}s)")

    if args.baseline_only:
        log(f"\n  --baseline-only flag set — skipping checkpoint images.")
        log(f"{'='*62}")
        log(f"  Done. Baseline images in {COMFY_OUTPUT_ROOT / character / 'baseline'}")
        log(f"{'='*62}\n")
        log.close()
        return

    # ── CHECKPOINT LORA IMAGES ────────────────────────────────────────────
    for ckpt_idx, (step, stem) in enumerate(checkpoints, 1):
        step_str   = str(step)
        lora_path  = lora_comfy_path(character, stem)
        ckpt_start = time.time()

        log(f"\n{'─'*62}")
        log(f"  Checkpoint : {step_str}  ({stem}.safetensors)  [{ckpt_idx}/{len(checkpoints)}]")
        log(f"{'─'*62}")

        combo_idx = 0

        for prompt_id, prompt_text in prompts:
            pid_int = int(prompt_id)

            for multiplier in args.multipliers:
                seed      = pid_int * multiplier
                combo_idx += 1
                pair_num  += 1

                if combo_idx <= args.resume:
                    log(f"  [SKIP] prompt {prompt_id} "
                        f"mult {multiplier:>5} seed {seed}")
                    continue

                log(f"\n  [{pair_num}/{total_pairs}] "
                    f"prompt={prompt_id} | mult={multiplier:>5} | seed={seed:>6}")
                log(f"    lora → {OUTPUT_ROOT}/{character}/checkpoint/{step_str}/{prompt_id}_{seed}")

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
                    NODE_PATH_LORA:   f"/checkpoint/{step_str}/",
                    NODE_LORA_LOADER: lora_comfy_path(character, stem),
                    NODE_FILENAME:    f"{prompt_id}_{seed}",
                }

                # Only queue the LoRA image — baseline already exists
                wf  = patch_ui_workflow(wf_ui_src, patches)
                patch_lora_strength(wf, NODE_LORA_LOADER, CHECKPOINT_LORA_STRENGTH)
                api = clean_api(ui_to_api(wf))
                pid = queue_prompt(args.host, args.port, api, client_id)
                log(f"    lora queued {pid[:8] if pid else 'ERROR'}", end="")
                wait_for_completion(args.host, args.port, [pid], args.check_interval)

        ckpt_elapsed = time.time() - ckpt_start
        ckpt_mins, ckpt_secs = divmod(int(ckpt_elapsed), 60)
        log(f"\n  ✓ Checkpoint {step_str} complete — "
            f"{len(prompts) * len(args.multipliers)} images  "
            f"({ckpt_mins}m {ckpt_secs:02d}s)")

    total_elapsed = time.time() - run_start
    total_hrs,  rem       = divmod(int(total_elapsed), 3600)
    total_mins, total_secs = divmod(rem, 60)
    total_str = (f"{total_hrs}h {total_mins}m {total_secs:02d}s"
                 if total_hrs else f"{total_mins}m {total_secs:02d}s")
    log(f"\n{'='*62}")
    log(f"  All done. {pair_num} pairs generated.")
    log(f"  Finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Run time : {total_str}")
    log(f"{'='*62}\n")
    log.close()


if __name__ == "__main__":
    main()
