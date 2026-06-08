"""Visualize evaluation results for all LLM runs.

Reads all .txt files from the evaluation/ directory and generates:
  - Per subtask: bar chart comparing all models × variants on Macro F1
  - Per model:   grouped bar chart showing all 4 variants across subtasks

Run from the scripts/ directory:
    python3 visualize_results.py
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EVAL_DIR = Path("../evaluation")
OUT_DIR  = Path("../evaluation/figures")

# Human-readable model names
MODEL_LABELS = {
    "llama31-8B":  "Llama 3.1 8B",
    "llama32-3B":  "Llama 3.2 3B",
    "gemma3-12B":  "Gemma 3 12B",
    "gemma4-4B":   "Gemma 4 4B",
}

VARIANT_LABELS = {
    "base_zero-shot":     "Base\nZero-shot",
    "base_few-shot":      "Base\nFew-shot",
    "enhanced_zero-shot": "Enhanced\nZero-shot",
    "enhanced_few-shot":  "Enhanced\nFew-shot",
}

SUBTASK_LABELS = {
    "detection":      "Subtask 1 – Fallacy Detection",
    "classification": "Subtask 2 – Fallacy Classification",
}

CLASSIFICATION_CLASSES = [
    "authority", "black-white", "hasty_generalization",
    "natural", "population", "slippery_slope", "tradition", "worse_problems",
]

# Colour palette — one per model
MODEL_COLORS = {
    "llama31-8B": "#4C72B0",
    "llama32-3B": "#55A868",
    "gemma3-12B": "#C44E52",
    "gemma4-4B":  "#DD8452",
}

VARIANT_COLORS = {
    "base_zero-shot":     "#4C72B0",
    "base_few-shot":      "#55A868",
    "enhanced_zero-shot": "#C44E52",
    "enhanced_few-shot":  "#DD8452",
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_file(path: Path) -> dict:
    """Parse an evaluate.py output file and return a metrics dict."""
    text = path.read_text()
    result = {"path": path}

    m = re.search(r"Macro F1\s*:\s*([\d.]+)", text)
    result["macro_f1"] = float(m.group(1)) if m else None

    m = re.search(r"Weighted F1\s*:\s*([\d.]+)", text)
    result["weighted_f1"] = float(m.group(1)) if m else None

    m = re.search(r"accuracy\s+([\d.]+)", text)
    result["accuracy"] = float(m.group(1)) if m else None

    per_class = {}
    for cls in CLASSIFICATION_CLASSES:
        pattern = rf"{re.escape(cls)}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
        m = re.search(pattern, text)
        if m:
            per_class[cls] = {
                "precision": float(m.group(1)),
                "recall":    float(m.group(2)),
                "f1":        float(m.group(3)),
            }
    result["per_class"] = per_class

    return result


def load_all_results() -> dict:
    """Load all evaluation files into a nested dict:
       results[subtask][model][variant] = metrics_dict
    """
    results = {}
    for path in sorted(EVAL_DIR.glob("*.txt")):
        stem = path.stem
        parts = stem.split("_", 2)
        if len(parts) < 3:
            print(f"  Skipping unrecognised filename: {path.name}", file=sys.stderr)
            continue
        subtask, model, variant = parts[0], parts[1], parts[2]
        results.setdefault(subtask, {}).setdefault(model, {})[variant] = parse_file(path)

    return results


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def bar_positions(n_groups, n_bars, width=0.18, gap=0.35):
    """Return x positions for a grouped bar chart."""
    group_width = n_bars * width
    starts = np.arange(n_groups) * (group_width + gap)
    offsets = np.arange(n_bars) * width
    return starts, offsets, width


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 1: Per subtask — all models × variants, Macro F1
# ---------------------------------------------------------------------------

def plot_subtask_overview(results: dict):
    """One chart per subtask: grouped bars per model, each group = 4 variants."""
    variants = list(VARIANT_LABELS.keys())

    for subtask, subtask_label in SUBTASK_LABELS.items():
        if subtask not in results:
            print(f"  No data for {subtask}, skipping.", file=sys.stderr)
            continue

        models = [m for m in MODEL_LABELS if m in results[subtask]]
        n_groups = len(models)
        n_bars   = len(variants)
        starts, offsets, width = bar_positions(n_groups, n_bars)

        fig, ax = plt.subplots(figsize=(15, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        for b, variant in enumerate(variants):
            xs = starts + offsets[b]
            ys = []
            for model in models:
                val = results[subtask].get(model, {}).get(variant, {}).get("macro_f1")
                ys.append(val if val is not None else 0.0)

            bars = ax.bar(xs, ys, width=width, color=VARIANT_COLORS[variant],
                          label=VARIANT_LABELS[variant].replace("\n", " "),
                          alpha=0.88, edgecolor="white", linewidth=0.4)

            for bar, y in zip(bars, ys):
                if y > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, y + 0.008,
                            f"{y:.3f}", ha="center", va="bottom",
                            fontsize=7, color="black", fontweight="bold")

        group_centers = starts + offsets.mean()
        ax.set_xticks(group_centers)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models],
                           color="black", fontsize=12, fontweight="bold")
        ax.set_ylabel("Macro F1", color="black", fontsize=12)
        ax.set_ylim(0, 1.08)
        ax.set_title(subtask_label, color="black", fontsize=15, fontweight="bold", pad=14)
        ax.tick_params(colors="black")
        ax.spines[:].set_color("#ccc")
        ax.yaxis.grid(True, color="#ddd", linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)

        legend = ax.legend(title="Variant", title_fontsize=9, fontsize=8,
                           loc="upper left", bbox_to_anchor=(1.01, 1),
                           borderaxespad=0, framealpha=0.9,
                           labelcolor="black", facecolor="white", edgecolor="#ccc")
        legend.get_title().set_color("black")

        fig.tight_layout()
        save(fig, f"overview_{subtask}_macro_f1.png")


# ---------------------------------------------------------------------------
# Chart 2: Per model — Macro F1 across both subtasks and all 4 variants
# ---------------------------------------------------------------------------

def plot_per_model(results: dict):
    """One chart per model: side-by-side subtasks, bars = variants."""
    variants = list(VARIANT_LABELS.keys())
    subtasks = list(SUBTASK_LABELS.keys())

    for model, model_label in MODEL_LABELS.items():
        n_groups = len(subtasks)
        n_bars   = len(variants)
        starts, offsets, width = bar_positions(n_groups, n_bars)

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        for b, variant in enumerate(variants):
            xs = starts + offsets[b]
            ys = []
            for subtask in subtasks:
                val = results.get(subtask, {}).get(model, {}).get(variant, {}).get("macro_f1")
                ys.append(val if val is not None else 0.0)

            bars = ax.bar(xs, ys, width=width, color=VARIANT_COLORS[variant],
                          label=VARIANT_LABELS[variant].replace("\n", " "),
                          alpha=0.88, edgecolor="white", linewidth=0.4)

            for bar, y in zip(bars, ys):
                if y > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, y + 0.008,
                            f"{y:.3f}", ha="center", va="bottom",
                            fontsize=8.5, color="black", fontweight="bold")

        group_centers = starts + offsets.mean()
        ax.set_xticks(group_centers)
        ax.set_xticklabels([SUBTASK_LABELS[s] for s in subtasks],
                           color="black", fontsize=11, fontweight="bold")
        ax.set_ylabel("Macro F1", color="black", fontsize=12)
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{model_label} — Performance per Subtask & Variant",
                     color="black", fontsize=13, fontweight="bold", pad=14)
        ax.tick_params(colors="black")
        ax.spines[:].set_color("#ccc")
        ax.yaxis.grid(True, color="#ddd", linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)

        legend = ax.legend(title="Variant", title_fontsize=9, fontsize=8,
                           loc="upper left", bbox_to_anchor=(1.01, 1),
                           borderaxespad=0, framealpha=0.9,
                           labelcolor="black", facecolor="white", edgecolor="#ccc")
        legend.get_title().set_color("black")

        fig.tight_layout()
        slug = model.replace(" ", "_")
        save(fig, f"model_{slug}_subtasks.png")


# ---------------------------------------------------------------------------
# Chart 3: Classification only — per-class F1 heatmap per model
# ---------------------------------------------------------------------------

def plot_classification_heatmap(results: dict):
    """Heatmap: rows = classes, columns = model×variant, value = F1."""
    if "classification" not in results:
        return

    subtask_data = results["classification"]
    models   = [m for m in MODEL_LABELS if m in subtask_data]
    variants = list(VARIANT_LABELS.keys())

    col_labels = []
    matrix = []

    for model in models:
        for variant in variants:
            metrics = subtask_data.get(model, {}).get(variant, {})
            col_labels.append(f"{MODEL_LABELS[model]}\n{VARIANT_LABELS[variant]}")
            col_f1 = [metrics.get("per_class", {}).get(cls, {}).get("f1", 0.0)
                      for cls in CLASSIFICATION_CLASSES]
            matrix.append(col_f1)

    matrix = np.array(matrix).T  # [n_classes, n_cols]

    fig, ax = plt.subplots(figsize=(max(14, len(col_labels) * 1.1), 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    for r in range(len(CLASSIFICATION_CLASSES)):
        for c in range(len(col_labels)):
            val = matrix[r, c]
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color="black", fontweight="bold")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, color="black", fontsize=7, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASSIFICATION_CLASSES)))
    ax.set_yticklabels(CLASSIFICATION_CLASSES, color="black", fontsize=10)
    ax.set_title("Subtask 2 – Per-Class F1 Heatmap (all models & variants)",
                 color="black", fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(colors="black")
    ax.spines[:].set_color("#ccc")

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("F1 Score", color="black", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")

    fig.tight_layout()
    save(fig, "classification_perclass_heatmap.png")


# ---------------------------------------------------------------------------
# Chart 4: Zero-shot vs Few-shot delta per model
# ---------------------------------------------------------------------------

def plot_zeroshot_vs_fewshot(results: dict):
    """Bar chart showing the F1 gain from zero-shot → few-shot, per model & variant prefix."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    fig.patch.set_facecolor("white")
    fig.suptitle("Few-shot vs Zero-shot Macro F1 gain (few − zero)",
                 color="black", fontsize=14, fontweight="bold")

    for ax, (subtask, subtask_label) in zip(axes, SUBTASK_LABELS.items()):
        ax.set_facecolor("white")
        if subtask not in results:
            ax.set_visible(False)
            continue

        models   = [m for m in MODEL_LABELS if m in results[subtask]]
        prefixes = ["base", "enhanced"]
        x = np.arange(len(models))
        width = 0.3

        for i, prefix in enumerate(prefixes):
            deltas = []
            for model in models:
                zs = results[subtask].get(model, {}).get(f"{prefix}_zero-shot", {}).get("macro_f1", 0) or 0
                fs = results[subtask].get(model, {}).get(f"{prefix}_few-shot",  {}).get("macro_f1", 0) or 0
                deltas.append(fs - zs)

            color = "#4C72B0" if prefix == "base" else "#DD8452"
            bars = ax.bar(x + i * width, deltas, width, label=prefix.capitalize(),
                          color=color, alpha=0.88, edgecolor="white", linewidth=0.4)
            for bar, d in zip(bars, deltas):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        d + (0.003 if d >= 0 else -0.012),
                        f"{d:+.3f}", ha="center", va="bottom" if d >= 0 else "top",
                        fontsize=8, color="black", fontweight="bold")

        ax.axhline(0, color="#999", linewidth=0.8, linestyle="--")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models],
                           color="black", fontsize=10, fontweight="bold")
        ax.set_ylabel("ΔMacro F1 (few − zero)", color="black", fontsize=10)
        ax.set_title(subtask_label, color="black", fontsize=11, fontweight="bold")
        ax.tick_params(colors="black")
        ax.spines[:].set_color("#ccc")
        ax.yaxis.grid(True, color="#ddd", linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)

        legend = ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1),
                           borderaxespad=0, framealpha=0.9,
                           labelcolor="black", facecolor="white", edgecolor="#ccc")

    fig.tight_layout()
    save(fig, "delta_fewshot_vs_zeroshot.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not EVAL_DIR.exists():
        print(f"ERROR: evaluation/ directory not found at {EVAL_DIR.resolve()}", file=sys.stderr)
        sys.exit(1)

    print("Loading evaluation files...")
    results = load_all_results()

    total = sum(len(v2) for v1 in results.values() for v2 in v1.values())
    print(f"  Loaded {total} result files across "
          f"{sum(len(v) for v in results.values())} models and "
          f"{len(results)} subtasks.\n")

    print("Generating charts...")
    plot_subtask_overview(results)       # Chart 1: per subtask, all models
    plot_per_model(results)              # Chart 2: per model, both subtasks
    plot_classification_heatmap(results) # Chart 3: per-class F1 heatmap
    plot_zeroshot_vs_fewshot(results)    # Chart 4: few-shot vs zero-shot delta

    print(f"\nDone! All figures saved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()