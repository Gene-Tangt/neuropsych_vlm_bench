#!/usr/bin/env python3
"""
Error consistency (Geirhos et al., 2020) between OpenAI models, in release order.

For every pair of models with a trial-by-trial record (trials_<model>.csv), trials are matched on
(task, trial) and scored only as correct / incorrect. Error consistency is Cohen's kappa:

    c_obs = fraction of trials where both models are correct or both are wrong
    c_exp = p_i * p_j + (1 - p_i) * (1 - p_j)          (agreement expected from accuracy alone)
    kappa = (c_obs - c_exp) / (1 - c_exp)

kappa = 1: identical errors; 0: no more overlap than independent models; < 0: anti-correlated errors.
95% CIs come from bootstrapping over trials.

Models are ordered by release date (ties: flagship before smaller tiers), so neighbouring cells of the
lower-triangle heatmap compare models released one after another. Outputs:
    output/error_consistency_openai_all.png      one heatmap, all tasks pooled
    output/error_consistency_openai_stage.png    1 x 3 heatmaps: low-, mid- and high-level tasks
    output/error_consistency_openai.csv          kappa, CI, accuracies and n for every pair and subset
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

from neuropsych_bench import TaskLoader, task_paths
from neuropsych_bench.analysis import STAGE_LABELS, STAGES, THREE_COLORS
from neuropsych_bench.longitudinal import short_name

OUTPUT_DIR = Path("results/longitudinal_openai")
MODELS_CSV = OUTPUT_DIR / "models.csv"

# Models run through run_all.py rather than the longitudinal runner (same settings: reasoning off, 1000 tokens)
EXTRA_TRIALS = {"gpt-6-sol": Path("results/trials_gpt-6-sol.csv")}
TIER_ORDER = ["flagship", "terra", "mini", "luna", "nano", "astra"]  # tie-break for models released the same day
CMAP = "RdBu_r"  # diverging around kappa = 0; one scale for every heatmap
N_BOOTSTRAP = 2000
SEED = 0


def load_trials() -> tuple:
    """Correctness matrix (trials x models, in release order) and the stage of each trial."""
    paths = {p.stem.removeprefix("trials_"): p for p in OUTPUT_DIR.glob("trials_*.csv")}
    paths.update({m: p for m, p in EXTRA_TRIALS.items() if p.exists()})

    n_trials = sum(len(TaskLoader(p).get_trials()) for p in task_paths())
    columns, stage = {}, None
    for model, path in paths.items():
        trials = pd.read_csv(path)
        if len(trials) != n_trials:
            print(f"Skipping {model}: {len(trials)}/{n_trials} trials (run incomplete)")
            continue
        trials = trials.set_index(["task", "trial"]).sort_index()
        columns[model] = trials["correct"].astype(bool)
        stage = trials["stage"] if stage is None else stage

    models = pd.read_csv(MODELS_CSV).set_index("model")
    order = sorted(columns, key=lambda m: (models.loc[m, "release_date"], TIER_ORDER.index(models.loc[m, "tier"])))
    correct = pd.DataFrame(columns)[order]
    if correct.isna().any().any():
        raise ValueError("Trial ids differ between models")
    return correct, stage, models.loc[order, "release_date"]


def kappa(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cohen's kappa on correctness along the last axis (works on bootstrap stacks too)."""
    p_a, p_b = a.mean(-1), b.mean(-1)
    c_obs = (a == b).mean(-1)
    c_exp = p_a * p_b + (1 - p_a) * (1 - p_b)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (c_obs - c_exp) / (1 - c_exp)


def error_consistency(correct: pd.DataFrame, subset: str, rng: np.random.Generator) -> pd.DataFrame:
    """Kappa with bootstrap CI for every model pair."""
    values = correct.to_numpy()
    boot = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))  # same resamples for every pair
    rows = []
    for i, j in combinations(range(values.shape[1]), 2):
        a, b = values[:, i], values[:, j]
        boot_k = kappa(a[boot], b[boot])
        rows.append({
            "subset": subset, "model_i": correct.columns[i], "model_j": correct.columns[j],
            "n_trials": len(a), "acc_i": a.mean(), "acc_j": b.mean(),
            "c_obs": (a == b).mean(), "kappa": kappa(a, b),
            "ci_low": np.nanpercentile(boot_k, 2.5), "ci_high": np.nanpercentile(boot_k, 97.5),
        })
    return pd.DataFrame(rows)


def kappa_matrix(pairs: pd.DataFrame, order: list) -> pd.DataFrame:
    """Square kappa matrix in release order (rows = later model, cols = earlier model)."""
    matrix = pd.DataFrame(np.nan, index=order, columns=order)
    for _, row in pairs.iterrows():
        matrix.loc[row["model_j"], row["model_i"]] = row["kappa"]
    return matrix


def luminance(color) -> float:
    """Relative luminance of a colour, used to pick readable text on it."""
    red, green, blue = to_rgb(color)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def draw_heatmap(ax, matrix: pd.DataFrame, labels: list, vlim: float, annotate_size: float):
    """Strict lower triangle: drop the first row and last column, which hold only the diagonal."""
    shown = matrix.iloc[1:, :-1]
    image = ax.imshow(shown.to_numpy(dtype=float), cmap=CMAP, vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(shown.shape[1]), labels[:-1], rotation=90, fontsize=7)
    ax.set_yticks(range(shown.shape[0]), labels[1:], fontsize=7)
    for (r, c), value in np.ndenumerate(shown.to_numpy(dtype=float)):
        if not np.isnan(value):
            # dark text on light cells, white on dark ones
            cell = image.cmap(image.norm(value))
            ax.text(c, r, f"{value:.2f}".replace("0.", ".", 1), ha="center", va="center",
                    fontsize=annotate_size, color="white" if luminance(cell) < 0.5 else "0.15")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    return image


def stage_title(ax, text: str, color: str):
    """Title in a box the width of the panel, filled with the stage colour."""
    ax.add_patch(Rectangle((0, 1.02), 1, 0.06, transform=ax.transAxes, facecolor=color,
                           edgecolor="none", clip_on=False))
    ax.text(0.5, 1.05, text, transform=ax.transAxes, ha="center", va="center", fontsize=11,
            fontweight="bold", color="white" if luminance(color) < 0.5 else "0.1")


def main(output_dir: str = "output"):
    """Save the heatmaps and the pairwise table, and print kappa for consecutive releases."""
    correct, stage, release = load_trials()
    order = list(correct.columns)
    labels = [short_name(m, order) for m in order]
    rng = np.random.default_rng(SEED)

    subsets = {"all": correct, **{s: correct[stage == s] for s in STAGES}}
    pairs = pd.concat([error_consistency(data, name, rng) for name, data in subsets.items()], ignore_index=True)
    vlim = np.ceil(pairs["kappa"].abs().max() * 10) / 10  # one symmetric scale shared by every panel

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # (1) all tasks pooled
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    image = draw_heatmap(ax, kappa_matrix(pairs[pairs["subset"] == "all"], order), labels, vlim, 6)
    ax.set_title(f"Error consistency between OpenAI models, all tasks (n = {len(correct)} trials)\n"
                 "models in release order; cells beside the diagonal compare consecutive releases",
                 fontsize=10, loc="left")
    fig.colorbar(image, ax=ax, shrink=0.6, label="Error consistency (Cohen's κ)")
    fig.savefig(out / "error_consistency_openai_all.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # (2) one panel per processing stage, shared colour scale; stage shown by the coloured title box
    fig, axes = plt.subplots(1, 3, figsize=(24, 8.5))
    for ax, s, color in zip(axes, STAGES, THREE_COLORS):
        image = draw_heatmap(ax, kappa_matrix(pairs[pairs["subset"] == s], order), labels, vlim, 4.5)
        stage_title(ax, f"{STAGE_LABELS[s]} tasks (n = {len(subsets[s])} trials)", color)
    fig.colorbar(image, ax=axes, shrink=0.6, label="Error consistency (Cohen's κ)")
    fig.savefig(out / "error_consistency_openai_stage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pairs.insert(3, "release_i", pairs["model_i"].map(release))
    pairs.insert(4, "release_j", pairs["model_j"].map(release))
    pairs.round(4).to_csv(out / "error_consistency_openai.csv", index=False)

    consecutive = pairs[pairs.apply(lambda r: order.index(r["model_j"]) == order.index(r["model_i"]) + 1, axis=1)]
    print(f"{len(order)} models: {', '.join(labels)}\n")
    print("Consecutive releases:")
    print(consecutive.pivot(index=["model_i", "model_j"], columns="subset", values="kappa")[["all", *STAGES]]
          .round(2).to_string())
    print(f"\nSaved figures and table to {out}")


if __name__ == "__main__":
    main()
