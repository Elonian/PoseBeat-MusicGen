#!/usr/bin/env python3
"""Generate PoseBeat training and evaluation curve visualizations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "visualiser" / "training_curves"

MODEL_SPECS = {
    "base": {
        "label": "Base 256x256",
        "color": "#2563eb",
        "train_log": REPO_ROOT / "logs" / "motion_to_music" / "train_metrics.jsonl",
        "config": "configs/motion_to_music.yaml",
        "final_metrics": REPO_ROOT
        / "outputs"
        / "motion_to_music"
        / "eval_runs"
        / "pipeline_final"
        / "metrics_rerun_2026-06-02.json",
        "fallback_final_metrics": REPO_ROOT
        / "outputs"
        / "motion_to_music"
        / "eval_runs"
        / "pipeline_final"
        / "metrics.json",
        "paired_fad": REPO_ROOT
        / "outputs"
        / "motion_to_music"
        / "eval_runs"
        / "pipeline_final"
        / "metrics_paired_fad_2026-06-02.json",
    },
    "hires": {
        "label": "High-res 512x512",
        "color": "#059669",
        "train_log": REPO_ROOT / "logs" / "motion_to_music_hires" / "train_metrics.jsonl",
        "config": "configs/motion_to_music_hires.yaml",
        "final_metrics": REPO_ROOT
        / "outputs"
        / "motion_to_music_hires"
        / "eval_runs"
        / "pipeline_final"
        / "metrics.json",
        "paired_fad": REPO_ROOT
        / "outputs"
        / "motion_to_music_hires"
        / "eval_runs"
        / "pipeline_final"
        / "metrics_paired_fad_2026-06-02.json",
    },
}

PAPER_OURS = {
    "beat_coverage": 0.935,
    "beat_hit": 0.860,
    "fad_dmd_style": 4.960,
    "genre_kld": 0.604,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_runs_by_step_restart(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_step: int | None = None
    for row in rows:
        step = int(row["step"])
        if current and previous_step is not None and step <= previous_step:
            runs.append(current)
            current = []
        current.append(row)
        previous_step = step
    if current:
        runs.append(current)
    return runs


def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    all_points: list[pd.DataFrame] = []
    run_summary: dict[str, Any] = {}

    for model_key, spec in MODEL_SPECS.items():
        rows = read_jsonl(spec["train_log"])
        runs = split_runs_by_step_restart(rows)
        main_run = max(runs, key=lambda run: int(run[-1]["step"]))
        first_time = float(main_run[0]["time"])
        records = []
        for row in main_run:
            record = dict(row)
            record["model"] = model_key
            record["model_label"] = spec["label"]
            record["epoch_number"] = int(row["epoch"]) + 1
            record["elapsed_hours"] = (float(row["time"]) - first_time) / 3600.0
            records.append(record)

        frame = pd.DataFrame.from_records(records)
        all_points.append(frame)

        min_loss = min(main_run, key=lambda row: float(row["loss"]))
        min_avg_loss = min(main_run, key=lambda row: float(row["avg_loss"]))
        final = main_run[-1]
        run_summary[model_key] = {
            "label": spec["label"],
            "source": str(spec["train_log"].relative_to(REPO_ROOT)),
            "rows_total": len(rows),
            "detected_runs": len(runs),
            "discarded_restart_rows": len(rows) - len(main_run),
            "main_run_rows": len(main_run),
            "first_step": int(main_run[0]["step"]),
            "final_step": int(final["step"]),
            "final_epoch": int(final["epoch"]) + 1,
            "duration_hours": round((float(final["time"]) - first_time) / 3600.0, 3),
            "final_loss": float(final["loss"]),
            "final_avg_loss": float(final["avg_loss"]),
            "min_loss": {
                "value": float(min_loss["loss"]),
                "step": int(min_loss["step"]),
                "epoch": int(min_loss["epoch"]) + 1,
            },
            "min_avg_loss": {
                "value": float(min_avg_loss["avg_loss"]),
                "step": int(min_avg_loss["step"]),
                "epoch": int(min_avg_loss["epoch"]) + 1,
            },
        }

    points = pd.concat(all_points, ignore_index=True)
    epoch_summary = (
        points.sort_values(["model", "step"])
        .groupby(["model", "model_label", "epoch_number"], as_index=False)
        .agg(
            step=("step", "max"),
            elapsed_hours=("elapsed_hours", "max"),
            loss_mean=("loss", "mean"),
            loss_min=("loss", "min"),
            loss_last=("loss", "last"),
            avg_loss_last=("avg_loss", "last"),
            learning_rate_last=("learning_rate", "last"),
        )
    )
    return points, epoch_summary, run_summary


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(payload: dict[str, Any], path: list[str], default: float | None = None) -> float | None:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return float(current)


def read_pair_count(payload: dict[str, Any]) -> str:
    inventory = payload.get("inventory", {})
    requested = inventory.get("requested", "?")
    paired = inventory.get("paired", "?")
    return f"{paired}/{requested}"


def extract_eval_row(
    *,
    model: str,
    model_label: str,
    pipeline: str,
    epoch: int | None,
    metrics_path: Path,
    paired_fad_path: Path | None = None,
) -> dict[str, Any]:
    payload = load_json(metrics_path)
    metrics = payload.get("metrics", {})
    paired_payload = load_json(paired_fad_path) if paired_fad_path and paired_fad_path.exists() else None

    dmd_fad = metric_value(metrics, ["fad_dmd_full_reference", "fad"])
    if dmd_fad is None:
        dmd_fad = metric_value(metrics, ["fad", "fad"])
    paired_fad = metric_value(metrics, ["fad", "fad"])
    if paired_payload:
        paired_fad = metric_value(paired_payload.get("metrics", {}), ["fad", "fad"])

    return {
        "model": model,
        "model_label": model_label,
        "pipeline": pipeline,
        "epoch": epoch,
        "pairs": read_pair_count(payload),
        "beat_coverage": metric_value(metrics, ["beat", "beat_coverage"]),
        "beat_hit": metric_value(metrics, ["beat", "beat_hit"]),
        "fad_dmd_style": dmd_fad,
        "paired_cdcd_fad": paired_fad,
        "genre_kld": metric_value(metrics, ["genre", "genre_kld"]),
        "metrics_path": str(metrics_path.relative_to(REPO_ROOT)),
    }


def load_evaluation_data() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_eval_root = REPO_ROOT / "outputs" / "motion_to_music" / "eval_runs"
    for metrics_path in sorted(base_eval_root.glob("pipeline_epoch_*/metrics.json")):
        epoch = int(metrics_path.parent.name.rsplit("_", 1)[1])
        rows.append(
            extract_eval_row(
                model="base",
                model_label=MODEL_SPECS["base"]["label"],
                pipeline=metrics_path.parent.name,
                epoch=epoch,
                metrics_path=metrics_path,
            )
        )

    for model_key, spec in MODEL_SPECS.items():
        metrics_path = spec["final_metrics"]
        if not metrics_path.exists() and "fallback_final_metrics" in spec:
            metrics_path = spec["fallback_final_metrics"]
        rows.append(
            extract_eval_row(
                model=model_key,
                model_label=spec["label"],
                pipeline="pipeline_final",
                epoch=None,
                metrics_path=metrics_path,
                paired_fad_path=spec.get("paired_fad"),
            )
        )
    return pd.DataFrame.from_records(rows)


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> list[str]:
    paths = []
    for suffix in ("png", "svg"):
        path = output_dir / f"{name}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(str(path.relative_to(REPO_ROOT)))
    plt.close(fig)
    return paths


def configure_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#d7dde8", linewidth=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#344054")
    ax.title.set_color("#111827")
    ax.xaxis.label.set_color("#344054")
    ax.yaxis.label.set_color("#344054")


def plot_training_dashboard(
    points: pd.DataFrame,
    epoch_summary: pd.DataFrame,
    eval_df: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("PoseBeat Training and Evaluation Dashboard", fontsize=18, fontweight="bold", y=0.99)

    ax = axes[0, 0]
    for model_key, spec in MODEL_SPECS.items():
        df = points[points["model"] == model_key].sort_values("step").copy()
        df["avg_loss_smooth"] = df["avg_loss"].rolling(window=21, min_periods=1).mean()
        ax.plot(df["step"], df["avg_loss_smooth"], label=spec["label"], color=spec["color"], linewidth=2.2)
    ax.set_title("Smoothed training avg_loss by optimizer step")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("avg_loss, log scale")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    configure_axes(ax)

    ax = axes[0, 1]
    for model_key, spec in MODEL_SPECS.items():
        df = epoch_summary[epoch_summary["model"] == model_key]
        ax.plot(
            df["epoch_number"],
            df["avg_loss_last"],
            marker="o",
            markersize=3,
            label=spec["label"],
            color=spec["color"],
            linewidth=1.8,
        )
    ax.set_title("End-of-epoch avg_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("avg_loss, log scale")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    configure_axes(ax)

    ax = axes[1, 0]
    base_epochs = eval_df[(eval_df["model"] == "base") & eval_df["epoch"].notna()].sort_values("epoch")
    ax.plot(base_epochs["epoch"], base_epochs["beat_coverage"], marker="o", label="Beat coverage", color="#2563eb")
    ax.plot(base_epochs["epoch"], base_epochs["beat_hit"], marker="o", label="Beat hit", color="#059669")
    ax.set_title("Base checkpoint beat metrics on CDCD")
    ax.set_xlabel("Saved pipeline epoch")
    ax.set_ylabel("Beat score, higher is better")
    ax.set_ylim(0.68, 0.92)
    ax.legend(frameon=False, loc="upper right")
    configure_axes(ax)

    ax = axes[1, 1]
    final_rows = eval_df[eval_df["pipeline"] == "pipeline_final"].copy()
    cell_text = []
    for _, row in final_rows.iterrows():
        cell_text.append(
            [
                row["model_label"],
                row["pairs"],
                f"{row['beat_coverage']:.3f}",
                f"{row['beat_hit']:.3f}",
                f"{row['fad_dmd_style']:.3f}",
                f"{row['genre_kld']:.3f}",
            ]
        )
    ax.axis("off")
    ax.set_title("Final model eval snapshot", pad=18)
    table = ax.table(
        cellText=cell_text,
        colLabels=["Model", "Pairs", "Beat cov", "Beat hit", "FAD", "KLD"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.7)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d7dde8")
        if row_idx == 0:
            cell.set_facecolor("#111827")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f8fafc")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, output_dir, "training_dashboard")


def plot_training_loss_curves(points: pd.DataFrame, output_dir: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Training Loss Curves", fontsize=17, fontweight="bold")

    for ax, y_col, title in (
        (axes[0], "loss", "Batch loss"),
        (axes[1], "avg_loss", "Running avg_loss"),
    ):
        for model_key, spec in MODEL_SPECS.items():
            df = points[points["model"] == model_key].sort_values("step").copy()
            smooth = df[y_col].rolling(window=21, min_periods=1).mean()
            ax.plot(df["step"], smooth, label=spec["label"], color=spec["color"], linewidth=2.0)
        ax.set_title(title)
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel(f"{y_col}, log scale")
        ax.set_yscale("log")
        ax.legend(frameon=False)
        configure_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, output_dir, "training_loss_curves")


def plot_base_eval_progression(eval_df: pd.DataFrame, output_dir: Path) -> list[str]:
    base_epochs = eval_df[(eval_df["model"] == "base") & eval_df["epoch"].notna()].sort_values("epoch")
    metrics = [
        ("beat_coverage", "Beat Coverage", "higher is better", "#2563eb"),
        ("beat_hit", "Beat Hit", "higher is better", "#059669"),
        ("fad_dmd_style", "DMD-style FAD", "lower is better", "#dc6803"),
        ("genre_kld", "Genre KLD", "lower is better", "#7c3aed"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Base Model Evaluation Across Saved Checkpoints", fontsize=17, fontweight="bold")
    for ax, (column, title, direction, color) in zip(axes.ravel(), metrics):
        ax.plot(base_epochs["epoch"], base_epochs[column], marker="o", color=color, linewidth=2.0)
        if column in PAPER_OURS:
            ax.axhline(PAPER_OURS[column], color="#64748b", linestyle="--", linewidth=1.3, label="Paper Ours")
            ax.legend(frameon=False)
        best_idx = base_epochs[column].idxmax() if direction.startswith("higher") else base_epochs[column].idxmin()
        best = base_epochs.loc[best_idx]
        ax.scatter([best["epoch"]], [best[column]], color="#111827", s=55, zorder=3)
        ax.set_title(f"{title} ({direction})")
        ax.set_xlabel("Saved pipeline epoch")
        ax.set_ylabel(title)
        configure_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, output_dir, "base_eval_progression")


def plot_final_model_comparison(eval_df: pd.DataFrame, output_dir: Path) -> list[str]:
    final_rows = eval_df[eval_df["pipeline"] == "pipeline_final"].copy().sort_values("model")
    metrics = [
        ("beat_coverage", "Beat Coverage", "higher is better"),
        ("beat_hit", "Beat Hit", "higher is better"),
        ("fad_dmd_style", "DMD-style FAD", "lower is better"),
        ("genre_kld", "Genre KLD", "lower is better"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    fig.suptitle("Final Base vs High-res Evaluation", fontsize=17, fontweight="bold")
    colors = [MODEL_SPECS[row["model"]]["color"] for _, row in final_rows.iterrows()]
    labels = list(final_rows["model_label"])

    for ax, (column, title, direction) in zip(axes.ravel(), metrics):
        values = list(final_rows[column])
        bars = ax.bar(labels, values, color=colors, width=0.55)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#111827",
            )
        ax.set_title(f"{title} ({direction})")
        ax.set_ylabel(title)
        configure_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, output_dir, "final_model_comparison")


def best_eval_summary(eval_df: pd.DataFrame) -> dict[str, Any]:
    base_epochs = eval_df[(eval_df["model"] == "base") & eval_df["epoch"].notna()].sort_values("epoch")
    summary: dict[str, Any] = {}
    for column, direction in (
        ("beat_coverage", "max"),
        ("beat_hit", "max"),
        ("fad_dmd_style", "min"),
        ("genre_kld", "min"),
    ):
        idx = base_epochs[column].idxmax() if direction == "max" else base_epochs[column].idxmin()
        row = base_epochs.loc[idx]
        summary[column] = {
            "epoch": int(row["epoch"]),
            "value": float(row[column]),
        }
    return summary


def final_eval_summary(eval_df: pd.DataFrame) -> dict[str, Any]:
    final_rows = eval_df[eval_df["pipeline"] == "pipeline_final"]
    result: dict[str, Any] = {}
    for _, row in final_rows.iterrows():
        result[row["model"]] = {
            "label": row["model_label"],
            "pairs": row["pairs"],
            "beat_coverage": float(row["beat_coverage"]),
            "beat_hit": float(row["beat_hit"]),
            "fad_dmd_style": float(row["fad_dmd_style"]),
            "paired_cdcd_fad": float(row["paired_cdcd_fad"]),
            "genre_kld": float(row["genre_kld"]),
        }
    return result


def write_report(
    output_dir: Path,
    summary: dict[str, Any],
    generated_files: list[str],
) -> Path:
    training = summary["training"]
    base_best = summary["base_checkpoint_best"]
    final_eval = summary["final_eval"]
    report_path = output_dir / "training_curves_report.md"
    text = f"""# PoseBeat Training and Evaluation Curve Report

Generated: {summary["generated_at"]}

## Outputs

| File | Purpose |
| --- | --- |
| `training_dashboard.png` / `.svg` | Four-panel overview of training loss, checkpoint eval, and final metrics. |
| `training_loss_curves.png` / `.svg` | Focused smoothed loss and avg_loss curves for base and high-res runs. |
| `base_eval_progression.png` / `.svg` | Base checkpoint metrics from epoch 10 through 100. |
| `final_model_comparison.png` / `.svg` | Final base vs high-res metric comparison. |
| `epoch_progression.csv` | Per-epoch training statistics from the completed runs. |
| `eval_metrics_progression.csv` | Checkpoint and final evaluation metrics. |
| `curve_summary.json` | Machine-readable summary of the best epochs and final metrics. |

## Key Reading

| Model | Completed Epochs | Final Step | Final avg_loss | Best avg_loss | Eval Pairs |
| --- | ---: | ---: | ---: | ---: | --- |
| Base 256x256 | {training["base"]["final_epoch"]} | {training["base"]["final_step"]} | {training["base"]["final_avg_loss"]:.6f} | {training["base"]["min_avg_loss"]["value"]:.6f} | {final_eval["base"]["pairs"]} |
| High-res 512x512 | {training["hires"]["final_epoch"]} | {training["hires"]["final_step"]} | {training["hires"]["final_avg_loss"]:.6f} | {training["hires"]["min_avg_loss"]["value"]:.6f} | {final_eval["hires"]["pairs"]} |

The high-res log contains {training["hires"]["discarded_restart_rows"]} rows from an aborted warm-up run. The plots and CSV use the completed high-res run only.

## Base Checkpoint Eval

| Metric | Best Saved Epoch | Value |
| --- | ---: | ---: |
| Beat coverage | {base_best["beat_coverage"]["epoch"]} | {base_best["beat_coverage"]["value"]:.6f} |
| Beat hit | {base_best["beat_hit"]["epoch"]} | {base_best["beat_hit"]["value"]:.6f} |
| DMD-style FAD | {base_best["fad_dmd_style"]["epoch"]} | {base_best["fad_dmd_style"]["value"]:.6f} |
| Genre KLD | {base_best["genre_kld"]["epoch"]} | {base_best["genre_kld"]["value"]:.6f} |

Training loss continues to improve late in the run, but the eval metrics do not all peak at the same epoch. For the base model, epoch 20 is best for beat matching, epoch 70 is best for DMD-style FAD, and epoch 100 is best for genre KLD.

## Final Model Eval

| Model | Beat Coverage | Beat Hit | DMD-style FAD | Paired-CDCD FAD | Genre KLD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base 256x256 | {final_eval["base"]["beat_coverage"]:.6f} | {final_eval["base"]["beat_hit"]:.6f} | {final_eval["base"]["fad_dmd_style"]:.6f} | {final_eval["base"]["paired_cdcd_fad"]:.6f} | {final_eval["base"]["genre_kld"]:.6f} |
| High-res 512x512 | {final_eval["hires"]["beat_coverage"]:.6f} | {final_eval["hires"]["beat_hit"]:.6f} | {final_eval["hires"]["fad_dmd_style"]:.6f} | {final_eval["hires"]["paired_cdcd_fad"]:.6f} | {final_eval["hires"]["genre_kld"]:.6f} |

The high-res final model is stronger on beat coverage and beat hit. The base final model is stronger on FAD and genre KLD.

## How To Read These Curves

- Training loss is the denoising/reconstruction objective used during optimization; lower means the model fits the latent training target better.
- Evaluation uses generated audio on the 31-file CDCD subset, then compares it to AIST++ test reference audio.
- DMD-style FAD uses all 186 reference test wavs against the 31 generated CDCD wavs, matching the existing base evaluation report.
- Paired-CDCD FAD uses only the same 31 reference/generated pairs; it is useful diagnostically but not the DMD paper-style number.
- BAS is not plotted because the existing base report also left BAS unavailable in this environment.
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path


def write_outputs(
    output_dir: Path,
    points: pd.DataFrame,
    epoch_summary: pd.DataFrame,
    eval_df: pd.DataFrame,
    run_summary: dict[str, Any],
    generated_files: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_csv = output_dir / "epoch_progression.csv"
    eval_csv = output_dir / "eval_metrics_progression.csv"
    epoch_summary.to_csv(epoch_csv, index=False)
    eval_df.to_csv(eval_csv, index=False)

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "training": run_summary,
        "base_checkpoint_best": best_eval_summary(eval_df),
        "final_eval": final_eval_summary(eval_df),
        "output_files": sorted(
            generated_files
            + [
                str(epoch_csv.relative_to(REPO_ROOT)),
                str(eval_csv.relative_to(REPO_ROOT)),
                str((output_dir / "training_curves_report.md").relative_to(REPO_ROOT)),
                str((output_dir / "curve_summary.json").relative_to(REPO_ROOT)),
            ]
        ),
    }
    report_path = write_report(output_dir, summary, generated_files)
    summary["output_files"] = sorted(set(summary["output_files"] + [str(report_path.relative_to(REPO_ROOT))]))
    summary_path = output_dir / "curve_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    points, epoch_summary, run_summary = load_training_data()
    eval_df = load_evaluation_data()

    generated_files: list[str] = []
    generated_files.extend(plot_training_dashboard(points, epoch_summary, eval_df, output_dir))
    generated_files.extend(plot_training_loss_curves(points, output_dir))
    generated_files.extend(plot_base_eval_progression(eval_df, output_dir))
    generated_files.extend(plot_final_model_comparison(eval_df, output_dir))

    summary = write_outputs(
        output_dir=output_dir,
        points=points,
        epoch_summary=epoch_summary,
        eval_df=eval_df,
        run_summary=run_summary,
        generated_files=generated_files,
    )
    print(json.dumps({"output_dir": str(output_dir), "files": summary["output_files"]}, indent=2))


if __name__ == "__main__":
    main()
