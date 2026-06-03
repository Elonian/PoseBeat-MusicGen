#!/usr/bin/env python3
"""Generate dataset overview visuals for the PoseBeat continuous-conditioned task."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "dmd_aistpp_legacy_2026-05-26"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "visualiser" / "dataset_overview"
CDCD_LIST = REPO_ROOT / "configs" / "cdcd_aist.txt"

FILENAME_RE = re.compile(
    r"g(?P<genre>[A-Z]{2})_s(?P<setting>[A-Z]{2})_c(?P<camera_set>[^_]+)"
    r"_d(?P<dance>\d+)_m(?P<music>[A-Z]{2}\d+)_ch(?P<channel>\d+)"
    r"(?:_slice(?P<slice>\d+))?$"
)

GENRE_NAMES = {
    "BR": "Break",
    "HO": "House",
    "JB": "Ballet Jazz",
    "JS": "Street Jazz",
    "KR": "Krump",
    "LH": "LA-style Hip-hop",
    "LO": "Lock",
    "MH": "Middle Hip-hop",
    "PO": "Pop",
    "WA": "Waack",
}

GENRE_ORDER = ["BR", "HO", "JB", "JS", "KR", "LH", "LO", "MH", "PO", "WA"]


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def base_key(stem: str) -> str:
    return re.sub(r"_slice\d+$", "", stem)


def parse_aist_name(stem: str) -> dict[str, Any]:
    match = FILENAME_RE.match(stem)
    if not match:
        return {
            "stem": stem,
            "base_key": base_key(stem),
            "genre": "??",
            "genre_name": "Unknown",
            "setting": "",
            "dance": "",
            "music": "",
            "channel": "",
            "slice": None,
        }
    data = match.groupdict()
    genre = data["genre"]
    return {
        "stem": stem,
        "base_key": base_key(stem),
        "genre": genre,
        "genre_name": GENRE_NAMES.get(genre, genre),
        "setting": data["setting"],
        "dance": int(data["dance"]),
        "music": data["music"],
        "channel": int(data["channel"]),
        "slice": int(data["slice"]) if data["slice"] is not None else None,
    }


def collect_wav_rows(data_root: Path, split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((data_root / split / "wavs_sliced").glob("*.wav")):
        row = parse_aist_name(path.stem)
        row.update(
            {
                "split": split,
                "path": str(path.relative_to(REPO_ROOT)),
                "file_size_bytes": path.stat().st_size,
                "tiny_or_empty": path.stat().st_size < 100,
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def collect_cdcd_rows(cdcd_list: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in read_lines(cdcd_list):
        stem = Path(name).stem
        row = parse_aist_name(stem)
        row.update(
            {
                "split": "cdcd_eval",
                "path": str(cdcd_list.relative_to(REPO_ROOT)),
                "file_size_bytes": None,
                "tiny_or_empty": False,
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def split_counts(data_root: Path) -> dict[str, int]:
    split_root = data_root / "aistplusplus_raw" / "splits"
    return {path.name: len(read_lines(path)) for path in sorted(split_root.glob("*.txt"))}


def raw_inventory(data_root: Path) -> dict[str, Any]:
    return {
        "raw_motion_pkls_present": len(list((data_root / "aistplusplus_raw" / "motions").glob("*.pkl"))),
        "ignore_list_entries": len(read_lines(data_root / "aistplusplus_raw" / "ignore_list.txt")),
        "official_split_counts": split_counts(data_root),
    }


def read_dataset_info_num_examples(path: Path) -> int | None:
    info_path = path / "train" / "dataset_info.json"
    if not info_path.exists():
        return None
    payload = json.loads(info_path.read_text(encoding="utf-8"))
    splits = payload.get("splits") or {}
    train = splits.get("train") or {}
    value = train.get("num_examples")
    return int(value) if value is not None else None


def mel_dataset_summary(data_root: Path, train_wav_count: int) -> dict[str, Any]:
    input_root = data_root / "data_and_model" / "input music"
    hires_full = input_root / "aistpp_hires_full_sorted"
    manifest_path = hires_full / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return {
        "aistpp_256_sorted_rows": read_dataset_info_num_examples(input_root / "aistpp_256_sorted")
        or train_wav_count,
        "aistpp_hires_sorted_real_rows": read_dataset_info_num_examples(input_root / "aistpp_hires_sorted")
        or manifest.get("reused_real_rows"),
        "aistpp_hires_full_rows": read_dataset_info_num_examples(hires_full) or manifest.get("rows"),
        "aistpp_hires_full_reused_real_rows": manifest.get("reused_real_rows"),
        "aistpp_hires_full_reconstructed_rows": manifest.get("reconstructed_rows"),
        "aistpp_hires_reconstruction_n_iter": manifest.get("reconstruction_n_iter"),
    }


def condition_summary(data_root: Path) -> tuple[dict[str, Any], np.ndarray | None, str | None]:
    condition_root = data_root / "data_and_model" / "conditions"
    test_path = condition_root / "normalized_all_test_data_01.pkl"
    train_path = condition_root / "normalized_all_train_data_01.pkl"
    summary: dict[str, Any] = {
        "test_condition_pickle": str(test_path.relative_to(REPO_ROOT)),
        "train_condition_pickle": str(train_path.relative_to(REPO_ROOT)),
        "train_condition_pickle_gb": round(train_path.stat().st_size / 1e9, 3) if train_path.exists() else None,
    }
    if not test_path.exists():
        return summary, None, None
    with test_path.open("rb") as handle:
        test_conditions = pickle.load(handle)
    keys = sorted(test_conditions)
    sample_key = keys[0] if keys else None
    sample = np.asarray(test_conditions[sample_key], dtype=np.float32) if sample_key else None
    summary.update(
        {
            "test_condition_entries": len(test_conditions),
            "condition_shape": list(sample.shape) if sample is not None else None,
            "sample_key": sample_key,
            "motion_channels": 360 if sample is not None and sample.shape[1] >= 370 else None,
            "extra_condition_channels": 10 if sample is not None and sample.shape[1] >= 370 else None,
        }
    )
    return summary, sample, sample_key


def finedance_summary(repo_root: Path) -> dict[str, Any]:
    base = repo_root / "data" / "FineDance" / "extracted" / "finedance"
    labels = sorted((base / "label_json").glob("*.json")) if (base / "label_json").exists() else []
    motions = sorted((base / "motion").glob("*")) if (base / "motion").exists() else []
    styles = Counter()
    frame_values: list[int] = []
    for path in labels:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        style = " / ".join(str(payload.get(key, "")).strip() for key in ("style1", "style2") if payload.get(key))
        if style:
            styles[style] += 1
        if payload.get("frames") is not None:
            frame_values.append(int(payload["frames"]))
    return {
        "present": base.exists(),
        "label_json_files": len(labels),
        "motion_files": len(motions),
        "style_counts": dict(styles.most_common()),
        "frames_min": min(frame_values) if frame_values else None,
        "frames_median": float(np.median(frame_values)) if frame_values else None,
        "frames_max": max(frame_values) if frame_values else None,
        "note": "FineDance is present locally but was not used by the DMD/PoseBeat paper-style pipeline.",
    }


def count_by_genre(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["split", "genre", "genre_name"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    all_rows = []
    for split in ["train", "test", "cdcd_eval"]:
        for genre in GENRE_ORDER:
            matching = counts[(counts["split"] == split) & (counts["genre"] == genre)]
            if matching.empty:
                all_rows.append(
                    {
                        "split": split,
                        "genre": genre,
                        "genre_name": GENRE_NAMES[genre],
                        "count": 0,
                    }
                )
            else:
                all_rows.append(matching.iloc[0].to_dict())
    return pd.DataFrame.from_records(all_rows)


def sequence_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["split", "base_key", "genre", "genre_name"], as_index=False)
        .size()
        .rename(columns={"size": "slice_count"})
        .sort_values(["split", "slice_count"], ascending=[True, False])
    )


def audio_quality_counts(wav_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ["train", "test"]:
        split_df = wav_df[wav_df["split"] == split]
        tiny = int(split_df["tiny_or_empty"].sum())
        nonempty = int(len(split_df) - tiny)
        rows.extend(
            [
                {"split": split, "status": "non-empty wav", "count": nonempty},
                {"split": split, "status": "tiny/empty wav", "count": tiny},
            ]
        )
    return pd.DataFrame.from_records(rows)


def configure_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#d7dde8", linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#344054")
    ax.title.set_color("#111827")
    ax.xaxis.label.set_color("#344054")
    ax.yaxis.label.set_color("#344054")


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> list[str]:
    paths = []
    for suffix in ("png", "svg"):
        path = output_dir / f"{name}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(str(path.relative_to(REPO_ROOT)))
    plt.close(fig)
    return paths


def plot_dataset_overview_dashboard(
    wav_df: pd.DataFrame,
    cdcd_df: pd.DataFrame,
    mel_summary: dict[str, Any],
    condition_info: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    all_eval_df = pd.concat([wav_df, cdcd_df], ignore_index=True)
    seq = sequence_counts(all_eval_df)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("PoseBeat AIST++ Dataset Overview", fontsize=18, fontweight="bold", y=0.99)

    ax = axes[0, 0]
    slice_counts = all_eval_df.groupby("split").size().reindex(["train", "test", "cdcd_eval"]).fillna(0)
    base_counts = seq.groupby("split").size().reindex(["train", "test", "cdcd_eval"]).fillna(0)
    x = np.arange(3)
    ax.bar(x - 0.18, slice_counts.values, width=0.36, label="5s slices", color="#2563eb")
    ax.bar(x + 0.18, base_counts.values, width=0.36, label="unique base dances", color="#059669")
    for offset, values in ((-0.18, slice_counts.values), (0.18, base_counts.values)):
        for xpos, value in zip(x + offset, values):
            ax.text(xpos, max(float(value), 1.0), f"{int(value):,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, ["train", "test", "CDCD eval"])
    ax.set_title("Split scale")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    configure_axes(ax)

    ax = axes[0, 1]
    quality = audio_quality_counts(wav_df)
    pivot = quality.pivot(index="split", columns="status", values="count").reindex(["train", "test"]).fillna(0)
    bottom = np.zeros(len(pivot))
    for status, color in (("non-empty wav", "#059669"), ("tiny/empty wav", "#dc6803")):
        values = pivot[status].values if status in pivot else np.zeros(len(pivot))
        ax.bar(pivot.index, values, bottom=bottom, label=status, color=color)
        bottom += values
    ax.set_title("Sliced WAV quality")
    ax.set_ylabel("Files")
    ax.legend(frameon=False)
    configure_axes(ax)

    ax = axes[1, 0]
    labels = ["256 mel rows", "512 real rows", "512 reconstructed", "512 full rows"]
    values = [
        mel_summary.get("aistpp_256_sorted_rows") or 0,
        mel_summary.get("aistpp_hires_full_reused_real_rows") or mel_summary.get("aistpp_hires_sorted_real_rows") or 0,
        mel_summary.get("aistpp_hires_full_reconstructed_rows") or 0,
        mel_summary.get("aistpp_hires_full_rows") or 0,
    ]
    colors = ["#2563eb", "#059669", "#dc6803", "#7c3aed"]
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(value):,}", ha="center", va="bottom")
    ax.set_title("Mel dataset materialization")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=15)
    configure_axes(ax)

    ax = axes[1, 1]
    ax.axis("off")
    facts = [
        ["Task type", "continuous conditioned generation"],
        ["Condition tensor", str(condition_info.get("condition_shape"))],
        ["Test condition entries", f"{condition_info.get('test_condition_entries', 'n/a')}"],
        ["Motion channels", f"{condition_info.get('motion_channels', 'n/a')}"],
        ["Extra channels", f"{condition_info.get('extra_condition_channels', 'n/a')}"],
        ["Train condition pickle", f"{condition_info.get('train_condition_pickle_gb', 'n/a')} GB"],
    ]
    table = ax.table(cellText=facts, colLabels=["Presentation fact", "Value"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#d7dde8")
        if row == 0:
            cell.set_facecolor("#111827")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f8fafc")
    ax.set_title("What the model sees", pad=16)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_figure(fig, output_dir, "dataset_overview_dashboard")


def plot_genre_distribution(all_df: pd.DataFrame, output_dir: Path) -> list[str]:
    counts = count_by_genre(all_df)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("AIST++ Genre Coverage By Split", fontsize=17, fontweight="bold")

    ax = axes[0]
    train_test = counts[counts["split"].isin(["train", "test"])].copy()
    split_totals = train_test.groupby("split")["count"].transform("sum").replace(0, np.nan)
    train_test["percent"] = train_test["count"] / split_totals * 100.0
    pivot = train_test.pivot(index="genre", columns="split", values="percent").reindex(GENRE_ORDER).fillna(0)
    x = np.arange(len(pivot))
    ax.bar(x - 0.18, pivot["train"], width=0.36, label="train share", color="#2563eb")
    ax.bar(x + 0.18, pivot["test"], width=0.36, label="test share", color="#059669")
    ax.set_xticks(x, [f"{g}\n{GENRE_NAMES[g]}" for g in GENRE_ORDER], fontsize=8)
    ax.set_title("Train/test genre share")
    ax.set_ylabel("Percent of split")
    ax.legend(frameon=False)
    configure_axes(ax)

    ax = axes[1]
    cdcd = counts[counts["split"] == "cdcd_eval"].set_index("genre").reindex(GENRE_ORDER).fillna(0)
    bars = ax.bar(cdcd.index, cdcd["count"], color=["#dc6803" if g == "WA" else "#7c3aed" for g in cdcd.index])
    for bar, value in zip(bars, cdcd["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(value)}", ha="center", va="bottom")
    ax.set_xticks(x, [f"{g}\n{GENRE_NAMES[g]}" for g in GENRE_ORDER], fontsize=8)
    ax.set_title("CDCD eval subset; WA is absent")
    ax.set_ylabel("Eval wavs")
    configure_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save_figure(fig, output_dir, "genre_distribution")


def plot_sequence_slice_distribution(sequence_df: pd.DataFrame, output_dir: Path) -> list[str]:
    train_seq = sequence_df[sequence_df["split"] == "train"]
    test_seq = sequence_df[sequence_df["split"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("How Raw Dances Become 5-second Training Examples", fontsize=17, fontweight="bold")

    ax = axes[0]
    bins = np.arange(0, max(train_seq["slice_count"].max(), test_seq["slice_count"].max()) + 2) - 0.5
    ax.hist(train_seq["slice_count"], bins=bins, alpha=0.72, label="train base dances", color="#2563eb")
    ax.hist(test_seq["slice_count"], bins=bins, alpha=0.72, label="test base dances", color="#059669")
    ax.set_title("Slices per base dance")
    ax.set_xlabel("Number of 5s slices")
    ax.set_ylabel("Base dance count")
    ax.legend(frameon=False)
    configure_axes(ax)

    ax = axes[1]
    top = train_seq.sort_values("slice_count", ascending=False).head(15).iloc[::-1]
    ax.barh(top["base_key"], top["slice_count"], color="#2563eb")
    ax.set_title("Longest train examples after slicing")
    ax.set_xlabel("5s slices")
    ax.tick_params(axis="y", labelsize=7)
    configure_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save_figure(fig, output_dir, "slice_distribution")


def plot_condition_heatmap(sample: np.ndarray | None, sample_key: str | None, output_dir: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle("Motion Conditioning Tensor Example", fontsize=17, fontweight="bold")
    if sample is None:
        axes[0].text(0.5, 0.5, "No condition sample found", ha="center", va="center")
        axes[1].axis("off")
    else:
        ax = axes[0]
        clipped = np.clip(sample, np.percentile(sample, 1), np.percentile(sample, 99))
        im = ax.imshow(clipped, aspect="auto", cmap="magma", interpolation="nearest")
        ax.axvline(359.5, color="white", linewidth=1.5, linestyle="--")
        ax.set_title(sample_key or "test condition sample")
        ax.set_xlabel("Condition channel: 0-359 motion, 360-369 extra conditioning")
        ax.set_ylabel("Motion frame at 30 fps")
        fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
        configure_axes(ax)

        ax = axes[1]
        ax.hist(sample[:, :360].ravel(), bins=60, color="#2563eb", alpha=0.78, label="motion channels")
        if sample.shape[1] > 360:
            ax.hist(sample[:, 360:].ravel(), bins=30, color="#dc6803", alpha=0.72, label="extra channels")
        ax.set_title("Value distribution")
        ax.set_xlabel("Normalized value")
        ax.set_ylabel("Count")
        ax.legend(frameon=False)
        configure_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save_figure(fig, output_dir, "condition_tensor_example")


def plot_preprocessing_flow(output_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis("off")
    fig.suptitle("Data Flow For Continuous Conditioned Generation", fontsize=17, fontweight="bold")
    steps = [
        ("AIST++ raw dance", "SMPL motion\npaired music"),
        ("5s slicing", "train/test wavs\nslice keys"),
        ("Motion condition", "150 frames\n370 channels"),
        ("Mel image dataset", "256x256 base\n512x512 hires"),
        ("Latent diffusion", "motion -> music\nUNet conditioned"),
        ("Evaluation", "CDCD 31 wavs\nbeat, FAD, KLD"),
    ]
    x_positions = np.linspace(0.07, 0.93, len(steps))
    for idx, ((title, body), x) in enumerate(zip(steps, x_positions)):
        rect = plt.Rectangle((x - 0.07, 0.38), 0.14, 0.24, facecolor="#f8fafc", edgecolor="#64748b", linewidth=1.6)
        ax.add_patch(rect)
        ax.text(x, 0.55, title, ha="center", va="center", fontsize=10, fontweight="bold", color="#111827")
        ax.text(x, 0.44, body, ha="center", va="center", fontsize=9, color="#344054")
        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - 0.08, 0.50),
                xytext=(x + 0.08, 0.50),
                arrowprops={"arrowstyle": "->", "linewidth": 1.8, "color": "#2563eb"},
            )
    ax.text(
        0.5,
        0.18,
        "Assignment task 4 framing: output is continuous audio; conditioning input is the AIST++ motion sequence.",
        ha="center",
        va="center",
        fontsize=11,
        color="#111827",
    )
    return save_figure(fig, output_dir, "preprocessing_flow")


def plot_finedance_context(finedance: dict[str, Any], output_dir: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Other Local Data Present: FineDance", fontsize=17, fontweight="bold")

    ax = axes[0]
    labels = ["FineDance labels", "FineDance motions"]
    values = [finedance.get("label_json_files", 0), finedance.get("motion_files", 0)]
    ax.bar(labels, values, color=["#7c3aed", "#059669"])
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{int(value)}", ha="center", va="bottom")
    ax.set_title("Local FineDance files")
    ax.set_ylabel("Count")
    configure_axes(ax)

    ax = axes[1]
    styles = list(finedance.get("style_counts", {}).items())[:10]
    if styles:
        names = [name for name, _count in styles][::-1]
        counts = [count for _name, count in styles][::-1]
        ax.barh(names, counts, color="#dc6803")
        ax.set_title("FineDance label styles")
        ax.set_xlabel("Label JSON files")
        ax.tick_params(axis="y", labelsize=8)
    else:
        ax.text(0.5, 0.5, "No FineDance labels found", ha="center", va="center")
    configure_axes(ax)
    fig.text(
        0.5,
        0.01,
        "FineDance is useful context, but the DMD/PoseBeat paper-style pipeline here uses AIST++.",
        ha="center",
        color="#344054",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    return save_figure(fig, output_dir, "finedance_context")


def write_report(
    output_dir: Path,
    summary: dict[str, Any],
    generated_files: list[str],
) -> Path:
    report_path = output_dir / "dataset_overview_report.md"
    aist = summary["aistpp"]
    mel = summary["mel_datasets"]
    cond = summary["conditions"]
    fd = summary["finedance"]
    genre_counts = summary["genre_counts"]
    report = f"""# PoseBeat Dataset Overview For Assignment Task 4

Generated: {summary["generated_at"]}

## Assignment Framing

I read the assignment PDF at `../ScoreVision-MIDI/outputs/153 _ 253 2026 Assignment 2 (1).pdf`. The relevant option is task 4, continuous conditioned generation. For this project, the continuous output is generated audio, and the conditioning signal is AIST++ dance motion.

The PDF asks the presentation to cover dataset context, preprocessing, and support the analysis with tables, plots, and statistics. These files are made for that section.

## Outputs

| File | Purpose |
| --- | --- |
| `dataset_overview_dashboard.png` / `.svg` | High-level split size, audio-quality, mel-row, and condition-shape dashboard. |
| `genre_distribution.png` / `.svg` | Genre coverage for train, test, and CDCD eval. |
| `slice_distribution.png` / `.svg` | How raw dances become many 5-second examples. |
| `condition_tensor_example.png` / `.svg` | Example 150x370 motion-conditioning tensor. |
| `preprocessing_flow.png` / `.svg` | Presentation-friendly data-flow diagram. |
| `finedance_context.png` / `.svg` | Shows FineDance is present locally but not used in this DMD-style pipeline. |
| `dataset_overview_summary.json` | Machine-readable summary. |
| `split_genre_counts.csv` | Genre counts by split. |
| `sequence_slice_counts.csv` | Slice count per base dance. |
| `audio_quality_counts.csv` | Tiny/empty versus non-empty sliced wav counts. |

## Key Dataset Facts

| Item | Value |
| --- | ---: |
| Raw AIST++ official `all.txt` entries | {aist["raw"]["official_split_counts"].get("all.txt")} |
| Local raw motion `.pkl` files present | {aist["raw"]["raw_motion_pkls_present"]} |
| Crossmodal train base dances | {aist["raw"]["official_split_counts"].get("crossmodal_train.txt")} |
| Crossmodal val base dances | {aist["raw"]["official_split_counts"].get("crossmodal_val.txt")} |
| Crossmodal test base dances | {aist["raw"]["official_split_counts"].get("crossmodal_test.txt")} |
| Train sliced wav files | {aist["train_slices"]} |
| Train unique base dances | {aist["train_unique_base_dances"]} |
| Test sliced wav files | {aist["test_slices"]} |
| Test unique base dances | {aist["test_unique_base_dances"]} |
| CDCD eval wavs | {aist["cdcd_slices"]} |
| CDCD unique base dances | {aist["cdcd_unique_base_dances"]} |

## Important Preprocessing Finding

| Split | Non-empty WAVs | Tiny/empty WAVs |
| --- | ---: | ---: |
| Train | {aist["train_nonempty_wavs"]} | {aist["train_tiny_or_empty_wavs"]} |
| Test | {aist["test_nonempty_wavs"]} | {aist["test_tiny_or_empty_wavs"]} |

This is why the high-resolution mel preprocessing matters: the real 512x512 mel render only had {mel["aistpp_hires_sorted_real_rows"]} usable rows, then the full high-res dataset was rebuilt to {mel["aistpp_hires_full_rows"]} rows by reconstructing {mel["aistpp_hires_full_reconstructed_rows"]} rows.

## Model Input Representation

| Representation | Value |
| --- | --- |
| Test condition entries | {cond.get("test_condition_entries")} |
| Condition tensor shape | {cond.get("condition_shape")} |
| Motion channels | {cond.get("motion_channels")} |
| Extra condition channels | {cond.get("extra_condition_channels")} |
| Train condition pickle size | {cond.get("train_condition_pickle_gb")} GB |

For presentation: say that each 5-second training/eval example is represented as 150 motion frames at 30 fps, with 370 conditioning channels per frame.

## Mel Dataset Materialization

| Dataset | Rows |
| --- | ---: |
| 256x256 base mel dataset | {mel["aistpp_256_sorted_rows"]} |
| 512x512 real rendered rows | {mel["aistpp_hires_sorted_real_rows"]} |
| 512x512 full rows | {mel["aistpp_hires_full_rows"]} |
| 512x512 reused real rows | {mel["aistpp_hires_full_reused_real_rows"]} |
| 512x512 reconstructed rows | {mel["aistpp_hires_full_reconstructed_rows"]} |

## Genre Coverage Notes

- Train covers all ten AIST++ genre codes: {", ".join(GENRE_ORDER)}.
- Test covers all ten AIST++ genre codes.
- CDCD eval covers nine of ten genre codes; `WA` is absent from the CDCD list used here.
- That means CDCD is useful for beat/FAD/KLD comparison, but it is not a full genre-balanced test set.

CDCD counts: {json.dumps(genre_counts["cdcd_eval"], sort_keys=True)}

## Other Local Dataset

FineDance is present under `data/FineDance`, with {fd["label_json_files"]} label JSON files and {fd["motion_files"]} motion files. It is not used by the DMD paper-style PoseBeat training/evaluation pipeline here, but it can be mentioned as related motion-dance data context.

## Suggested Presentation Talking Points

- Task definition: continuous conditioned generation, motion-to-audio.
- Data source: AIST++ dance/music pairs, processed into 5-second examples.
- Conditioning: each generated waveform is driven by a 150-frame motion tensor.
- Preprocessing risk: many train sliced wavs are tiny/empty; the high-res dataset rebuild explicitly handles this.
- Evaluation data: CDCD is a 31-wav subset, not the entire 186-wav test set.
- Related work: DMD uses AIST++ and reports beat/FAD/genre/BAS-style metrics; our visuals show how our local data supports that protocol.
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def make_summary(
    *,
    data_root: Path,
    wav_df: pd.DataFrame,
    cdcd_df: pd.DataFrame,
    mel: dict[str, Any],
    condition_info: dict[str, Any],
    raw: dict[str, Any],
    finedance: dict[str, Any],
    output_files: list[str],
) -> dict[str, Any]:
    train = wav_df[wav_df["split"] == "train"]
    test = wav_df[wav_df["split"] == "test"]
    all_df = pd.concat([wav_df, cdcd_df], ignore_index=True)
    genre_counts_df = count_by_genre(all_df)
    genre_counts: dict[str, dict[str, int]] = {}
    for split in ["train", "test", "cdcd_eval"]:
        split_df = genre_counts_df[genre_counts_df["split"] == split]
        genre_counts[split] = {row["genre"]: int(row["count"]) for _, row in split_df.iterrows()}

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_root": str(data_root.relative_to(REPO_ROOT)),
        "aistpp": {
            "raw": raw,
            "train_slices": int(len(train)),
            "train_unique_base_dances": int(train["base_key"].nunique()),
            "test_slices": int(len(test)),
            "test_unique_base_dances": int(test["base_key"].nunique()),
            "cdcd_slices": int(len(cdcd_df)),
            "cdcd_unique_base_dances": int(cdcd_df["base_key"].nunique()),
            "train_tiny_or_empty_wavs": int(train["tiny_or_empty"].sum()),
            "train_nonempty_wavs": int((~train["tiny_or_empty"]).sum()),
            "test_tiny_or_empty_wavs": int(test["tiny_or_empty"].sum()),
            "test_nonempty_wavs": int((~test["tiny_or_empty"]).sum()),
        },
        "mel_datasets": mel,
        "conditions": condition_info,
        "finedance": finedance,
        "genre_counts": genre_counts,
        "output_files": sorted(output_files),
    }
    return summary


def write_tables(output_dir: Path, all_df: pd.DataFrame, wav_df: pd.DataFrame) -> list[str]:
    genre_csv = output_dir / "split_genre_counts.csv"
    sequence_csv = output_dir / "sequence_slice_counts.csv"
    quality_csv = output_dir / "audio_quality_counts.csv"
    count_by_genre(all_df).to_csv(genre_csv, index=False)
    sequence_counts(all_df).to_csv(sequence_csv, index=False)
    audio_quality_counts(wav_df).to_csv(quality_csv, index=False)
    return [str(path.relative_to(REPO_ROOT)) for path in (genre_csv, sequence_csv, quality_csv)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cdcd-list", type=Path, default=CDCD_LIST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root if args.data_root.is_absolute() else REPO_ROOT / args.data_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_df = pd.concat(
        [collect_wav_rows(data_root, "train"), collect_wav_rows(data_root, "test")],
        ignore_index=True,
    )
    cdcd_df = collect_cdcd_rows(args.cdcd_list if args.cdcd_list.is_absolute() else REPO_ROOT / args.cdcd_list)
    all_df = pd.concat([wav_df, cdcd_df], ignore_index=True)

    raw = raw_inventory(data_root)
    mel = mel_dataset_summary(data_root, train_wav_count=int((wav_df["split"] == "train").sum()))
    condition_info, condition_sample, condition_key = condition_summary(data_root)
    finedance = finedance_summary(REPO_ROOT)

    generated_files: list[str] = []
    generated_files.extend(plot_dataset_overview_dashboard(wav_df, cdcd_df, mel, condition_info, output_dir))
    generated_files.extend(plot_genre_distribution(all_df, output_dir))
    generated_files.extend(plot_sequence_slice_distribution(sequence_counts(all_df), output_dir))
    generated_files.extend(plot_condition_heatmap(condition_sample, condition_key, output_dir))
    generated_files.extend(plot_preprocessing_flow(output_dir))
    generated_files.extend(plot_finedance_context(finedance, output_dir))
    generated_files.extend(write_tables(output_dir, all_df, wav_df))

    summary = make_summary(
        data_root=data_root,
        wav_df=wav_df,
        cdcd_df=cdcd_df,
        mel=mel,
        condition_info=condition_info,
        raw=raw,
        finedance=finedance,
        output_files=generated_files,
    )
    report_path = write_report(output_dir, summary, generated_files)
    summary["output_files"] = sorted(set(summary["output_files"] + [str(report_path.relative_to(REPO_ROOT))]))
    summary_path = output_dir / "dataset_overview_summary.json"
    summary["output_files"] = sorted(set(summary["output_files"] + [str(summary_path.relative_to(REPO_ROOT))]))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "files": summary["output_files"]}, indent=2))


if __name__ == "__main__":
    main()
