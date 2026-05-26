#!/usr/bin/env python3
"""
Analyze MCM address entropy logger CSVs.

Inputs:
  - mcm_entropy_summary.csv
  - mcm_entropy_distribution.csv
  - mcm_address_bit_entropy.csv

Outputs:
  - text summary report
  - PNG plots for destination/DRAM imbalance and address-bit entropy
"""

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze MCM entropy/imbalance CSV files."
    )
    parser.add_argument(
        "--input-dir",
        default=".",
        help="Directory containing the three MCM entropy CSV files.",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Path to mcm_entropy_summary.csv. Defaults to input-dir file.",
    )
    parser.add_argument(
        "--distribution",
        default=None,
        help="Path to mcm_entropy_distribution.csv. Defaults to input-dir file.",
    )
    parser.add_argument(
        "--bit-entropy",
        default=None,
        help="Path to mcm_address_bit_entropy.csv. Defaults to input-dir file.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Defaults to input-dir/mcm_entropy_analysis.",
    )
    parser.add_argument(
        "--kernel-index",
        type=int,
        default=-1,
        help="Kernel occurrence to analyze. -1 means aggregate all rows in each CSV.",
    )
    parser.add_argument(
        "--top-bits",
        type=int,
        default=16,
        help="Number of highest-entropy and lowest-entropy address bits to list.",
    )
    return parser.parse_args()


def ensure_numeric(df, exclude=("kernel", "source_chiplet", "traffic")):
    for col in df.columns:
        if col not in exclude:
            converted = pd.to_numeric(df[col], errors="coerce")
            if not converted.isna().all():
                df[col] = converted
    return df


def source_sort_key(value):
    if str(value) == "all":
        return 10**9
    try:
        return int(value)
    except ValueError:
        return 10**8


def select_kernel_occurrence(df, kernel_index):
    if kernel_index < 0 or "kernel" not in df.columns:
        return df.copy()

    kernels = list(df["kernel"].drop_duplicates())
    if not kernels:
        return df.copy()
    if kernel_index >= len(kernels):
        raise ValueError(
            f"--kernel-index {kernel_index} is out of range. "
            f"Found {len(kernels)} kernel occurrence(s)."
        )
    return df[df["kernel"] == kernels[kernel_index]].copy()


def aggregate_summary(summary):
    numeric_cols = [
        "total_cycle",
        "total_insn",
        "total_requests",
        "remote_requests",
        "remote_frac",
        "unique_dest_chiplets",
        "dest_chiplet_entropy",
        "remote_dest_chiplet_entropy",
        "max_dest_chiplet_frac",
        "max_remote_dest_chiplet_frac",
        "dominant_dest_chiplet",
        "dominant_remote_dest_chiplet",
        "unique_dram_ids",
        "dram_id_entropy",
        "remote_dram_id_entropy",
        "max_dram_id_frac",
        "max_remote_dram_id_frac",
        "dominant_dram_id",
        "dominant_remote_dram_id",
    ]
    summary = summary.copy()
    for col in numeric_cols:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")
    return summary


def write_report(path, summary, dist, bits, top_bits):
    lines = []
    lines.append("MCM Entropy / Imbalance Analysis")
    lines.append("=" * 36)
    lines.append("")

    all_rows = summary[summary["source_chiplet"].astype(str) == "all"]
    if not all_rows.empty:
        all_row = all_rows.iloc[-1]
    else:
        all_row = summary.iloc[-1] if not summary.empty else None

    if all_row is not None:
        lines.append("Aggregate Traffic")
        lines.append("-" * 17)
        lines.append(f"Total requests: {int(all_row['total_requests']):,}")
        lines.append(f"Remote requests: {int(all_row['remote_requests']):,}")
        lines.append(f"Remote fraction: {all_row['remote_frac']:.4f}")
        lines.append(
            "Remote destination entropy: "
            f"{all_row['remote_dest_chiplet_entropy']:.4f}"
        )
        lines.append(
            "Hottest remote destination fraction: "
            f"{all_row['max_remote_dest_chiplet_frac']:.4f}"
        )
        lines.append(
            "Remote DRAM entropy: "
            f"{all_row['remote_dram_id_entropy']:.4f}"
        )
        lines.append(
            "Hottest remote DRAM fraction: "
            f"{all_row['max_remote_dram_id_frac']:.4f}"
        )
        lines.append(
            "Dominant remote destination chiplet: "
            f"{int(all_row['dominant_remote_dest_chiplet'])}"
        )
        lines.append(
            "Dominant remote DRAM ID: "
            f"{int(all_row['dominant_remote_dram_id'])}"
        )
        lines.append("")

        dest_entropy = all_row["remote_dest_chiplet_entropy"]
        dram_entropy = all_row["remote_dram_id_entropy"]
        dest_hot = all_row["max_remote_dest_chiplet_frac"]
        dram_hot = all_row["max_remote_dram_id_frac"]

        lines.append("Interpretation")
        lines.append("-" * 14)
        if all_row["remote_frac"] < 0.05:
            lines.append(
                "Remote traffic is low in this run, so MCM mapping effects may be hard "
                "to see."
            )
        else:
            lines.append(
                "Remote traffic is present, so destination/DRAM imbalance is meaningful."
            )

        if dest_entropy < 0.5 and dest_hot > 0.8:
            lines.append(
                "Remote destination-chiplet traffic is highly concentrated. This is a "
                "strong chiplet-level valley/hotspot signal."
            )
        elif dest_hot > 0.5:
            lines.append(
                "Remote destination-chiplet traffic is moderately concentrated."
            )
        else:
            lines.append("Remote destination-chiplet traffic is relatively spread.")

        if dram_entropy < 1.0 and dram_hot > 0.5:
            lines.append(
                "Remote DRAM traffic is highly concentrated. This is a strong "
                "memory-partition valley/hotspot signal."
            )
        elif dram_hot > 0.25:
            lines.append("Remote DRAM traffic is moderately concentrated.")
        else:
            lines.append("Remote DRAM traffic is relatively spread.")
        lines.append("")

    per_source = summary[summary["source_chiplet"].astype(str) != "all"].copy()
    if not per_source.empty:
        lines.append("Per-Source Chiplet Hotspots")
        lines.append("-" * 27)
        per_source = per_source.sort_values(
            ["max_remote_dram_id_frac", "max_remote_dest_chiplet_frac"],
            ascending=False,
        )
        cols = [
            "source_chiplet",
            "total_requests",
            "remote_frac",
            "max_remote_dest_chiplet_frac",
            "dominant_remote_dest_chiplet",
            "max_remote_dram_id_frac",
            "dominant_remote_dram_id",
        ]
        lines.append(per_source[cols].to_string(index=False))
        lines.append("")

    if not dist.empty:
        remote_dist = dist[dist["remote_requests"] > 0].copy()
        if not remote_dist.empty:
            lines.append("Top Remote Buckets")
            lines.append("-" * 18)
            remote_dist = remote_dist.sort_values("remote_requests", ascending=False)
            cols = [
                "source_chiplet",
                "dest_chiplet",
                "dram_id",
                "remote_requests",
                "remote_fraction",
            ]
            lines.append(remote_dist[cols].head(15).to_string(index=False))
            lines.append("")

    bit_remote = bits[
        (bits["source_chiplet"].astype(str) == "all") & (bits["traffic"] == "remote")
    ].copy()
    if bit_remote.empty:
        bit_remote = bits[bits["traffic"] == "remote"].copy()

    if not bit_remote.empty:
        bit_remote = bit_remote.sort_values("entropy", ascending=False)
        lines.append("Highest-Entropy Remote Address Bits")
        lines.append("-" * 35)
        lines.append(
            bit_remote[["bit", "one_frac", "entropy", "total"]]
            .head(top_bits)
            .to_string(index=False)
        )
        lines.append("")

        lines.append("Lowest-Entropy Remote Address Bits")
        lines.append("-" * 34)
        lines.append(
            bit_remote[["bit", "one_frac", "entropy", "total"]]
            .sort_values("entropy", ascending=True)
            .head(top_bits)
            .to_string(index=False)
        )
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def save_bar(path, labels, values, title, ylabel):
    plt.figure(figsize=(max(7, len(labels) * 0.7), 4.5))
    plt.bar([str(x) for x in labels], values)
    plt.title(title)
    plt.xlabel("Source chiplet")
    plt.ylabel(ylabel)
    plt.ylim(0, max(1.0, np.nanmax(values) * 1.1 if len(values) else 1.0))
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_heatmap(path, table, title, xlabel, ylabel):
    if table.empty:
        return

    data = table.to_numpy(dtype=float)
    fig_w = max(6, table.shape[1] * 0.65)
    fig_h = max(4, table.shape[0] * 0.55)
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(data, aspect="auto", cmap="viridis")
    plt.colorbar(label="Remote requests")
    plt.xticks(range(table.shape[1]), table.columns, rotation=45, ha="right")
    plt.yticks(range(table.shape[0]), table.index)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if table.shape[0] <= 12 and table.shape[1] <= 24:
        for r in range(table.shape[0]):
            for c in range(table.shape[1]):
                value = data[r, c]
                if value > 0:
                    color = "white" if value > np.nanmax(data) * 0.5 else "black"
                    plt.text(c, r, f"{int(value)}", ha="center", va="center",
                             color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def generate_plots(out_dir, summary, dist, bits):
    per_source = summary[summary["source_chiplet"].astype(str) != "all"].copy()
    if not per_source.empty:
        per_source = per_source.sort_values(
            "source_chiplet", key=lambda s: s.map(source_sort_key)
        )
        save_bar(
            out_dir / "remote_fraction_by_source.png",
            per_source["source_chiplet"],
            per_source["remote_frac"],
            "Remote Traffic Fraction by Source Chiplet",
            "Remote fraction",
        )
        save_bar(
            out_dir / "max_remote_dest_fraction_by_source.png",
            per_source["source_chiplet"],
            per_source["max_remote_dest_chiplet_frac"],
            "Hottest Remote Destination Fraction by Source",
            "Fraction of remote traffic",
        )
        save_bar(
            out_dir / "max_remote_dram_fraction_by_source.png",
            per_source["source_chiplet"],
            per_source["max_remote_dram_id_frac"],
            "Hottest Remote DRAM Fraction by Source",
            "Fraction of remote traffic",
        )

    remote_dist = dist[dist["remote_requests"] > 0].copy()
    if not remote_dist.empty:
        no_all = remote_dist[remote_dist["source_chiplet"].astype(str) != "all"]
        src_dest = no_all.pivot_table(
            index="source_chiplet",
            columns="dest_chiplet",
            values="remote_requests",
            aggfunc="sum",
            fill_value=0,
        )
        src_dest = src_dest.sort_index(key=lambda s: s.map(source_sort_key))
        src_dest = src_dest.reindex(sorted(src_dest.columns), axis=1)
        save_heatmap(
            out_dir / "remote_source_to_dest_heatmap.png",
            src_dest,
            "Remote Requests: Source Chiplet to Destination Chiplet",
            "Destination chiplet",
            "Source chiplet",
        )

        src_dram = no_all.pivot_table(
            index="source_chiplet",
            columns="dram_id",
            values="remote_requests",
            aggfunc="sum",
            fill_value=0,
        )
        src_dram = src_dram.sort_index(key=lambda s: s.map(source_sort_key))
        src_dram = src_dram.reindex(sorted(src_dram.columns), axis=1)
        save_heatmap(
            out_dir / "remote_source_to_dram_heatmap.png",
            src_dram,
            "Remote Requests: Source Chiplet to DRAM ID",
            "DRAM ID",
            "Source chiplet",
        )

        all_remote = remote_dist[remote_dist["source_chiplet"].astype(str) == "all"]
        if not all_remote.empty:
            dest_dram = all_remote.pivot_table(
                index="dest_chiplet",
                columns="dram_id",
                values="remote_requests",
                aggfunc="sum",
                fill_value=0,
            )
            dest_dram = dest_dram.reindex(sorted(dest_dram.index))
            dest_dram = dest_dram.reindex(sorted(dest_dram.columns), axis=1)
            save_heatmap(
                out_dir / "remote_dest_to_dram_heatmap.png",
                dest_dram,
                "Remote Requests: Destination Chiplet to DRAM ID",
                "DRAM ID",
                "Destination chiplet",
            )

    bit_remote_all = bits[
        (bits["source_chiplet"].astype(str) == "all") & (bits["traffic"] == "remote")
    ].copy()
    if bit_remote_all.empty:
        bit_remote_all = bits[bits["traffic"] == "remote"].copy()

    if not bit_remote_all.empty:
        bit_remote_all = (
            bit_remote_all.groupby("bit", as_index=False)
            .agg({"entropy": "mean", "one_frac": "mean", "total": "sum"})
            .sort_values("bit")
        )
        plt.figure(figsize=(10, 4.5))
        plt.plot(bit_remote_all["bit"], bit_remote_all["entropy"], marker="o")
        plt.ylim(-0.02, 1.02)
        plt.xlabel("Address bit")
        plt.ylabel("Entropy")
        plt.title("Remote Address Bit Entropy")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "remote_address_bit_entropy.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10, 4.5))
        plt.plot(bit_remote_all["bit"], bit_remote_all["one_frac"], marker="o")
        plt.axhline(0.5, color="black", linewidth=1, linestyle="--")
        plt.ylim(-0.02, 1.02)
        plt.xlabel("Address bit")
        plt.ylabel("Fraction of ones")
        plt.title("Remote Address Bit One Fraction")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "remote_address_bit_one_fraction.png", dpi=200)
        plt.close()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.out).resolve() if args.out else input_dir / "mcm_entropy_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.summary) if args.summary else input_dir / "mcm_entropy_summary.csv"
    dist_path = (
        Path(args.distribution)
        if args.distribution
        else input_dir / "mcm_entropy_distribution.csv"
    )
    bit_path = (
        Path(args.bit_entropy)
        if args.bit_entropy
        else input_dir / "mcm_address_bit_entropy.csv"
    )

    for path in [summary_path, dist_path, bit_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    summary = ensure_numeric(pd.read_csv(summary_path))
    dist = ensure_numeric(pd.read_csv(dist_path))
    bits = ensure_numeric(pd.read_csv(bit_path))

    summary = select_kernel_occurrence(summary, args.kernel_index)
    dist = select_kernel_occurrence(dist, args.kernel_index)
    bits = select_kernel_occurrence(bits, args.kernel_index)

    summary = aggregate_summary(summary)
    write_report(out_dir / "mcm_entropy_report.txt", summary, dist, bits, args.top_bits)
    generate_plots(out_dir, summary, dist, bits)

    print(f"Analysis saved to: {out_dir}")
    print(f"Text report: {out_dir / 'mcm_entropy_report.txt'}")


if __name__ == "__main__":
    main()
