#!/usr/bin/env python3
"""
Analyze MCM GPU prefetch opportunities from current and future simulator logs.

Current supported inputs:
  - mcm_entropy_summary.csv
  - mcm_entropy_distribution.csv
  - mcm_address_bit_entropy.csv
  - l2_dram_batch_summary.csv

Future supported inputs:
  - mcm_prefetch_window_stats.csv
  - mcm_access_trace.csv

The script writes a text report and PNG figures. It is intentionally tolerant of
missing files so it can be used before the full prefetch instrumentation exists.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KNOWN_FILES = {
    "summary": "mcm_entropy_summary.csv",
    "distribution": "mcm_entropy_distribution.csv",
    "bit_entropy": "mcm_address_bit_entropy.csv",
    "batching": "l2_dram_batch_summary.csv",
    "window": "mcm_prefetch_window_stats.csv",
    "access": "mcm_access_trace.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Characterize MCM local/remote phases and prefetch slack."
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        default=[],
        help="Directory to recursively search for known CSV files.",
    )
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        help="Explicit mcm_entropy_summary.csv path. Can be repeated.",
    )
    parser.add_argument(
        "--distribution",
        action="append",
        default=[],
        help="Explicit mcm_entropy_distribution.csv path. Can be repeated.",
    )
    parser.add_argument(
        "--bit-entropy",
        action="append",
        default=[],
        help="Explicit mcm_address_bit_entropy.csv path. Can be repeated.",
    )
    parser.add_argument(
        "--batching",
        action="append",
        default=[],
        help="Explicit l2_dram_batch_summary.csv path. Can be repeated.",
    )
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        help="Explicit mcm_prefetch_window_stats.csv path. Can be repeated.",
    )
    parser.add_argument(
        "--access",
        action="append",
        default=[],
        help="Explicit mcm_access_trace.csv path. Can be repeated.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Defaults to input-dir/mcm_prefetch_characterization.",
    )
    parser.add_argument(
        "--window-cycles",
        type=int,
        default=1000,
        help="Cycle window size when deriving windows from access traces.",
    )
    parser.add_argument(
        "--window-accesses",
        type=int,
        default=1000,
        help="Access-count window size if an access trace has no cycle column.",
    )
    parser.add_argument(
        "--remote-low",
        type=float,
        default=0.20,
        help="Remote fraction below this is considered remote-slack.",
    )
    parser.add_argument(
        "--remote-high",
        type=float,
        default=0.60,
        help="Remote fraction above this is considered remote-heavy.",
    )
    parser.add_argument(
        "--pressure-high",
        type=float,
        default=0.50,
        help="Queue utilization above this is considered high pressure.",
    )
    return parser.parse_args()


def find_known_files(input_dirs):
    found = {key: [] for key in KNOWN_FILES}
    for input_dir in input_dirs:
        root = Path(input_dir).resolve()
        if not root.is_dir():
            continue
        for key, filename in KNOWN_FILES.items():
            found[key].extend(sorted(p for p in root.glob(f"**/{filename}") if p.is_file()))
    return found


def unique_paths(paths):
    result = []
    seen = set()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def case_name(path):
    parent = path.parent.name
    if parent:
        return parent
    return path.stem


def read_csv_group(paths):
    frames = []
    for path in unique_paths(paths):
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        df["case"] = case_name(path)
        frames.append(normalize_numeric(df))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize_numeric(df):
    df = df.copy()
    for col in df.columns:
        if col in ("kernel", "case", "source_file", "traffic", "access_type", "cache_status"):
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if not converted.isna().all():
            df[col] = converted
    return df


def aggregate_per_source(summary_df):
    if summary_df.empty or "source_chiplet" not in summary_df.columns:
        return pd.DataFrame()
    per_source = summary_df[summary_df["source_chiplet"].astype(str) != "all"].copy()
    if per_source.empty:
        return per_source

    for col in ["total_requests", "remote_requests"]:
        if col in per_source.columns:
            per_source[col] = pd.to_numeric(per_source[col], errors="coerce").fillna(0)

    if {"total_requests", "remote_requests"}.issubset(per_source.columns):
        grouped = (
            per_source.groupby(["case", "source_chiplet"], as_index=False)[
                ["total_requests", "remote_requests"]
            ]
            .sum()
            .reset_index(drop=True)
        )
        grouped["remote_frac"] = [
            compute_remote_fraction(total - remote, remote)
            for total, remote in zip(grouped["total_requests"], grouped["remote_requests"])
        ]
        return grouped

    return per_source


def output_dir(args):
    if args.out:
        return Path(args.out).resolve()
    if args.input_dir:
        return Path(args.input_dir[0]).resolve() / "mcm_prefetch_characterization"
    return Path(".").resolve() / "mcm_prefetch_characterization"


def load_inputs(args):
    found = find_known_files(args.input_dir)
    paths = {
        "summary": found["summary"] + args.summary,
        "distribution": found["distribution"] + args.distribution,
        "bit_entropy": found["bit_entropy"] + args.bit_entropy,
        "batching": found["batching"] + args.batching,
        "window": found["window"] + args.window,
        "access": found["access"] + args.access,
    }
    return {key: read_csv_group(value) for key, value in paths.items()}


def safe_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def compute_remote_fraction(local, remote):
    total = local + remote
    if total <= 0:
        return 0.0
    return float(remote) / float(total)


def fmt_id(value):
    if pd.isna(value):
        return "na"
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value)


def derive_windows_from_access(access_df, window_cycles, window_accesses):
    if access_df.empty:
        return pd.DataFrame()

    df = access_df.copy()
    src_col = safe_col(df, ["source_chiplet", "src_chiplet", "src"])
    dst_col = safe_col(df, ["dest_chiplet", "dst_chiplet", "destination_chiplet", "dest"])
    if src_col is None or dst_col is None:
        return pd.DataFrame()

    df["is_remote"] = df[src_col].astype(str) != df[dst_col].astype(str)
    if "cycle" in df.columns:
        df["window_start"] = (pd.to_numeric(df["cycle"], errors="coerce").fillna(0) // window_cycles) * window_cycles
    else:
        df["window_start"] = (np.arange(len(df)) // window_accesses) * window_accesses

    group_cols = ["case", "window_start"]
    if "kernel" in df.columns:
        group_cols.insert(1, "kernel")

    grouped = df.groupby(group_cols, dropna=False)
    rows = []
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_cols, group_key))
        row["local_requests"] = int((~group["is_remote"]).sum())
        row["remote_requests"] = int(group["is_remote"].sum())
        row["total_requests"] = int(len(group))
        row["remote_frac"] = compute_remote_fraction(row["local_requests"], row["remote_requests"])
        if "cycle" in group.columns:
            row["window_end"] = int(pd.to_numeric(group["cycle"], errors="coerce").max())
        rows.append(row)

    return pd.DataFrame(rows)


def normalize_window_df(window_df):
    if window_df.empty:
        return pd.DataFrame()

    df = window_df.copy()
    local_col = safe_col(df, ["local_requests", "local_reqs", "local_accesses"])
    remote_col = safe_col(df, ["remote_requests", "remote_reqs", "remote_accesses"])
    total_col = safe_col(df, ["total_requests", "requests", "accesses"])
    remote_frac_col = safe_col(df, ["remote_frac", "remote_fraction"])

    if remote_frac_col is None and local_col and remote_col:
        df["remote_frac"] = [
            compute_remote_fraction(l, r)
            for l, r in zip(pd.to_numeric(df[local_col], errors="coerce").fillna(0), pd.to_numeric(df[remote_col], errors="coerce").fillna(0))
        ]
    elif remote_frac_col and remote_frac_col != "remote_frac":
        df["remote_frac"] = pd.to_numeric(df[remote_frac_col], errors="coerce").fillna(0)

    if local_col and local_col != "local_requests":
        df["local_requests"] = pd.to_numeric(df[local_col], errors="coerce").fillna(0)
    if remote_col and remote_col != "remote_requests":
        df["remote_requests"] = pd.to_numeric(df[remote_col], errors="coerce").fillna(0)
    if total_col and total_col != "total_requests":
        df["total_requests"] = pd.to_numeric(df[total_col], errors="coerce").fillna(0)
    elif "total_requests" not in df.columns and {"local_requests", "remote_requests"}.issubset(df.columns):
        df["total_requests"] = df["local_requests"] + df["remote_requests"]

    if "window_start" not in df.columns:
        time_col = safe_col(df, ["cycle", "start_cycle", "window", "interval"])
        if time_col:
            df["window_start"] = pd.to_numeric(df[time_col], errors="coerce").fillna(0)
        else:
            df["window_start"] = df.groupby("case").cumcount()

    if "source_chiplet" in df.columns:
        all_rows = df[df["source_chiplet"].astype(str) == "all"].copy()
        if not all_rows.empty:
            df = all_rows

    return df


def latest_summary_rows(summary_df):
    if summary_df.empty:
        return pd.DataFrame()
    df = summary_df.copy()
    sort_cols = [c for c in ["case", "total_cycle", "total_insn"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    all_rows = df[df.get("source_chiplet", "").astype(str) == "all"] if "source_chiplet" in df.columns else df
    if all_rows.empty:
        all_rows = df
    return all_rows.groupby("case", as_index=False).tail(1).reset_index(drop=True)


def write_report(out_path, data, windows, args):
    summary = data["summary"]
    distribution = data["distribution"]
    bit_entropy = data["bit_entropy"]
    batching = data["batching"]

    lines = []
    lines.append("MCM Prefetch Opportunity Characterization")
    lines.append("=========================================")
    lines.append("")

    lines.append("Input Files")
    lines.append("-----------")
    for key, df in data.items():
        if df.empty:
            lines.append(f"{key}: not found")
        else:
            files = sorted(df["source_file"].drop_duplicates())
            lines.append(f"{key}: {len(files)} file(s), {len(df):,} row(s)")
            for path in files[:5]:
                lines.append(f"  {path}")
            if len(files) > 5:
                lines.append(f"  ... {len(files) - 5} more")
    lines.append("")

    if not summary.empty:
        latest = latest_summary_rows(summary)
        lines.append("Aggregate Remote Traffic")
        lines.append("------------------------")
        for _, row in latest.sort_values("case").iterrows():
            remote_frac = float(row.get("remote_frac", 0.0))
            remote_reqs = int(row.get("remote_requests", 0))
            total_reqs = int(row.get("total_requests", 0))
            hot_dest = float(row.get("max_remote_dest_chiplet_frac", 0.0))
            hot_dram = float(row.get("max_remote_dram_id_frac", 0.0))
            lines.append(
                f"{row['case']}: remote_frac={remote_frac:.4f}, "
                f"remote={remote_reqs:,}/{total_reqs:,}, "
                f"hottest_remote_dest={hot_dest:.4f}, "
                f"hottest_remote_dram={hot_dram:.4f}"
            )
        lines.append("")

        per_source = aggregate_per_source(summary)
        if not per_source.empty:
            per_source["remote_frac"] = pd.to_numeric(per_source["remote_frac"], errors="coerce").fillna(0)
            hot = per_source.sort_values("remote_frac", ascending=False).head(8)
            lines.append("Most Remote-Heavy Source Chiplets")
            lines.append("---------------------------------")
            for _, row in hot.iterrows():
                lines.append(
                    f"{row['case']} source {fmt_id(row['source_chiplet'])}: "
                    f"remote_frac={float(row['remote_frac']):.4f}, "
                    f"remote_requests={int(row.get('remote_requests', 0)):,}"
                )
            lines.append("")

    if not windows.empty and "remote_frac" in windows.columns:
        lines.append("Phase / Slack Windows")
        lines.append("---------------------")
        for case, group in windows.groupby("case"):
            remote_frac = pd.to_numeric(group["remote_frac"], errors="coerce").fillna(0)
            low = int((remote_frac <= args.remote_low).sum())
            high = int((remote_frac >= args.remote_high).sum())
            low_pressure = count_low_pressure_windows(group, args.pressure_high)
            lines.append(
                f"{case}: {len(group):,} window(s), "
                f"{low:,} remote-slack <= {args.remote_low:.2f}, "
                f"{high:,} remote-heavy >= {args.remote_high:.2f}, "
                f"mean_remote_frac={remote_frac.mean():.4f}, "
                f"{low_pressure:,} low-pressure window(s)"
            )

            seq_score = count_slack_before_remote(group, args.remote_low, args.remote_high)
            safe_seq_score = count_safe_slack_before_remote(
                group, args.remote_low, args.remote_high, args.pressure_high
            )
            lines.append(
                f"  slack-before-remote transitions within next 3 windows: {seq_score:,}"
            )
            if has_pressure_columns(group):
                lines.append(
                    "  low-pressure slack-before-remote transitions within next "
                    f"3 windows: {safe_seq_score:,}"
                )
        lines.append("")
    else:
        lines.append("Phase / Slack Windows")
        lines.append("---------------------")
        lines.append(
            "No per-window or per-access timing CSV was found. Current analysis can "
            "show static remote imbalance, but it cannot yet prove prefetch timing "
            "opportunities."
        )
        lines.append("")

    if not batching.empty:
        latest_batch = batching.groupby("case", as_index=False).tail(1)
        lines.append("L2/DRAM Batching Signals")
        lines.append("------------------------")
        for _, row in latest_batch.sort_values("case").iterrows():
            req_red = 100.0 * float(row.get("request_packet_reduction", 0.0))
            rep_red = 100.0 * float(row.get("reply_packet_reduction", 0.0))
            lines.append(
                f"{row['case']}: request_packet_reduction={req_red:.2f}%, "
                f"reply_packet_reduction={rep_red:.2f}%"
            )
        lines.append("")

    if not bit_entropy.empty:
        remote_bits = bit_entropy[bit_entropy.get("traffic", "") == "remote"].copy()
        if not remote_bits.empty:
            remote_bits["entropy"] = pd.to_numeric(remote_bits["entropy"], errors="coerce")
            low_bits = remote_bits.sort_values("entropy").head(8)
            lines.append("Lowest-Entropy Remote Address Bits")
            lines.append("----------------------------------")
            for _, row in low_bits.iterrows():
                lines.append(
                    f"{row['case']} source {fmt_id(row.get('source_chiplet', 'all'))} "
                    f"bit {int(row['bit'])}: entropy={float(row['entropy']):.4f}, "
                    f"one_frac={float(row.get('one_frac', 0.0)):.4f}"
                )
            lines.append("")

    lines.append("Interpretation")
    lines.append("--------------")
    lines.append(
        "A prefetcher is worth implementing only if the static imbalance is paired "
        "with time-varying slack. The strongest signal is remote-heavy demand "
        "appearing shortly after remote-slack windows, especially when queue, MSHR, "
        "NoC, or cache occupancy is low during the slack window."
    )
    lines.append(
        "If only aggregate entropy files are available, the next simulator change "
        "should be a compact per-window logger rather than the prefetcher itself."
    )

    out_path.write_text("\n".join(lines) + "\n")


def count_slack_before_remote(group, low_threshold, high_threshold):
    group = group.sort_values("window_start")
    values = pd.to_numeric(group["remote_frac"], errors="coerce").fillna(0).to_numpy()
    count = 0
    for index, value in enumerate(values):
        if value > low_threshold:
            continue
        lookahead = values[index + 1 : index + 4]
        if len(lookahead) and np.any(lookahead >= high_threshold):
            count += 1
    return count


def has_pressure_columns(df):
    pressure_cols = [
        "remote_request_queue_util",
        "l2_dram_queue_util",
        "dram_queue_util",
        "interchiplet_busy",
        "icnt_busy",
    ]
    return any(col in df.columns for col in pressure_cols)


def pressure_is_low(row, high_threshold):
    util_cols = [
        "remote_request_queue_util",
        "l2_dram_queue_util",
        "dram_queue_util",
    ]
    for col in util_cols:
        if col in row and pd.notna(row[col]) and float(row[col]) > high_threshold:
            return False
    for col in ["interchiplet_busy", "icnt_busy"]:
        if col in row and pd.notna(row[col]) and int(row[col]) != 0:
            return False
    return True


def count_low_pressure_windows(group, high_threshold):
    if not has_pressure_columns(group):
        return 0
    count = 0
    for _, row in group.iterrows():
        if pressure_is_low(row, high_threshold):
            count += 1
    return count


def count_safe_slack_before_remote(group, low_threshold, high_threshold, pressure_high):
    if not has_pressure_columns(group):
        return 0
    group = group.sort_values("window_start").reset_index(drop=True)
    remote_frac = pd.to_numeric(group["remote_frac"], errors="coerce").fillna(0).to_numpy()
    count = 0
    for index, value in enumerate(remote_frac):
        if value > low_threshold:
            continue
        if not pressure_is_low(group.iloc[index], pressure_high):
            continue
        lookahead = remote_frac[index + 1 : index + 4]
        if len(lookahead) and np.any(lookahead >= high_threshold):
            count += 1
    return count


def save_bar(df, x_col, y_cols, title, ylabel, out_path, scale=1.0):
    existing = [col for col in y_cols if col in df.columns]
    if df.empty or not existing or x_col not in df.columns:
        return
    plot_df = df[[x_col] + existing].copy()
    for col in existing:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce").fillna(0) * scale
    ax = plot_df.set_index(x_col)[existing].plot(kind="bar", figsize=(max(8, len(plot_df) * 1.2), 4.8))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x_col)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_line(df, x_col, y_cols, title, ylabel, out_path):
    existing = [col for col in y_cols if col in df.columns]
    if df.empty or not existing or x_col not in df.columns:
        return
    ax = plt.figure(figsize=(10, 4.8)).gca()
    for case, group in df.sort_values(x_col).groupby("case"):
        for col in existing:
            y = pd.to_numeric(group[col], errors="coerce").fillna(0)
            ax.plot(group[x_col], y, label=f"{case}:{col}", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_scatter(df, x_col, y_col, title, xlabel, ylabel, out_path):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return
    ax = plt.figure(figsize=(7.5, 5.2)).gca()
    for case, group in df.groupby("case"):
        x = pd.to_numeric(group[x_col], errors="coerce").fillna(0)
        y = pd.to_numeric(group[y_col], errors="coerce").fillna(0)
        ax.scatter(x, y, s=18, alpha=0.75, label=case)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_plots(out_dir, data, windows):
    summary = data["summary"]
    distribution = data["distribution"]

    if not summary.empty:
        latest = latest_summary_rows(summary)
        save_bar(
            latest,
            "case",
            ["remote_frac"],
            "Aggregate Remote Traffic Fraction",
            "Remote fraction",
            out_dir / "remote_fraction_by_case.png",
        )
        save_bar(
            latest,
            "case",
            ["max_remote_dest_chiplet_frac", "max_remote_dram_id_frac"],
            "Remote Hotspot Strength",
            "Fraction of remote traffic",
            out_dir / "remote_hotspot_summary.png",
        )

        per_source = aggregate_per_source(summary)
        if not per_source.empty:
            per_source["case_source"] = per_source.apply(
                lambda row: f"{row['case']}:S{fmt_id(row['source_chiplet'])}",
                axis=1,
            )
            save_bar(
                per_source,
                "case_source",
                ["remote_frac"],
                "Remote Fraction By Source Chiplet",
                "Remote fraction",
                out_dir / "per_source_remote_fraction.png",
            )

    if not distribution.empty and {"source_chiplet", "dest_chiplet", "remote_requests"}.issubset(distribution.columns):
        all_dist = distribution[distribution["source_chiplet"].astype(str) != "all"].copy()
        if not all_dist.empty:
            pivot = all_dist.pivot_table(
                values="remote_requests",
                index="source_chiplet",
                columns="dest_chiplet",
                aggfunc="sum",
                fill_value=0,
            )
            plt.figure(figsize=(6.8, 5.4))
            plt.imshow(pivot.values, aspect="auto", cmap="viridis")
            plt.colorbar(label="Remote requests")
            plt.title("Remote Requests: Source Chiplet To Destination Chiplet")
            plt.xlabel("Destination chiplet")
            plt.ylabel("Source chiplet")
            plt.xticks(range(len(pivot.columns)), pivot.columns)
            plt.yticks(range(len(pivot.index)), pivot.index)
            plt.tight_layout()
            plt.savefig(out_dir / "remote_source_dest_heatmap.png", dpi=200)
            plt.close()

    if not windows.empty and "remote_frac" in windows.columns:
        save_line(
            windows,
            "window_start",
            ["remote_frac"],
            "Remote Fraction Over Time",
            "Remote fraction",
            out_dir / "phase_remote_fraction_timeline.png",
        )
        save_line(
            windows,
            "window_start",
            ["local_requests", "remote_requests"],
            "Local vs Remote Requests Over Time",
            "Requests",
            out_dir / "local_remote_requests_timeline.png",
        )

        local_bw_col = safe_col(windows, ["local_bw_util", "local_dram_bw_util", "local_bandwidth_util"])
        remote_bw_col = safe_col(windows, ["remote_bw_util", "remote_dram_bw_util", "remote_bandwidth_util"])
        if local_bw_col and remote_bw_col:
            save_scatter(
                windows,
                local_bw_col,
                remote_bw_col,
                "Local vs Remote Bandwidth Utilization",
                local_bw_col,
                remote_bw_col,
                out_dir / "bandwidth_slack_scatter.png",
            )

        pressure_cols = [
            "mshr_occupancy",
            "memory_queue_occupancy",
            "dram_queue_occupancy",
            "remote_request_queue_util",
            "l2_dram_queue_util",
            "dram_queue_util",
            "remote_ingress_pending",
            "remote_request_queue_pending",
            "l2_icnt_remote_pending",
            "request_tracker_pending",
            "noc_util",
            "l2_occupancy",
        ]
        existing_pressure = [col for col in pressure_cols if col in windows.columns]
        if existing_pressure:
            save_line(
                windows,
                "window_start",
                existing_pressure,
                "Queue / Cache / Network Pressure Over Time",
                "Occupancy or utilization",
                out_dir / "mshr_or_queue_pressure.png",
            )


def main():
    args = parse_args()
    if not args.input_dir and not any(
        [
            args.summary,
            args.distribution,
            args.bit_entropy,
            args.batching,
            args.window,
            args.access,
        ]
    ):
        raise SystemExit("Provide --input-dir or one or more explicit CSV paths.")

    out_dir = output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_inputs(args)
    windows = normalize_window_df(data["window"])
    if windows.empty:
        windows = derive_windows_from_access(data["access"], args.window_cycles, args.window_accesses)
    windows = normalize_window_df(windows)

    if not windows.empty:
        windows.to_csv(out_dir / "derived_or_loaded_windows.csv", index=False)

    write_report(out_dir / "mcm_prefetch_opportunity_report.txt", data, windows, args)
    make_plots(out_dir, data, windows)

    print(f"Analysis written to: {out_dir}")
    print(f"Text report: {out_dir / 'mcm_prefetch_opportunity_report.txt'}")


if __name__ == "__main__":
    main()
