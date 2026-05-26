#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def find_csvs(input_dir, filename):
    paths = sorted(Path(input_dir).resolve().glob(f"**/{filename}"))
    return [p for p in paths if p.is_file()]


def read_inputs(args):
    paths = []
    if args.csv:
        paths.extend(Path(p).resolve() for p in args.csv)
    if args.input_dir:
        paths.extend(find_csvs(args.input_dir, args.filename))

    unique_paths = []
    seen = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)

    frames = []
    for path in unique_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        parent = path.parent.name
        df["case"] = parent if parent else path.stem
        frames.append(df)

    if not frames:
        raise SystemExit("No batching CSV files found.")

    return pd.concat(frames, ignore_index=True)


def latest_per_case(df):
    sort_cols = ["case", "total_cycle", "total_insn"]
    existing = [c for c in sort_cols if c in df.columns]
    if existing:
        df = df.sort_values(existing)
    return df.groupby("case", as_index=False).tail(1).reset_index(drop=True)


def percent(series):
    return 100.0 * series


def write_report(path, df):
    lines = []
    lines.append("L2/DRAM Inter-Chiplet Batching Report")
    lines.append("======================================")
    lines.append("")
    lines.append(f"Cases analyzed: {len(df)}")
    lines.append("")

    if "interval_ipc" in df.columns and len(df):
        baseline = df[df["case"].str.contains("baseline", case=False, na=False)]
        base_ipc = None
        if len(baseline):
            base_ipc = float(baseline.iloc[-1]["interval_ipc"])
            lines.append(f"Baseline IPC: {base_ipc:.6f}")
        lines.append("")
        lines.append("Performance")
        lines.append("-----------")
        for _, row in df.sort_values("case").iterrows():
            ipc = float(row.get("interval_ipc", 0.0))
            if base_ipc and base_ipc > 0:
                speedup = ipc / base_ipc
                lines.append(f"{row['case']}: IPC {ipc:.6f}, speedup {speedup:.4f}x")
            else:
                lines.append(f"{row['case']}: IPC {ipc:.6f}")
        lines.append("")

    lines.append("Batching Effects")
    lines.append("----------------")
    for _, row in df.sort_values("case").iterrows():
        lines.append(f"\n{row['case']}:")
        lines.append(
            "  Request packets: "
            f"{int(row.get('request_baseline_packets', 0)):,} baseline -> "
            f"{int(row.get('request_actual_packets', 0)):,} actual "
            f"({100.0 * float(row.get('request_packet_reduction', 0.0)):.2f}% reduction)"
        )
        lines.append(
            "  Request bytes: "
            f"{int(row.get('request_bytes_before', 0)):,} before -> "
            f"{int(row.get('request_bytes_after', 0)):,} after "
            f"({100.0 * float(row.get('request_byte_savings', 0.0)):.2f}% savings)"
        )
        lines.append(
            "  Request batches: "
            f"{int(row.get('request_batches', 0)):,}, "
            f"avg size {float(row.get('avg_request_batch_size', 0.0)):.2f}"
        )
        lines.append(
            "  Reply packets: "
            f"{int(row.get('reply_baseline_packets', 0)):,} baseline -> "
            f"{int(row.get('reply_actual_packets', 0)):,} actual "
            f"({100.0 * float(row.get('reply_packet_reduction', 0.0)):.2f}% reduction)"
        )
        lines.append(
            "  Reply bytes: "
            f"{int(row.get('reply_bytes_before', 0)):,} before -> "
            f"{int(row.get('reply_bytes_after', 0)):,} after "
            f"({100.0 * float(row.get('reply_byte_savings', 0.0)):.2f}% savings)"
        )
        lines.append(
            "  Reply batches: "
            f"{int(row.get('reply_batches', 0)):,}, "
            f"avg size {float(row.get('avg_reply_batch_size', 0.0)):.2f}"
        )

    lines.append("")
    lines.append("Interpretation")
    lines.append("--------------")
    lines.append(
        "Packet reduction shows how much NoP arbitration/injection pressure batching "
        "removed. Byte savings shows modeled traffic-volume reduction. If IPC improves "
        "mostly when reply batching is enabled, the reply network was likely important; "
        "if request-only improves, small request packet pressure was likely important."
    )

    path.write_text("\n".join(lines) + "\n")


def plot_bar(df, columns, title, ylabel, out_path, scale=1.0):
    existing = [c for c in columns if c in df.columns]
    if not existing:
        return
    plot_df = df.set_index("case")[existing] * scale
    ax = plot_df.plot(kind="bar", figsize=(max(8, len(df) * 1.4), 4.8))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Case")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_plots(out_dir, df):
    plot_bar(
        df,
        ["request_packet_reduction", "reply_packet_reduction"],
        "Packet Count Reduction",
        "Reduction (%)",
        out_dir / "packet_reduction.png",
        scale=100.0,
    )
    plot_bar(
        df,
        ["request_byte_savings", "reply_byte_savings"],
        "Modeled Byte Savings",
        "Savings (%)",
        out_dir / "byte_savings.png",
        scale=100.0,
    )
    plot_bar(
        df,
        ["avg_request_batch_size", "avg_reply_batch_size"],
        "Average Multi-Request Batch Size",
        "mem_fetches per batch",
        out_dir / "avg_batch_size.png",
    )
    plot_bar(
        df,
        ["interval_ipc"],
        "Interval IPC",
        "IPC",
        out_dir / "ipc.png",
    )

    packet_cols = [
        "request_baseline_packets",
        "request_actual_packets",
        "reply_baseline_packets",
        "reply_actual_packets",
    ]
    plot_bar(
        df,
        packet_cols,
        "Baseline vs Actual Packets",
        "Packets",
        out_dir / "packet_counts.png",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze L2/DRAM inter-chiplet batching summary CSV files."
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory to recursively search for l2_dram_batch_summary.csv.",
    )
    parser.add_argument(
        "--filename",
        default="l2_dram_batch_summary.csv",
        help="CSV filename to search for with --input-dir.",
    )
    parser.add_argument(
        "--csv",
        action="append",
        help="Explicit CSV path. Can be used multiple times.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Defaults to input-dir/l2_dram_batch_analysis.",
    )
    args = parser.parse_args()

    if not args.input_dir and not args.csv:
        raise SystemExit("Provide --input-dir or at least one --csv.")

    df = read_inputs(args)
    latest = latest_per_case(df)

    if args.out:
        out_dir = Path(args.out).resolve()
    elif args.input_dir:
        out_dir = Path(args.input_dir).resolve() / "l2_dram_batch_analysis"
    else:
        out_dir = Path(".").resolve() / "l2_dram_batch_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    latest.to_csv(out_dir / "batching_latest_by_case.csv", index=False)
    write_report(out_dir / "l2_dram_batch_report.txt", latest)
    make_plots(out_dir, latest)

    print(f"Analysis written to: {out_dir}")
    print(f"Text report: {out_dir / 'l2_dram_batch_report.txt'}")


if __name__ == "__main__":
    main()
