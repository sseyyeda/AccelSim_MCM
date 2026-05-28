#!/usr/bin/env python3
"""Analyze ACCELSIM_REQUEST_TRACE CSV output."""

import argparse
import csv
import sys
from contextlib import redirect_stdout
from collections import Counter, defaultdict


class TeeOutput:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_int(value):
    value = value.strip()
    if value.startswith(("0x", "0X")):
        return int(value, 16)
    return int(value)


def load_rows(path, remote_only):
    rows = []
    with open(path, newline="") as trace_file:
        reader = csv.DictReader(trace_file)
        for row in reader:
            parsed = {key: parse_int(value) for key, value in row.items()}
            if remote_only and parsed["is_local"]:
                continue
            rows.append(parsed)
    rows.sort(key=lambda item: (item["cycle"], item["request_uid"]))
    return rows


def print_counter(title, counter, total, top):
    print("\n" + title)
    print("-" * len(title))
    for key, count in counter.most_common(top):
        pct = 100.0 * count / total if total else 0.0
        print(f"{key}: {count} ({pct:.2f}%)")


def summarize_basic(rows, top):
    total = len(rows)
    local = sum(1 for row in rows if row["is_local"])
    remote = total - local
    print("Basic Summary")
    print("-------------")
    print(f"requests: {total}")
    print(f"local_requests: {local}")
    print(f"remote_requests: {remote}")
    if total:
        print(f"remote_fraction: {remote / total:.4f}")
        print(f"first_cycle: {rows[0]['cycle']}")
        print(f"last_cycle: {rows[-1]['cycle']}")
    print_counter(
        "Top Source->Destination Routes",
        Counter((row["source_chiplet"], row["dest_chiplet"]) for row in rows),
        total,
        top,
    )
    print_counter("Top Destinations", Counter(row["dest_chiplet"] for row in rows), total, top)
    print_counter("Top Sources", Counter(row["source_chiplet"] for row in rows), total, top)


def summarize_locality(rows, top):
    total = len(rows)
    pages = Counter(row["page_start"] for row in rows)
    l1_lines = Counter(row["l1_line_start"] for row in rows)
    l2_lines = Counter(row["l2_line_start"] for row in rows)

    print("\nLocality Summary")
    print("----------------")
    print(f"unique_4kb_pages: {len(pages)}")
    print(f"unique_l1_lines: {len(l1_lines)}")
    print(f"unique_l2_lines: {len(l2_lines)}")
    print(f"avg_requests_per_page: {total / len(pages):.4f}" if pages else "avg_requests_per_page: 0.0000")
    print(f"avg_requests_per_l1_line: {total / len(l1_lines):.4f}" if l1_lines else "avg_requests_per_l1_line: 0.0000")
    print(f"avg_requests_per_l2_line: {total / len(l2_lines):.4f}" if l2_lines else "avg_requests_per_l2_line: 0.0000")
    print(f"requests_crossing_4kb_pages: {sum(1 for row in rows if row['page_start'] != row['page_end'])}")
    print(f"requests_crossing_l1_lines: {sum(1 for row in rows if row['l1_line_start'] != row['l1_line_end'])}")
    print(f"requests_crossing_l2_lines: {sum(1 for row in rows if row['l2_line_start'] != row['l2_line_end'])}")
    print_counter("Top 4KB Pages", pages, total, top)
    print_counter("Top L1 Lines", l1_lines, total, top)
    print_counter("Top L2 Lines", l2_lines, total, top)


def summarize_batching(rows, window, top, header_bytes, data_bytes):
    if not rows:
        return
    first_cycle = rows[0]["cycle"]
    groups = defaultdict(list)
    for row in rows:
        window_id = (row["cycle"] - first_cycle) // window
        groups[(row["source_chiplet"], window_id, row["dest_chiplet"])].append(row)

    batch_sizes = [len(batch) for batch in groups.values()]
    multi = [size for size in batch_sizes if size > 1]
    same_page = sum(1 for batch in groups.values() if len(batch) > 1 and len({row["page_start"] for row in batch}) == 1)
    same_l2 = sum(1 for batch in groups.values() if len(batch) > 1 and len({row["l2_line_start"] for row in batch}) == 1)
    avg_batch = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0.0
    avg_multi = sum(multi) / len(multi) if multi else 0.0
    gain = (header_bytes + data_bytes) / (header_bytes / avg_batch + data_bytes) if avg_batch else 1.0

    route_counts = Counter()
    for (source, _window, dest), batch in groups.items():
        route_counts[(source, dest)] += len(batch)

    print("\nDestination Batching Summary")
    print("----------------------------")
    print(f"window_cycles: {window}")
    print(f"candidate_batches: {len(batch_sizes)}")
    print(f"candidate_batches_size_gt_1: {len(multi)}")
    print(f"avg_batch_size: {avg_batch:.4f}")
    print(f"avg_batch_size_when_gt_1: {avg_multi:.4f}")
    print(f"max_batch_size: {max(batch_sizes) if batch_sizes else 0}")
    print(f"same_page_multi_request_batches: {same_page}")
    print(f"same_l2_line_multi_request_batches: {same_l2}")
    print(f"estimated_header_efficiency_gain: {gain:.4f} (H={header_bytes}, D={data_bytes})")
    print_counter("Top Batched Routes", route_counts, len(rows), top)


def summarize_stride_predictability(rows, top):
    by_route = defaultdict(list)
    for row in rows:
        by_route[(row["source_chiplet"], row["dest_chiplet"])].append(row["l2_line_start"])

    scores = []
    for route, lines in by_route.items():
        if len(lines) < 3:
            continue
        deltas = [lines[i] - lines[i - 1] for i in range(1, len(lines))]
        delta, count = Counter(deltas).most_common(1)[0]
        scores.append((count / len(deltas), route, delta, len(lines)))

    print("\nSimple Stride Predictability")
    print("----------------------------")
    print("score = fraction of consecutive L2-line deltas matching most common delta")
    for score, route, delta, count in sorted(scores, reverse=True)[:top]:
        print(f"{route}: score={score:.4f}, common_delta={delta}, requests={count}")


def build_local_remote_series(rows, window):
    if not rows:
        return []
    first_cycle = rows[0]["cycle"]
    last_cycle = rows[-1]["cycle"]
    series = [
        {
            "window_id": window_id,
            "cycle_start": first_cycle + window_id * window,
            "cycle_end": first_cycle + (window_id + 1) * window - 1,
            "local": 0,
            "remote": 0,
            "total": 0,
        }
        for window_id in range(((last_cycle - first_cycle) // window) + 1)
    ]
    for row in rows:
        bucket = series[(row["cycle"] - first_cycle) // window]
        bucket["local" if row["is_local"] else "remote"] += 1
        bucket["total"] += 1
    return series


def write_series(series, path):
    with open(path, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["window_id", "cycle_start", "cycle_end", "local", "remote", "total"])
        writer.writeheader()
        writer.writerows(series)


def plot_series(series, path, window):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for --plot-out") from exc

    x_values = [item["cycle_start"] for item in series]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_values, [item["remote"] for item in series], label="Remote accesses", linewidth=1.8)
    ax.plot(x_values, [item["local"] for item in series], label="Local accesses", linewidth=1.8)
    ax.set_title(f"Local vs Remote Memory Requests, window={window} cycles")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Requests per window")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_csv")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--remote-only", action="store_true")
    parser.add_argument("--header-bytes", type=float, default=16.0)
    parser.add_argument("--data-bytes", type=float, default=64.0)
    parser.add_argument("--time-window", type=int)
    parser.add_argument("--series-out")
    parser.add_argument("--plot-out")
    parser.add_argument("--summary-out")
    args = parser.parse_args()

    if args.summary_out:
        with open(args.summary_out, "w") as summary_file:
            with redirect_stdout(TeeOutput(sys.stdout, summary_file)):
                run_analysis(args)
        return

    run_analysis(args)


def run_analysis(args):
    rows = load_rows(args.trace_csv, args.remote_only)
    summarize_basic(rows, args.top)
    summarize_locality(rows, args.top)
    summarize_batching(rows, args.window, args.top, args.header_bytes, args.data_bytes)
    summarize_stride_predictability(rows, args.top)

    if args.series_out or args.plot_out:
        series_window = args.time_window or args.window
        series = build_local_remote_series(rows, series_window)
        if args.series_out:
            write_series(series, args.series_out)
            print(f"\nWrote local/remote time series CSV: {args.series_out}")
        if args.plot_out:
            plot_series(series, args.plot_out, series_window)
            print(f"Wrote local/remote line graph: {args.plot_out}")


if __name__ == "__main__":
    main()
