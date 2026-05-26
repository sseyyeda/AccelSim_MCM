#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


DEFAULT_PATTERNS = [
    "icnt",
    "inter",
    "flit",
    "packet",
    "traffic",
    "stall",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract selected Accel-Sim/GPGPU-Sim log lines from each results "
            "configuration directory into one CSV file."
        )
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing one subdirectory per configuration.",
    )
    parser.add_argument(
        "--out",
        default="result.csv",
        help="Output CSV filename.",
    )
    parser.add_argument(
        "--log-name",
        action="append",
        default=["sim_stdout.log", "sim.log"],
        help=(
            "Log filename to search for inside each configuration directory. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help=(
            "Case-insensitive regex pattern to match. Can be passed multiple "
            "times. Defaults to icnt/inter/flit/packet/traffic/stall."
        ),
    )
    return parser.parse_args()


def first_existing_log(case_dir, log_names):
    for log_name in log_names:
        log_path = case_dir / log_name
        if log_path.is_file():
            return log_path
    return None


def parse_metric_and_value(line):
    cleaned = line.strip()

    metric_match = re.match(r"^\s*([^:=,]+?)\s*(?:=|:|,)\s*(.*)$", cleaned)
    if metric_match:
        metric = metric_match.group(1).strip()
        rest = metric_match.group(2)
    else:
        parts = cleaned.split()
        metric = parts[0] if parts else ""
        rest = cleaned

    value_match = re.search(
        r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", rest
    )
    value = value_match.group(0) if value_match else ""

    return metric, value


def extract_case(case_dir, log_path, matcher):
    rows = []

    with log_path.open("r", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            if not matcher.search(line):
                continue

            metric, value = parse_metric_and_value(line)
            rows.append(
                {
                    "case": case_dir.name,
                    "log_file": str(log_path),
                    "line_number": line_number,
                    "metric": metric,
                    "value": value,
                    "raw_line": line.strip(),
                }
            )

    return rows


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    patterns = args.pattern if args.pattern else DEFAULT_PATTERNS
    matcher = re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)

    rows = []
    missing = []

    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")

    for case_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        log_path = first_existing_log(case_dir, args.log_name)
        if log_path is None:
            missing.append(case_dir.name)
            continue

        rows.extend(extract_case(case_dir, log_path, matcher))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case",
        "log_file",
        "line_number",
        "metric",
        "value",
        "raw_line",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} matching log lines to {out_path}")
    if missing:
        print("No log file found for:")
        for name in missing:
            print(f"  {name}")


if __name__ == "__main__":
    main()
