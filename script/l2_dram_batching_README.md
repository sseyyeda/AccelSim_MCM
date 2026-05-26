# L2/DRAM Batching Analysis Script

This script analyzes compact batching summary CSV files generated with:

```text
-gpgpu_l2_dram_batch_log 1
-gpgpu_l2_dram_batch_log_file l2_dram_batch_summary.csv
```

The simulator CSV reports how many request/reply packets entered the batching
path, how many packets were actually injected after batching, the average batch
size, and modeled byte savings.

## Run

From the repository root:

```bash
python3 script/analyze_l2_dram_batching.py --input-dir results
```

This recursively finds:

```text
l2_dram_batch_summary.csv
```

under `results/` and creates:

```text
results/l2_dram_batch_analysis/
```

## Outputs

Text report:

```text
l2_dram_batch_report.txt
```

Summary table:

```text
batching_latest_by_case.csv
```

Figures:

```text
packet_reduction.png
byte_savings.png
avg_batch_size.png
packet_counts.png
ipc.png
```

## What To Look For

Strong batching evidence looks like:

```text
high request_packet_reduction or reply_packet_reduction
high avg_request_batch_size or avg_reply_batch_size
lower total cycles / higher IPC compared with baseline
```

If packet reduction is high but IPC does not improve, the network was probably
not the dominant bottleneck, or the batching wait window added latency that
cancelled the traffic benefit.

If reply batching helps more than request batching, read-response traffic is
likely the heavier inter-chiplet bottleneck. If request batching helps more,
small request packet pressure or request-side NoP arbitration was likely more
important.

## Useful Options

Analyze explicit files:

```bash
python3 script/analyze_l2_dram_batching.py \
  --csv results/batch_request_only/l2_dram_batch_summary.csv \
  --csv results/batch_reply_only/l2_dram_batch_summary.csv \
  --csv results/batch_both/l2_dram_batch_summary.csv
```

Write output somewhere else:

```bash
python3 script/analyze_l2_dram_batching.py \
  --input-dir results \
  --out results/my_batching_analysis
```
