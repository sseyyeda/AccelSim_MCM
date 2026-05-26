# MCM Entropy Analysis Script

This folder contains a helper script for analyzing the three MCM entropy logger
CSV files:

```text
mcm_entropy_summary.csv
mcm_entropy_distribution.csv
mcm_address_bit_entropy.csv
```

It also contains `analyze_l2_dram_batching.py` for the inter-chiplet batching
summary CSV:

```text
l2_dram_batch_summary.csv
```

See `script/l2_dram_batching_README.md` for details.

The script produces a text report and PNG figures that help explain whether the
baseline address mapping creates remote-memory traffic imbalance.

## Run

From the repository root:

```bash
python3 script/analyze_mcm_entropy.py --input-dir test
```

This creates:

```text
test/mcm_entropy_analysis/
```

## Outputs

The main text output is:

```text
mcm_entropy_report.txt
```

The main figures are:

```text
remote_fraction_by_source.png
max_remote_dest_fraction_by_source.png
max_remote_dram_fraction_by_source.png
remote_source_to_dest_heatmap.png
remote_source_to_dram_heatmap.png
remote_dest_to_dram_heatmap.png
remote_address_bit_entropy.png
remote_address_bit_one_fraction.png
```

Some figures are skipped automatically when the corresponding input data is
empty.

## What To Look For

In `mcm_entropy_report.txt`, the strongest motivation signal is:

```text
remote_frac is high
remote_dest_chiplet_entropy is low
max_remote_dest_chiplet_frac is high
remote_dram_id_entropy is low
max_remote_dram_id_frac is high
```

That means remote traffic exists, but the baseline mapping concentrates it into
one or a few destination chiplets or DRAM partitions.

In the heatmaps, look for one bright cell dominating a row. That means one
source chiplet sends most of its remote requests to a single destination or
DRAM ID.

In `remote_address_bit_entropy.png`, bits near entropy `0` are almost constant
and are poor mapping bits. Bits near entropy `1` change often and are better
candidates for an entropy-aware address mapping.

## Useful Options

Write output somewhere else:

```bash
python3 script/analyze_mcm_entropy.py --input-dir test --out test/my_analysis
```

Analyze one kernel occurrence only:

```bash
python3 script/analyze_mcm_entropy.py --input-dir test --kernel-index 0
```

Use explicit file paths:

```bash
python3 script/analyze_mcm_entropy.py \
  --summary test/mcm_entropy_summary.csv \
  --distribution test/mcm_entropy_distribution.csv \
  --bit-entropy test/mcm_address_bit_entropy.csv \
  --out test/mcm_entropy_analysis
```
