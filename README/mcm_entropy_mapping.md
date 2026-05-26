# MCM Address Entropy Logging

This file documents the baseline motivation logger for MCM GPU address
mapping. The logger does not change address mapping or simulator behavior. It
only records compact CSV summaries when explicitly enabled.

## Goal

The purpose is to show whether the baseline address mapping creates an entropy
valley or traffic imbalance for remote MCM memory accesses. This is the
measurement step before implementing a PAE-style entropy-aware mapping.

The logger helps answer:

- Do remote requests from one chiplet concentrate on one destination chiplet?
- Do requests concentrate on one `dram_id` or memory partition?
- Do address bits used by the current mapping have low entropy?
- Is there enough entropy in other address bits to motivate a new mapping?

## Configuration

Enable logging with:

```text
-gpgpu_mcm_entropy_log_enable 1
```

The output files are controlled by:

```text
-gpgpu_mcm_entropy_summary_file mcm_entropy_summary.csv
-gpgpu_mcm_entropy_distribution_file mcm_entropy_distribution.csv
-gpgpu_mcm_entropy_bit_file mcm_address_bit_entropy.csv
```

By default, logging is disabled:

```text
-gpgpu_mcm_entropy_log_enable 0
```

## Mapping Controls

The baseline page placement policy is controlled with:

```text
-gpgpu_page_alloc_policy 1
```

Available policies:

```text
0 = round-robin page home chiplet
1 = first-touch page home chiplet
2 = balanced first-touch page home chiplet
```

Policy `2` assigns each newly touched page to the chiplet with the fewest
allocated pages so far. If the first-touch source chiplet is tied for least
used, it keeps the page local to that source. This is intended to reduce cases
where many remote requests converge on one destination chiplet.

DRAM-ID remapping is controlled independently:

```text
-gpgpu_mcm_dram_id_remap_enable 0
```

Available values:

```text
0 = disabled
1 = enabled
```

When enabled, the simulator keeps the selected page-home chiplet unchanged, but
remaps the DRAM offset inside that chiplet using deterministic page/address
bits. This lets you test whether spreading traffic across the memory
controllers inside a chiplet helps after, or separate from, page-home balancing.

## Recommended Baseline Run

For motivation data, run baseline mapping without reordering and batching:

```text
-gpgpu_l2_dram_reorder_enable 0
-gpgpu_l2_dram_batch_enable 0
-gpgpu_page_alloc_policy 1
-gpgpu_mcm_dram_id_remap_enable 0
-gpgpu_mcm_entropy_log_enable 1
```

Then compare against one design point at a time:

```text
# Balanced page-home mapping only
-gpgpu_page_alloc_policy 2
-gpgpu_mcm_dram_id_remap_enable 0

# DRAM-ID remapping only
-gpgpu_page_alloc_policy 1
-gpgpu_mcm_dram_id_remap_enable 1

# Both mechanisms
-gpgpu_page_alloc_policy 2
-gpgpu_mcm_dram_id_remap_enable 1
```

Then run the simulator normally from `test/`:

```bash
./runme.sh
```

## Output Files

### `mcm_entropy_summary.csv`

One compact summary per source chiplet plus one aggregate `all` row.

Important columns:

- `total_requests`: requests observed at injection to the memory system.
- `remote_requests`: requests whose source chiplet differs from destination.
- `remote_frac`: remote request fraction.
- `dest_chiplet_entropy`: entropy of destination-chiplet distribution.
- `remote_dest_chiplet_entropy`: same, but only for remote requests.
- `max_dest_chiplet_frac`: fraction going to the hottest destination.
- `max_remote_dest_chiplet_frac`: fraction of remote traffic going to the
  hottest destination.
- `dram_id_entropy`: entropy of DRAM partition distribution.
- `remote_dram_id_entropy`: same, but only for remote requests.
- `max_dram_id_frac`: fraction going to the hottest DRAM partition.
- `max_remote_dram_id_frac`: fraction of remote traffic going to the hottest
  DRAM partition.

Low entropy plus high max fraction is the strongest imbalance signal.

### `mcm_entropy_distribution.csv`

Distribution by source chiplet, destination chiplet, and `dram_id`.
It also includes `remote_requests`, so the aggregate `all` row can still show
which destination/DRAM buckets are caused by remote traffic.

Use it to plot heatmaps such as:

```text
source_chiplet -> dest_chiplet
source_chiplet -> dram_id
dest_chiplet -> dram_id
```

This file is useful for showing that a source chiplet is falling into one
remote destination or one remote memory partition.

### `mcm_address_bit_entropy.csv`

Bit-level entropy of the original request address. The logger emits both:

```text
traffic = all
traffic = remote
```

Important columns:

- `bit`: address bit index.
- `ones`: number of requests where this bit was 1.
- `total`: number of requests in the group.
- `one_frac`: fraction of ones.
- `entropy`: Shannon entropy from 0 to 1.

Interpretation:

```text
entropy near 0: bit is almost constant
entropy near 1: bit changes frequently
```

If current `dram_id` or destination-chiplet selection uses low-entropy bits,
the mapping can concentrate requests into a small number of remote partitions.

## Paper Story

The baseline motivation should be built from three observations:

1. `mcm_address_bit_entropy.csv` shows an entropy valley in the address bits
   used by the current chiplet or DRAM partition mapping.
2. `mcm_entropy_summary.csv` shows low destination/DRAM entropy and high
   hottest-destination fraction.
3. `mcm_entropy_distribution.csv` shows the actual imbalance as a heatmap.

After this baseline is confirmed, compare first-touch against balanced
first-touch and DRAM-ID remapping. A useful result should increase remote
destination-chiplet entropy, reduce the hottest remote destination fraction, or
spread the hottest DRAM-ID bucket without increasing total cycles.
