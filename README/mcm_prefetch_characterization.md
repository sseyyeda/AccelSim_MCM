# MCM Prefetch Characterization

This is the first step before implementing any prefetcher. The goal is to
measure whether the MCM GPU has enough stable local/remote traffic imbalance to
justify a resource-aware remote prefetching mechanism.

The important idea is:

```text
Do not start by proving that a predictor is clever.
First prove that useful bandwidth slack exists.
```

If the workload has local-heavy windows where the remote path is idle, followed
by remote-heavy windows where warps stall on remote memory, then a conservative
remote prefetcher has a real opportunity.

## Research Question

The characterization should answer:

- Do kernels alternate between local-heavy and remote-heavy phases?
- Is the remote link or remote DRAM underused before future remote demand?
- Is local/remote traffic concentrated on specific source/destination chiplets?
- Are the same remote destinations or DRAM partitions repeatedly hot?
- Is there enough cache/MSHR/NoC slack to issue prefetches safely?
- Would prefetched data likely arrive early enough and survive until demand?

## First-Step Method

Run the simulator without any prefetcher and collect statistics. The first
analysis script is:

```bash
python3 script/analyze_mcm_prefetch_opportunities.py \
  --input-dir test \
  --out test/mcm_prefetch_characterization
```

The script writes:

```text
mcm_prefetch_opportunity_report.txt
remote_fraction_by_case.png
per_source_remote_fraction.png
remote_hotspot_summary.png
```

If per-window or per-access logs are present, it also writes phase and slack
figures such as:

```text
phase_remote_fraction_timeline.png
local_remote_requests_timeline.png
bandwidth_slack_scatter.png
mshr_or_queue_pressure.png
```

The script is intentionally tolerant of partial data. With the CSVs already
available in this repository, it can analyze aggregate MCM imbalance. With
future instrumentation, it will automatically add phase/slack analysis.

## Existing Inputs

The script understands the existing entropy logger files:

```text
mcm_entropy_summary.csv
mcm_entropy_distribution.csv
mcm_address_bit_entropy.csv
```

These files answer the static MCM imbalance question:

- How much traffic is remote?
- Which source chiplets generate remote traffic?
- Which destination chiplet or DRAM partition is hottest?
- Are address bits low-entropy in a way that explains imbalance?

This is useful, but it is not enough for prefetching. Prefetching needs time.
That means the next instrumentation target is a per-window or per-access log.

## Recommended Future Input: Window Log

The most useful next file is:

```text
mcm_prefetch_window_stats.csv
```

Enable it with:

```text
-gpgpu_mcm_prefetch_window_log_enable 1
-gpgpu_mcm_prefetch_window_cycles 1000
-gpgpu_mcm_prefetch_window_file mcm_prefetch_window_stats.csv
```

Recommended columns:

```text
kernel
window_start
window_end
source_chiplet
total_requests
local_requests
remote_requests
remote_frac
local_bytes
remote_bytes
read_requests
write_requests
remote_read_requests
remote_write_requests
unique_dest_chiplets
hottest_dest_chiplet
hottest_dest_frac
unique_remote_dest_chiplets
hottest_remote_dest_chiplet
hottest_remote_dest_frac
unique_dram_ids
hottest_dram_id
hottest_dram_frac
unique_remote_dram_ids
hottest_remote_dram_id
hottest_remote_dram_frac
remote_ingress_pending
remote_request_queue_pending
remote_request_queue_util
l2_dram_queue_pending
l2_dram_queue_util
l2_icnt_local_pending
l2_icnt_remote_pending
request_tracker_pending
dram_queue_pending
dram_queue_util
dram_read_full_partitions
dram_write_full_partitions
icnt_busy
interchiplet_busy
```

The current logger emits these traffic-window and queue-pressure fields. The
script can work with any subset of columns, but the most important fields are
`window_start`, `local_requests`, `remote_requests`, and the queue utilization
columns.

## Optional Future Input: Access Trace

For deeper analysis, emit:

```text
mcm_access_trace.csv
```

Recommended columns:

```text
cycle
sm_id
warp_id
source_chiplet
dest_chiplet
dram_id
address
access_type
cache_status
```

The script can derive local/remote windows from this file if the explicit window
stats file is not present.

## What Counts As A Good Prefetch Opportunity?

A strong opportunity window looks like this:

```text
Current window:
  local traffic high
  remote traffic low
  inter-chiplet network not busy
  remote request queues not full
  L2-to-DRAM and DRAM queues not highly occupied

Future window:
  remote traffic high
  remote demand latency or memory stalls high
```

This supports a prefetch policy like:

```text
prefetch only when remote bandwidth is idle
and NoC/MSHR/cache pressure is safe
and recent accesses predict future remote demand
```

## Decision Point Before Implementing Prefetching

Move to a prefetcher only if the report shows at least one of these:

- clear local-heavy and remote-heavy phases,
- significant remote-heavy windows after remote-slack windows,
- high remote miss or stall pressure,
- repeated hot remote destinations or DRAM partitions,
- slack windows with low queue/MSHR/NoC pressure.

If the workload is uniformly saturated or remote traffic has no predictable
structure, then a hardware prefetcher will likely increase congestion instead
of helping.

## Suggested Paper Framing

The first paper-quality claim should be a characterization claim:

```text
MCM GPU workloads expose time-varying local/remote bandwidth imbalance.
This imbalance creates exploitable slack windows for conservative remote
prefetching.
```

Then the later prefetcher can be framed as:

```text
Bandwidth-slack-aware remote prefetching for chiplet-based GPUs.
```
