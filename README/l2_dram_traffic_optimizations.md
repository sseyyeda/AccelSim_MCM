# L2-to-DRAM Traffic Optimizations

This note documents the L2-to-DRAM traffic optimizations added for MCM GPU
experiments: scheduling-only reordering plus request/reply inter-chiplet
batching.

The goal is to test whether remote L2 miss traffic can be issued and transported
more efficiently. Reordering groups already-generated misses by destination and
DRAM locality. Batching groups multiple inter-chiplet packets with the same
destination to reduce modeled NoP packet/header overhead. These mechanisms are
not prefetchers and do not create new memory requests.

## Where the Change Is

The implementation is in:

- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc`
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.h`
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc`
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.h`

There is also a build-system compatibility fix in:

- `gpu-simulator/gpgpu-sim/src/cuda-sim/Makefile`

That Makefile fix prevents newer `bison` output from generating an invalid
`ptx_parser_decode.def`.

## What It Does

Normally, the memory partition picks one request from the L2-to-DRAM queues and
pushes it into the DRAM latency queue in round-robin/FIFO style.

With reordering enabled, eligible requests are first placed into a small
per-memory-partition reorder buffer. The scheduler then chooses a request from
that buffer using the configured policy. The default policy tries to continue
issuing requests with the same:

```text
destination chiplet + DRAM id + bank + row
```

This should increase the chance that DRAM receives nearby row-buffer-friendly
requests close together.

The implementation is bounded:

- It has a maximum buffer size.
- It has a maximum waiting window.
- Requests are forced out when they become too old.
- By default, only remote requests are reordered.

## Recommended Configuration

Add these lines to the `gpgpusim.config` you are using:

```text
-gpgpu_l2_dram_reorder_enable 1
-gpgpu_l2_dram_reorder_window 40
-gpgpu_l2_dram_reorder_buffer_size 8
-gpgpu_l2_dram_reorder_policy 3
-gpgpu_l2_dram_reorder_remote_only 1
-gpgpu_l2_dram_reorder_log 1
-gpgpu_l2_dram_reorder_log_file l2_dram_reorder_summary.csv
```

Keep FR-FCFS enabled:

```text
-gpgpu_dram_scheduler 1
```

For the FR-FCFS DRAM scheduler queue size, there is no single standard value.
In this codebase, the option default is:

```text
-gpgpu_frfcfs_dram_sched_queue_size 0
```

where `0` means unlimited. However, the tested configs in this repository
commonly use:

```text
-gpgpu_frfcfs_dram_sched_queue_size 64
```

For experiments, a reasonable sweep is:

```text
64  baseline/common tested value
16  smaller queue, often useful for seeing scheduling pressure
8   aggressive small queue, may amplify the reordering effect
```

Going much smaller than `8` can make the result dominated by artificial
backpressure instead of useful row-buffer behavior.

## Configuration Knobs

### `-gpgpu_l2_dram_reorder_enable`

```text
0 = disabled
1 = enabled
```

Default: `0`

Use `0` for baseline runs and `1` for reordering runs.

### `-gpgpu_l2_dram_reorder_window`

Maximum number of cycles a request can wait in the reorder buffer before it is
forced out in age order.

Default: `40`

Suggested sweep:

```text
10  conservative
20  moderate
40  default, matches the offline trace-analysis window you used
80  aggressive
```

If this is too small, there is little chance to form useful groups. If it is too
large, the scheduler may add too much waiting time.

### `-gpgpu_l2_dram_reorder_buffer_size`

Maximum number of requests held in the reorder buffer per memory partition.

Default: `8`

Suggested sweep:

```text
4
8
16
```

Larger buffers give more opportunity to find same-row or same-destination
requests, but they can also increase waiting time.

### `-gpgpu_l2_dram_reorder_policy`

```text
0 = FIFO
1 = destination chiplet
2 = destination chiplet + DRAM id
3 = destination chiplet + DRAM id + bank + row
```

Default: `3`

For your idea, start with `3`. It is the closest to testing whether row-buffer
locality improves after reordering. Use `1` or `2` only if you want to isolate
NoP/destination grouping from DRAM row grouping.

### `-gpgpu_l2_dram_reorder_remote_only`

```text
0 = reorder local and remote requests
1 = reorder only remote requests
```

Default: `1`

For your current motivation, use `1`. That keeps local HBM traffic closer to
the original behavior and focuses the experiment on remote NoP/LLC miss traffic.

### `-gpgpu_l2_dram_reorder_log`

```text
0 = disabled
1 = enabled
```

Default: `0`

Use `1` for both baseline and reordering runs when you want a compact CSV file
for comparison.

### `-gpgpu_l2_dram_reorder_log_file`

Output path for the compact CSV summary.

Default:

```text
l2_dram_reorder_summary.csv
```

The file is intentionally small. It writes one line when the simulator prints
stats, not one line per memory request.

## Suggested Experiment Matrix

First run a baseline:

```text
-gpgpu_l2_dram_reorder_enable 0
-gpgpu_l2_dram_reorder_log 1
-gpgpu_l2_dram_reorder_log_file baseline_l2_dram_reorder_summary.csv
-gpgpu_dram_scheduler 1
-gpgpu_frfcfs_dram_sched_queue_size 64
```

Then run the main experiment:

```text
-gpgpu_l2_dram_reorder_enable 1
-gpgpu_l2_dram_reorder_window 40
-gpgpu_l2_dram_reorder_buffer_size 8
-gpgpu_l2_dram_reorder_policy 3
-gpgpu_l2_dram_reorder_remote_only 1
-gpgpu_l2_dram_reorder_log 1
-gpgpu_l2_dram_reorder_log_file reorder_l2_dram_reorder_summary.csv
-gpgpu_dram_scheduler 1
-gpgpu_frfcfs_dram_sched_queue_size 64
```

Then repeat with smaller FR-FCFS queues:

```text
-gpgpu_frfcfs_dram_sched_queue_size 16
-gpgpu_frfcfs_dram_sched_queue_size 8
```

A compact useful sweep is:

```text
baseline: reorder=0, frfcfs_queue=64
main:     reorder=1, window=40, buffer=8,  frfcfs_queue=64
pressure: reorder=1, window=40, buffer=8,  frfcfs_queue=16
pressure: reorder=1, window=40, buffer=8,  frfcfs_queue=8
tuning:   reorder=1, window=20, buffer=8,  frfcfs_queue=16
tuning:   reorder=1, window=40, buffer=16, frfcfs_queue=16
```

## Compact CSV Log

Enable the compact log with:

```text
-gpgpu_l2_dram_reorder_log 1
```

The default output file is:

```text
l2_dram_reorder_summary.csv
```

The CSV contains these columns:

- `kernel`: executed kernel name or names.
- `total_cycle`: total simulated cycles so far.
- `interval_cycle`: cycles in the current stats interval.
- `total_insn`: total simulated instructions so far.
- `interval_insn`: instructions in the current stats interval.
- `interval_ipc`: `interval_insn / interval_cycle`.
- `reorder_enable`: whether reordering was enabled for the run.
- `window`, `buffer_size`, `policy`, `remote_only`: reorder configuration.
- `frfcfs_queue`: DRAM scheduler queue size.
- `buffered`: requests inserted into the reorder buffer.
- `issued`: reordered-buffer requests issued toward DRAM.
- `forced`: requests forced out because they reached the age window.
- `same_key_issued`: issued requests matching the previous reorder key.
- `same_key_frac`: `same_key_issued / issued`.
- `forced_frac`: `forced / issued`.
- `avg_wait`: average cycles spent in the reorder buffer.
- `max_wait`: maximum reorder-buffer wait.
- `pending`: requests still in the reorder buffer at stats print time.

For baseline runs, enable the log but keep:

```text
-gpgpu_l2_dram_reorder_enable 0
```

That gives you the same performance columns with zero reorder activity. Then
compare against the reordering run. A good result usually looks like:

```text
interval_ipc improves
total_cycle or interval_cycle decreases
same_key_frac is meaningfully above zero
forced_frac is not too high
avg_wait is not large enough to cancel the row-locality benefit
```

## What You Should See

When reordering is enabled, the memory partition printout includes a line like:

```text
L2->DRAM reorder: buffered=... issued=... forced=... same_key_issued=... pending=... avg_wait=... max_wait=...
```

The fields mean:

- `buffered`: number of requests inserted into the reorder buffer.
- `issued`: number of buffered requests eventually sent toward DRAM.
- `forced`: number of requests forced out because they reached the age window.
- `same_key_issued`: number of issued requests that matched the previous
  scheduling key.
- `pending`: requests still waiting in the reorder buffer at print time.
- `avg_wait`: average cycles spent in the reorder buffer.
- `max_wait`: maximum observed wait in the reorder buffer.

For a useful result, you want to see:

```text
same_key_issued increases
forced is not too high
avg_wait remains modest
IPC or total runtime improves
DRAM row-buffer locality improves if those stats are available
```

If `forced` is very high, the window is probably too small or the stream does
not have enough locality. If `avg_wait` is high but performance does not
improve, the reorder buffer is adding delay without creating enough DRAM benefit.

## How to Interpret Performance

The simulator already models DRAM scheduling and row-buffer effects through the
DRAM scheduler. Reordering can help only if it changes the order seen by the
DRAM controller enough to increase row hits or reduce queue conflicts.

You are most likely to see an effect when:

- The workload has many remote L2 misses.
- Several requests target the same destination/DRAM/bank/row close in time.
- The DRAM scheduler queue is not so large that FR-FCFS can already find all
  the row hits by itself.
- The reorder buffer window is large enough to group requests, but not so large
  that it adds excessive waiting.

You may see little or no benefit when:

- FR-FCFS queue size is large enough to already expose the locality.
- The workload has little row reuse.
- The access stream is dominated by local requests and `remote_only=1`.
- The NoP or another pipeline stage is the real bottleneck.

## Option B: Inter-Chiplet Packet Batching

Option B adds a real inter-chiplet batching model for remote traffic in
memory-side LLC mode. The simulator keeps each original `mem_fetch`, but can
group several remote requests or replies going to the same chiplet into one NoP
packet. When that packet arrives, it is unpacked back into individual
`mem_fetch` objects before entering the destination LLC queue or the source
cluster reply queue.

Add or change these options in `gpgpusim.config`:

```text
-gpgpu_l2_dram_batch_request_enable 1
-gpgpu_l2_dram_batch_reply_enable 0
-gpgpu_l2_dram_batch_window 40
-gpgpu_l2_dram_batch_max_size 8
-gpgpu_l2_dram_batch_header_size 16
-gpgpu_l2_dram_batch_min_payload 0
-gpgpu_l2_dram_batch_log 1
-gpgpu_l2_dram_batch_log_file l2_dram_batch_summary.csv
```

Use both batching enables as `0` for the baseline:

```text
# Request network only
-gpgpu_l2_dram_batch_request_enable 1
-gpgpu_l2_dram_batch_reply_enable 0

# Reply network only
-gpgpu_l2_dram_batch_request_enable 0
-gpgpu_l2_dram_batch_reply_enable 1

# Both request and reply networks
-gpgpu_l2_dram_batch_request_enable 1
-gpgpu_l2_dram_batch_reply_enable 1
```

The batching window limits how long the oldest packet can wait for more traffic
to the same destination. The max size limits the number of `mem_fetch` objects in
one batch packet. The header size is the shared packet header charged once per
batch. The minimum payload keeps a small per-request cost inside the batch if
you do not want reads to become completely free after the shared header.
The batch log writes a compact CSV summary with packet-count and byte-count
reductions for the request and reply networks.

Request batching groups packets traveling:

```text
source chiplet -> destination chiplet
```

Reply batching groups packets traveling:

```text
destination chiplet -> source chiplet
```

With batching enabled, the benefit should mainly appear in inter-chiplet NoC
pressure: fewer/flatter request or reply flits, lower NoP contention, and
possibly better runtime if remote inter-chiplet traffic was the bottleneck. It
is separate from Option A; you can run batching alone, reordering alone, or both
together.

### Batching Summary CSV

Enable:

```text
-gpgpu_l2_dram_batch_log 1
-gpgpu_l2_dram_batch_log_file l2_dram_batch_summary.csv
```

Important columns:

```text
request_baseline_packets
request_actual_packets
request_batches
request_batched_mfs
avg_request_batch_size
request_packet_reduction
request_bytes_before
request_bytes_after
request_byte_savings
reply_baseline_packets
reply_actual_packets
reply_batches
reply_batched_mfs
avg_reply_batch_size
reply_packet_reduction
reply_bytes_before
reply_bytes_after
reply_byte_savings
```

`*_baseline_packets` and `*_bytes_before` are what the traffic would have been
without batching for the packets that entered the batching path. `*_actual`
values are what was actually injected after batching. `*_batches` counts only
batch packets with more than one `mem_fetch`; single requests that timed out or
could not find a partner are counted as actual packets but not as batches.
