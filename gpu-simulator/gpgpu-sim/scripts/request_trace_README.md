# Request Trace Logger and Analyzer

This update adds CSV tracing for memory requests and a Python analyzer.

## What Was Updated

- `gpgpu-sim/src/gpgpu-sim/shader.cc`
  - Reads `ACCELSIM_REQUEST_TRACE`.
  - Logs each memory request at interconnect injection after source/destination chiplet IDs are assigned.
  - Records address, 4KB page, L1/L2 line, source chiplet, destination chiplet, cycle, and request type data.

- `gpgpu-sim/scripts/analyze_request_trace.py`
  - Parses the generated CSV.
  - Reports local/remote traffic, source-to-destination traffic, page locality, cache-line locality, batching potential, and simple stride/reuse behavior.
  - Can generate a local-vs-remote line graph.

## Generate the CSV Trace

Run your normal Accel-Sim command with:

```bash
ACCELSIM_REQUEST_TRACE=request_trace.csv <your normal Accel-Sim command>
```

The trace file is written only when `ACCELSIM_REQUEST_TRACE` is set.

## CSV Columns

- `cycle`: cycle when the memory request is injected.
- `request_uid`: unique ID from `mem_fetch`.
- `addr`: memory address.
- `access_size`: memory access size in bytes.
- `packet_size`: packet size injected into the interconnect.
- `packet_header_size`: control/header bytes in the injected packet.
- `packet_body_size`: data/body bytes in the injected packet. This is `0` for read requests at request injection, and nonzero for write/atomic request packets.
  `packet_size = packet_header_size + packet_body_size`.
- `page_start`, `page_end`: 4KB page numbers for the first and last byte.
- `page_base_start`, `page_base_end`: 4KB page base addresses.
- `l1_line_start`, `l1_line_end`: L1 cache-line base addresses.
- `l2_line_start`, `l2_line_end`: L2 cache-line base addresses.
- `source_chiplet`: chiplet that generated the request.
- `dest_chiplet`: chiplet selected by address decoding.
- `is_local`: `1` if source and destination are the same, otherwise `0`.
- `dram_id`: decoded DRAM chip ID.
- `sub_partition_id`: decoded memory sub-partition.
- `cluster_id`: source cluster/TPC ID.
- `shader_id`: source shader core ID.
- `warp_id`: source warp ID.
- `access_type`: GPGPU-Sim memory access type enum value.
- `mf_type`: GPGPU-Sim `mem_fetch` type enum value.
- `is_write`: `1` for stores.
- `is_atomic`: `1` for atomics.

## Run the Analyzer

From the simulator root:

```bash
python3 gpgpu-sim/scripts/analyze_request_trace.py request_trace.csv
```

From a benchmark/test directory:

```bash
python3 ../gpu-simulator/gpgpu-sim/scripts/analyze_request_trace.py request_trace.csv
```

Analyze only remote accesses:

```bash
python3 ../gpu-simulator/gpgpu-sim/scripts/analyze_request_trace.py request_trace.csv --remote-only
```

Analyze batching with a chosen cycle window:

```bash
python3 ../gpu-simulator/gpgpu-sim/scripts/analyze_request_trace.py request_trace.csv --remote-only --window 100
```

Generate a local-vs-remote graph:

```bash
python3 ../gpu-simulator/gpgpu-sim/scripts/analyze_request_trace.py \
  request_trace.csv \
  --time-window 1000 \
  --series-out local_remote_1000.csv \
  --plot-out local_remote_1000.png
```

Save the terminal summary to a text file at the same time:

```bash
python3 ../gpu-simulator/gpgpu-sim/scripts/analyze_request_trace.py \
  request_trace.csv \
  --time-window 5000 \
  --series-out local_remote_5000.csv \
  --plot-out local_remote_5000.png \
  --summary-out analysis_summary_5000.txt
```

Do not use `--remote-only` for the local-vs-remote graph, because it removes local accesses before plotting.

## Window Options

The analyzer has two different window options:

- `--window`: controls the `Destination Batching Summary`.
- `--time-window`: controls the local-vs-remote graph and optional time-series CSV.

For batching, `--window` groups requests by:

```python
batch_key = (source_chiplet, window_id, dest_chiplet)
window_id = (cycle - first_cycle) // window
```

For example:

```bash
python3 ../gpu-simulator/gpgpu-sim/scripts/analyze_request_trace.py \
  request_trace.csv \
  --remote-only \
  --window 500
```

This makes the batching output report:

```text
Destination Batching Summary
----------------------------
window_cycles: 500
```

For the local-vs-remote graph, `--time-window` bins requests by cycle range:

```bash
python3 ../gpu-simulator/gpgpu-sim/scripts/analyze_request_trace.py \
  request_trace.csv \
  --time-window 5000 \
  --series-out local_remote_5000.csv \
  --plot-out local_remote_5000.png
```

This does not change the batching summary. It only changes the graph/time-series bin size.

Recommended starting points:

- `--window 50`, `--window 100`, `--window 500`, `--window 1000` for batching sensitivity.
- `--time-window 1000`, `--time-window 5000`, `--time-window 10000` for local/remote traffic over time.

Larger batching windows usually increase `avg_batch_size`, `max_batch_size`, and estimated header efficiency, but they also represent more waiting latency before requests are injected.

## How Metrics Are Calculated

### Basic Summary

- `requests = number of CSV rows`
- `local_requests = count(is_local == 1)`
- `remote_requests = count(is_local == 0)`
- `remote_fraction = remote_requests / requests`
- `first_cycle = first cycle in sorted trace`
- `last_cycle = last cycle in sorted trace`

### Source/Destination Routes

Each request contributes:

```python
route = (source_chiplet, dest_chiplet)
```

The analyzer counts how often each route appears.

### 4KB Page Locality

The logger computes:

```cpp
page_start = addr >> 12;
page_end = (addr + access_size - 1) >> 12;
```

Two requests touch the same 4KB page if their `page_start` values match.

### Cache-Line Locality

The logger computes:

```cpp
line_start = addr & ~(line_size - 1);
```

for L1 and L2.

### Destination Batching

Requests are grouped by:

```python
batch_key = (source_chiplet, window_id, dest_chiplet)
window_id = (cycle - first_cycle) // window
```

Header efficiency estimate:

```text
gain = (H + D) / (H / k + D)
```

where `k` is `avg_batch_size`.

### Simple Stride Predictability

For each source/destination route:

```python
delta_i = l2_line_i - l2_line_(i-1)
score = count(most_common_delta) / number_of_deltas
```

If `common_delta = 0`, the route repeatedly accesses the same L2 line. If `common_delta` is nonzero and score is high, the route has a repeated stride.

### Local vs Remote Graph

Requests are binned by:

```python
window_id = (cycle - first_cycle) // time_window
```

For each bin:

- `local = count(is_local == 1)`
- `remote = count(is_local == 0)`
- `total = local + remote`

The graph plots local and remote counts over time.
