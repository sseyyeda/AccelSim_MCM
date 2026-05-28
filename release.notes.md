
# Release v0.2-pagealloc-prefetch-profiling

May 28, 2026.

MCM research changes since `v0.1-batching-entropy`:

1. Added page allocation policy `3`, a region remote-cost-aware first-touch
   policy controlled by `-gpgpu_page_region_shift`,
   `-gpgpu_page_remote_threshold`, `-gpgpu_page_balance_threshold`, and
   `-gpgpu_page_region_min_accesses`.
2. Added MCM prefetch characterization window logging with queue, DRAM, request
   tracker, and interconnect pressure fields.
3. Added `script/analyze_mcm_prefetch_opportunities.py` to generate text
   reports and figures from entropy and prefetch-window logs.
4. Added optional request-level profiling via `ACCELSIM_REQUEST_TRACE` and the
   `gpu-simulator/gpgpu-sim/scripts/analyze_request_trace.py` analyzer.

# Release v1.3.0

Oct 19, 2021.

Major changes since v1.2.0:

1. Release and integration of AccelWattch v1.0 with Accel-Sim.

# Release v1.2.0

May 26, 2021.

Major changes since v1.1.0:

1. First release of Accel-Sim's tuner component
2. Nvbit 1.5.2 support
3. Partial support of cutting-edge MLPerf traces
4. Turing architecture traces
5. Ampere architecture traces (without copy-async feature: still in progress)
6. GPGPU-Sim 4.1 release
7. Detailed manual

# Release v1.1.0

Oct 23, 2020.

Major changes since v1.0.0:

1. Advanced disk space format to reduce the traces size (up to 5-10X reduction).
2. Optimized runtime memory footprint (up to 5-10X reduction).
3. Support for Nvbit 1.5
4. Fixed the generic loads address mapping
5. Fixed atomic instruction parsing bug
6. Fixed the max_insn_limit bug
7. Adding support to Turing instruction traces

---
