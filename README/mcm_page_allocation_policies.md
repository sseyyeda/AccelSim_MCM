# MCM Page Allocation Policies

This simulator supports four page-home placement policies for MCM GPU memory.
The selected policy decides which chiplet owns a page the first time that page
is touched. Later accesses reuse the same page-home chiplet from the page table.

The main knob is:

```text
-gpgpu_page_alloc_policy <policy>
```

Related knobs:

```text
-gpgpu_page_size 4096
-gpgpu_page_fault_overhead 1000
```

Policy `3` also uses:

```text
-gpgpu_page_region_shift 4
-gpgpu_page_remote_threshold 0.25
-gpgpu_page_balance_threshold 64
-gpgpu_page_region_min_accesses 32
```

The implementation is mainly in:

```text
gpu-simulator/gpgpu-sim/src/gpgpu-sim/addrdec.cc
```

## Policy 0: Round-Robin

```text
-gpgpu_page_alloc_policy 0
```

Each new page is assigned to chiplets in order:

```text
page 0 -> chiplet 0
page 1 -> chiplet 1
page 2 -> chiplet 2
page 3 -> chiplet 3
page 4 -> chiplet 0
```

This gives strong page-count balance, but it ignores the source chiplet that
first touched the page. It can turn naturally local accesses into remote
accesses.

Use this as a balance-oriented baseline.

## Policy 1: First-Touch

```text
-gpgpu_page_alloc_policy 1
```

The first chiplet that accesses a page becomes the page-home chiplet:

```text
first touch from chiplet 2 -> page home chiplet 2
```

This preserves locality for private pages. The weakness is imbalance: if many
pages are first touched by one chiplet, that chiplet can become a memory hotspot.

Use this as the locality-oriented baseline.

## Policy 2: Balanced First-Touch

```text
-gpgpu_page_alloc_policy 2
```

This policy chooses the chiplet with the fewest allocated pages so far. If the
first-touch source chiplet is tied for least-loaded, it keeps the page local to
the source.

This improves page-count balance compared with pure first-touch, but it can be
too aggressive. If an application has mostly private pages, this policy may move
pages away from their first-touch chiplet and increase remote traffic.

Use this to measure the cost/benefit of naive page balancing.

## Policy 3: Region Remote-Cost-Aware First-Touch

```text
-gpgpu_page_alloc_policy 3
```

This policy is locality-first and balance-aware. It keeps first-touch placement
unless history from the surrounding address region suggests that balancing is
useful.

The region is computed from the page number:

```text
region_id = page_addr >> gpgpu_page_region_shift
```

With:

```text
-gpgpu_page_region_shift 4
```

one region contains:

```text
2^4 = 16 pages
```

The simulator tracks region history:

```text
total accesses in the region
remote accesses in the region
accesses from each source chiplet
home chiplets used by pages in the region
```

When a new page is allocated, policy `3` does:

```text
if source chiplet is invalid:
    use round-robin fallback
else if region has fewer than gpgpu_page_region_min_accesses:
    use first-touch
else if region remote fraction < gpgpu_page_remote_threshold:
    use first-touch
else if source page count <= min page count + gpgpu_page_balance_threshold:
    use first-touch
else:
    use the least-loaded chiplet
```

So a page is moved away from first-touch only when:

```text
the region has enough history
the region is already remote/shared enough
the source chiplet is meaningfully overloaded
```

This avoids the main weakness of policy `2`: it does not blindly balance private
regions that were mostly local under first-touch.

## Recommended Experiments

For a clean comparison, run:

```text
policy 1: first-touch baseline
policy 2: naive balanced first-touch
policy 3: remote-cost-aware first-touch
```

Look at:

```text
IPC / total cycles
remote fraction
destination chiplet entropy
DRAM ID entropy
final MCM Page Placement Summary
```

At the end of the simulator output, the code prints:

```text
=== MCM Page Placement Summary ===
mcm_total_pages_allocated = ...
mcm_pages_allocated_chiplet_0 = ... (...)
mcm_pages_allocated_chiplet_1 = ... (...)
mcm_pages_allocated_chiplet_2 = ... (...)
mcm_pages_allocated_chiplet_3 = ... (...)
```

This tells you how many pages each chiplet owns and the fraction of total pages.

## Suggested Starting Values

Conservative policy `3` settings:

```text
-gpgpu_page_alloc_policy 3
-gpgpu_page_region_shift 4
-gpgpu_page_remote_threshold 0.25
-gpgpu_page_balance_threshold 64
-gpgpu_page_region_min_accesses 32
```

More aggressive balancing:

```text
-gpgpu_page_remote_threshold 0.15
-gpgpu_page_balance_threshold 16
```

More locality-preserving:

```text
-gpgpu_page_remote_threshold 0.40
-gpgpu_page_balance_threshold 128
```

Start conservative, then make the policy more aggressive only if first-touch
shows a strong page-home hotspot.
