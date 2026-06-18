---
title: "Tiled Matrix Multiplication in CUDA: Why a 200KB Cache Beats Brute Force"
excerpt: "A 10,000×10,000 matmul on a GPU isn't fast because of more cores — it's fast because of a tiny on-chip cache most engineers never think about. Part 1 of a 3-part series on hardware-aware algorithms."
date: "2026-05-29"
readTime: "13 min read"
category: "GPU / CUDA"
---


A few days ago I was reading about **State Space Models** and kept tripping over the same phrase: *"hardware-aware algorithm."* Papers used it like everyone already knew what it meant. I didn't — not at the hardware level. So I went down the rabbit hole, and the first stop was the most fundamental operation in all of deep learning: **how does a GPU actually multiply two matrices fast?**

The answer turned out to be a single idea — *tiling* — and once it clicked, I started seeing the same pattern everywhere: in Mamba's selective scan, in FlashAttention, in basically every kernel that wins. This post is where it starts.

---

## 1. The operation everyone takes for granted

Matrix multiplication is deceptively simple. To compute `C = A × B` where all three are N×N:

```
for i in 0..N:
  for j in 0..N:
    sum = 0
    for k in 0..N:
      sum += A[i][k] * B[k][j]
    C[i][j] = sum
```

Every output element `C[i][j]` is a dot product of one row of `A` and one column of `B`. There are N² outputs, each costing N multiply-adds, so the whole thing is **O(N³) arithmetic**. That number is what most people fixate on.

It's also the wrong number to fixate on.

---

## 2. The naive CUDA kernel (and why it's slow)

The textbook first GPU kernel gives each output element its own thread:

```cuda
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];   // two global-memory reads, every iteration
        }
        C[row * N + col] = sum;
    }
}
```

This is *correct*. It is also leaving most of the GPU on the table. Here's the problem, and it has nothing to do with the math.

Look at the inner loop. Each iteration does **two reads from global memory** (`A` and `B`) and **two floating-point operations** (one multiply, one add). On an NVIDIA A100:

- Peak compute: **~19.5 TFLOP/s** (FP32)
- Peak memory bandwidth (HBM): **~1.5 TB/s**

To feed 19.5 trillion FP32 ops per second, you'd need to move ~156 TB/s of operands. You have 1.5 TB/s. **You are starved for data by roughly 100×.** The cores spend almost all their time *waiting on memory*, not computing. The kernel is **memory-bound**, even though matmul is "supposed" to be compute-bound.

This is the single most important reframe in the whole series: **the bottleneck is not FLOPs, it's bytes.**

---

## 3. The number that actually matters: arithmetic intensity

There's a clean way to make this precise. **Arithmetic intensity** is:

$$
\text{AI} = \frac{\text{FLOPs performed}}{\text{bytes moved from HBM}}
$$

In the naive kernel, every multiply-add reads two fresh 4-byte floats from global memory: 2 FLOPs for 8 bytes, so AI ≈ **0.25 FLOP/byte**. That is catastrophically low.

The crossover point — where a GPU stops being memory-bound and starts being compute-bound — is:

$$
\text{AI}_{\text{crit}} = \frac{\text{peak FLOP/s}}{\text{peak bytes/s}} = \frac{19.5 \times 10^{12}}{1.5 \times 10^{12}} \approx 13 \text{ FLOP/byte}
$$

So we need to be near ~13 FLOP/byte to saturate the cores. We're at 0.25. **We need to read each operand from HBM ~50× less often.** Not compute less — *read* less.

This is the entire game. Everything below is just *how* you read less.

---

## 4. The GPU memory hierarchy (the missing context)

The reason "read less from HBM" is even possible is that HBM isn't the only memory on a GPU. There's a hierarchy, and it spans about five orders of magnitude in speed:

```
        ┌─────────────┐
        │  Registers  │   ~20 TB/s, per-thread, ~KB         fastest
        ├─────────────┤
        │   SRAM      │   ~19 TB/s, per-SM, ~100-200KB      (shared memory)
        │ (shared mem)│
        ├─────────────┤
        │  L2 cache   │   ~5 TB/s, chip-wide, ~40MB
        ├─────────────┤
        │  HBM / DRAM │   ~1.5 TB/s, chip-wide, ~40-80GB    slowest
        └─────────────┘
```

![The GPU memory hierarchy: registers, SRAM/shared memory, L2 cache, and HBM, ordered by speed and size](/blog/diagrams/memory-hierarchy.svg)
*Five orders of magnitude separate the top of the hierarchy from the bottom. Tiling is the art of keeping the working set as high up this pyramid as possible.*

The key players:

- **HBM (global memory):** huge (tens of GB), but "slow" — 1.5 TB/s. This is where your matrices live by default.
- **SRAM (shared memory):** tiny (~100–200KB *per Streaming Multiprocessor*), but ~10× the bandwidth of HBM and far lower latency. Crucially, it's **programmer-controlled** — you decide what goes in it. It's also shared by all threads in a block, so it's a scratchpad for *cooperation*.
- **Registers:** fastest of all, but private to a single thread and scarce.

The naive kernel never touches shared memory. Every thread independently re-reads `A` and `B` straight from HBM. And here's the kicker: **neighboring threads re-read the exact same data.** Thread `(0,0)` and thread `(0,1)` both stream the entire first row of `A`. Thousands of threads, redundantly hammering HBM for bytes their neighbors already fetched.

That redundancy is the opportunity.

![Naive vs tiled: in the naive kernel every thread reads HBM directly with redundant overlapping traffic; in the tiled kernel a tile is loaded into shared memory once and reused by all threads](/blog/diagrams/naive-vs-tiled.svg)
*Same matrices, same math. The only difference is whether threads hammer HBM individually (left) or cooperate through shared memory (right).*

---

## 5. The fix: tiling

The idea is simple to state: **load a small block (a "tile") of `A` and `B` from HBM into shared memory once, let the whole thread block reuse it many times, then move on to the next tile.**

Instead of each thread computing its output in one long pass over HBM, the block walks across the matrices in tiles. For each tile:

1. Cooperatively load a `TILE × TILE` chunk of `A` and a `TILE × TILE` chunk of `B` from HBM into shared memory.
2. Synchronize so every thread sees the full tile.
3. Each thread computes the partial dot product *using only shared memory* — no HBM access in the inner loop.
4. Synchronize again, then load the next pair of tiles.

![Tiled matrix multiplication: a tile from a row of A and a tile from a column of B are multiplied and accumulated into the corresponding tile of C, stepping across the K dimension](/blog/diagrams/tiling-matmul.svg)
*A tile of A (row i) and a tile of B (col j) combine into a tile of C (i, j). The block walks across the K dimension one tile at a time, accumulating partial sums.*

Here's a classic `TILE = 16` (or 32) kernel:

```cuda
#define TILE 16

__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // Walk across the K dimension, one tile at a time.
    for (int t = 0; t < N / TILE; ++t) {
        // Each thread loads ONE element of each tile into shared memory.
        As[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];

        __syncthreads();  // make sure the whole tile is loaded

        // Inner loop now reads from SRAM, not HBM.
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();  // don't overwrite the tile before everyone's done
    }

    C[row * N + col] = sum;
}
```

The change is small. The consequence is not.

---

## 6. Why this works: counting the reads

Let's redo the arithmetic intensity, because this is where the payoff becomes concrete.

In the **naive** kernel, computing one tile's worth of output (TILE² elements) reads operands from HBM for *every* multiply-add. With `TILE = 16`, each element re-streams its full row/column.

In the **tiled** kernel, you load each `TILE×TILE` block of `A` and `B` from HBM **exactly once**, and then reuse every loaded value `TILE` times (once for each output column/row it contributes to). 

The effect: **HBM traffic drops by a factor of `TILE`.** With `TILE = 16`, that's ~16× fewer reads from global memory. With `TILE = 32`, ~32×. Arithmetic intensity jumps from ~0.25 to several FLOP/byte, dragging the kernel from deep in memory-bound territory up toward the compute-bound regime — exactly where we wanted to be.

Same FLOPs. Same answer. The *only* thing that changed is **where the data lived during the computation.** That's the whole trick, and it's worth sitting with: we made matmul dramatically faster without doing a single fewer multiplication.

---

## 7. Gotchas — what I got wrong

This is the part I'd skip in a textbook and the part that actually taught me something.

**"Bigger tiles are always faster."** I assumed this. They're not. Shared memory is a *fixed, tiny* budget (~100–200KB per SM). Two `TILE×TILE` float tiles cost `2 × TILE² × 4` bytes. But more importantly, shared memory is shared across all the thread blocks resident on an SM at once. Use too much per block and **fewer blocks fit on the SM** — your *occupancy* drops, the GPU has fewer warps to hide latency with, and performance gets *worse*. Tile size is a balancing act between reuse (bigger is better) and occupancy (smaller is better). The sweet spot is a property of **the SM, not the matrix.**

**Shared memory bank conflicts.** SRAM is split into 32 banks. If threads in a warp hit the same bank, the accesses *serialize*. A naively laid-out tile can quietly cost you 2–8× on shared-memory reads. The classic fix is padding (`__shared__ float As[TILE][TILE + 1];`) to shift the access pattern off the conflicting stride.

**The `__syncthreads()` are not optional.** Drop either barrier and you get a race: threads compute on a tile that's half-loaded, or that's already been overwritten by the next iteration. The bug is nondeterministic and miserable to find. Both barriers are load-bearing.

**This is still not cuBLAS.** The tiled kernel above is the *concept*, not the state of the art. Production GEMM (cuBLAS / CUTLASS) layers on register blocking (each thread computes a small *grid* of outputs, not one), double-buffering (prefetch the next tile while computing the current one), vectorized loads, and Tensor Cores. But every one of those is an *optimization on top of tiling*. Tiling is the foundation.

---

## 8. The pattern (and what's next)

Strip away the CUDA syntax and the idea is almost embarrassingly general:

> **Identify the working set. Pull it into the fastest memory you can. Reuse it as much as possible before you write back. Touch slow memory as few times as you can get away with.**

That's it. That's what "hardware-aware" means. It's not about clever math — the matmul did the *exact same multiplications*. It's about respecting the memory hierarchy.

And once you see it, you can't unsee it. In **Part 2**, I'll show how **Mamba** uses this identical principle for something that looks nothing like a matmul — a *sequential scan* — fusing the whole operation into one kernel that keeps its state in SRAM and only writes the final result to HBM. The surprising punchline: a "slow," inherently sequential algorithm can crush a "fast," parallel one, purely by respecting the hierarchy.

In **Part 3**, **FlashAttention** generalizes the trick to attention itself — and that's the algorithm that inspired Mamba's scan in the first place. Three algorithms, one pattern.

---

### References & further reading

- NVIDIA CUDA C++ Programming Guide — *Shared Memory* and *Tiled Matrix Multiplication*
- *Programming Massively Parallel Processors* (Kirk & Hwu) — the canonical tiling chapter
- CUTLASS — NVIDIA's open-source, production-grade GEMM templates
- The Roofline Model (Williams et al.) — the formal framework behind arithmetic intensity

*Code from this post is on my GitHub: [link]. Questions or corrections welcome — this is a learning journey and I'm sharing it as I go.*
