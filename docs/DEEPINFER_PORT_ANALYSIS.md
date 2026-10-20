# DeepInfer → TensorFold port analysis (2026-10-20)

Question: which concepts in DeepSeek's DeepInfer (`DeepInfer/` in this repo) can be ported
into TensorFold to make GLM-5.3-Flash run faster on 2x DGX Sparks?

## DeepInfer's six pillars

1. **DFlash / DSpark — confidence-scheduled speculative verification**
   - `deepinfer/model_executor/models/mtp/dflash.py` (DFlash model + `ConfidenceHead` per-step
     acceptance-probability probe), `csrc/engine/mtp_varlen.hpp:170-265` (`MTPVarlen::schedule`).
   - Mechanism: collect per-request, per-position acceptance rates; sort; for each candidate total
     query budget, score `expected_tokens / ms` on a profiled decode perf curve
     (`_decode_perf_curve`) corrected by a fitted `ForwardRuntimeEstimator` and a sliding-median
     `SamplerRuntimeEstimator`; pick the best budget. Low-confidence tails are not verified that
     round. Optional GPU-side budget redistribution: `compute_seqlens_q_by_confidence_rates`
     (cumprod of confidence rates → global stable sort → rank < extra budget), a TileLang kernel.
   - Flags: `DEEPINFER_ENABLE_MTP_VARLEN=1`, `DEEPINFER_USE_NEXTN_DECODE` (max draft len, default 5),
     `DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE`, `DEEPINFER_MAX_BACKTRACKING_TOKENS`.
2. **Decoder SWA bounded replay (YOCO prefill truncation)**
   - `deepinfer/model_executor/input_metadata/yoco.py`, `csrc/engine/input_metadata/prepare_tensor_input_v4.cpp`,
     `deepinfer/worker/cache_engine.py:98-114`. On a prefix-cache hit, decoder layers replay only the
     last `KEEP_TOKENS` (140 = 128 window + backtracking + margin) instead of the full window; result
     used for decode only, never cached. Modes `mask` / `compact`. Flags:
     `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE`, `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS`.
3. **ZOS — zero-overhead asynchronous scheduling**
   - `deepinfer/worker/worker.py` (`execute_model_step`), `csrc/engine/zos.cpp`, `csrc/engine/scheduler.hpp`.
   - Round reordered to Forward(N) → [CPU: schedule(N+1), launch forward(N+1), post-process(N-1)] →
     Sample(N). CPU `seqlens_q` are one-step-lag "optimistic estimates", corrected on GPU by TileLang
     kernels (`_correct_seqlens_qk`, `compute_seqlens_k_zos`), read back via pinned memory + CUDA event.
4. **Batch-invariant inference & deterministic sampling**
   - Deterministic-mode GEMM (`deep_gemm`/`tile_kernels`), batch-invariant FlashMLA decode reduction
     order, per-row seeded sampler (`random_sampling.py`: SplitMix64, randomness from
     `(seq_seed, sample_position, tag)`), disabling NCCL multi-reduction fusion. NVIDIA-only.
5. **MoE EP + EPLB, attention DP / balanced CP**
   - `deepinfer/model_executor/eplb/policy.py` (greedy replication of hot experts),
     `shuffle.py` (in-place weight remap via NCCL batched P2P or symm-mem/IPC mirror);
     mirror-CP TileLang kernels in `layers/tilelang_ops/prepare_cp_mirror.py`.
6. **Kernel infrastructure**: DeepSelect (fast deterministic vocab top-k), TileLang dual-backend
   metadata kernels, PDL everywhere (`set_pdl(True)`), full draft-path CUDA graph capture.

## What TensorFold already has (no port needed)

| DeepInfer feature | TensorFold equivalent |
| --- | --- |
| Confidence-scheduled draft depth | `DRAFT_POLICY=fnc7:0.3` (patches 0018, 0021) |
| Draft-path CUDA graphs | 0025 (draft ring), 0027 (per-stream drafter), 0032 (code copy drafts), 0100 (prelaunch) |
| Batch invariance / deterministic sampling | 0105/0106/0108 (PTX-exact CUDA ports), 0034 (nucleus union), prompt-derived RNG keys |
| Kept prompt states + replay | 0042, 0015, 0063/0077/0078/0089/0097 + spill tier (0088) |
| FP8 KV pool | 0038 `KV=fp8`, shared 2.9M-token pool (DeepInfer ships no KV cache at all) |
| Chunked prefill with budget | 0062 sliced fill (`FILL_BUDGET_MS`), 0049 multi-prefill |
| Fast DSA select | 0101 one-pass floor select, 0105/0106/0107 bit-exact CUDA kernels |

## Portable concepts, ranked

### 1. Global verify-budget throughput search — TOP PICK
Port `MTPVarlen::schedule`'s CPU-side search into TensorFold's round planner (`engine.py`).
Today the verify window is a fixed `TF_GLM_MULTI_WINDOW` (32/64 rows) with per-request policy stop.
Instead: per round, gather per-stream acceptance-rate histories, sort, and choose the total verify
row budget maximizing `expected_tokens / ms` against a profiled GB10 decode perf curve
(per `GRAPH_ROWS` 1..8 / 16..64) plus runtime estimators (item 2). High-confidence prose streams get
short windows; hard streams get the full 64. README data: window matters for code (141.2/160.3/167.0
tok/s at 32/48/64 rows), little for prose — reallocation targets exactly the PARALLEL=4–8
per-request throughput cliff (60.4 → 27.9 tok/s at 4 streams).
Constraints: the search must read only state both ranks hold (rank-consensus model, cf. patch 0065);
do NOT port the GPU-side ZOS correction path.

### 2. Runtime estimator pattern — cheap, do first
`ForwardRuntimeEstimator` (two-level median per graph size) and `SamplerRuntimeEstimator`
(9-sample sliding median) from `mtp_varlen.hpp`, ~100 lines each. Feed item 1; also auto-tune
`FILL_BUDGET_MS`, and pick `TF_GLM_EXL3_DEC_ORDER` / prompt matmul tile (0102) by measurement
instead of static env.

### 3. ZOS-lite deferred post-process — medium
TensorFold already prelaunches the draft block pass at round end (0100). The remaining serial CPU
segment is rank-0 emit/plan/all-gather. Deferring rank-0-only emit and per-stream bookkeeping into
the next round's draft window mirrors DeepInfer's Forward(N)→post-process(N-1) overlap without
touching rank consensus. A few ms per round.

### 4. PDL (programmatic dependent launch) — small, broad
`runtime_config.py:67-70` enables PDL for DeepGEMM/TileKernels. TensorFold's own CUDA kernels
(`exl3.cu` dec_kernel, `seg_scores.cu`, indexer kernels) can take
`cudaLaunchAttributeProgrammaticStreamSerialization` on GB10 (SM121). Bit-identical; style of patch 0090.

## Not portable

| Concept | Reason |
| --- | --- |
| SWA bounded replay (YOCO) | GLM-5.3-Flash has KDA recurrent + DSA sparse layers, no SWA; nothing to truncate. Bounded replay on cache hit already covered by kept prompt states (0042). |
| EPLB + in-place expert shuffle | TensorFold is TP, not EP; experts are replicated EXL3 trellis weights; 2-GPU EP would add RoCE traffic EPLB exists to avoid. |
| Mirror-CP / attention DP | 4+ GPU production math; TP2 one-shot RoCE all-gather is already the optimal topology. |
| DeepSelect top-k | DSA indexer select is already a tuned bit-exact CUDA kernel; replacing breaks the batch-invariance invariant. |
| ZOS GPU-side seqlen correction | Conflicts with sealed rank-message consensus (0065); only the CPU-side budget search survives. |
| Disaggregation / 3FS / wide-EP | Out of scope at 2-GPU scale; not in the DeepInfer release either. |

## Recommended order

2 (estimators) → 1 (adaptive verify budget) → 4 (PDL) → 3 (ZOS-lite).
Expected impact concentrated in PARALLEL=4–8; no new kernels, no checkpoint changes, rank-consensus
and batch-invariance preserved.
