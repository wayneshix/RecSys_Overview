# Part 12 — Serving & Inference Efficiency

A synthesis note on the **systems / serving paradigm** of industrial recommendation. The fundamentals notes (01–08) and the frontier notes (09–11) describe *what* the models compute; this note is about *how those models are made to run* inside a strict latency budget at hundreds of thousands of QPS. Every architectural idea in the archive — the three-tower pre-rank ([Note 03 — Ranking Models](03_Ranking_Models.md) §7), the two-tower retriever with ANN ([Note 02 — Candidate Retrieval](02_Candidate_Retrieval.md) §5, §8), the unified Transformers and scaling laws ([Note 04 — Feature Interaction](04_Feature_Interaction.md) Expanded Reading; [Note 10 — Scaling Laws & Large-Model Architectures](10_Scaling_Laws_and_Large_Model_Architectures.md)), and generative / semantic-ID retrieval ([Note 09 — Generative Recommendation & Semantic IDs](09_Generative_Recommendation_and_Semantic_IDs.md)) — is ultimately constrained by serving. This note pulls the recurring efficiency tricks scattered across those chapters into one place and frames the central tension: **scaling laws say "make the model bigger," real-time serving says "you have ~10 ms."**

This is a conceptual/synthesis note. It does **not** re-derive the per-paper detail that lives in the Expanded Reading sections of Notes 02, 03, and 04 — it cross-references them and explains the shared serving *pattern*.

---

## Table of Contents

1. [Why Serving Is the Binding Constraint](#1-why-serving-is-the-binding-constraint)
2. [Generative-Retrieval Serving: Constrained Decoding over a Trie](#2-generative-retrieval-serving-constrained-decoding-over-a-trie)
3. [Attention & Architecture Efficiency for Big Rankers](#3-attention--architecture-efficiency-for-big-rankers)
4. [Custom GPU Kernels: GDPA](#4-custom-gpu-kernels-gdpa)
5. [MoE Serving: Inference-Time Load Balancing](#5-moe-serving-inference-time-load-balancing)
6. [The Efficiency Toolbox](#6-the-efficiency-toolbox)
7. [Interview Cheat-Sheet](#7-key-insights)
8. [References](#8-references)

---

## 1. Why Serving Is the Binding Constraint

### 1.1 The per-stage latency budget

A recommendation request flows through a funnel, and **each stage has a hard latency budget** measured in milliseconds. The corpus of hundreds of millions of items shrinks stage by stage:

| Stage | #items in → out | Budget pressure |
|---|---|---|
| Retrieval (召回) | $\sim 10^8 \to$ a few thousand | dozens of channels in parallel, each must be cheap |
| Pre-rank (粗排) | few thousand → few hundred | a *lightweight* model run on **thousands** of items |
| Fine-rank (精排) | few hundred → few hundred (scored) | a *heavy* model run on **hundreds** of items |
| Re-rank (重排) | few hundred → a few dozen | diversity sampling ([Note 06](06_Reranking_and_Diversity.md)) |

The whole pipeline must return in roughly the time it takes a page to render. There is no slack to "just run a bigger model on everything."

### 1.2 The candidate-count asymmetry

The single most important structural fact for serving:

> **One user request must be scored against many candidates.**

There is **one** user per request but **$n$** items to score. This asymmetry is the lever behind almost every optimization in this note. It explains:

- **Why two-tower retrieval works** ([Note 02](02_Candidate_Retrieval.md) §8.1): the item tower is run *offline* over the whole corpus and cached in a vector DB; only the **user tower runs online, once**. Score = $\cos(\mathbf a, \mathbf b)$ via ANN.
- **Why the three-tower pre-rank is shaped the way it is** ([Note 03](03_Ranking_Models.md) §7.3): push capacity into the **user tower (runs once)** and the **cacheable item tower**; keep the **cross tower + upper network (run $n$ times)** tiny, because dynamic statistical features can't be cached.
- **Why modern big rankers cache user-side computation**: in a unified Transformer the user/history tokens are *invariant* across all $n$ candidates in a request, so they should be computed once and reused (KV-cache, request-level batching — §3).

The design principle is constant across eras: **do per-user work once; do per-item work as cheaply as possible.**

### 1.3 QPS and the cost of large models

Industrial systems serve hundreds of thousands of QPS. Cost scales with (QPS) × (FLOPs per request) × (#candidates). A fine-rank model that is $2\times$ more accurate but $10\times$ more expensive may be un-shippable not because of latency on a single request but because of **fleet cost** at full traffic.

### 1.4 The tension scaling laws create

[Note 10](10_Scaling_Laws_and_Large_Model_Architectures.md) establishes that recommendation models now follow **scaling laws** — accuracy improves near-log-linearly with parameters/FLOPs, à la LLMs. The serving reality says the opposite: latency and cost are fixed. The entire 2026 serving literature is the resolution of this tension, and it resolves along two axes:

1. **Raise utilization, not just size.** The headline metric is **MFU (Model FLOPs Utilization, 模型算力利用率)** — the fraction of peak hardware FLOPs the model actually uses. A model at 17% MFU wastes 83% of the GPU; doubling MFU is "free" capacity. Kunlun's central result is **MFU 17% → 37%** (Hou et al., 2026), roughly doubling scaling efficiency without changing the model's accuracy-per-FLOP.
2. **Spend FLOPs only where they pay.** Cache the invariant, skip the redundant, sparsify the attention, prune the cold experts. Almost every technique below is a way to *delete work* that a naïve implementation would do.

> **Interview framing.** "Scaling laws make bigger models *worth* building; serving efficiency makes them *possible* to build and run. The bridge between the two is MFU and the one-user-many-candidates asymmetry."

---

## 2. Generative-Retrieval Serving: Constrained Decoding over a Trie

### 2.1 The paradigm shift

Classic retrieval ([Note 02](02_Candidate_Retrieval.md)) represents items as dense vectors and uses **ANN** (Faiss / HNSW / ScaNN) to find nearest neighbors. Generative retrieval ([Note 09](09_Generative_Recommendation_and_Semantic_IDs.md)) instead represents each item as a short sequence of **semantic IDs (语义 ID / SIDs)** and *autoregressively decodes* the SID tokens of the items to recommend. This obviates the external ANN index — but introduces a new serving problem.

### 2.2 Why you need *constrained* decoding

A free-running autoregressive model will confidently emit SID sequences that **don't correspond to any real item**, or that violate business logic (out-of-stock, stale, wrong region/category). Post-generation filtering is wasteful — the model can burn its whole decode budget producing invalid items and return *zero* valid recommendations.

The fix: the set of valid SID sequences forms a **prefix tree (trie, 前缀树)**, and **constrained / beam decoding** masks the invalid next-tokens at each step so only valid continuations survive. This is conceptually the autoregressive **beam search over paths** of Deep Retrieval ([Note 02](02_Candidate_Retrieval.md) §10.5) — scoring $p(a,b,c\mid\mathbf x)$ token-by-token — now over a semantic-ID vocabulary instead of a hand-built path index.

### 2.3 Why naïve trie traversal is slow on accelerators

A trie implemented as pointers/linked nodes is **hostile to TPUs/GPUs** (Su et al., 2026):

- **Memory-bound:** pointer-chasing produces non-contiguous, random memory access, defeating HBM burst/coalescing and hardware prefetchers → stalled execution units.
- **Compilation-incompatible:** XLA-style accelerators need **static computation graphs**; data-dependent control flow and recursive branching can't be compiled end-to-end.

In Su et al.'s experiments a CPU-offloaded trie *doubled* inference time — unusable against a target of **≤ 10 ms per decoding step**.

### 2.4 STATIC — vectorizing the trie

**STATIC (Sparse Transition matrix-Accelerated Trie Index for Constrained decoding)** flattens the prefix tree into a **static Compressed Sparse Row (CSR) sparse matrix**, turning the per-step "which children are valid?" lookup into a fully vectorized **sparse matrix operation** that XLA compiles to fixed shapes. The I/O complexity is $O(1)$ in the constraint-set size (vs. logarithmic for binary-search baselines), via coalesced reads and a branch-free, host-device-roundtrip-free decode.

- **Cost:** only **0.033 ms per step (~0.25% of inference time)**, under the ≤10 ms target.
- **Speedup:** **948× over a CPU trie**, **47–1033× over accelerator binary-search baselines**.
- **Scale:** vocabularies of **~20 million fresh items**, enabling business-logic constraints (freshness, in-stock, category) directly in decoding.
- **Trade-off:** the CSR matrix is *static* and must be **rebuilt when the valid item set changes** → index-maintenance cost under high churn. It targets throughput, not recall (orthogonal to retrieval quality).

> STATIC is the **serving-side counterpart to the ANN index**: where two-tower retrieval needs Faiss/HNSW to make dense lookup tractable, generative retrieval needs an accelerator-friendly trie to make constrained decoding tractable. See [Note 02](02_Candidate_Retrieval.md) §5.5, §8.1 for the ANN side and its Expanded Reading for STATIC's full write-up.

### 2.5 TrieRec — trie-aware positional encodings

STATIC makes the trie *fast*; **TrieRec** (Xu et al., 2026) makes the model *aware* of the trie. Standard Transformers flatten SID tokens into a linear stream and ignore the tree topology the tokenization induces. TrieRec injects structural inductive bias via two positional encodings:

1. a **trie-aware absolute positional encoding** folding a node's local structure (depth, ancestors, descendants) into its token representation;
2. a **topology-aware relative positional encoding** adding pairwise structural relations (e.g. lowest-common-ancestor distance $d_{\text{LCA}}$) into self-attention.

It is **model-agnostic and hyperparameter-free**; dropped into TIGER / CoST / LETTER it gives an **average +8.83%** across four datasets at negligible cost. This refines the path-scoring view of Deep Retrieval ([Note 02](02_Candidate_Retrieval.md) §10): the same prefix tree that STATIC *vectorizes for serving*, TrieRec *teaches the model to respect*.

> **One-liner:** STATIC = make the constraint *cheap to evaluate*; TrieRec = make the model *understand* the constraint's tree. Both live on the same semantic-ID trie.

---

## 3. Attention & Architecture Efficiency for Big Rankers

As fine-rank moves from MLP-era shared-bottom / MMoE ([Note 03](03_Ranking_Models.md) §1, §3) to **unified Transformers** that jointly model feature interaction and behavior sequences ([Note 04](04_Feature_Interaction.md) Expanded Reading; [Note 10](10_Scaling_Laws_and_Large_Model_Architectures.md)), the attention stack becomes the serving bottleneck. The recurring tricks below all exploit the §1.2 asymmetry or prune redundant attention. (Architecture detail lives in Notes 04 and 10; this is the *serving* synthesis.)

### 3.1 Cache the invariant user-side computation

In a unified token stream, the **user / history tokens are identical across all $n$ candidates** in a request. Recomputing them per candidate is the single biggest waste.

- **KV-cache for invariant tokens.** **OneTrans** (ByteDance) shares one set of Q/K/V across the homogeneous sequence (S) tokens and **caches the S-token KV across candidates**, cutting per-session complexity from $O(C)$ to $O(1)$ for $C$ candidates — the Transformer analogue of two-tower's "compute the user once." (See [Note 04](04_Feature_Interaction.md) Expanded Reading.)
- **Request-Level Batching (RLB) + User-Item Decoupling.** **MixFormer** (ByteDance) separates user-side from item-side computation and batches at the request level so user-side work is reused across candidates → **≈ 36% FLOPs** saved on the UI-decoupled variant.

These are exactly [Note 03](03_Ranking_Models.md) §7.3's "push capacity into the part that runs once," re-expressed for attention models.

### 3.2 Sparsify / shorten attention

Full self-attention is $O(L^2)$ in sequence length — prohibitive for lifelong behavior sequences.

- **Sliding-window / semilocal attention (SLA).** Restrict attention to a local window. **ULTRA-HSTU** uses SLA to reach **21.4× inference** speedup; **Kunlun** uses sliding-window attention as one of its kernel-level levers.
- **Attention truncation.** Cap the sequence (e.g. lifelong history truncated to a few thousand via GSU-style retrieval), so attention sees only the most relevant span.
- **Cross-only / content-sparse attention.** Prune redundant same-type interaction blocks and attend only to the high-signal cross block or top-similar neighbors (the EST line in [Note 04](04_Feature_Interaction.md)), turning $O(L^2) \to O(LK)$.

### 3.3 Skip and compress

- **Computation Skip (CompSkip).** **Kunlun** avoids recomputing parts that don't change (e.g. event-level personalization caching), skipping redundant work outright.
- **Hierarchical Seed Pooling (HSP).** **Kunlun** pools/compresses tokens hierarchically before expensive interaction, shrinking the effective sequence the heavy layers see.
- **Mixed precision (BF16/FP8).** Standard LLM-inherited trick: lower-precision matmuls roughly double tensor-core throughput; used across OneTrans, Kunlun, and the GDPA kernel (§4).

### 3.4 MFU as the headline metric

The single number that ties §3 together is **MFU (Model FLOPs Utilization)** — actual ÷ peak FLOPs. Kunlun's diagnosis is that *poor scaling efficiency is a low-MFU problem, not a model-capacity problem*: with kernel fusion (GDPA), HSP, SLA, and CompSkip it lifts **MFU 17% → 37%** on B200 and ~2× scaling efficiency. The lesson for interviews: **when a big ranker is "too slow," profile MFU first** — you are often leaving more than half the hardware idle, and reclaiming it beats shrinking the model.

> **Synthesis.** Every trick here is one of three moves: (1) **reuse** invariant user-side compute (KV-cache, RLB, decoupling); (2) **prune** attention (SLA, truncation, cross-only/sparse); (3) **skip/compress** redundant work (CompSkip, HSP, mixed precision). All are measured by MFU. Architecture specifics: [Note 04](04_Feature_Interaction.md), [Note 10](10_Scaling_Laws_and_Large_Model_Architectures.md).

---

## 4. Custom GPU Kernels: GDPA

### 4.1 The problem fused kernels solve

Off-the-shelf attention kernels (even SOTA FlashAttention-4, FA4) are tuned for **LLM-style traffic**: long, regular, uniform-length sequences in modest batches. **RecSys traffic is the opposite** — short, asymmetric, **jagged** sequences in **very large batches**, driven by user behavior with no fixed distribution. Xu et al. (2026) measure a **2.6× forward / 1.6× backward** gap (up to 4× worst-case) between a generic kernel on real production data and the benchmark, purely from this shape mismatch (reduced pipeline occupancy, poor compute–memory overlap on short K/V loops).

### 4.2 What GDPA is

**Generalized Dot-Product Attention (GDPA)** generalizes standard (softmax) attention by **replacing softmax with arbitrary element-wise activations** — the pattern inside Meta's RecSys foundation models: **GEM** (Meta's largest RecSys training model), **HSTU** (SiLU), **InterFormer**, and **Kunlun** (GELU in its PFFN). Structurally, self-attention, PMA, and PFFN are all "**two matmuls with an optional activation in between**," so GDPA **unifies them into one fused kernel** built on **FA4**, with workload-driven optimizations for RecSys shapes (e.g. removing the softmax-correction warp stage, outer-loop software pipelining for short K/V sequences).

### 4.3 Training vs. inference throughput

GDPA is primarily a **training kernel** (the headline numbers are training throughput):

- Up to **2× forward** (1,145 BF16 TFLOPs, ~97% tensor-core utilization) and **1.6× backward** vs. the original Triton kernel;
- up to **3.5× forward / 1.6× backward** vs. FA4 under some production traffic;
- **>30% end-to-end training throughput** when applied across the full model.

It is hardware- and shape-specific (tuned for B200 + Meta's irregular shapes) and only helps the **non-softmax sequence-ranker** regime, not classic MLP rankers.

> **Why this matters.** GDPA is the **infrastructure layer beneath** the §3 architectures. Kunlun's MFU 17%→37% (§3.4) is largely *because* GDPA fuses the attention/PFFN math. When a paper says "we made the big ranker trainable at scale," a fused, shape-aware kernel like GDPA is usually what they mean. See [Note 03](03_Ranking_Models.md) and [Note 04](04_Feature_Interaction.md) Expanded Reading for the model-level context.

---

## 5. MoE Serving: Inference-Time Load Balancing

### 5.1 The load-imbalance problem

Sparse Mixture-of-Experts (MoE) scales capacity by activating only a few experts per token — but routing is **heavy-tailed**: a minority of "heavy-hitter" experts receive most tokens while others sit idle (Liu et al., 2026). At serving this is a *latency* problem, not just a quality one:

- overloaded experts become **stragglers** that gate the batch's completion time;
- idle experts **waste compute** and depress effective parallelism;
- skew adds synchronization/buffer overhead → higher cost.

Crucially, the paper finds (i) imbalance **persists and worsens at inference** (and grows with batch size), and (ii) **selection frequency ≠ importance** — a heavily-used expert is not necessarily a high-value one.

This is the MoE-serving twin of the **polarization / dead-expert** failure mode of MMoE ([Note 03](03_Ranking_Models.md) §3.2). There the fix is **dropout on gate outputs at training time**; here we need a fix at *inference* time, because the deployed router is frozen and the input distribution drifts.

### 5.2 Replicate-and-Quantize (R&Q)

**R&Q** (Liu et al., 2026) is a **training-free, inference-time, near-lossless** rebalancing method requiring only a small calibration set and no router changes:

- **Replicate** heavy-hitter experts → extra parallel capacity removes the straggler.
- **Quantize** the less-important experts *and* the replicas → stay within the **original memory budget**.
- Measure imbalance with the per-layer **Load-Imbalance Score (LIS)**: compares tokens sent to the heavy-hitter against an equal split; **LIS = 1 means perfectly balanced**, larger = worse.

**Result:** up to **1.4× reduction in load imbalance** with accuracy within **±0.6%**. (Evaluated on general LLMs; Meta AI + UNC.) The recsys relevance: as MoE-based rankers ([Note 03](03_Ranking_Models.md) §3, [Note 10](10_Scaling_Laws_and_Large_Model_Architectures.md)) deploy, R&Q is the serving-time analogue of MMoE's anti-polarization dropout — balance expert workload so a few experts don't bottleneck latency.

---

## 6. The Efficiency Toolbox

A consolidated map of technique → what it saves → where it applies → source. Use this as the §1.2 asymmetry made concrete.

| Technique | What it saves | Where it applies | Source / cross-ref |
|---|---|---|---|
| **Two-tower + ANN** (item tower offline, user tower online) | per-item online inference (only user runs once) | retrieval | [Note 02](02_Candidate_Retrieval.md) §5.5, §8.1 |
| **Three-tower pre-rank** (heavy user tower runs once, cacheable item tower, tiny cross tower) | compute on the $n$-times path | pre-rank | [Note 03](03_Ranking_Models.md) §7.3 (COLD) |
| **Distillation** (fine-rank teacher → light student) | model size while keeping ordering | pre-rank, SLM rankers | [Note 03](03_Ranking_Models.md) §7.4 |
| **STATIC** (trie → CSR sparse matrix) | trie-traversal latency on accelerators (948× vs CPU trie) | generative-retrieval constrained decoding | Su et al. (2026); [Note 02](02_Candidate_Retrieval.md) |
| **TrieRec** (trie-aware positional encodings) | accuracy (+8.83%) at ~0 cost — *quality*, not latency | generative recommendation | Xu et al. (2026); [Note 02](02_Candidate_Retrieval.md) |
| **KV-cache for invariant S tokens** | recomputing user/history per candidate; $O(C)\to O(1)$ | unified-Transformer fine-rank (OneTrans) | [Note 04](04_Feature_Interaction.md) |
| **Request-Level Batching + User-Item Decoupling** | redundant user-side FLOPs (≈36%) | big sequence rankers (MixFormer) | [Note 04](04_Feature_Interaction.md) |
| **Sliding-window / semilocal attention (SLA)** | $O(L^2)$ attention on long sequences (21.4× inference) | lifelong-sequence rankers (ULTRA-HSTU, Kunlun) | [Note 04](04_Feature_Interaction.md), [Note 10](10_Scaling_Laws_and_Large_Model_Architectures.md) |
| **Attention truncation / cross-only sparse attention** | $O(L^2)\to O(LK)$ | long-sequence CTR (EST) | [Note 04](04_Feature_Interaction.md) |
| **Computation Skip (CompSkip)** | redundant recomputation of stable parts | Kunlun | Hou et al. (2026); [Note 04](04_Feature_Interaction.md) |
| **Hierarchical Seed Pooling (HSP)** | sequence length into heavy layers | Kunlun | Hou et al. (2026) |
| **Mixed precision (BF16/FP8)** | matmul time (~2× tensor-core throughput) | all big models | OneTrans, Kunlun, GDPA |
| **GDPA fused kernel** (softmax→activation, FA4-based) | training throughput (>30% e2e; 2× fwd) | non-softmax RecSys foundation models | Xu et al. (2026) |
| **MFU (model-flops-utilization)** | *the metric*: surfaces idle hardware (17%→37%) | any accelerator-served model | Hou et al. (2026); §3.4 |
| **Replicate-and-Quantize (R&Q)** | MoE straggler latency (1.4× less imbalance) within memory budget | MoE serving | Liu et al. (2026) |
| **Bloom filter exposure dedup** | $O(nr)$ exposure comparison | post-retrieval filtering | [Note 02](02_Candidate_Retrieval.md) §12 |
| **Cache retrieval** (reuse unexposed fine-rank results) | re-running retrieval/ranking | retrieval channel | [Note 02](02_Candidate_Retrieval.md) §11.3 |

---

## 7. Key Insights

- **The one fact that explains everything:** *one user request, many candidates.* Do per-user work **once**; do per-item work **cheaply** or **offline**. (Two-tower, three-tower, KV-cache, RLB all instantiate this.)
- **Per-stage latency budget:** retrieval → pre-rank → fine-rank → re-rank, each measured in milliseconds; you can't "just run a bigger model on everything."
- **The scaling-law tension:** scaling laws ([Note 10](10_Scaling_Laws_and_Large_Model_Architectures.md)) say go bigger; serving says ~10 ms. Resolved by **raising MFU** (use the GPU you already pay for) and **deleting redundant work** (cache / prune / skip / sparsify).
- **MFU is the headline metric.** "Too slow" usually means low MFU (e.g. 17%), not too many parameters. Kunlun: **17% → 37%**.
- **Generative-retrieval serving = constrained beam decoding over a semantic-ID trie.** Naïve trie = pointer-chasing, accelerator-hostile. **STATIC** flattens it to a static CSR matrix (948× vs CPU trie, 0.033 ms/step). **TrieRec** makes the model aware of the tree (+8.83%). Trie ≈ the generative analogue of the ANN index ([Note 02](02_Candidate_Retrieval.md)) and of Deep Retrieval's beam search over paths (§10).
- **Big-ranker attention tricks (one-liners):** KV-cache invariant user tokens (OneTrans, $O(C)\to O(1)$); RLB + User-Item Decoupling (MixFormer, ~36% FLOPs); sliding-window/semilocal attention (ULTRA-HSTU 21.4×); CompSkip + HSP + mixed precision (Kunlun).
- **Custom kernels:** generic FA4 is tuned for long regular LLM sequences; RecSys traffic is **short, jagged, large-batch**. **GDPA** = softmax replaced by element-wise activation, fuses self-attn/PMA/PFFN into one FA4-based kernel; **training** throughput >30% e2e, up to 2× forward.
- **MoE serving:** heavy-hitter experts are **stragglers**; selection frequency ≠ importance. **R&Q** replicates hot experts and quantizes cold ones within the memory budget (training-free, ±0.6% accuracy), measured by **LIS** (=1 when balanced). This is the *inference-time* twin of MMoE's anti-polarization dropout ([Note 03](03_Ranking_Models.md) §3.2).
- **Map serving fixes to model failure modes:** dead/polarized experts (training: dropout on gates → inference: R&Q); per-item recompute (→ two-tower / three-tower / KV-cache); $O(L^2)$ attention (→ SLA / truncation / cross-only); invalid generative outputs (→ trie-constrained decoding).

---

## 8. References

- Su, Z., Katsman, I., Wang, Y., He, R., Heldt, L., Keshavan, R., Wang, S.-C., Yi, X., Gao, M., Dalal, O., Hong, L., Chi, E. H., & Han, N. (2026). *Vectorizing the trie: Efficient constrained decoding for LLM-based generative retrieval on accelerators.* arXiv. https://arxiv.org/abs/2602.22647
- Xu, Z., Chen, J., Chen, S., He, Y., Yang, J., Yuan, C., Ding, K., & Wang, C. (2026). *Trie-aware transformers for generative recommendation.* arXiv. https://arxiv.org/abs/2602.21677
- Liu, Z., Peng, J., Duan, J., Liu, Z., Zhou, K., Liang, M., Simon, L., Liu, X., Xu, Z., & Chen, T. (2026). *A replicate-and-quantize strategy for plug-and-play load balancing of sparse mixture-of-experts LLMs.* arXiv. https://arxiv.org/abs/2602.19938
- Xu, J., Chen, C., Yang, S., Hoehnerbach, M., Liu, X., Zadouri, T., Dao, T., et al. (2026). *Generalized Dot-Product Attention: Tackling real-world challenges in GPU training kernels.* PyTorch / Meta Engineering Blog.
- Hou, B., et al. (2026). *Kunlun: Establishing scaling laws for massive-scale recommendation systems through unified architecture design.* arXiv. https://arxiv.org/abs/2602.10016
- Ding, Q., et al. (2026). *Bending the scaling law curve in large-scale recommendation systems.* arXiv. https://arxiv.org/abs/2602.16986

## Pinterest in Practice (2024–2026)

Pinterest's serving work makes the §1.3 fleet-cost point literal: every model-scaling win below ships only because it is held to a **cost-neutral (or cost-down) serving budget** as a hard constraint, not an afterthought.

### PinFM — DCAT + int4 PTQ

> Chen, X., Rajesh, K., Lawhon, M., Wang, Z., Li, H., Li, H., Joshi, S. V., Eksombatchai, P., Yang, J., Hsu, Y.-P., Xu, J., & Rosenberg, C. (2025). *PinFM: Foundation model for user activity sequences at a billion-scale visual discovery platform*. In *Proceedings of the Nineteenth ACM Conference on Recommender Systems (RecSys '25)* (pp. 381–390). ACM. https://doi.org/10.1145/3705328.3748050

A 20B-parameter sequence foundation model is only servable because its **Deduplicated Cross-Attention Transformer (DCAT)** is a direct instance of this chapter's §3.1 reuse pattern: it runs the transformer over the user history **once and KV-caches it**, then cross-attends each candidate against the cached KV — exploiting the §1.2 asymmetry, which at serving reaches a 1:1000 unique-sequence-to-candidate ratio. A fixed 256-length window with KV-cache rotation (replace oldest tokens with the candidate) avoids re-concatenation, and the whole thing rides custom Triton kernels for **+600% serving throughput** vs. FlashAttention self-attention. On top, **post-training int4 quantization** (FBGEMM min-max) shrinks the 20B embedding table to 31.25% of size for **−7% API latency** (lower IO) at A/B-neutral quality — the deploy lands cost-neutral.

### TransAct V2 — serving full lifelong sequences

> Xia, X., Joshi, S. V., Rajesh, K., Li, K., Lu, Y., Pancha, N., Badani, D. D., Xu, J., & Eksombatchai, P. (2025). *TransAct V2: Lifelong user action sequence modeling on Pinterest recommendation*. In *Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM '25)* (pp. 6881–6882). ACM. https://doi.org/10.1145/3746252.3761433

The Homefeed CTR ranker attends **end-to-end over an $O(10^4)$ lifelong sequence** inside a real-time latency budget — refusing the two usual escape hatches this chapter flags (§3.2): lossy **compression** (TWIN/SIM-style) and **stale offline caching**. Instead it leans on data-processing and model-serving optimizations to keep the full sequence on the live path under production latency/storage/network limits, with ablations attributing the efficiency to each. It is the counterpoint to attention-truncation (§3.2): rather than shorten the sequence, engineer the serving stack so the long sequence stays affordable.

### InteractRank — post-dot-product cross features at ~21× fewer FLOPs

> Khandagale, S., Juneja, B., Agarwal, P., Subramanian, A., Yang, J., & Wang, Y. (2025). *InteractRank: Personalized web-scale search pre-ranking with cross interaction features*. arXiv. https://arxiv.org/abs/2504.06609

Search pre-ranking must score $O(10^4{-}10^5)$ candidates in milliseconds, so InteractRank keeps the two-tower form (item embeddings precomputed offline into the forward index) and injects query–item cross signal **after** the dot product via a small affine projection plus precomputed engagement-prior (IQP) features, realized as a **WAND structured-query** operator at request time. Because the towers stay decoupled and the cross features are cached, interaction costs essentially **zero added latency** — yielding more accuracy than early-interaction IntTower at **~21× fewer serving FLOPs** (142 vs. 3069). This is the §1.2/§6 "spend FLOPs only where they pay" budget made concrete: avoid the early-interaction blowup, buy the cross signal for a constant.
