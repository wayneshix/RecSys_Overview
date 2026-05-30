# Part 10 — Scaling Laws & Large-Model Architectures

This is a **synthesis note**. The previous notes built the fundamentals — feature crossing ([Note 04 — Feature Interaction](04_Feature_Interaction.md)) and behavior-sequence modeling ([Note 05 — Sequence Modeling](05_Sequence_Modeling.md)) — and their *Expanded Reading* sections already carry per-paper detail for the recent wave of large models. This note steps back and explains the **paradigm**: what a "scaling law" (扩展律 / 缩放定律) means for recommendation, why classic Deep-Learning Recommendation Models (DLRMs, 深度学习推荐模型) historically did *not* scale, how the new generative/large architectures "bend the curve," and how the whole `*Former` family fits into one map. It cross-references the per-paper write-ups rather than repeating them.

The one-sentence thesis: **LLMs taught us that loss falls as a predictable power law in compute/params/data; recommendation is now re-engineering its architectures (unified Transformers, generative sequence backbones) and its efficiency (MFU, KV-cache, sparse attention) so that recsys metrics obey the same kind of law.**

---

## Table of Contents

1. [What a Scaling Law Is (and Is Not)](#1-what-a-scaling-law-is-and-is-not)
2. [Why Classic DLRMs Did Not Scale](#2-why-classic-dlrms-did-not-scale)
3. [Bending the Curve: How Large Architectures Restore Scaling](#3-bending-the-curve-how-large-architectures-restore-scaling)
4. [The Central Debate: Encode-then-Interaction vs. Unified Modeling](#4-the-central-debate-encode-then-interaction-vs-unified-modeling)
5. [Taxonomy of the Unified-Transformer (`*Former`) Family](#5-taxonomy-of-the-unified-transformer-former-family)
6. [The HSTU Lineage & Generative Sequence Backbones](#6-the-hstu-lineage--generative-sequence-backbones)
7. [Lifelong / Ultra-Long Sequence Taxonomy](#7-lifelong--ultra-long-sequence-taxonomy)
8. [Efficiency Is the Real Bottleneck](#8-efficiency-is-the-real-bottleneck)
9. [Trade-offs & Open Questions](#9-trade-offs--open-questions)
10. [Key Insights](#10-key-insights)
11. [References](#11-references)
12. [Pinterest in Practice (2024–2026)](#pinterest-in-practice-20242026)
   - [PinFM](#pinfm)

---

## 1. What a Scaling Law Is (and Is Not)

A **scaling law** is an empirical, smooth, *power-law* relationship between a model's quality and the resources spent on it. In language modeling (Kaplan et al., 2020; Hoffmann et al., 2022) the canonical form is

$$
L(N) \approx L_\infty + \left(\frac{N_c}{N}\right)^{\alpha_N},
$$

where $L$ is test loss, $N$ is the number of (non-embedding) parameters, $L_\infty$ is the irreducible loss, and $\alpha_N$ is a small positive exponent. Analogous laws hold in dataset size $D$ and total training compute $C \approx 6ND$. Two consequences matter:

- **Predictability.** Plot the metric against $\log(\text{compute})$ and you get a (nearly) **straight line** — "near log-linear gains." You can fit it on small models and *extrapolate* to decide whether a 10× spend is worth it.
- **Compute-optimal allocation.** Given a fixed compute budget $C$, there is an optimal split between making the model bigger ($N$) vs. training on more data ($D$) — the **Chinchilla** insight (Hoffmann et al., 2022). You do *not* just grow params; you co-scale params and data along the compute-optimal frontier.

**For recommendation, "loss" is replaced by a task metric** — most papers track **Normalized Entropy (NE, 归一化熵)** or **AUC / UAUC / GAUC** versus FLOPs or parameters. The dream is the same: a straight log-linear line so that "spend more compute → predictably better ranking."

**A scaling law is *not*** simply "bigger is better." A model "scales" only if the line is (a) *monotone* (more compute keeps helping) and (b) has a *useful slope*. A flat or quickly-saturating curve means the architecture does **not** scale — which is exactly what happened to classic DLRMs. The key quantity the new papers obsess over is **scaling efficiency** — formally defined (Ding et al., 2026) as **the slope of the fitted line between metric and log-compute**. "Bending the curve" means *increasing that slope*.

---

## 2. Why Classic DLRMs Did Not Scale

A classic DLRM (e.g. Wide&Deep → DeepFM → DCNv2 → a multi-task tower, see [Note 04](04_Feature_Interaction.md)) is dominated by **ID-embedding lookups** plus a relatively small **feature-interaction** dense network. Throwing more parameters/FLOPs at it gives diminishing — often flat — returns, for several reasons:

1. **Memorization, not compute, drives most of the quality.** The bulk of a DLRM's "parameters" live in sparse **ID embedding tables** (用户ID/物品ID embedding). These memorize per-entity statistics; their size scales with the *vocabulary*, not with reasoning depth. Growing the embedding table helps coverage but does not give a power-law in *dense* compute, and it mostly memorizes rather than generalizes (the cold-start / long-tail weakness — recall [Note 04 §1.3 FM](04_Feature_Interaction.md), [Note 05 §2.4](05_Sequence_Modeling.md)).
2. **Feature-interaction saturation.** Explicit crossing modules (FM second-order, DCNv2 bounded-degree, FiBiNet bilinear) capture interactions efficiently but **plateau**: stacking more cross layers or widening the MLP yields little after a point, because the useful high-order crosses are few and the modules were designed for *cheapness*, not *scalability*. Wukong (Zhang et al., 2024) was the first to show a non-sequential interaction stack (stacked-FM + linear compression) that *does* obey a scaling law — proving the saturation was an architecture problem, not a law of nature.
3. **No end-to-end signal to scale into.** The sequence side was handled by a *separate*, deliberately cheap module (LastN-average, DIN, SIM — [Note 05](05_Sequence_Modeling.md)) that compresses the history *before* it ever meets the interaction network. Compression throws away the very long-range signal that more compute could exploit.
4. **Low hardware utilization.** Even when you add FLOPs, recsys models run at **3–15% Model FLOPs Utilization (MFU)** versus 40–60% for LLMs (Hou et al., 2026), because of small embedding dims, irregular tensor shapes, and memory-bound ops. Low MFU means "more FLOPs on paper" rarely translates into more *effective* compute — so the curve looks flat for reasons that are half-systems, half-architecture.

**Interview framing.** DLRMs scale in *memorization* (table size) but not in *computation/reasoning* (dense FLOPs). LLMs scale in computation. The 2025–2026 wave's whole project is to give recsys an architecture whose *dense* compute obeys a law — by making it a Transformer (so it can reason over raw tokens) and by raising MFU (so the FLOPs are real).

---

## 3. Bending the Curve: How Large Architectures Restore Scaling

"Bending the scaling-law curve" (Ding et al., 2026) = *raising the slope* of metric-vs-log-compute. The recipes split into two complementary levers, and every paper in §5–§7 is some mix of the two:

- **Algorithmic effectiveness** (performance gain per FLOP): replace hand-designed crossing/compression with **learned Transformer interaction** over **raw tokens**, and **fuse the behavior sequence into the same backbone** so depth/width buy long-range reasoning, not just memorization. This is the move from "DCNv2 + DIN bolted together" to "one Transformer that reads features *and* history."
- **Computational efficiency** (effective FLOPs per nominal FLOP): raise **MFU** with fused kernels, mixed precision, sparse/linear attention, and KV-cache reuse, so nominal FLOPs become *effective* FLOPs. Kunlun (Hou et al., 2026) explicitly names **low MFU as the primary barrier** to a clean power law and pushes MFU **17% → 37%** for ~**2× scaling efficiency**; OneTrans reports MFU **13.4% → 30.8%** (Zhang et al., 2025).

Because $C \approx 6ND$, "compute-optimal" reasoning carries over: you co-scale model size with data, and — crucially for recsys — you must **co-scale the sequence (S) and non-sequence (NS) capacities** rather than dumping all FLOPs into one (the MixFormer "co-scaling" problem, §5). The payoff is the LLM-style promise: fit the line on a small model, extrapolate, and *predictably* buy quality with compute.

---

## 4. The Central Debate: Encode-then-Interaction vs. Unified Modeling

The architectural fault line running through the entire `*Former` wave is **how** the two information sources — the user **behavior sequence (S, 序列特征)** and the **non-sequence/context features (NS, 非序列特征)**: user profile, item attributes, scene/context — should meet.

**Classic pipeline — "encode-then-interaction" (先编码后交叉):**

```
behavior sequence (S) ──► sequence encoder ──► COMPRESS to one vector ──┐
                                                                         ├─► concat ─► feature-interaction (DCNv2 / DHEN / RankMixer) ─► multi-task heads
non-sequence features (NS) ──────────────────────────────────────────────┘
```

The sequence is summarized **early** into a single candidate-aware vector (DIN/SIM-style, [Note 05](05_Sequence_Modeling.md)), then concatenated with NS features and passed to a crossing module. Two structural weaknesses:

- **Unidirectional, late fusion.** NS features cannot reshape the sequence representation; information flows S → NS only, and only *after* compression. Long-range sequence signal is discarded before interaction sees it.
- **Fragmented execution.** Two separate modules with separate parameters fragment the compute graph, raise latency, and cannot share LLM optimizations (KV-cache, FlashAttention).

**Unified modeling (统一建模):** put S and NS into **one Transformer stack** so sequence modeling and feature interaction co-evolve with **bidirectional**, **early**, **fine-grained** fusion. The entire model is then scaled like an LLM — just grow depth/width — and inherits KV-caching, FlashAttention, and mixed precision (Zhang et al., 2025; Hou et al., 2026).

The wave does **not** agree on *how* unified to go. The live sub-debate:

- **Merge everything into one token stream** (OneTrans) — maximally LLM-like, but heterogeneous S and NS tokens compete in one attention space.
- **Keep sequences independent, interact via cross-attention / query decoding** (HyFormer, MixFormer, EST, HeMix) — preserves semantic boundaries, often via NS-as-Query over S-as-Key/Value, and sparsifies attention to the high-signal S–NS cross block.

This "split-then-merge vs. interact-then-split vs. cross-attend" disagreement is exactly the taxonomy axis in §5. Full per-paper detail lives in [Note 04 — Expanded Reading](04_Feature_Interaction.md).

---

## 5. Taxonomy of the Unified-Transformer (`*Former`) Family

All of these take [Note 04](04_Feature_Interaction.md)'s explicit crossing (FM → DCNv2 → FiBiNet → RankMixer) and make it (a) a **learned Transformer interaction**, (b) **fused with the behavior sequence** ([Note 05](05_Sequence_Modeling.md)), tuned to (c) **follow a scaling law** by raising MFU and pruning redundant compute. The table is a map; **cells are deliberately short — see [Note 04 — Expanded Reading](04_Feature_Interaction.md) for the full per-paper write-ups** (method, results, trade-offs, lineage).

| Model | Org / Year | Core idea | How S & NS interact | Key efficiency trick | Detail |
|---|---|---|---|---|---|
| **InterFormer** | Meta, 2024 | Interleave S & NS with **bidirectional** flow; avoid early aggregation | NS-summary guides S via PFFN+MHA, S-summary guides NS (two-way) | One-to-one token mapping; +24% QPS vs prior SOTA | [N04](04_Feature_Interaction.md) |
| **OneTrans** | ByteDance, 2025 | **Fully unified**: one tokenizer → single token stream | Merged stream; HiFormer-style **mixed params** (S shares Q/K/V/FFN, NS token-specific) | **Cross-request KV cache** $O(C)\toO(1)$; pyramid token pruning | [N04](04_Feature_Interaction.md) |
| **HyFormer** | ByteDance, 2026 | Critiques merged stream; **alternating** Query Decoding + Query Boosting | NS Global Tokens cross-attend layer-wise K/V of *independent* sequences | HeadMixing token mixing; sequence-specific K/V | [N04](04_Feature_Interaction.md) |
| **MixFormer** | ByteDance, 2026 | **Co-scaling** S & NS in one parameter space | Decoder cross-attn: **NS = Query, S = Key/Value** | User-Item decoupling + **request-level batching** (≈36% FLOPs saved) | [N04](04_Feature_Interaction.md) |
| **TokenMixer-Large** | ByteDance, 2026 | Pure **feature-interaction** backbone (RankMixer) made depth-stable | (S handled upstream; NS/dense tower) | Mixing-and-reverting + interval residuals; **Sparse per-token MoE**; 7B online | [N04](04_Feature_Interaction.md) |
| **Kunlun** | Meta, 2026 | InterFormer + **model-efficiency codesign** for true power-law scaling | Bidirectional S↔NS (Wukong/Adv-FM interaction core) | **GDPA** fused kernel, HSP, sliding-window attn, **CompSkip**; MFU 17→37% | [N04](04_Feature_Interaction.md) |
| **EST** | Alibaba, 2026 | **Effective-rank** analysis → keep only high-signal **S–NS cross block** | Lightweight Cross-Attention (prune self-blocks) | Content Sparse Attention (top-k neighbors); $O(L^2)\toO(LK)$ | [N04](04_Feature_Interaction.md) |
| **HeMix** | Alibaba/AMAP, 2026 | **Interact-then-split** (opposite of OneTrans) to preserve semantics | Q-Former-style learnable query tokens ($Q_G$ global, $Q_R$ real-time) extract interests; HeteroMixer | **Low-rank MLP token mixer** ($d_r \ll N\cdotd_h$) | [N04](04_Feature_Interaction.md) |
| **HSTU lineage** (HSTU, ULTRA-HSTU) | Meta, 2024–26 | **Generative** sequence backbone; replace feature engineering with self-attn | Sequence-first; candidates appended / target-aware (see §6) | Semi-local attention $O((K_1{+}K_2)L)$; FP8/INT4; Attention Truncation | [N05](05_Sequence_Modeling.md) |

**Reading the table for interviews.** Two axes organize everything: **(1) how unified** — one merged stream (OneTrans) vs. independent sequences with cross-attention/decoding (HyFormer, MixFormer, EST, HeMix) vs. interaction-only towers (TokenMixer-Large) vs. sequence-first generative (HSTU); and **(2) where to spend FLOPs** — full attention vs. cross-only/sparse attention, dense MLP vs. sparse-MoE expansion, and architectural vs. kernel-level efficiency (Kunlun's GDPA is mostly the latter). Note the **org clustering**: Meta favors the bidirectional InterFormer→Kunlun + generative HSTU lineages; ByteDance favors token-mixing (RankMixer→TokenMixer-Large) and unified Transformers (OneTrans/HyFormer/MixFormer); Alibaba favors rank-/cross-pruned attention (EST, HeMix).

---

## 6. The HSTU Lineage & Generative Sequence Backbones

The `*Former` family in §5 grows out of the **feature-interaction** side. A parallel lineage grows out of the **sequence** side and is more radical: treat recommendation as **generative sequence modeling** — feed the raw interleaved (item, action) history into a causal Transformer and predict the next interaction, à la a language model, *removing* most hand-engineered features.

- **HSTU** (Hierarchical Sequential Transduction Units; Zhai et al., 2024, Meta) was the **first transformer-style recsys architecture to demonstrate favorable scaling** with sequence length, attention density, and depth. It reframes ranking/retrieval as generative transduction over the behavior sequence.
- **ULTRA-HSTU** ("HSTU 2.0"; Ding et al., 2026) is the **"bend the curve"** flagship. Where most industry teams dodge the $O(L^2)$ self-attention cost with **cross-attention** (queries = candidates or truncated history), ULTRA-HSTU argues self-attention over the *whole* sequence is strictly stronger and just makes it cheap (DeepSeek-V2-inspired): merge item+action into one token (halve $L$), **semi-local attention** ($O((K_1{+}K_2)L)$ — local window + global window), **mixed precision** (BF16/FP8/INT4), **Attention Truncation** (full sequence for the first layers, a selected high-value segment after), and **Mixture of Transducers** (route behavior types to separate transducers). Result: **5.3× training / 21.4× inference scaling efficiency** vs. vanilla HSTU; 18 self-attention layers over 16k-length sequences in production.

The generative backbone makes scoring **candidate-dependent** the self-attention way — *append the candidate to the sequence end under a causal mask* — replacing DIN's separate target-attention head ([Note 05 §4](05_Sequence_Modeling.md)). This connects directly to the **generative-retrieval / Semantic-ID** paradigm in [Note 09 — Generative Recommendation & Semantic IDs](09_Generative_Recommendation_and_Semantic_IDs.md): both treat recsys as next-token prediction; §6 here is the ranking-side, §09 is the retrieval/ID-side. Full ULTRA-HSTU and lifelong write-ups are in [Note 05 — Expanded Reading](05_Sequence_Modeling.md).

**S-side vs NS-side, reconciled.** The two lineages are converging: OneTrans/Kunlun pull NS features *into* a sequence Transformer; ULTRA-HSTU pulls everything *into* the sequence and drops NS features. The open question (§9) is whether NS context features survive at all, or whether enough scale makes raw-sequence modeling subsume them.

---

## 7. Lifelong / Ultra-Long Sequence Taxonomy

Scaling the sequence axis means modeling **lifelong** histories — $O(10^4)$–$O(10^5)$ events per user. The Kuaishou + USTC survey (Zhou et al., 2026) organizes this not by architecture but by the **Efficiency–Effectiveness Balance (EEB)**: efficiency methods cut **Cost-per-Mille (CPM)** ~linearly, effectiveness methods raise **Revenue-per-Mille (RPM)** ~logarithmically, and the objective is **ROI = RPM/CPM**. Methods fall into three families:

| Family | Idea | Representative | Maps to |
|---|---|---|---|
| **Search-based (检索式)** | For each candidate, **retrieve** the relevant sub-sequence (GSU → ESU), then attend on the top-k | SIM, TWIN, QARM-V2 | [Note 05 §5 SIM](05_Sequence_Modeling.md) |
| **Compression-based (压缩式)** | Summarize the whole sequence into fixed-size **memory / clusters** | memory-network / cluster summaries; HSP in Kunlun | [Note 05](05_Sequence_Modeling.md) |
| **Hybrid (混合式)** | Combine retrieval + compression, or sparse self-attention over the full sequence | ULTRA-HSTU's SLA + Attention Truncation | [Note 05 §ULTRA-HSTU](05_Sequence_Modeling.md) |

The survey's empirical message validates §2's premise: **GAUC keeps rising for SIM and TWIN as sequence length grows 10k → 100k** — longer histories reliably help; the only question is the efficiency price. One-third of users (>10k interactions) contribute >95% of watch time, so lifelong modeling is where the value concentrates. SIM is the canonical *search-based* member; ULTRA-HSTU is the *hybrid/self-attention* answer to the same problem. Full detail: [Note 05 — Expanded Reading](05_Sequence_Modeling.md).

---

## 8. Efficiency Is the Real Bottleneck

The recurring lesson of the whole wave: **for recsys, scaling is gated by efficiency, not by ideas.** The architectures are known; getting them to a *useful slope under a real latency/cost budget* is the hard part. The levers (all referenced above, summarized here):

- **MFU — Model FLOPs Utilization (模型算力利用率).** The fraction of peak hardware FLOPs the model actually uses. Recsys sits at 3–15% vs LLMs' 40–60% (Hou et al., 2026), because heterogeneous features give small embedding dims, irregular tensor shapes, and memory-bound kernels. **Raising MFU is the single biggest scaling lever** — it converts nominal FLOPs into effective FLOPs (Kunlun 17→37%, OneTrans 13.4→30.8%). Fixes: fused kernels (Kunlun's GDPA), regularized tensor shapes, mixed precision.
- **KV-cache & request-level batching.** A user is scored against many candidates per request; the **user-side computation is identical** across them. **Cross-request / cross-candidate KV caching** (OneTrans) reuses it, cutting per-session cost $O(C) \to O(1)$; **User-Item decoupling + request-level batching** (MixFormer, ≈36% FLOPs saved) does the same structurally.
- **Sparse / linear attention.** Full self-attention is $O(L^2)$ — fatal for lifelong $L$. **Sliding-window / semi-local attention** (Kunlun; ULTRA-HSTU's SLA = local window $K_1$ + global window $K_2$) gives $O((K_1{+}K_2)L)$; **cross-only / content-sparse attention** (EST) keeps just the high-effective-rank S–NS block, $O(L^2)\to O(LK)$.
- **Computation skip & dynamic topology.** **CompSkip** (Kunlun) selects which components run per layer; **Attention Truncation** (ULTRA-HSTU) runs full-sequence attention only in early layers; **Sparse MoE** (TokenMixer-Large) expands parameters without paying dense FLOPs.

These are the *architectural* efficiency tricks. The full **serving / inference** deep-dive — quantization deployment, kernel engineering, latency budgeting, CPU+GPU serving — belongs in [Note 12 — Serving & Inference Efficiency](12_Serving_and_Inference_Efficiency.md).

---

## 9. Trade-offs & Open Questions

- **Capacity competition between S and NS.** In a single merged stream (OneTrans) heterogeneous sequence and context tokens compete for the same attention/parameter budget — HyFormer and HeMix argue this **blurs semantic boundaries**, motivating "interact-then-split" or independent-sequence cross-attention. Where exactly to draw the S/NS boundary is unresolved.
- **Co-scaling.** With separate S and NS modules they **fight over a fixed FLOPs budget** (MixFormer's core problem). A unified parameter space helps, but the *optimal allocation* between sequence reasoning and feature interaction is empirical and domain-dependent — the recsys analogue of Chinchilla's params-vs-data frontier.
- **When does scaling stop helping?** Recsys data is *non-stationary* (interests drift, items churn) and feedback is noisy/biased, unlike a fixed text corpus. The irreducible-loss floor $L_\infty$ may be hit far sooner, and a steep offline slope need not translate to online GMV. Online A/B gains are real but single-digit-percent, not the order-of-magnitude leaps LLMs saw.
- **Do NS features survive scale?** ULTRA-HSTU removes most human features; OneTrans/Kunlun keep them as first-class tokens (citing business + infra dependencies). Whether sufficient sequence scale subsumes engineered NS features is an open empirical bet.
- **Effectiveness vs. effective rank.** EST's finding that same-type (NS–NS, S–S) attention blocks suffer **rank collapse** while the S–NS cross block carries most signal suggests much of "more attention" is wasted — but aggressively pruning self-attention is risky in domains where NS–NS crosses matter (echoing FM/Bilinear's pairwise-cross philosophy, [Note 04 §4](04_Feature_Interaction.md)).
- **Infra cost.** Most "bend the curve" gains are systems work (custom kernels, FP8/INT4, distributed load balancing) far beyond most teams' infrastructure — the slope you can achieve is partly a function of your hardware budget, not just your architecture.

---

## 10. Key Insights

- **Scaling law (one line):** metric improves as a **power law** in compute/params/data → a **near-log-linear** line you can fit small and extrapolate; "**scaling efficiency**" = the **slope** of that line; "**bending the curve**" = raising the slope.
- **Why DLRMs didn't scale:** they scale in **memorization** (ID-embedding table size) not **dense computation**; feature-interaction **saturates**; the sequence is **compressed early**; and **MFU is 3–15%** so added FLOPs aren't effective. Wukong first showed an interaction stack *can* obey a law.
- **Encode-then-interaction vs. unified:** classic = encode sequence → **compress** → concat NS → DCNv2/DHEN/RankMixer (late, **unidirectional** fusion). Unified = one Transformer over S+NS tokens (early, **bidirectional**, inherits KV-cache/FlashAttention, scales like an LLM).
- **`*Former` map:** organize by **(1) how unified** (OneTrans merged stream ↔ HyFormer/MixFormer/EST/HeMix cross-attention over independent sequences ↔ TokenMixer-Large interaction-only ↔ HSTU sequence-first) and **(2) where FLOPs go** (full vs. sparse/cross-only attention; dense vs. MoE; architecture vs. kernel). Memorize one distinguishing trick each (table in §5).
- **HSTU lineage:** **generative** next-interaction prediction; ULTRA-HSTU keeps **self-attention** (made linear via **semi-local attention**) instead of the cross-attention shortcuts, 5×/21× scaling efficiency; candidate appended to sequence end = candidate-dependent scoring without a DIN head.
- **Lifelong taxonomy:** **search-based** (SIM/TWIN — GSU→ESU retrieval), **compression-based** (memory/cluster summaries), **hybrid** (sparse self-attention). Longer = better (GAUC rises 10k→100k); EEB trade-off ROI = RPM/CPM.
- **Efficiency is the bottleneck:** **MFU** (raise it = biggest lever), **KV-cache / request-level batching** (reuse user-side compute $O(C)\to O(1)$), **sparse/linear attention** ($O(L^2)\to O(LK)$), **CompSkip / MoE**. Serving deep-dive → [Note 12](12_Serving_and_Inference_Efficiency.md).
- **Open questions:** S-vs-NS capacity competition, co-scaling allocation, non-stationary data hitting $L_\infty$ early, whether NS features survive scale, rank-collapse of same-type attention.

**Key Chinese terms:** 扩展律/缩放定律 (scaling law) · 算力 (compute) · 幂律 (power law) · 模型算力利用率 (MFU) · 序列特征 (sequence features, S) · 非序列特征 (non-sequence features, NS) · 先编码后交叉 (encode-then-interaction) · 统一建模 (unified modeling) · 生成式推荐 (generative recommendation) · 终身行为序列 (lifelong behavior sequence) · 检索式/压缩式 (search-based/compression-based) · 稀疏注意力 (sparse attention) · 滑动窗口注意力 (sliding-window attention).

---

## 11. References

- Ding, Q., Course, K., Ma, L., Sun, J., Liu, R., Zhu, Z., et al. (2026). *Bending the scaling law curve in large-scale recommendation systems.* arXiv. https://arxiv.org/abs/2602.16986
- Hou, B., Liu, X., Liu, X., Xu, J., Badr, Y., Hang, M., et al. (2026). *Kunlun: Establishing scaling laws for massive-scale recommendation systems through unified architecture design.* arXiv. https://arxiv.org/abs/2602.10016
- Huang, X., et al. (2026). *MixFormer: Co-scaling up dense and sequence in industrial recommenders.* arXiv. https://arxiv.org/abs/2602.14110
- Huang, Y., et al. (2026). *HyFormer: Revisiting the roles of sequence modeling and feature interaction in CTR prediction.* arXiv. https://arxiv.org/abs/2601.12681
- Jiang, Y., et al. (2026). *TokenMixer-Large: Scaling up large ranking models in industrial recommenders.* arXiv. https://arxiv.org/abs/2602.06563
- Liu, M., et al. (2026). *EST: Towards efficient scaling laws in click-through rate prediction via unified modeling.* arXiv. https://arxiv.org/abs/2602.10811
- Wang, F., et al. (2026). *Query-mixed interest extraction and heterogeneous interaction: A scalable CTR model for industrial recommender systems.* arXiv. https://arxiv.org/abs/2602.09387
- Zeng, Z., Liu, X., Hang, M., et al. (2024). *InterFormer: Effective heterogeneous interaction learning for click-through rate prediction.* arXiv. https://arxiv.org/abs/2411.09852
- Zhang, Z., Pei, H., Guo, J., Wang, T., Feng, Y., Sun, H., Liu, S., & Sun, A. (2025). *OneTrans: Unified feature interaction and sequence modeling with one transformer in industrial recommender.* arXiv. https://arxiv.org/abs/2510.26104
- Zhou, R., Jia, Q., Chen, B., Xu, P., Sun, Y., Lou, S., et al. (2026). *A survey of user lifelong behavior modeling: Perspectives on efficiency and effectiveness.* Preprints.org. https://doi.org/10.20944/preprints202601.1559.v1

*Foundational scaling-law works referenced inline:* Kaplan et al. (2020), *Scaling laws for neural language models*; Hoffmann et al. (2022), *Training compute-optimal large language models* (Chinchilla); Zhang et al. (2024), *Wukong: Towards a scaling law for large-scale recommendation*; Zhai et al. (2024), *HSTU / Actions speak louder than words*.

## Pinterest in Practice (2024–2026)

### PinFM

> Chen, X., Rajesh, K., Lawhon, M., Wang, Z., Li, H., Li, H., Joshi, S. V., Eksombatchai, P., Yang, J., Hsu, Y.-P., Xu, J., & Rosenberg, C. (2025). *PinFM: Foundation model for user activity sequences at a billion-scale visual discovery platform*. In *Proceedings of the Nineteenth ACM Conference on Recommender Systems (RecSys '25)* (pp. 381–390). ACM. https://doi.org/10.1145/3705328.3748050

PinFM is Pinterest's concrete embodiment of this chapter's **foundation-model reuse** thesis: rather than training a standalone large sequence model per surface, they **pretrain once** a 20B+ parameter GPT-2-style decoder on 2 years of ID-based user activity sequences, then **fine-tune cheaply** into multiple downstream DLRM/DCN rankers (Home Feed, Related Items), serving >500M users. The behavior sequence is fused into the ranker with **early fusion** — the candidate item is appended to the user sequence so cross-attention conditions the history on the candidate — the same "unified, candidate-dependent self-attention" idea as the HSTU lineage (§6), but in a pretrain-then-finetune wrapper. Their serving innovation, the **Deduplicated Cross-Attention Transformer (DCAT)**, is a sharp instance of §8's *efficiency-as-bottleneck* lesson: because unique user sequences vastly outnumber candidates (≈1:1000 at serving), they run the user-history transformer **once and KV-cache it**, then **cross-attend each candidate** against the deduplicated cached context — yielding **+600% serving throughput** vs FlashAttention self-attention, the same "reuse the identical user-side computation" principle as OneTrans's cross-request KV cache and MixFormer's request-level batching. **Post-training int4 quantization** (FBGEMM min-max) shrinks the 20B-param embedding table to ~31% of its size with only 0.06% offline metric drop, making the whole system **cost-neutral** in production — exactly the quantization/MFU-style lever that turns nominal FLOPs into an affordable serving budget. Their ablations also recover this chapter's scaling-law shape on the *vocabulary* and *iteration* axes (e.g. growing the embedding table 20M→160M rows adds +1.98% Save, and fine-tuning is essential — a frozen PinFM gives only ~+0.10% Save vs +3.76% fine-tuned). For the deployment-side detail — int4 PTQ, KV-cache rotation, and CPU+GPU split serving — see [Note 12 — Serving & Inference Efficiency](12_Serving_and_Inference_Efficiency.md).
