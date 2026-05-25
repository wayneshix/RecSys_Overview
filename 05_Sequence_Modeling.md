# Part 05 — User Behavior Sequences / LastN (用户行为序列, "LastN")

Recommendation systems course notes — Part 05: modeling the user's behavior sequence.

A user's **LastN** is the sequence of the last $n$ items the user interacted with (clicked, liked, saved, shared, etc.). This sequence is one of the strongest signals about what kind of item a user is interested in. Adding LastN features to retrieval and ranking models tends to lift *all* metrics noticeably. This part traces the evolution of how we turn that sequence into a usable feature:

- **LastN averaging** — embed the last-$n$ items, average the vectors. Simple, candidate-independent, works everywhere (双塔/三塔/精排).
- **DIN (Deep Interest Network)** — replace the plain average with an *attention-weighted* average, where weights are the similarity of each LastN item to the **candidate item** (候选物品). Better accuracy, but candidate-dependent, so it only works where the model can see the candidate (i.e. ranking, not the two-tower user tower).
- **SIM (Search-based Interest Model)** — for *long* sequences ($n$ in the thousands), first do a cheap **search** (查找) to keep only the $k$ items most relevant to the candidate, then run attention on those $k$. This captures long-term interest while keeping compute bounded.

---

## Table of Contents

1. [Background: Where LastN Fits in the Pipeline](#1-background-where-lastn-fits-in-the-pipeline)
2. [LastN Averaging (取平均): The Baseline Feature](#2-lastn-averaging-the-baseline-feature)
3. [DIN — Deep Interest Network (注意力机制)](#3-din--deep-interest-network)
4. [Simple Average vs. Attention: Where Each Is Allowed](#4-simple-average-vs-attention-where-each-is-allowed)
5. [SIM — Search-based Interest Model (长序列建模)](#5-sim--search-based-interest-model)
6. [Interview Summary & Talking Points](#6-key-insights)

---

## 1. Background: Where LastN Fits in the Pipeline

Recall the multi-objective ranking model (多目标排序模型) from Part 04. The model concatenates several feature groups — **user features** (用户特征), **item features** (物品特征), **statistical features** (统计特征), **scene/context features** (场景特征) — feeds them through a neural network, and outputs predictions for click-through rate (点击率), like rate (点赞率), favorite rate (收藏率), share rate (转发率), etc. (each via its own fully-connected + sigmoid head). Those predicted scores are then fused to rank items.

This entire Part focuses on **one slice of the user features: the LastN behavior sequence**. The question is: given the IDs of the last $n$ items a user interacted with, how do we compress them into a vector (or a few vectors) that the network can consume?

> Wang's note: every company implements this differently, mostly because the answer depends on **engineering infrastructure** (系统基建). Stronger infra → more expensive, more accurate methods.

---

## 2. LastN Averaging: The Baseline Feature

### 2.1 Definition

- **LastN (LastN 特征):** the item IDs (物品 ID) of the user's last $n$ interactions, where "interaction" (交互) covers click, like, favorite/save, share, comment, watch-completion, etc.
- **Embed** each of the $n$ item IDs → $n$ vectors.
- **Average** the $n$ vectors → a single vector.
- Use that averaged vector as one of the user's features (a representation of "what kind of item this user has historically liked").

Formally, let the LastN item embeddings be $\mathbf{x}_1, \dots, \mathbf{x}_n \in \mathbb{R}^d$. The feature is:

$$
\mathbf{u}_{\text{avg}} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{x}_i .
$$

Reference: Covington, Adams, and Sargin, *Deep Neural Networks for YouTube Recommendations*, RecSys 2016 — this is the original "average of watched-video embeddings as a user vector" idea.

### 2.2 Where it applies

Averaging is **candidate-independent** — it depends only on the user's own history, not on which item we are scoring. So it can be placed anywhere:

- Retrieval **two-tower model** (召回双塔模型) — put $\mathbf{u}_{\text{avg}}$ into the **user tower** (用户塔).
- Pre-ranking **three-tower model** (粗排三塔模型).
- Fine-ranking model (精排模型).

### 2.3 Multiple sequences, one per behavior type (小红书 practice)

In practice you don't keep a single LastN; you keep **one sequence per behavior type** and average each separately:

- LastN of **clicked** items (点击的 LastN) → average → one vector.
- LastN of **liked** items (点赞的 LastN) → average → one vector.
- LastN of **favorited** items (收藏的 LastN) → average → one vector.
- ... (shares, comments, watch-completion, etc.)

Concatenate these per-behavior averaged vectors into the user feature block.

### 2.4 Two practical refinements (from the audio)

1. **Don't average raw IDs only.** Alongside each item's ID embedding, also include embeddings of other item attributes — e.g. the item's **category** (物品类目). Concatenate (ID embedding ⊕ attribute embeddings) before averaging. Empirically richer than ID-only.
2. **Averaging is the early/simple approach** and is still widely used. The better-performing alternative is **attention**, at the cost of more compute — which motivates DIN next.

### 2.5 Why averaging is lossy (the motivation for DIN)

Averaging throws away two things:

- **It is order- and importance-blind.** Every interacted item gets weight $1/n$ regardless of how relevant it is to *what we're scoring right now*.
- **It is candidate-agnostic.** When scoring a *food* (美食) candidate, a user's beauty (美妆) and car (汽车) history get the same weight as their food history, even though only the food history is predictive for this candidate. The average lands somewhere in the "center of mass" of all interests — possibly far from any actual cluster (see §3.3).

This is exactly the gap DIN closes.

---

## 3. DIN — Deep Interest Network

> **One-liner:** DIN replaces the simple average with a **weighted average** (加权平均), where the weight on each LastN item is its **similarity to the candidate item**. This is, mechanically, an **attention mechanism**. Reference: Zhou et al., *Deep Interest Network for Click-Through Rate Prediction*, KDD 2018 (Alibaba).

### 3.1 Setup

- LastN item vectors (the user's history): $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n$.
- **Candidate item** vector (候选物品向量): $\mathbf{q}$.

> **Candidate item:** the item currently being scored. Example: pre-ranking hands ranking ~500 candidates; the ranking model must assign each a score reflecting the user's interest, then keep the top few dozen to show. *Do not confuse a LastN item (history) with a candidate item (the thing we're scoring now).*

### 3.2 The attention computation

**Step 1 — similarities (相似度).** For each LastN vector $\mathbf{x}_i$, compute a scalar similarity $\alpha_i$ to the candidate $\mathbf{q}$:

$$
\alpha_i = \text{sim}(\mathbf{x}_i, \mathbf{q}), \qquad i = 1, \dots, n.
$$

The similarity function can be an **inner product** $\mathbf{x}_i^\top \mathbf{q}$, **cosine similarity**, or something more elaborate (e.g. a small MLP on $[\mathbf{x}_i, \mathbf{q}, \mathbf{x}_i - \mathbf{q}, \mathbf{x}_i \odot \mathbf{q}]$, as in the original paper). Each $\alpha_i$ is a real number; the more similar a LastN item is to the candidate, the larger its weight.

**Step 2 — weighted sum (加权和).** Use the similarities as weights and sum the LastN vectors:

$$
\mathbf{u}_{\text{DIN}} = \sum_{i=1}^{n} \alpha_i \mathbf{x}_i .
$$

(If the $\alpha_i$ are normalized — e.g. via softmax — this is a weighted *average*; the lecture states it as a weighted sum / weighted average. The key point is that weights are candidate-dependent.)

**Step 3 — use as a feature.** The resulting vector $\mathbf{u}_{\text{DIN}}$ is treated as a user-behavior feature, concatenated with the other features, and fed to the ranking model to predict click rate, like rate, etc. — *for that specific candidate*.

### 3.3 DIN = single-head attention (本质是注意力机制)

Map DIN onto the standard attention vocabulary:

| Attention role | DIN object |
|---|---|
| **Query** (query) | candidate item vector $\mathbf{q}$ (just one vector) |
| **Keys** (key) | LastN vectors $\mathbf{x}_1, \dots, \mathbf{x}_n$ |
| **Values** (value) | LastN vectors $\mathbf{x}_1, \dots, \mathbf{x}_n$ (same as keys) |

Feed all of these into a **single-head attention layer** (单头注意力层). Because there is exactly one query, the output is a single vector. If you know attention, DIN is just attention with the candidate as the query and the history as keys/values.

### 3.4 Why DIN works (geometric intuition)

Picture the embedding space with three interest clusters: beauty, food, cars.

- **Simple average (简单平均):** the user vector lands at the centroid of *all* the user's history — somewhere in the middle, not inside any cluster. It is the *same* vector no matter what we're scoring.
- **DIN with a food candidate:** the candidate (a food item) is near the food cluster. Similarities to the food-history items are high; similarities to beauty/car items are near zero. The weighted average is pulled into the **food cluster** — a representation that says "this user, *with respect to food*, looks like this." Strong signal that the food candidate is a good match.
- **DIN with a news candidate** (a category the user has *never* engaged with): the candidate is far from all clusters, all similarities are low, and the weighted average sits near the origin / between clusters — correctly signaling weak interest.

So DIN's user representation **adapts to the candidate**, surfacing the slice of history that is actually relevant. That adaptivity is what averaging cannot do.

---

## 4. Simple Average vs. Attention: Where Each Is Allowed

This distinction is a classic interview point.

| | **Simple Average** | **DIN / Attention** |
|---|---|---|
| Inputs needed | LastN only | LastN **+ candidate item** |
| Candidate-dependent? | No (user's own feature) | Yes |
| Fine-ranking (精排)? | ✅ | ✅ |
| Two-tower retrieval (双塔)? | ✅ (user-tower input) | ❌ |
| Three-tower pre-ranking (三塔)? | ✅ | ❌ |

**Why attention can't go in the two-tower / three-tower user tower:** the whole point of the two-tower architecture is that the **user tower never sees the candidate item**. (At serving time there may be hundreds of millions of candidate items; the user vector is computed *once* and reused via ANN search across all of them.) DIN needs the candidate as its query, so it cannot live in the user tower. Simple averaging needs only the user's own LastN, so it can.

> Mnemonic: **average = user-only feature → goes anywhere; attention = needs the candidate → ranking only.**

---

## 5. SIM — Search-based Interest Model

> **One-liner:** SIM keeps a *very long* behavior sequence (long-term interest) but, for each candidate, first **searches** the sequence to retain only the $k$ most relevant items, then runs DIN-style attention on those $k$. Reference: Qi et al., *Search-based User Interest Modeling with Lifelong Sequential Behavior Data for CTR Prediction*, CIKM 2020 (Alibaba).

### 5.1 The problem with DIN: it can't go long

DIN's attention layer has compute **proportional to $n$**, the sequence length:

$$
\text{cost}_{\text{DIN}} \propto n .
$$

So DIN can only afford to keep the last ~100–200 items, otherwise the cost explodes. The downside (缺点): a short window captures only **short-term interest** (短期兴趣) and **forgets long-term interest** (遗忘长期兴趣).

Experiments show that **lengthening the sequence reliably lifts all recommendation metrics** — long-term interest matters. But *brute-force* lengthening (just feeding DIN a sequence of thousands) is a bad trade: the added compute is huge and the marginal gain doesn't justify it. We want large $n$ (e.g. thousands) **and** bounded compute.

### 5.2 The key insight (how to improve DIN)

DIN computes a weighted average where weights are candidate–history similarities. **If a LastN item is very dissimilar to the candidate, its weight is ≈ 0** — so dropping it barely changes the result.

> Example: candidate is a **food** note; a particular history item is a **women's-beauty** note. They're clearly dissimilar → weight ≈ 0. Removing that beauty note from the sequence has almost no effect on DIN's weighted average.

Therefore: **quickly throw away the history items that are irrelevant to the candidate, and run attention only on the relevant ones.** This shrinks the attention layer's input from $n$ down to a small $k$ while leaving the output essentially unchanged.

### 5.3 The two stages

SIM = **two steps**: (1) **search** to reduce LastN → TopK, then (2) **attention** on the TopK.

- Keep a long history: $n$ can be **a few thousand**. (Without SIM, $n$ is typically only 100–200.)
- For each candidate, do a **fast search** in the user's LastN to find the $K$ most similar items ($K$ small, e.g. 100).
- Feed those TopK (not all $n$) into the attention layer.
- Compute drops from $\propto n$ to $\propto k$.

> Worked example: store $n = 1000$ recent item IDs. Using the candidate's category, quickly discard the vast majority, keeping the ~100 most relevant. Search turned LastN (1000) into TopK (100); attention runs on 100.

*(Wang's original papers' naming: the search stage is the **GSU — General Search Unit**, and the attention stage is the **ESU — Exact Search Unit**. The lecture refers to them simply as "step 1: search" and "step 2: attention.")*

### 5.4 Step 1 — Search: Hard vs. Soft

**Method 1 — Hard Search (硬搜索):**
- Rule-based filtering. E.g., using the **candidate's category** (类目), keep only the LastN items with the *same category*.
- **Simple, fast, no training required** (无需训练). Trivial to implement; very fast online.

**Method 2 — Soft Search (软搜索):**
- Embed every item into a vector.
- Use the **candidate vector as the query** and do **$k$-nearest-neighbor (k-NN) search** (最近邻查找) over the LastN vectors, keeping the $k$ closest.
- **Better accuracy** (higher AUC for CTR-type targets in the paper) but **more complex to implement and more expensive to compute**.

**Which to use?** Depends on engineering infrastructure (工程基建). Soft search is worth it only if your infra is strong; otherwise hard search (rule-based, easy, fast online) is the pragmatic default.

### 5.5 Step 2 — Attention on the TopK

Mechanically **identical to DIN**, with one change: the input is the **TopK** items ($\mathbf{x}_1, \dots, \mathbf{x}_k$) rather than the full LastN. As argued in §5.2, the discarded items had near-zero attention weight, so the output vector is essentially the same as if we'd run attention on all $n$ — but far cheaper. The output (purple) vector becomes a user-behavior feature for the ranking model (predict CTR etc.).

### 5.6 Time-interval features (使用时间信息) — a useful trick

A practical SIM trick is to inject **how long ago** each interaction happened:

- Let $\delta$ = time since the user interacted with a given LastN item (e.g. "1000 hours ago" → $\delta = 1000$). $\delta$ is continuous.
- **Discretize** $\delta$ into buckets (离散化) — e.g. within the last 1 day, 7 days, 30 days, 1 year, more than 1 year — then **embed** the bucket into a vector $\mathbf{d}$.
- Now each LastN item is represented by **two vectors concatenated**: the item embedding $\mathbf{x}$ ⊕ the time embedding $\mathbf{d}$.
- Per the figure: each of the TopK items contributes $[\mathbf{d}_i; \mathbf{x}_i]$; feed all $k$ of these (plus candidate $\mathbf{q}$, which needs **no** time embedding) into the attention layer.

**Why SIM needs time but DIN doesn't:**
- DIN's sequence is short (~100–200 items) — almost all interactions are recent, so "how long ago" carries little information.
- SIM's sequence is long-term: an item could be from a year ago or ten minutes ago, and **older interactions are generally less important**. The model should down-weight stale interactions when predicting CTR, so it needs the time signal. The paper shows time information gives a significant lift.

### 5.7 SIM paper conclusions (结论)

1. **Long sequence (long-term interest) > short sequence (recent interest)** — verified at 小红书 too.
2. **Attention > simple average** — the paper compares "search then average" vs. "search then attention"; attention wins.
3. **Soft search > hard search** in accuracy, but hard search is far easier to implement (just rules) — choice depends on infra.
4. **Using time information helps.**

---

## 6. Key Insights

**The progression and *why* each step happens:**

1. **Average (YouTube-DNN-style):** embed LastN, average. Candidate-independent → usable in the two-tower user tower and everywhere else. *Lossy:* gives every history item equal weight, ignores which slice of history is relevant to the candidate.
2. **DIN (KDD 2018):** weighted average; weights = similarity(candidate, history item) = attention with candidate as query, history as key/value. Adapts the user representation to the candidate (geometric intuition: pulls toward the relevant cluster). *Cost $\propto n$* → can only keep ~100–200 items → forgets long-term interest. *Candidate-dependent* → cannot go in the two-tower/three-tower user tower (the tower never sees the candidate).
3. **SIM (CIKM 2020):** to get long-term interest ($n$ ~ thousands) *without* paying $O(n)$ attention, first **search** (GSU) to keep the $K$ candidate-relevant items, then run DIN-style **attention** (ESU) on those $K$. Safe to drop the rest because their attention weights are ≈ 0. Search is **hard** (category-rule, fast, no training) or **soft** (vector k-NN, more accurate, more expensive). Adds **time-interval embeddings** because long sequences span very different recencies.

**Likely interview questions:**
- *Why does averaging lose information?* → equal weights, candidate-agnostic; centroid may sit between clusters (§2.5, §3.4).
- *Why can't DIN be used in the two-tower retrieval model?* → it needs the candidate as the query; the user tower is computed once without seeing the candidate (§4).
- *Why does SIM need time features but DIN doesn't?* → DIN's window is short/recent; SIM's is long-term, and recency should down-weight stale items (§5.6).
- *Why is it safe to drop most of the sequence in SIM's search step?* → dropped items are dissimilar to the candidate → near-zero attention weight → negligible effect on the weighted average (§5.2).
- *Complexity:* DIN attention $\propto n$; SIM reduces it to $\propto k$ with $k \ll n$.

**Key Chinese terms:** 用户行为序列 (user behavior sequence) · LastN / 行为序列 · 取平均 (averaging) · 候选物品 (candidate item) · 注意力机制 (attention) · 加权平均/加权和 (weighted average/sum) · 相似度 (similarity) · 单头注意力层 (single-head attention layer) · 短期/长期兴趣 (short-/long-term interest) · 查找 (search) · Hard Search · Soft Search · 最近邻查找 (nearest-neighbor search) · TopK · 离散化 (discretization) · 时间信息 (time information) · 用户塔 (user tower) · 双塔/三塔/精排 (two-tower / three-tower / fine-ranking).

## Expanded Reading — Recent Advances (2026)

The 2018–2020 lineage in this chapter (LastN averaging → DIN → SIM) optimizes *how* to read a behavior sequence; the 2026 frontier instead scales the sequence model itself along three axes — **HSTU-style generative sequence backbones** that replace feature engineering with end-to-end self-attention, **lifelong / ultra-long sequence efficiency** (sequences of $10^4$–$10^5$ events), and **user foundation models** that pre-train one reusable representation from long action sequences.

### ULTRA-HSTU: bending the scaling-law curve with efficient self-attention

> Ding, Q., Course, K., Ma, L., Sun, J., Liu, R., Zhu, Z., … Li, R. (2026). *Bending the scaling law curve in large-scale recommendation systems*. arXiv. https://arxiv.org/abs/2602.16986

**Method.** ULTRA-HSTU ("HSTU 2.0", Meta) is a next-generation generative sequence ranker that, unlike the cross-attention shortcuts most industry teams adopt to dodge the $O(L^2)$ self-attention bottleneck, keeps full *self*-attention and makes it cheap via model–system co-design (inspired by DeepSeek-V2). Four ideas: (1) **Action encoding** — instead of vanilla HSTU's interleaved item/action tokens (which double ranking-sequence length), merge each item and its action into a *single* token by adding their embeddings ($x_{i,j} = I_{i,j} + a_{i,j}$, with candidate actions masked to avoid label leakage), halving $L$. (2) **Semi-local attention (SLA)** — a sparse mask combining a *local window* $K_1$ (recent local patterns) and a *global window* $K_2$ (earliest long-term context), cutting attention from $O(L^2)\to O((K_1+K_2)\cdot L)$ while, unlike DeepSeek's local-only NSA, preserving long-term interest. (3) **Mixed precision** — BF16 for stability, FP8 for the dominant GEMMs, INT4 embedding quantization. (4) Dynamic topology: **Attention Truncation** runs the first $N_1$ layers on the full sequence, then selects a shorter high-value segment for $N_2$ more layers (avoiding $O(D\cdot L)$ depth cost), and **Mixture of Transducers (MoT)** routes heterogeneous behavior types into separate transducers.

**Key results.** 5.3× training and 21.4× inference scaling efficiency vs. vanilla HSTU. Deployed at scale (18 self-attention layers over 16k-length sequences, hundreds of H100s) serving billions of users daily; +4–8% consumption/engagement and +0.217% topline. SLA alone gives >5× inference scaling efficiency; LBSL load-balancing adds 15% training throughput; the FP8/INT4 stack adds 10% training / 40% serving throughput.

**Trade-offs / limitations.** Heavy infrastructure cost (custom FlashAttention-V3-style SLA kernels for SiLU attention and non-standard masks, multi-format quantization, distributed load balancing) — far beyond most teams' infra. SLA's quality hinges on tuning $K_1, K_2$, and the value-segment heuristics in Attention Truncation are domain-specific.

**Connection to this chapter.** This is the self-attention answer to the same problem SIM (§5) solves with search: how to model very long histories under bounded compute. SIM keeps DIN-style *target-aware cross-attention* over a GSU-selected TopK; ULTRA-HSTU argues self-attention over the *whole* sequence is strictly stronger if you make it linear (SLA) rather than truncating queries. Attention Truncation is a learned, layered analogue of SIM's GSU "keep only the valuable part," and MoT generalizes the §2.3 "one sequence per behavior type" idea into separate transducers.

### A Survey of User Lifelong Behavior Modeling (ULBM)

> Zhou, R., Jia, Q., Chen, B., Xu, P., Sun, Y., Lou, S., … Chen, E. (2026). *A survey of user lifelong behavior modeling: Perspectives on efficiency and effectiveness*. Preprints.org. https://doi.org/10.20944/preprints202601.1559.v1

**Method.** A Kuaishou + USTC survey (44 pp.) organizing ultra-long-sequence modeling not by architecture but by the **Efficiency–Effectiveness Balance (EEB)**: efficiency-oriented methods cut Cost-per-Mille (CPM) roughly linearly, effectiveness-oriented methods raise Revenue-per-Mille (RPM) roughly logarithmically, and ROI = RPM/CPM. It taxonomizes methods into **search-based** (GSU/ESU retrieval of relevant sub-sequences, e.g. SIM, TWIN), **compression-based** (summarize the whole sequence into fixed-size memory/clusters), and **hybrid** approaches, then splits efficiency into algorithmic vs. system-level and effectiveness into intrinsic sequential dependencies vs. external/contextual signals.

**Key results.** Documents lifelong sequences reaching length $\sim 10^5$ on Kuaishou; one-third of users (>10k interactions) contribute >95% of total watch time; >82% of users show ≥5 behavior types. Reproduces the curve that GAUC keeps rising for both SIM and TWIN as sequence length grows 10k→100k — empirical justification for going lifelong. Maintains a living repo of methods/datasets.

**Trade-offs / limitations.** A survey, not a method — no new model; coverage is industrially biased toward production-validated work (lighter on academic-only methods).

**Connection to this chapter.** Places SIM (§5) precisely: SIM is the canonical **search-based** family member, and its hard/soft GSU search is one point on the EEB frontier. The survey's message — *longer sequences reliably help, the only question is the efficiency price* — is exactly §5.1's "lengthening the sequence lifts all metrics," now formalized as a CPM/RPM trade-off and extended from $n\sim$ thousands to $n\sim 10^5$.

### Feed SR: an industrial-scale sequential recommender for LinkedIn Feed

> Hertel, L., Srivastava, G., Naqvi, S. A., Kumar, S., Zhang, Y., Ocejo, B., … Ghosh, S. (2026). *An industrial-scale sequential recommender for LinkedIn Feed ranking*. arXiv. https://arxiv.org/abs/2602.12354

**Method.** Feed SR replaces LinkedIn Feed's DCNv2-based ranker with a transformer sequential ranker. It interleaves the member's last $T=1000$ impressed posts with the actions taken on each (HSTU/generative style), processes them through causal-masked transformer blocks, *discards* the action-position outputs, then concatenates context features and feeds a multi-task "head" DNN (passive tasks: click/skip/long-dwell; active tasks: like/comment/share). Evolving interests are captured via **RoPE** positional encoding, incremental daily training, and a recency-weighted loss; the dynamic billion-post corpus is handled by **late-fusing numeric features** (interaction/affinity counts) and static text/content embeddings rather than learning everything in-sequence; member profile embeddings help long- and short-history members alike. Candidates are appended to the sequence end and scored in one pass.

**Key results.** +2.10% time spent in online A/B vs. the DCNv2 production model; now the primary ranker on LinkedIn Feed. The team reports that fine-tuned LLM rankers and TransAct-style alternatives lost to Feed SR on the online-metrics-vs-production-efficiency trade-off, requiring CPU+GPU serving optimizations to meet real-time latency over 1.2B members.

**Trade-offs / limitations.** Length capped at $T=1000$ (not lifelong); much signal is *late-fused* engineered/numeric features rather than learned end-to-end (the opposite philosophy to ULTRA-HSTU's "remove human features"); strict real-time-on-CPU constraints drove architecture choices over raw quality.

**Connection to this chapter.** Same generative-sequence idea as ULTRA-HSTU but a pragmatic mid-scale instance, and a direct successor to DIN's target-awareness: appending candidates to the sequence end under a causal mask is the self-attention way to make scoring *candidate-dependent* (§4) without a separate DIN attention head. The retained late-fused numeric/affinity features echo §2.4's "don't use IDs alone — add attributes."

### QARM V2: reasoning-aligned multimodal Semantic IDs for GSU/ESU

> Xia, T., Zhang, J., Liu, Y., Dou, H., Yin, T., Cao, J., … Gai, K. (2026). *QARM V2: Quantitative alignment multi-modal recommendation for reasoning user sequence modeling*. arXiv. https://arxiv.org/abs/2602.08559

**Method.** QARM V2 (Kuaishou) keeps the SIM/TWIN **GSU→ESU** paradigm but replaces brittle ID embeddings with multimodal-LLM semantics, fixing two failures of naively bolting on frozen LLM embeddings: *representation unmatch* (LLM captioning objectives ≠ recsys objectives) and *representation unlearning* (frozen embeddings can't update end-to-end). At the **GSU** side, a **reasoning item-alignment** mechanism fine-tunes an LLM on contrastive item pairs mined from Swing (I2I) and two-tower (U2I) retrieval, *filtered by a reasoning LLM* (Qwen3) to drop hot-popular-biased / unrelated pairs (rejecting ~10% of I2I and ~70% of U2I), plus a three-segment attention mask (input / `<EMB>` compression / QA) that unifies next-token prediction with embedding compression — yielding *business-aligned* LLM embeddings for semantic GSU retrieval of relevant sub-sequences. At the **ESU** side, **Res-KmeansFSQ** quantizes those embeddings into multi-level **Semantic IDs**: residual K-means for the first two coarse layers (category/usage) and **Finite Scalar Quantization** for the last layer (a distribution-independent uniform grid that cuts codebook collisions, which exceeded 30% with naive Res-Kmeans). The SIDs are learnable discrete features trained end-to-end with the ranking model.

**Key results.** Significant offline and online A/B gains across short-video, advertising, and live-streaming; deployed serving 400M daily active users over sequences exceeding $10^5$ per user.

**Trade-offs / limitations.** Requires an LLM fine-tuning + reasoning-filtering data pipeline (Qwen3-0.6B/8B, Qwen2.5-VL-72B, Gemini) and periodic codebook retraining — substantial offline cost; quality depends on the reasoning model's pair-filtering judgments.

**Connection to this chapter.** A direct descendant of SIM (§5) and its TWIN successors: it inherits the exact two-stage GSU search → ESU attention structure (§5.3), but upgrades the *similarity signal* the search runs on. SIM's hard search uses category rules and soft search uses ID-embedding k-NN (§5.4); QARM V2's GSU instead retrieves on reasoning-aligned multimodal embeddings, and its Semantic IDs give the ESU richer, generalizable item representations than raw ID embeddings — addressing the cold-start / long-tail weakness of ID-only LastN noted in §2.4.

### Scaling Personalization with User Foundation Models

> Edezhath, R. (2026). *Scaling personalization with user foundation models*. Coinbase Engineering Blog.

**Method.** Coinbase pre-trains one **user foundation model**: a Transformer **two-tower** encoder over a single long sequence containing nearly every user interaction over several months (batch history + last-24h streaming actions, PII-free tokens). Training is **self-supervised** via *sequence-pair classification* — sample two non-overlapping activity windows; positive pairs come from the same user, negatives from different users; the model must decide whether two sequences belong to the same user. The resulting embedding is reused three ways downstream: (1) **direct similarity** (zero/few-shot cosine retrieval), (2) **static embedding features** into a supervised model, or (3) **fine-tuning** the pre-trained block (best performing).

**Key results.** Same-user-pair pretraining beat masked-token and BERT-style alternatives on a multi-domain eval suite (ROC-AUC / PR-AUC across notification clicks, segmentation, ranking). The "Smart Targeting" explore-train-target loop **more than doubled engagement** vs. hand-designed heuristics for cold-start new-asset notifications, and collapsed new-use-case setup from days of feature engineering to minutes.

**Trade-offs / limitations.** The standard Transformer encoder captures only event *order*, not *timing/intervals* — explicitly flagged as the major open problem (recall §5.6: SIM's whole point was injecting time-interval embeddings into long sequences). The pairwise-classification objective is a proxy, not a recsys task, so downstream value must be checked via the eval suite.

**Connection to this chapter.** This is the LastN-as-foundation idea: where §2 averages LastN into a *task-specific* user feature, Coinbase encodes the long sequence *once, self-supervised,* into a *task-agnostic* reusable user vector — the two-tower setup keeps it candidate-independent (§4), so like LastN-averaging it can sit in a user tower or seed many downstream models. Its open problem (no time signal) is exactly the gap §5.6's time-interval embeddings fill, confirming that for long sequences recency must be modeled explicitly.

---

## Pinterest in Practice (2024–2026)

### TransAct V2 — lifelong sequences attended end-to-end (no GSU/ESU compression)

> Xia, X., Joshi, S. V., Rajesh, K., Li, K., Lu, Y., Pancha, N., Badani, D. D., Xu, J., & Eksombatchai, P. (2025). *TransAct V2: Lifelong user action sequence modeling on Pinterest recommendation*. In *Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM '25)* (pp. 6881–6882). ACM. https://doi.org/10.1145/3746252.3761433

TransAct V2 is Pinterest's production Homefeed CTR ranker (the successor to TransAct, KDD '23), and it jointly feeds a real-time sequence and a **lifelong sequence of length $O(10^4)$** through a transformer encoder, with each action embedded as $E_{act}+E_{surf}+E_{pos}$ over PinSage Pin embeddings to produce a user representation $u(t)$. It adds an auxiliary **Next Action Loss (NAL)** — predicting the user's next action $p(t+1)$ from $u(t)$ — where impression-based negatives ($\text{NAL}_{imp}$) beat in-batch negatives and notably lift diversity. The headline result is **+13.31% offline HIT@3/repin** and an online **Homefeed Repin Volume +6.35%**, **Hide Volume −12.80%**, with **Time Spent +1.41%** (a 1% repin lift is considered substantial). The key **contrast with this chapter**: SIM/TWIN (§5) tackle long histories with a *search-then-attention* pipeline — a GSU compresses the $O(10^3)$ sequence down to a candidate-relevant TopK, then the ESU attends only over that small set. TransAct V2 instead attends over the *entire* $O(10^4)$ lifelong sequence **end-to-end**, with no GSU compression and no stale offline-inference cache, relying on data-processing and serving optimizations to stay within production latency. In short, where SIM bounds compute by *shrinking the sequence*, Pinterest bounds it by *making full-sequence attention cheap enough to serve* — and the NAL objective is an extra ingredient SIM/TWIN do not have, turning the ranker into a partially self-supervised next-action predictor.
