# Recommendation Systems — Part 06: Re-ranking & Diversity (重排)

> Recommendation systems course notes — Part 06: Re-ranking & Diversity.
> Covers item similarity measures, the role of diversity, MMR, business-rule constraints, and DPP.
> English with key Chinese terms in parentheses. ML-interview depth.

---

## Table of Contents

1. [Where Diversity Lives in the Pipeline](#1-where-diversity-lives-in-the-pipeline)
2. [Measuring Item Similarity](#2-measuring-item-similarity)
   - 2.1 [Attribute-tag based](#21-attribute-tag-based)
   - 2.2 [Embedding based](#22-embedding-based)
   - 2.3 [Content-based embeddings via CLIP](#23-content-based-embeddings-via-clip)
3. [Why Diversity Matters](#3-why-diversity-matters)
4. [MMR — Maximal Marginal Relevance](#4-mmr--maximal-marginal-relevance)
   - 4.1 [The objective](#41-the-objective)
   - 4.2 [The greedy algorithm](#42-the-greedy-algorithm)
   - 4.3 [Sliding window](#43-sliding-window)
   - 4.4 [Complexity](#44-complexity)
5. [Business-Rule Constraints](#5-business-rule-constraints)
   - 5.1 [Three rule families](#51-three-rule-families)
   - 5.2 [Combining rules with the diversity algorithm](#52-combining-rules-with-the-diversity-algorithm)
6. [DPP — Determinantal Point Process](#6-dpp--determinantal-point-process)
   - 6.1 [Math foundation: hyper-parallelotope volume](#61-math-foundation-hyper-parallelotope-volume)
   - 6.2 [Volume ↔ determinant](#62-volume--determinant)
   - 6.3 [The DPP / k-DPP objective](#63-the-dpp--k-dpp-objective)
   - 6.4 [Greedy algorithm](#64-greedy-algorithm)
   - 6.5 [Brute force vs. Cholesky speedup (Hulu fast algorithm)](#65-brute-force-vs-cholesky-speedup-hulu-fast-algorithm)
   - 6.6 [Sliding window + rules for DPP](#66-sliding-window--rules-for-dpp)
   - 6.7 [MGS — a simpler equivalent (bonus)](#67-mgs--a-simpler-equivalent-bonus)
7. [Reward vs. Diversity Trade-off — Intuition Summary](#7-reward-vs-diversity-trade-off--intuition-summary)
8. [Interview Cheat-Sheet](#8-key-insights)

---

## 1. Where Diversity Lives in the Pipeline

The recommendation pipeline (链路) is:

```
recall (召回)  →  pre-rank (粗排) → post-proc → rank (精排) → post-proc → items 1..k
  ~hundreds of M     ~thousands               ~hundreds
```

- **Pre-rank and rank** use multi-objective models to score each item **pointwise**: for item `i` they predict CTR, interaction rate, etc., fuse them into a single score `reward_i`. `reward_i` represents the **user's interest in / intrinsic value of** item `i`. "Pointwise" means each item is scored independently, ignoring relations between items.
- The **post-processing (后处理)** step takes the `n` candidate items with scores `reward_1, …, reward_n` and selects `k` of them so that the chosen set has **both high total score and good diversity**.
- The **post-processing after rank is called re-ranking**. The post-processing after pre-rank also needs a diversity step, but it is not usually treated as a separate stage and uses a simpler/faster algorithm.

**Why diversity is applied as post-processing.** The ranking models are deliberately kept pointwise (their only job is accurate score prediction). Diversity is a property of the *set*, so it is layered on top as a separate selection step. MMR and DPP have non-trivial cost, so they are practical for rank post-processing (pick tens out of hundreds) but **too slow for pre-rank post-processing** (pick hundreds out of thousands), where a coarser, faster variant is used.

**Business impact.** Industry experience shows diversity strongly affects business metrics — user dwell time, DAU, retention all improve when diversity is good. Intuition: even a die-hard NBA fan gets bored of a screen full of only NBA clips, shortens sessions, and may churn.

---

## 2. Measuring Item Similarity

To improve diversity we must first **quantify how similar two items are**. Two practical families exist.

### 2.1 Attribute-tag based

Simplest and fastest. Use attribute tags (属性标签): first-level category (一级类目), second-level category (二级类目), brand (品牌), keywords (关键词)…

For each attribute, similarity is 1 if the tags match, else 0. Example:

- Item `i`: (美妆 beauty, 彩妆 makeup, 香奈儿 Chanel)
- Item `j`: (美妆 beauty, 香水 perfume, 香奈儿 Chanel)

$$
\text{sim}_1(i,j) = 1,\quad \text{sim}_2(i,j) = 0,\quad \text{sim}_3(i,j) = 1.
$$

The overall similarity is a **weighted sum** of the per-attribute similarities, with weights tuned empirically.

> Caveat: category/brand tags are usually inferred by NLP/CV algorithms from the item's content, so they may be inaccurate.

### 2.2 Embedding based

Represent items `i, j` as vectors `v_i, v_j`. Larger inner product / cosine `⟨v_i, v_j⟩` ⇒ more similar. The key question is **how to learn the embedding**.

**A wrong approach: reuse the two-tower (双塔) recall model's item embeddings.** It works but performs poorly, for two reasons:

1. **Head effect (头部效应):** exposures/clicks concentrate on a few popular items; the vast majority of items have very few exposures, so the model cannot learn their embeddings well (new and long-tail items suffer most).
2. **Wrong semantics:** the two-tower item vector is learned from user–item interactions, so it captures the item's **interest profile**, not its **content**. Two items with similar cover image and title may end up with dissimilar two-tower vectors.

### 2.3 Content-based embeddings via CLIP

Best in practice — embed the item from its **image + text content**. Use a CV model (e.g. CNN) to get image vector `a_i`, an NLP model (e.g. BERT) to get text vector `b_i`. Final item embedding can be `a_i`, `b_i`, the concatenation `[a_i; b_i]`, or the sum `a_i + b_i`.

**Training: CLIP** (Radford et al., ICML 2021) — currently the recognized best pre-training method.

- **Idea:** for an (image, text) pair, predict whether image and text match.
- **No human labels needed** — perfect for Xiaohongshu (小红书) notes, which naturally pair images with related text.
- **Batch construction (batch-internal negative sampling):** in a batch of `m` notes, the matching (image, text) pair from the same note is a **positive sample (正样本)**. Pairing one image with the `m − 1` texts of *other* notes gives **negatives (负样本)** — so a batch has `m` positives and `m(m−1)` negatives. Training maximizes `|⟨a_i, b_i⟩|` and minimizes `|⟨a_i, b_j⟩|` for `i ≠ j`.
- The pre-trained model can be used to extract features **without fine-tuning**.

---

## 3. Why Diversity Matters

We want the exposed items to be diverse, i.e. pairwise **low similarity**. Given:

- `n` candidate items from rank with fused scores `reward_1, …, reward_n` (intrinsic value).
- pairwise similarity `sim(i, j)` (from tags or embeddings).

**Goal:** select `k` of the `n` items (set `S`) maximizing both **total value** and **diversity**. MMR and DPP differ in *how they measure diversity*:

- **MMR** uses **pairwise** similarities.
- **DPP** uses a **determinant** that measures the diversity of the **whole set at once**.

---

## 4. MMR — Maximal Marginal Relevance

MMR (Carbonell & Goldstein, SIGIR 1998) originated in **search ranking** (hence "relevance"), later adopted in recommendation re-ranking.

Notation: `S` = already-**selected** items (选中), `R` = **remaining**/unselected items (未选中). `θ ∈ [0,1]` balances value vs. diversity.

### 4.1 The objective

For each candidate `i ∈ R`, define the **Marginal Relevance** score:

$$
\text{MR}_i = \underbrace{\theta \cdot \text{reward}_i}_{\text{item value (精排分数)}} - \underbrace{(1-\theta)\cdot \max_{j \in S} \text{sim}(i, j)}_{\text{diversity penalty (多样性分数)}}.
$$

MMR then selects:

$$
\text{MMR} = \arg\max_{i \in R} \text{MR}_i = \arg\max_{i \in R}\left[ \theta \cdot \text{reward}_i - (1-\theta)\cdot \max_{j \in S}\text{sim}(i,j)\right].
$$

**Reading the formula.**
- First term: high `reward_i` favors selecting `i` (high CTR/like-rate etc.).
- Second term: `max_{j∈S} sim(i,j)` is `i`'s similarity to its *most similar already-selected item*. If `i` is similar to some `j ∈ S`, this term is large and **suppresses** `i`. So a chosen item must have **high reward AND be dissimilar to everything already chosen**.
- `θ` larger ⇒ value dominates, diversity matters less; `θ` smaller ⇒ diversity dominates.

### 4.2 The greedy algorithm

1. Initialize `S = ∅` (empty), `R = {1, …, n}` (full set).
2. Select the item with the **highest `reward_i`** (no diversity term yet since `S` is empty); move it from `R` to `S`.
3. Repeat `k − 1` rounds:
   - (a) Compute `{MR_i}` for all `i ∈ R`.
   - (b) Pick the highest-scoring item, move it from `R` to `S`.

After `k − 1` rounds, `S` holds the `k` items to expose. Because `S` changes every round, all `MR_i` must be recomputed each round.

### 4.3 Sliding window

**Problem with standard MMR:** as `S` grows, it becomes nearly impossible to find any `i ∈ R` dissimilar to *all* of `S`. With `sim ∈ [0,1]`, when `S` is large `max_{j∈S} sim(i,j) ≈ 1` for every `i`, so the diversity term saturates and MMR **degenerates** (only `reward` matters).

**Fix:** keep a fixed-size **sliding window** `W` (e.g. the 10 most recently selected items), a subset of `S`, and replace `S` with `W` in the diversity term:

$$
\arg\max_{i \in R}\left[ \theta \cdot \text{reward}_i - (1-\theta)\cdot \max_{j \in W}\text{sim}(i,j)\right].
$$

Now a new item only needs to differ from the `|W|` *recent* selections, not all of `S`. (Intuition: a user scrolling a feed only notices repetition among nearby items.)

### 4.4 Complexity

Let `n` = #candidates, `k` = #selected, `d` = embedding dim, `w = |W| ≤ k`.

- Computing similarities: when an item is selected we compute its sim to all of `R` and cache; over `k−1` rounds that is `(n−1)+(n−2)+…+(n−k) = O(nk)` similarity scores, each costing `O(d)` ⇒ `O(nkd)`.
- Standard MMR: each round computes `|R|` scores of cost `|S|` ⇒ `O(|R|·|S|) = O(nk)` per round, `O(nk²)` total ⇒ **`O(nk² + nkd)`**.
- With sliding window: each `MR_i` costs `O(w)` ⇒ **`O(nkd + nkw)`**, lower since `w ≤ k`.

---

## 5. Business-Rule Constraints

Real re-ranking adds many **hard business rules** (硬性规则) for user experience — they **must** be satisfied and take priority over the diversity algorithm. (Numbers below are illustrative, not real Xiaohongshu data.)

### 5.1 Three rule families

1. **"At most `k` consecutive notes of one type" (最多连续出现 k 篇某种笔记).**
   Notes are image-text or video. E.g. at most `k = 5` consecutive image-text, at most `k = 5` consecutive video. If positions `i … i+4` are all image-text, position `i+5` **must** be video.

2. **"At most 1 of a type per `k` notes" (每 k 篇笔记最多出现 1 篇某种笔记).**
   Promoted/operations notes (运营推广笔记) get their score multiplied by a **boost** coefficient `> 1` to gain exposure. To prevent boost from hurting UX, limit to ≤ 1 promoted note per `k = 9`. If position `i` is promoted, positions `i+1 … i+8` cannot be.

3. **"At most `k` of a type within the top `t`" (前 t 篇笔记最多出现 k 篇某种笔记).**
   The top `t` positions are most visible (Xiaohongshu's top 4 = first screen). For e-commerce-card notes (带电商卡片的笔记): top `t=1` allows `k=0` (position 1 cannot be an e-commerce note), top `t=4` allows at most `k=1`.

### 5.2 Combining rules with the diversity algorithm

Re-ranking maximizes MR **subject to** the rules. The trick is simple and works the same for MMR and DPP:

- Each round, **first apply the rules to filter `R`**, removing items that would violate a rule, producing a subset `R' ⊆ R`.
- Then run the usual selection over `R'`:

$$
\arg\max_{i \in R'}\left[ \theta \cdot \text{reward}_i - (1-\theta)\cdot \max_{j \in W}\text{sim}(i,j)\right].
$$

The selected item is now rule-compliant **and** high value + diverse. The only change vs. plain MMR is `R → R'`. (Shrinking `R` to `R'` also reduces compute slightly.)

---

## 6. DPP — Determinantal Point Process

DPP is a classical statistical-ML method (roots in 1970s; popular in the kernel-methods era ~mid-2000s) designed to **select a diverse subset**. It is the recognized **best diversity algorithm** for recommendation. Hulu (NIPS 2018) applied it to recommendation re-ranking. The math is heavier than MMR.

### 6.1 Math foundation: hyper-parallelotope volume

A **hyper-parallelotope** generalizes the parallelogram (2D, 平行四边形) and parallelepiped (3D, 平行六面体). Given vectors `v_1, …, v_k ∈ ℝᵈ`:

$$
\mathcal{P}(v_1, \dots, v_k) = \lbrace \alpha_1 v_1 + \dots + \alpha_k v_k \mid 0 \le \alpha_1, \dots, \alpha_k \le 1 \rbrace.
$$

- The edge vectors uniquely determine the body. Requires **`k ≤ d`** (a 2D parallelogram can sit in 3D, but a 3D parallelepiped cannot fit in 2D).
- If `v_1, …, v_k` are **linearly dependent**, the body collapses onto a `(k−1)`-dim subspace and its **volume = 0**.

**Volume via orthogonalization (Gram-Schmidt projection).** For a parallelogram with base `v_1`, the height is `v_2` minus its projection onto `v_1`:

$$
q_2 = v_2 - \text{Proj}_{v_1}(v_2), \qquad \text{Proj}_{v_1}(v_2) = \frac{v_1^\top v_2}{\lVert v_1 \rVert_2^2} v_1,
$$

with `q_2 ⟂ v_1`, and `vol(P) = ‖v_1‖₂ · ‖q_2‖₂` (base × height). For 3D:

$$
q_3 = v_3 - \text{Proj}_{v_1}(v_3) - \text{Proj}_{q_2}(v_3),\qquad
\text{vol}(\mathcal{P}) = \lVert v_1\rVert_2 \cdot \lVert q_2\rVert_2 \cdot \lVert q_3\rVert_2.
$$

**Diversity interpretation.** Take all item vectors to be **unit vectors** (`‖v_i‖ = 1`). Assume: `v_i, v_j` near-parallel (`v_iᵀv_j ≈ ±1`) ⇒ items similar; near-orthogonal (`v_iᵀv_j ≈ 0`) ⇒ dissimilar. Then the volume of `P(v_1,…,v_k)` lies in `[0, 1]`:

- All vectors **pairwise orthogonal** (best diversity) ⇒ unit hypercube ⇒ **vol = 1 (max)**.
- Vectors **linearly dependent / `v_i = ±v_j`** (poor diversity) ⇒ **vol = 0 (min)**.

So **volume measures set diversity**.

### 6.2 Volume ↔ determinant

Stack the `k` chosen unit vectors as columns of `V_S ∈ ℝ^{d×k}`. Then `A_S = V_Sᵀ V_S` is a `k×k` symmetric positive-semidefinite matrix (a Gram matrix; entry `(i,j)` is `v_iᵀ v_j`). For `k ≤ d`:

$$
\boxed{\det\big(V_S^\top V_S\big) = \mathrm{vol}\big(\mathcal{P}(v_1,\dots,v_k)\big)^2}
$$

(When `k = d`, `V_S` is square and `|\det(V_S)| = \mathrm{vol}(\mathcal P)`; the `k ≤ d` case is proven via an orthogonal rotation `R` that maps the vectors into a `k`-dim subspace, giving `Vᵀ V = Uᵀ U` with `U` square, then `det(VᵀV) = det(U)²`.)

Taking logs: `log det(A_S) = 2 · log vol(P(V_S))`. Therefore **`log det(A_S)` measures the diversity** of set `S` (larger = more diverse).

### 6.3 The DPP / k-DPP objective

Pure (k-)DPP — diversity only:

$$
\arg\max_{S:|S|=k} \log\det\big(V_S^\top V_S\big) = \arg\max_{S:|S|=k} \log\det(A_S).
$$

Hulu's recommendation objective — **value + diversity**:

$$
\boxed{\arg\max_{S:|S|=k} \theta \cdot \Big(\textstyle\sum_{j \in S} \text{reward}_j\Big) + (1-\theta)\cdot \log\det(A_S)}
$$

where:
- `A` is the full `n×n` similarity matrix, `a_{ij} = v_iᵀ v_j`; building it costs `O(n²d)`.
- `A_S` is the `k×k` principal submatrix of `A` indexed by `S`.
- First term = total value (like MMR's reward term). Second term = set diversity.

This is a **combinatorial optimization** (choose a size-`k` subset of `{1,…,n}`); it is NP-hard, so we solve it **greedily**.

### 6.4 Greedy algorithm

`S` = selected, `R` = remaining. Each round add the item maximizing the marginal gain:

$$
\arg\max_{i \in R} \theta \cdot \text{reward}_i + (1-\theta)\cdot \log\det\big(A_{S\cup\lbrace i\rbrace}\big).
$$

**Intuition for the determinant term.** Adding item `i` appends one row and one column to `A_S`. We want the new item to *increase* the determinant (volume), i.e. `v_i` should be **as orthogonal as possible** to the span of the already-selected vectors. If `v_i` is similar to some selected item, the new det → 0 and `log det → −∞`, strongly discouraging the pick.

### 6.5 Brute force vs. Cholesky speedup (Hulu fast algorithm)

**Brute force.** Computing `det(A_{S∪{i}})` by eig/Cholesky decomposition from scratch costs `O(|S|³)`. Over all `i ∈ R`: `O(|S|³·|R|)`. Repeating `k` times to grow `S` from 1 to `k`:

$$
O(|S|^3 \cdot |R| \cdot k) = O(nk^4), \quad\text{plus } O(n^2 d)\text{ to build } A \Rightarrow O(n^2 d + n k^4).
$$

**Hulu's fast numerical algorithm — incremental Cholesky.** Total cost only **`O(n²d + nk²)`**:
`O(n²d)` to build `A`, plus `O(nk²)` to compute all determinants via incremental Cholesky updates.

Key facts:
- **Cholesky decomposition** `A_S = L Lᵀ`, `L` lower-triangular (下三角, zeros above the diagonal).
- `det(L)` = product of its diagonal entries, so `det(A_S) = det(L)² = ∏_i l_{ii}²`.
- Given `A_S = L Lᵀ`, the Cholesky factor of `A_{S∪{i}}` can be obtained **incrementally** (cheaply) for every `i ∈ R`, so every `log det(A_{S∪{i}})` is fast.

**Incremental update derivation.** With `a_{ii} = v_iᵀ v_i = 1` (unit vectors),

$$
A_{S\cup\lbrace i\rbrace} = \begin{bmatrix} A_S & a_i \\ a_i^\top & 1 \end{bmatrix}
= \begin{bmatrix} L & 0 \\ c_i^\top & d_i \end{bmatrix}\begin{bmatrix} L & 0 \\ c_i^\top & d_i \end{bmatrix}^\top
= \begin{bmatrix} LL^\top & Lc_i \\ c_i^\top L^\top & c_i^\top c_i + d_i^2 \end{bmatrix}.
$$

Matching blocks gives two equations:

$$
a_i = L c_i \quad\text{and}\quad 1 = c_i^\top c_i + d_i^2.
$$

Since `L` is lower-triangular, solve `c_i = L^{-1} a_i` by forward substitution in `O(|S|²)`, then `d_i² = 1 - c_iᵀ c_i`. The new determinant is:

$$
\det\big(A_{S\cup\lbrace i\rbrace}\big) = \det(L)^2 \times d_i^2.
$$

Because `det(L)²` is the **same for all `i`** (independent of `i`), the greedy step simplifies to:

$$
\boxed{ i^\star = \arg\max_{i \in R} \theta \cdot \text{reward}_i + (1-\theta)\cdot \log d_i^2 }
$$

**Algorithm:**
1. Input: vectors `v_1…v_n ∈ ℝᵈ`, scores `reward_1…reward_n`.
2. Build `A` (`a_{ij} = v_iᵀ v_j`) — `O(n²d)`.
3. Pick the highest-`reward` item `i`; set `S = {i}`, `L = [1]` (1×1, since `a_{ii} = 1`).
4. For `t = 1 … k−1`:
   - (a) For each `i ∈ R`: solve `a_i = L c_i` for `c_i` (`O(|S|²)`); compute `d_i² = 1 − c_iᵀ c_i`.
   - (b) `i⋆ = argmax_{i∈R} θ·reward_i + (1−θ)·log d_i²`.
   - (c) `S ← S ∪ {i⋆}`; update `L ← [[L, 0],[c_{i⋆}ᵀ, d_{i⋆}]]`.
5. Return `S` (`k` items).

Naively this is `O(n²d + nk³)`; reusing the previous round's solve of `a_i = L c_i` drops it to **`O(n²d + nk²)`**.

### 6.6 Sliding window + rules for DPP

The **sliding window is even more necessary for DPP** than for MMR. As `S` grows it inevitably contains similar items ⇒ vectors approach linear dependence ⇒ `det(A_S) → 0` ⇒ `log det → −∞`, blowing up the objective. Fix: replace `S` with a small window `W`:

$$
\arg\max_{i \in R} \theta \cdot \text{reward}_i + (1-\theta)\cdot \log\det\big(A_{W\cup\lbrace i\rbrace}\big).
$$

**Rules** apply identically: filter `R → R'` first, then optimize over `R'`. The numerical algorithm is essentially unchanged.

### 6.7 MGS — a simpler equivalent (bonus)

DPP's numerical algorithm is fiddly to implement. **Modified Gram-Schmidt (MGS)** (Xiaohongshu, KDD 2021, "Sliding Spectrum Decomposition") solves the **exact same** objective with a much simpler implementation.

Idea: instead of tracking Cholesky factors, directly maintain orthogonalized residual vectors. Since `vol(S) = ‖q_1‖₂ · ‖q_2‖₂ ⋯ ‖q_t‖₂` and `vol(S∪{i}) = vol(S)·‖q_i‖₂`, and `log vol(S)` is independent of `i`, the greedy step becomes:

$$
\arg\max_{i \in R} \theta \cdot \text{reward}_i + (1-\theta)\cdot \log\big(\lVert q_i \rVert_2\big),
$$

where `q_i` is item `i`'s component orthogonal to the span of already-selected vectors. MGS is mathematically equivalent to GS but numerically more stable (avoids error accumulation), and runs in **`O(ndk)`**: each of the `k−1` rounds orthogonalizes the `O(n)` remaining vectors against the newest basis vector at `O(d)` each.

---

## 7. Reward vs. Diversity Trade-off — Intuition Summary

Both MMR and DPP optimize the **same shape**:

$$
\text{select } i = \arg\max_i  \underbrace{\theta \cdot \text{value}}_{\text{user interest}} + \underbrace{(1-\theta)\cdot \text{diversity}}_{\text{set freshness}}.
$$

- **`reward` term:** "Is this item interesting to the user *on its own*?" Maximizing reward alone gives a list that may be relevant but monotonous (all NBA clips).
- **Diversity term:** "Does this item add something *new* relative to what is already on screen?"
  - MMR penalizes the *max pairwise similarity* to the selected set (or window).
  - DPP rewards the *increase in volume* (determinant) of the selected set — a richer, holistic notion: it captures redundancy across the whole set, not just nearest-neighbor pairs.
- **`θ`** is the single knob: `θ → 1` chases pure relevance; `θ → 0` chases pure diversity. Tune for business metrics (dwell time, retention).
- **Geometric picture:** each item is a unit vector. A diverse set ⇒ vectors spread out / orthogonal ⇒ large hyper-volume (det large). A redundant set ⇒ vectors collinear ⇒ volume collapses to 0 (det → 0, log-det → −∞), which is exactly why selecting a near-duplicate is heavily penalized.

---

## 8. Key Insights

| Topic | Key point |
|---|---|
| Pipeline position | Diversity = post-processing after rank, called **re-ranking**; ranking stays pointwise. |
| Similarity: tags | category/brand/keyword match → weighted sum of 0/1 sims; tags may be NLP-inferred & noisy. |
| Similarity: embeddings | inner product of item vectors. **Don't** use two-tower vectors (head effect + interest- not content-semantics). |
| Best embedding | **CLIP** content embeddings (CNN image + BERT text), batch-internal negatives, no labels. |
| MMR objective | `argmax_{i∈R} [θ·reward_i − (1−θ)·max_{j∈S} sim(i,j)]`. |
| MMR pitfall + fix | large `S` ⇒ diversity term saturates to 1 ⇒ degenerate; fix = **sliding window `W`** (replace `S` with last ~10). |
| MMR complexity | `O(nk² + nkd)`; with window `O(nkd + nkw)`. |
| Rules | hard constraints; each round filter `R → R'`, then optimize over `R'`. Three families: consecutive-`k`, 1-per-`k`, `k`-in-top-`t`. |
| DPP diversity metric | `log det(A_S)`, `A_S = V_Sᵀ V_S`; `det(V_SᵀV_S) = vol(P)²`. |
| DPP objective | `argmax_{|S|=k} θ·Σ_{j∈S} reward_j + (1−θ)·log det(A_S)`. |
| DPP greedy step | `argmax_{i∈R} θ·reward_i + (1−θ)·log det(A_{S∪{i}})`. |
| DPP speedup | incremental **Cholesky** `A_S=LLᵀ`, `det=∏l_{ii}²`; brute `O(n²d+nk⁴)` → Hulu `O(n²d+nk²)`. |
| det as diversity | orthogonal vectors ⇒ vol=1 (max diversity); dependent ⇒ vol=0 (`log det → −∞`). |
| MGS | equivalent to DPP, simpler & stabler, `O(ndk)`, greedy via `log‖q_i‖₂`. |

**References:** [1] Carbonell & Goldstein, SIGIR 1998 (MMR). [2] Chen, Zhang, Zhou, NIPS 2018 (fast greedy MAP for DPP). [3] Huang et al., KDD 2021 (Sliding Spectrum Decomposition / MGS). [5] Radford et al., ICML 2021 (CLIP).

## Expanded Reading — Recent Advances (2026): Generative & Agentic Re-Ranking

The classical re-ranking toolkit in this chapter optimizes **hand-crafted set objectives** — MMR's pairwise penalty and DPP's log-determinant, both steered by a single scalar `θ`. The 2025–2026 frontier replaces (or wraps) these fixed objectives with **LLM reasoning re-rankers** and **agentic / self-evolving systems** that learn *what* to optimize: a model can read item semantics and user history, reason in natural language about ordering, be tuned with reinforcement learning (RL, 强化学习), and even autonomously rewrite its own objectives and architecture.

### GR2 — Generative Reasoning Re-Ranker (生成式推理重排)

> Liang, M., Li, Y., Xu, J., Asadi, K., Liu, X., Gu, S., Rangadurai, K., Shyu, F., Wang, S., Yang, S., Li, Z., Liu, J., Sun, M., Tian, F., Wei, X., Sun, C., Tao, J., Mei, S., Chen, W., Kolay, S., Pandey, S., Firooz, H., & Simon, L. (2026). *Generative reasoning re-ranker*. arXiv. https://arxiv.org/abs/2602.07774

**Method.** GR2 (Meta AI) targets the **re-ranking stage specifically** — the same stage this chapter calls re-ranking — with an LLM and a **three-stage training pipeline**:
1. **Tokenized mid-training on Semantic IDs (语义 ID).** Raw non-semantic item IDs (billions, blowing up the LLM vocabulary) are mapped to short discrete code sequences via an RQ-VAE tokenizer (à la TIGER), targeting **≥99% ID uniqueness**, with a contrastive loss $L_{ctr}$ to encode co-engagement history. A student LLM (e.g. Qwen3-8B) is mid-trained on a mixture of these SIDs and world knowledge.
2. **SFT on reasoning traces (推理轨迹).** A larger teacher LLM (e.g. Qwen3-32B) generates hierarchical reasoning traces via re-ranking-specific prompts + **rejection sampling** (filter noisy traces); the student is supervised-fine-tuned on the curated traces.
3. **RL via DAPO.** Decoupled Clip and Dynamic sAmpling Policy Optimization with **verifiable rewards** designed for re-ranking, combining a ranking reward $R_{rank}$ and a format reward $R_{fmt}$.

**Key results.** Surpasses the SOTA baseline OneRec-Think by **+2.4% Recall@5** and **+1.3% NDCG@5** on two real-world datasets (incl. Amazon Beauty). Ablations show advanced reasoning traces drive most of the gain; RL adds further lift on top of SFT.

**Trade-offs / limitations.** A key failure mode is **reward hacking (奖励作弊)**: naively summing $R_{rank}+R_{fmt}$ lets the model just **preserve the input order** to collect the format reward — fixed by *conditional* verifiable rewards (only grant format reward when ranking actually changes). Evaluation is offline academic datasets, not a live A/B test; running an 8B-parameter autoregressive LLM per request is **orders of magnitude more expensive** than greedy MMR/DPP.

**Connection to this chapter.** GR2 is the modern successor to §4–§6. Where MMR/DPP score a set with a *fixed* formula (`θ·reward + (1−θ)·diversity`) computed from precomputed similarity vectors, GR2 *reasons* over item semantics and user context to produce an ordering, and *learns* its objective via RL rather than having it hand-designed. Diversity here is implicit — emergent from reasoning over content semantics rather than from an explicit $\log\det(A_S)$ or pairwise $\max_{j} \text{sim}(i,j)$ term. The price is **latency and cost**: DPP's incremental-Cholesky greedy step is $O(n^2 d + nk^2)$ of cheap linear algebra, whereas autoregressive LLM decoding of a reasoning trace per request is far heavier — viable for small candidate sets ($n$ in the tens) and where world-knowledge reasoning beats vector geometry.

### RecoWorld — Simulated Environments for Agentic Recommenders (智能体推荐的模拟环境)

> Liu, F., Lin, X., Yu, H., Wu, M., Wang, J., Zhang, Q., Zhao, Z., Xia, Y., Zhang, Y., Li, W., Gao, M., Wang, Q., Zhang, L., Zhang, B., & Fan, X. (2025). *RecoWorld: Building simulated environments for agentic recommender systems*. arXiv. https://arxiv.org/abs/2509.10397

**Method.** A **blueprint / position paper** (Meta MRS), not a single trained model. RecoWorld is a **dual-view simulated environment**: a **simulated user** (LLM) and an **agentic recommender** engage in **multi-turn interactions** aimed at maximizing **user retention**. The simulated user reviews recommended items, updates a "mindset," and on sensing disengagement issues **reflective natural-language instructions** ("too many hairstyling clips; show something different but related"). The recommender adapts via reasoning traces, forming a feedback loop — a Gym-like RL framework for recommenders. Supports text / multimodal / Semantic-ID content modeling and multi-agent (creator) simulation; engagement statistics (watch time, clicks) become **pseudo-rewards** for multi-turn RL.

**Key results.** No headline accuracy numbers — it is a framework/blueprint. Claims to be the **first** environment enabling **instruction-following recommenders** in a Gym-like RL setup, with four use cases (instruction-following evaluation, creator publishing strategies, new-user interest exploration via bandits, community leaderboard).

**Trade-offs / limitations.** Simulated user feedback **cannot perfectly replicate real interactions** (sim-to-real gap); validity must be checked against real RecSys datasets and human annotators. Largely conceptual — no production deployment reported. LLM-driven simulation of every user turn is computationally heavy.

**Connection to this chapter.** This reframes re-ranking from a *one-shot static set-selection* (the entire premise of §3–§6: pick `k` of `n` once) into a **multi-turn, interactive** loop where the user can *instruct* the recommender to re-diversify ("show me something different"). The diversity/relevance trade-off that MMR's `θ` encodes statically becomes a **learned, dynamic policy** trained against a simulated user's retention signal — diversity emerges because monotony triggers disengagement instructions, not because a $\log\det$ term penalizes it. It also tackles re-ranking's evaluation problem: where we tune `θ` against offline dwell-time proxies, RecoWorld offers a cheaper-than-A/B sandbox to train and test such policies.

### Self-Evolving Recommendation System — Autonomous Model Optimization with LLM Agents (自进化推荐系统)

> Wang, H., Wu, Y., Chang, D., Wei, L., & Heldt, L. (2026). *Self-evolving recommendation system: End-to-end autonomous model optimization with LLM agents*. arXiv. https://arxiv.org/abs/2602.10226

**Method.** Google's system (deployed at YouTube) frames model improvement as **bi-level optimization** and automates the human ML-engineer role with **LLM agents** (Gemini 2.5 Pro). Lower level: train ranking weights $\theta^{\ast}(\Phi) = \arg\min_\theta L_{proxy}(D;\theta,\Phi)$ on a proxy reward. Upper level: find the meta-configuration $\Phi^{\ast} = \arg\max_\Phi \mathbb{E}[M(\theta^{\ast}(\Phi))]$ s.t. $G(\Phi)\le C$, where $M$ is delayed north-star business metrics and $\Phi$ = {optimizer, architecture, reward definition}. Two synchronized loops: an **Offline Agent (inner loop)** does high-throughput hypothesis generation against proxy metrics, and an **Online Agent (outer loop)** validates survivors against delayed business metrics in live production. Agents act as **MLEs**: they read production code, propose **structural mutations** (e.g. new activations, DCN/Transformer interaction layers) and **novel reward functions**, beyond what AutoML/NAS can do (those only select from a fixed menu).

**Key results.** Several **successful production launches at YouTube**, with agentic evolution surpassing hand-tuned baselines in both **development velocity** and model performance (offline + online + production validation). Ablations relate model reasoning power (full vs. lightweight Gemini) to discovery quality.

**Trade-offs / limitations.** RecSys lacks a **clear oracle** for success — the true objective (user satisfaction) is latent, observed only through **delayed, noisy, sparse** feedback on the order of $\Theta(\text{days})$–$\Theta(\text{weeks})$, so the outer loop is slow and expensive. Requires strict safety guardrails to let agents touch production code; reward hacking and proxy-vs-north-star misalignment are live risks.

**Connection to this chapter.** This is the most radical departure: instead of a human choosing MMR vs. DPP and hand-tuning `θ`, an **LLM agent autonomously designs and deploys the objective itself** — exactly the kind of multi-objective reward and architecture (including diversity/exploration terms) that this chapter constructs by hand. It also operationalizes our §1/§7 note that *diversity must be tuned for business metrics (dwell time, retention)*: the outer loop optimizes directly for delayed north-star metrics, automating the manual `θ`-tuning loop. The trade-off is no longer per-request latency but a **slow, costly meta-optimization loop** bounded by how fast real-world retention signals arrive.
