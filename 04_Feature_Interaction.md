# Part 04 — Feature Cross (特征交叉)

Feature crossing means letting features **multiply** each other (interact), not just add up. Linear models only add weighted features; cross techniques explicitly model second- (and higher-) order interactions, which dramatically increases expressive power for ranking and retrieval. This note covers four building blocks used in industrial recommenders:

1. **Factorized Machine (FM)** — cheap second-order crossing via low-rank factorization.
2. **Deep & Cross Network (DCN)** — a learnable "cross layer" stacked into a cross network, combined with a deep (fully-connected) network.
3. **LHUC / PPNet** — per-user multiplicative scaling of hidden units (personalization).
4. **SENet + Bilinear Cross (FiBiNet)** — field-wise importance reweighting plus pairwise field interaction.

---

## Table of Contents

- [1. Factorized Machine (FM)](#1-factorized-machine-fm)
  - [1.1 From Linear Model to Second-Order Cross](#11-from-linear-model-to-second-order-cross)
  - [1.2 The Parameter Explosion Problem](#12-the-parameter-explosion-problem)
  - [1.3 FM: Low-Rank Factorization of the Cross Matrix](#13-fm-low-rank-factorization-of-the-cross-matrix)
  - [1.4 Properties and Interview Points](#14-properties-and-interview-points)
- [2. Deep & Cross Network (DCN)](#2-deep--cross-network-dcn)
  - [2.1 Where DCN Sits](#21-where-dcn-sits)
  - [2.2 The Cross Layer](#22-the-cross-layer)
  - [2.3 The Cross Network](#23-the-cross-network)
  - [2.4 Deep & Cross Network (Full Model)](#24-deep--cross-network-full-model)
- [3. LHUC / PPNet](#3-lhuc--ppnet)
  - [3.1 Origin: LHUC in Speech Recognition](#31-origin-lhuc-in-speech-recognition)
  - [3.2 LHUC Structure](#32-lhuc-structure)
  - [3.3 LHUC in Recommendation Ranking (PPNet)](#33-lhuc-in-recommendation-ranking-ppnet)
- [4. SENet & Bilinear Cross (FiBiNet)](#4-senet--bilinear-cross-fibinet)
  - [4.1 SENet: Field-Wise Importance Reweighting](#41-senet-field-wise-importance-reweighting)
  - [4.2 Field-to-Field Feature Cross](#42-field-to-field-feature-cross)
  - [4.3 Bilinear Cross](#43-bilinear-cross)
  - [4.4 FiBiNet: Putting It Together](#44-fibinet-putting-it-together)
- [5. Summary Cheat-Sheet](#5-key-insights)

---

## 1. Factorized Machine (FM)

### 1.1 From Linear Model to Second-Order Cross

Given $d$ features $\mathbf{x} = [x_1, \dots, x_d]$, the **linear model** (线性模型) predicts:

$$
p = b + \sum_{i=1}^{d} w_i x_i
$$

- Parameters: $\mathbf{w} = [w_1, \dots, w_d]$ and bias $b$, so $d+1$ parameters total.
- The prediction is just a **weighted sum** of features — only addition, no multiplication. It cannot capture "feature A is good *only when* feature B is also present."

To add **second-order cross features** (二阶交叉特征) we let every *pair* of features multiply, each pair getting its own weight $u_{ij}$:

$$
p = b + \sum_{i=1}^{d} w_i x_i + \sum_{i=1}^{d} \sum_{j=i+1}^{d} u_{ij} x_i x_j
$$

The cross term $x_i x_j$ is nonzero only when both features fire, so $u_{ij}$ learns the interaction strength of that specific pair.

### 1.2 The Parameter Explosion Problem

The cross weights $u_{ij}$ form a $d \times d$ matrix $\mathbf{U}$ (only the upper triangle is used). The number of cross parameters is $O(d^2)$.

In recommendation, $d$ (number of features / one-hot dimensions) can be huge, so $O(d^2)$ is intractable — both for memory and for learning (most pairs are seen rarely, so weights are poorly estimated).

### 1.3 FM: Low-Rank Factorization of the Cross Matrix

**Key idea:** approximate the dense $d \times d$ cross matrix $\mathbf{U}$ by a **low-rank factorization** $\mathbf{U} \approx \mathbf{V}\mathbf{V}^T$, where $\mathbf{V}$ is $d \times k$ with $k \ll d$.

Concretely, each feature $i$ gets a $k$-dimensional latent vector $\mathbf{v}_i$ (the $i$-th row of $\mathbf{V}$), and the cross weight is replaced by an inner product:

$$
u_{ij} \approx \mathbf{v}_i^T \mathbf{v}_j
$$

The **Factorized Machine** prediction is therefore:

$$
\boxed{p = b + \sum_{i=1}^{d} w_i x_i + \sum_{i=1}^{d} \sum_{j=i+1}^{d} \left(\mathbf{v}_i^T \mathbf{v}_j\right) x_i x_j}
$$

- Parameter count drops from $O(d^2)$ to $O(kd)$ — linear in $d$.
- Reference: Steffen Rendle, *Factorization Machines*, ICDM 2010.

**Why factorization helps beyond just saving memory:** because each pair weight is composed from shared per-feature vectors $\mathbf{v}_i$, FM can estimate the strength of a pair $(i,j)$ even if that exact pair was never co-observed in training — it generalizes through the shared latent space. This is the same trick as matrix completion in collaborative filtering.

### 1.4 Properties and Interview Points

- FM is a **drop-in replacement for linear / logistic regression** (线性回归、逻辑回归): anywhere you'd use those, you can use FM, and it is strictly more expressive because it models second-order crosses.
- FM is a *shallow* model. It was important historically (pre-deep-learning ranking) and is a conceptual ancestor of the embedding-based deep crossing models below. In modern deep recommenders it is often subsumed by DCN / FiBiNet, but it remains a frequent interview baseline.
- Complexity note: the FM second-order term can be computed in $O(kd)$ time (not $O(kd^2)$) using the identity $\sum_{i<j}\langle\mathbf{v}_i,\mathbf{v}_j\rangle x_i x_j = \tfrac{1}{2}\sum_{f}\big[(\sum_i v_{i,f}x_i)^2 - \sum_i v_{i,f}^2 x_i^2\big]$. (Standard FM result; not on Wang's slides but commonly asked.)

---

## 2. Deep & Cross Network (DCN)

### 2.1 Where DCN Sits

DCN is a concrete **network architecture** that can be plugged in anywhere a generic neural net is used in a recommender:

- As the **shared bottom** of a multi-task ranking model (the block that feeds click-rate / like-rate / favorite-rate / share-rate heads, each "fully-connected + Sigmoid").
- As a **tower** in the two-tower retrieval model (user tower / item tower).
- As an **expert** inside MMoE.

In other words, wherever the slides earlier said you may use any network structure, DCN is one of the choices. DCN's distinguishing feature is the **cross network**, which performs bounded-degree explicit feature crossing efficiently.

### 2.2 The Cross Layer

A **cross layer** (交叉层) takes two inputs:

- $\mathbf{x}_0$ — the original input vector to the *whole* cross network (kept around at every layer).
- $\mathbf{x}_i$ — the output of the previous cross layer (the input to *this* layer).

Internally:
1. Pass $\mathbf{x}_i$ through a fully-connected layer (全连接层): $\mathbf{y} = \mathbf{W}\mathbf{x}_i + \mathbf{b}$.
2. Take the **Hadamard product** (element-wise multiply, 哈达玛乘积) of $\mathbf{x}_0$ with $\mathbf{y}$: $\mathbf{z} = \mathbf{x}_0 \circ \mathbf{y}$.
3. Add a **residual** connection of $\mathbf{x}_i$.

The cross-layer update (this is the **Cross Network V2 / DCN-V2** form on the slides) is:

$$
\boxed{\mathbf{x}_{i+1} = \mathbf{x}_0 \circ \big(\mathbf{W}\mathbf{x}_i + \mathbf{b}\big) + \mathbf{x}_i}
$$

where $\circ$ is the Hadamard product and $\mathbf{W}, \mathbf{b}$ are that layer's parameters.

Using the task's requested notation (per-layer subscripts):

$$
\mathbf{x}_{l+1} = \mathbf{x}_0 \mathbf{x}_l^T \mathbf{w}_l + \mathbf{b}_l + \mathbf{x}_l
$$

> **Note on two DCN formulations.** Wang's slides teach **DCN-V2**, where the per-layer transform is a full **matrix** $\mathbf{W}$ (so $\mathbf{W}\mathbf{x}_i$ is a matrix-vector product). The *original* DCN-V1 (Wang et al., ADKDD 2017) uses a **weight vector** $\mathbf{w}_l$, giving $\mathbf{x}_{l+1} = \mathbf{x}_0 (\mathbf{x}_l^T \mathbf{w}_l) + \mathbf{b}_l + \mathbf{x}_l$, where $\mathbf{x}_l^T \mathbf{w}_l$ is a scalar. The task prompt's formula $x_{l+1} = x_0 x_l^T w_l + b_l + x_l$ is the **V1** form; the slide diagram (with a square matrix $\mathbf{W}$) is the **V2** form. Both share the same intuition: multiply $\mathbf{x}_0$ element-wise by a learned transform of $\mathbf{x}_l$, then add a residual.

**Key structural points:**
- $\mathbf{x}_0$ is re-injected at *every* cross layer (it is an input to the cross layer along with $\mathbf{x}_i$). This is what makes crossing happen explicitly.
- The residual $+\mathbf{x}_i$ makes it easy to train deep stacks (like ResNet skip connections).
- All cross outputs have the **same shape** as $\mathbf{x}_0$.

### 2.3 The Cross Network

A **cross network** (交叉网络) is just several cross layers stacked, each with its own parameters $(\mathbf{W}_l, \mathbf{b}_l)$, all sharing the same $\mathbf{x}_0$:

$$
\mathbf{x}_1 = \mathbf{x}_0 \circ (\mathbf{W}_0 \mathbf{x}_0 + \mathbf{b}_0) + \mathbf{x}_0
$$
$$
\mathbf{x}_2 = \mathbf{x}_0 \circ (\mathbf{W}_1 \mathbf{x}_1 + \mathbf{b}_1) + \mathbf{x}_1
$$
$$
\mathbf{x}_3 = \mathbf{x}_0 \circ (\mathbf{W}_2 \mathbf{x}_2 + \mathbf{b}_2) + \mathbf{x}_2
$$

**Intuition / why this matters:** because $\mathbf{x}_0$ is multiplied in at each layer, the *polynomial degree* of feature interactions grows by one per layer. After $L$ cross layers the network represents feature crosses up to order $L+1$ — but with only $O(L \cdot d)$ (V1) or $O(L \cdot d^2)$ (V2) parameters, far cheaper than enumerating all high-order crosses. This is the deep-learning analogue of FM's "cheap explicit crossing."

### 2.4 Deep & Cross Network (Full Model)

The full **Deep & Cross Network** runs two branches in parallel on the same concatenated input (user features 用户特征 + item features 物品特征 + other features 其它特征):

```
                ┌──> Fully-Connected (Deep) Network ──┐
input vector ───┤                                      ├──> concat ──> FC layer ──> output
                └──> Cross Network ───────────────────┘
```

- **Deep network** (全连接网络): a standard MLP — captures arbitrary, implicit high-order interactions.
- **Cross network**: captures bounded-degree *explicit* interactions efficiently.
- Their output vectors are **concatenated** and passed through a final fully-connected layer to produce the prediction.

This whole thing — deep tower + cross tower + concat + FC — *is* "the DCN" and can serve as the shared bottom / tower / expert as described in §2.1.

References:
1. Ruoxi Wang et al. *DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems.* WWW 2021. (The version taught.)
2. Ruoxi Wang et al. *Deep & Cross Network for Ad Click Predictions.* ADKDD 2017. (Original.)

---

## 3. LHUC / PPNet

LHUC = **Learning Hidden Unit Contributions**. It is structurally similar to the cross network (both rely on Hadamard products) but its purpose is **personalization via multiplicative gating**. It is used **only for ranking** (精排), not retrieval.

- Origin: speech recognition (Swietojanski, Li & Renals, IEEE/ACM TASLP 2016).
- Kuaishou (快手) applied LHUC to recommendation ranking and named it **PPNet**.

### 3.1 Origin: LHUC in Speech Recognition

In speech recognition the input is a **speech signal** (语音信号). We want a network to transform it into a good representation and recognize the words. But different people sound different, so we add **personalization** using the **speaker's features** (说话者的特征) — simplest is the speaker ID embedded into a vector.

### 3.2 LHUC Structure

LHUC interleaves two pathways:

- **Main pathway (speech signal / item features):** a stack of **fully-connected layers** that transform the signal.
- **Gating pathway (speaker features / user features):** a separate **neural network** (multiple FC layers). Its **last layer's activation is `Sigmoid × 2`**, applied element-wise — so every output element lies in $(0, 2)$.

At each gating point, the main vector and the gating vector (which must have identical shape) are combined with a **Hadamard product** (element-wise multiply):

$$
\mathbf{h}_{\text{out}} = \mathbf{h}_{\text{main}} \circ \big(2\cdot\sigma(\text{gate-net}(\mathbf{u}))\big)
$$

Because each gate value is in $(0, 2)$, this **amplifies some hidden units (>1) and shrinks others (<1)** in a per-user (or per-speaker) way — hence "learning hidden unit *contributions*." The output keeps the same shape as the input.

The block (FC on main → Hadamard with gate → next FC → Hadamard with a *new* gate, …) can be repeated as many times as you like to build a deeper network. The example on the slides repeats two modules (two Hadamard products).

**Important detail:** each gating sub-network takes the **speaker/user features as its input** (the gate is re-derived from the personalization features at each stage), and each ends in `Sigmoid × 2`.

### 3.3 LHUC in Recommendation Ranking (PPNet)

The mapping to recommendation is direct:

| Speech recognition | Recommendation ranking |
|---|---|
| Speech signal → main pathway | **Item features** → main pathway |
| Speaker features → gating pathway | **User features** → gating pathway |

So the item representation is transformed by FC layers, and at each stage it is element-wise scaled by a user-derived gate (Sigmoid×2). The result is a **per-user personalized** item representation. Structurally identical to the speech version. Kuaishou reports gains; this is their PPNet.

**Interview framing:** LHUC/PPNet is multiplicative *feature gating* conditioned on the user — it lets the same item produce different hidden activations for different users, which is a cheap, effective form of personalization on top of any ranking backbone.

---

## 4. SENet & Bilinear Cross (FiBiNet)

Two techniques that both give ranking gains; combined they form **FiBiNet** (Huang, Zhang & Zhang, RecSys 2019). SENet originates from computer vision (Hu, Shen & Sun, *Squeeze-and-Excitation Networks*, CVPR 2018).

### 4.1 SENet: Field-Wise Importance Reweighting

**Setup.** We have $m$ discrete features (user ID, item ID, item category, item keywords, …). Each is embedded into a vector. If we (for explanation) make all embeddings $k$-dimensional, we get an $m \times k$ matrix.

**SENet pipeline** (Squeeze → Excitation → Reweight):

1. **Squeeze — AvgPool over each row:** average-pool each embedding (row) to one scalar → an $m \times 1$ vector (one number per field).
2. **Excitation — FC + ReLU:** compress to $\frac{m}{r} \times 1$, where $r > 1$ is the compression ratio (压缩比例). (This is the "bottleneck.")
3. **FC + Sigmoid:** expand back to $m \times 1$; every element lies in $(0,1)$. This is the per-field **importance weight** vector.
4. **Reweight — row-wise multiply:** multiply each original embedding row by its scalar weight, producing a new $m \times k$ matrix of the same shape.

$$
\text{e.g. } \text{row}_2^{\text{out}} = w_2 \cdot \text{row}_2^{\text{in}}, \quad w \in (0,1)^m
$$

**Key concept — field-wise weighting:**
- A **field** is one discrete feature. E.g. the user-ID embedding is a 64-dim vector; all 64 elements form **one field** and receive the **same** weight.
- With $m$ fields, the weight vector is **$m$-dimensional** — one weight per field.
- SENet *learns* each field's importance automatically from all the features: important fields get high weight, unimportant fields (e.g. an item-ID embedding that doesn't help the task) get down-weighted.

**Note:** the $m$ embeddings do **not** need the same dimension — SENet avg-pools each to a single scalar regardless, so different-width embeddings are fine.

### 4.2 Field-to-Field Feature Cross

After (or in parallel with) SENet's reweighting, we cross **pairs of fields** to get new features (e.g. cross the item-location embedding with the user-location embedding). Two simple primitives, given two field embeddings $\mathbf{x}_i, \mathbf{x}_j$:

**Inner product (内积):**
$$
f_{ij} = \mathbf{x}_i^T \mathbf{x}_j \quad (\text{a scalar})
$$
- $m$ fields → $m^2$ scalars. Since $m$ in recommenders is small (tens), $m^2$ scalars is acceptable.

**Hadamard product:**
$$
\mathbf{f}_{ij} = \mathbf{x}_i \circ \mathbf{x}_j \quad (\text{a vector, same size as } \mathbf{x}_i,\mathbf{x}_j)
$$
- $m$ fields → $m^2$ **vectors** — too large to use for all pairs. In practice you must **hand-pick a subset of pairs** to cross.

Both inner and Hadamard products **require the two embeddings to have the same shape** ($k$-dim). If shapes differ, neither product is defined — motivating Bilinear Cross.

### 4.3 Bilinear Cross

**Bilinear Cross** is a more advanced cross that inserts a **learned parameter matrix** $\mathbf{W}_{ij}$ between the two field embeddings, so the embeddings need **not** have the same shape.

**Inner-product form:**
$$
\boxed{f_{ij} = \mathbf{x}_i^T \mathbf{W}_{ij} \mathbf{x}_j} \quad (\text{a scalar})
$$
- $m$ fields → $m^2$ scalar cross features (fine, small).
- But you need many $\mathbf{W}_{ij}$ matrices: $\approx \frac{m^2}{2}$ of them. With $\sim 1000$ matrices each $64\times64$, that is $\sim 4$ million parameters — large. To control this, **manually specify which important, related field pairs to cross** rather than crossing all pairs.

**Hadamard-product form** (replace the final inner product with a Hadamard product):
$$
\boxed{\mathbf{f}_{ij} = \mathbf{x}_i \circ \big(\mathbf{W}_{ij} \mathbf{x}_j\big)} \quad (\text{a vector})
$$
- First compute $\mathbf{W}_{ij}\mathbf{x}_j$ (a vector), then Hadamard with $\mathbf{x}_i$.
- $m$ fields → $m^2$ vectors. Concatenating all of them gives an enormous, mostly-useless vector — again, **hand-pick a subset of pairs** to both shrink the output and cut parameters.

### 4.4 FiBiNet: Putting It Together

**FiBiNet = Feature Importance (SENet) + Bilinear feature interaction.** Pipeline:

```
discrete features ──Embedding──> embeddings ──┬─> Concatenate ────────────────┐
                                              │                                │
                                              ├─> Bilinear ───────────────────►├─ Concatenate ─> upper network
                                              │                                │
                                              └─> SENet ─> Bilinear ──────────►┤
                                                                                │
continuous features ──(transform)──────────────────────────────────────────────┘
```

- One branch keeps the raw embeddings (concatenated).
- One branch applies **Bilinear Cross** directly on the embeddings.
- One branch applies **SENet** (reweight) then **Bilinear Cross** on the reweighted embeddings.
- All resulting vectors, plus transformed **continuous features** (连续特征), are concatenated and fed to the upper network of the ranking model (排序模型的上层网络).
- Versus a simple ranking model, FiBiNet's additions are exactly the **SENet reweighting** and **Bilinear Cross**.

**Practitioner caveat (from the lecturer, Xiaohongshu/小红书 experience):** SENet weighting and Bilinear Cross do give real gains on ranking models, but the speaker considers the *extra* Bilinear-Cross branch (on raw embeddings) somewhat redundant — at Xiaohongshu they did not use that branch and just concatenated directly. They borrowed FiBiNet's ideas without copying the architecture; their model and Bilinear-Cross implementation differ from the paper.

---

## 5. Key Insights

| Technique | Core formula | What it does | Cost / params | Used where |
|---|---|---|---|---|
| **Linear** | $b + \sum w_i x_i$ | weighted sum, no crossing | $O(d)$ | baseline |
| **FM** | $b + \sum w_i x_i + \sum_{i<j}(\mathbf{v}_i^T\mathbf{v}_j)x_i x_j$ | 2nd-order cross via low-rank | $O(kd)$ vs $O(d^2)$ | LR/LogReg replacement |
| **DCN cross layer** | $\mathbf{x}_{i+1}=\mathbf{x}_0\circ(\mathbf{W}\mathbf{x}_i+\mathbf{b})+\mathbf{x}_i$ | explicit bounded-degree cross + residual | $O(Ld)$ (V1) | shared bottom / tower / expert |
| **DCN (full)** | deep MLP ‖ cross net → concat → FC | explicit + implicit crosses | — | ranking & retrieval backbone |
| **LHUC / PPNet** | $\mathbf{h}\circ(2\sigma(\text{gate}(\mathbf{u})))$ | per-user multiplicative gating (Sigmoid×2 ∈ (0,2)) | small | ranking only |
| **SENet** | AvgPool → FC+ReLU → FC+Sigmoid → row-wise mult | field-wise importance reweighting ($w\in(0,1)^m$) | $O(m^2/r)$ | ranking |
| **Bilinear Cross (inner)** | $f_{ij}=\mathbf{x}_i^T\mathbf{W}_{ij}\mathbf{x}_j$ | learnable pairwise field cross → scalars | $\sim m^2/2$ matrices | ranking |
| **Bilinear Cross (Hadamard)** | $\mathbf{f}_{ij}=\mathbf{x}_i\circ(\mathbf{W}_{ij}\mathbf{x}_j)$ | learnable pairwise field cross → vectors | $\sim m^2/2$ matrices | ranking (pick pairs) |
| **FiBiNet** | SENet + Bilinear Cross + concat | importance + interaction | — | ranking |

**Recurring threads to remember for interviews:**
- The **Hadamard product** ($\circ$) is the common element-wise-multiply primitive across DCN cross layers, LHUC gating, and one Bilinear-Cross variant.
- **Low-rank factorization** (FM) and **bounded-degree stacking** (DCN) are two ways to get crossing power without $O(d^2)$ enumeration.
- **Sigmoid×2 ∈ (0,2)** in LHUC vs **Sigmoid ∈ (0,1)** in SENet: LHUC can amplify (>1) and shrink; SENet only attenuates (≤1).
- A **field** = one discrete feature's whole embedding; SENet weights and Bilinear crosses operate at field granularity, with $m$ = number of fields (small, tens).
- For Hadamard-based crosses (plain or Bilinear), $m^2$ outputs blow up, so **hand-select pairs**; inner-product crosses produce scalars and are cheaper.

---

## Expanded Reading — Recent Advances (2026): The Unified-Transformer / Scaling-Law Wave

The classic methods above bake in **fixed, explicit crosses**: FM enumerates second-order pairs, DCN/DCNv2 grows polynomial degree one layer at a time, FiBiNet hand-picks field pairs. The 2025–2026 wave replaces these hand-designed crossing rules with **learned Transformer interaction** and, crucially, fuses feature interaction with **user-behavior sequence modeling** (cross-reference note 05) inside one backbone that obeys **scaling laws** — accuracy improving predictably (near log-linear) with parameters/FLOPs, à la LLMs. Two design philosophies frame the debate: **encode-then-interaction** (compress the sequence first, then run a DCNv2/RankMixer-style interaction module on the summary — late, unidirectional fusion) versus **unified modeling** (sequence and non-sequence tokens share one Transformer stack so interaction and sequence modeling co-evolve bidirectionally). Almost all of these models descend from the **RankMixer / DCNv2 / Wukong** feature-interaction lineage scaled up — think of them as "DCNv2 grown into a Transformer that also reads your click history."

### InterFormer (Meta, 2024)

> Zeng, Z., Liu, X., Hang, M., Liu, X., Zhou, Q., Yang, C., … Yang, J. (2024). *InterFormer: Effective heterogeneous interaction learning for click-through rate prediction*. arXiv. https://arxiv.org/abs/2411.09852

**Method.** Learns sequence (S) and non-sequence (NS) information in an **interleaving** style with **bidirectional information flow**, the seed idea the whole wave builds on. NS summarization (a CLS-token-like summary) guides sequence learning through a **Personalized FeedForward Network (PFFN)** plus multihead attention; a sequence summary then instructs NS learning. It deliberately **avoids early/aggressive aggregation** (summation, pooling, concat) by keeping one-to-one input→output token mappings via MHA. The NS-side interaction module is pluggable — **DCNv2 or DHEN**.

**Key results.** Up to +0.14% AUC on benchmarks, +0.15% NE on industrial data; deployed across Meta Ads with +0.15% performance gain and **+24% QPS** vs. prior SOTA.

**Trade-offs / limitations.** Still keeps S and NS as distinct modes glued by interaction modules rather than a single token stream — a half-step toward full unification; serving cost of retaining full token resolution is non-trivial.

**Connection to this chapter.** The interaction core is literally **DCNv2 (§2)**; InterFormer wraps it in a Transformer that also consumes the behavior sequence and reinjects sequence signal into the cross — the bridge from "DCN as a tower" to "DCN inside a sequence-aware Transformer."

### OneTrans (ByteDance, 2025)

> Zhang, Z., Pei, H., Guo, J., Wang, T., Feng, Y., Sun, H., Liu, S., & Sun, A. (2025). *OneTrans: Unified feature interaction and sequence modeling with one transformer in industrial recommender*. arXiv. https://arxiv.org/abs/2510.26104

**Method.** The canonical **unified** answer to InterFormer's half-step. A **unified tokenizer** turns sequential and non-sequential features into a **single token stream** processed by a **pyramid of causal blocks** (progressively pruning sequence tokens). It uses **mixed parameterization** (HiFormer-style): all homogeneous S tokens **share one set of Q/K/V and FFN** weights, while each NS token gets **token-specific** parameters. **Cross-request KV caching** of S tokens reuses user-side compute across candidates, cutting time complexity from $O(C)$ to $O(1)$ for a session of $C$ candidates, and inherits LLM tricks (FlashAttention, mixed precision).

**Key results.** Near log-linear scaling with model size; **+5.68% per-user GMV** in online A/B (accepted at WWW 2026).

**Trade-offs / limitations.** Merging all features into one `[SEP]`-style stream means heterogeneous sequences and NS tokens compete in a shared attention space — exactly the choice HyFormer and HeMix later critique as blurring semantic boundaries.

**Connection to this chapter.** Generalizes **DCNv2 + RankMixer** into a single Transformer: the feature-interaction module is no longer a separate cross network but emergent from attention/FFN over the joint token stream.

### HyFormer (ByteDance, 2026)

> Huang, Y., Hong, S., Xiao, X., Jin, J., Luo, X., Wang, Z., Chai, Z., Wu, S., Zheng, Y., & Lin, J. (2026). *HyFormer: Revisiting the roles of sequence modeling and feature interaction in CTR prediction*. arXiv. https://arxiv.org/abs/2601.12681

**Method.** Critiques OneTrans' single merged stream and the prevailing "encode-then-interaction" pipeline. HyFormer instead frames ranking as **alternating optimization** of two modules: **Query Decoding** — NS features expand into **Global Tokens** that cross-attend over **layer-wise K/V of long behavior sequences** (each sequence processed with its own representation rather than pre-compressed into one query); and **Query Boosting** — strengthens cross-query/cross-sequence heterogeneous interaction via efficient **token mixing (HeadMixing)**. This restores bidirectional, early, fine-grained fusion that a compressed late-fusion pipeline loses.

**Key results.** Beats strong **LONGER** and **RankMixer** baselines under matched params/FLOPs with better scaling slope; significant online A/B lifts in high-traffic production.

**Trade-offs / limitations.** More moving parts (alternating two modules, layer-wise K/V) than OneTrans' uniform stack; sequence-specific K/V storage grows with the number of sequences.

**Connection to this chapter.** Query Boosting's token mixing is the **RankMixer** descendant of FiBiNet-style field interaction; vs. **OneTrans**, HyFormer argues *interact across, not merge into one stream* — keep sequences independent and let Global Tokens decode them.

### MixFormer (ByteDance, 2026)

> Huang, X., Zhang, H., Fan, Z., Huang, Y., Wei, Z., Chai, Z., Ni, J., Zheng, Y., & Chen, Q. (2026). *MixFormer: Co-scaling up dense and sequence in industrial recommenders*. arXiv. https://arxiv.org/abs/2602.14110

**Method.** Targets the **co-scaling** problem: when S and NS modules have separate parameters they fight over a fixed FLOPs budget. MixFormer unifies them in **one parameter space** with a **decoder-style cross-attention** (NS features = **Query**, sequence = **Key/Value**), so high-order feature semantics directly drive sequence aggregation. All sub-modules (**Query Mixer**, Cross Attention, **Output Fusion**) are **multi-head with per-head FFNs**. For industrial practicality it adds **User-Item Decoupling + Request-Level Batching** to reuse user-side computation across candidates.

**Key results.** Consistently better accuracy/efficiency and scaling than hierarchical/parallel hybrids; the decoupling + request-level batching removes large redundant compute (≈**36% FLOPs** saved on the UI-decoupled variant); online A/B gains on Douyin and Douyin Lite (active days, usage duration).

**Trade-offs / limitations.** A pure NS=Q, S=KV decoder gives the sequence less ability to be reshaped by interaction than HyFormer's alternating scheme; gains depend on request-level batching being feasible in serving.

**Connection to this chapter.** The Query Mixer / per-head FFN are direct **RankMixer** descendants; the NS-as-Query cross-attention is the Transformer reincarnation of **DCNv2's** "let non-sequence features cross against everything." Vs. **OneTrans**: shared params but an explicit Q/KV split instead of one undifferentiated stream.

### TokenMixer-Large (ByteDance, 2026)

> Jiang, Y., Zhu, J., Han, X., Lu, H., Bai, K., Yang, M., … Xu, P. (2026). *TokenMixer-Large: Scaling up large ranking models in industrial recommenders*. arXiv. https://arxiv.org/abs/2602.06563

**Method.** A pure **feature-interaction backbone** (the RankMixer family) fixed for extreme depth. The original RankMixer's **HeadMixing** suffered residual **misalignment** / vanishing gradients in deep stacks; TokenMixer-Large introduces a **mixing-and-reverting** operation that **restores the $T\times D$ token layout** after mixing, **inter-layer (interval) residuals** + an auxiliary residual loss for stable gradient flow, and a **Sparse Per-token MoE** with **per-token SwiGLU sub-experts** for cheap parameter expansion.

**Key results.** Scales to **7B parameters online / 15B offline**. Deployed across ByteDance: **+1.66% orders** and **+2.98% per-capita preview-payment GMV** (e-commerce), **+2.0% ADSS** (ads), **+1.4% revenue** (live streaming).

**Trade-offs / limitations.** Focuses on the NS/dense feature-interaction tower — sequence modeling is largely handled by separate modules upstream, so it does not itself resolve the encode-then-interaction vs. unified debate the way OneTrans/MixFormer do.

**Connection to this chapter.** This is **the most direct DCNv2/FiBiNet successor**: a token-mixing operator that replaces self-attention for explicit feature crossing, now made depth-stable and sparsely scaled. Many sibling papers (HyFormer, MixFormer, HeMix) use a RankMixer-style mixer as their interaction primitive.

### Kunlun (Meta, 2026)

> Hou, B., Liu, X., Liu, X., Xu, J., Badr, Y., Hang, M., … Li, H. (2026). *Kunlun: Establishing scaling laws for massive-scale recommendation systems through unified architecture design*. arXiv. https://arxiv.org/abs/2602.10016

**Method.** Builds directly on **InterFormer** and diagnoses **poor scaling efficiency** (low Model FLOPs Utilization, MFU) as the barrier to clean power-law scaling. Low-level kernel work: **Generalized Dot-Product Attention (GDPA)** — a fused kernel that folds the **PFFN** weight-generation into attention; **Hierarchical Seed Pooling (HSP)**; **Sliding-Window Attention**. High-level: **Event-Level Personalization** and **Computation Skip (CompSkip)** to avoid redundant compute. Interaction core remains Wukong/Advanced-FM blocks.

**Key results.** **MFU 17% → 37%** on NVIDIA B200 GPUs and roughly **2× scaling efficiency** vs. SOTA; deployed in major Meta Ads models.

**Trade-offs / limitations.** Many gains are systems/kernel-level (hardware-specific, e.g. B200) rather than purely architectural; inherits InterFormer's mode-separated structure rather than going fully token-unified like OneTrans.

**Connection to this chapter.** The interaction blocks are **Wukong = stacked-FM (FM, §1) + DCN-style linear compression**; Kunlun's contribution is making that scale efficiently. Vs. **InterFormer** (its parent): same bidirectional S↔NS philosophy, but re-engineered for MFU and predictable scaling laws.

### EST (Alibaba, 2026)

> Liu, M., Bai, Y., Chan, Z., Chen, S., Sheng, X.-R., Zhu, H., Xu, J., & Chen, X. (2026). *EST: Towards efficient scaling laws in click-through rate prediction via unified modeling*. arXiv. https://arxiv.org/abs/2602.10811

**Method.** Starts from an **effective-rank analysis** of the attention matrix blocks $\langle B,B\rangle,\langle B,N\rangle,\langle N,B\rangle,\langle N,N\rangle$ (behavior B vs. non-behavior N): the **$\langle N,B\rangle$ cross block carries far higher effective rank** (most information), while same-type blocks suffer rank collapse. This motivates **Lightweight Cross-Attention (LCA)** — prune redundant self-interactions and keep mainly the high-signal **S–NS cross block**. **Content Sparse Attention (CSA)** computes intra-sequence content similarity and attends only to the **top similar neighbors** (≈top-5), using **frozen content embeddings** to keep it cheap. Together they decouple scaling from the cost of long sequences.

**Key results.** Stable, predictable **power-law scaling**; on Taobao display ads **+3.27% RPM** and **+1.22% CTR**. Handles lifelong sequences truncated to 5,000 with GSU top-50 retrieval.

**Trade-offs / limitations.** Aggressively dropping self-attention blocks is a bet that same-type interactions are nearly redundant — riskier on domains where NS–NS crosses matter; frozen content embeddings can't adapt during training.

**Connection to this chapter.** EST is an **information-theoretic justification for cross-only interaction** — echoing FM/Bilinear (§4), which also model *pairwise cross* terms while skipping pure self-terms. Vs. **OneTrans** (full unified attention), EST keeps the unified stream but **sparsifies attention to just the cross block** $O(L^2)\to O(L K)$.

### HeMix (Alibaba / AMAP, 2026)

> Wang, F., Yang, G., Zhou, X., Yang, S., & Wang, P. (2026). *Query-mixed interest extraction and heterogeneous interaction: A scalable CTR model for industrial recommender systems*. arXiv. https://arxiv.org/abs/2602.09387

**Method.** Two parts. **Query-Mixed Interest Extraction** uses two groups of **learnable query tokens** (Q-Former / BLIP-2 style), $Q=[Q_G, Q_R]$, to extract **context-independent (global/long-term)** and **context-aware (real-time)** interests from the behavior sequences. **HeteroMixer** is a self-attention *replacement* following an **"interact-then-split"** paradigm: first apply an implicit high-order **bit-wise interaction** across all features, *then* **split** NS tokens for global vs. real-time sequence groups — opposite to OneTrans' split-then-merge, preserving semantic boundaries. It uses a **low-rank MLP token mixer** $\hat G_m = W^R_m\mathrm{ReLU}(W^L_m G_m)$ with rank $d_r \ll N\cdot d_h$ for cheap multi-granularity interaction.

**Key results.** Favorable parameter-scaling; deployed on AMAP with **+3.61% GMV, +2.78% PV_CTR, +2.12% UV_CVR** over DLRM (abstract figures; the body also reports +0.61% GMV / +2.32% PV_CTR / +0.81% UV_CVR on another A/B).

**Trade-offs / limitations.** "Interact-then-split" plus dual query groups add design complexity; benefits hinge on the global vs. real-time interest distinction being meaningful for the domain.

**Connection to this chapter.** HeteroMixer's **low-rank token mixer** is the spiritual heir of **FM's low-rank factorization (§1.3)** and **RankMixer**, applied to token interaction; vs. **OneTrans**, HeMix flips the order to *interact first, split later* to avoid the merged-stream semantic blur that HyFormer also flags.

**Synthesis for interviews.** All eight share one storyline: take the chapter's explicit crossing (FM → DCNv2 → FiBiNet → RankMixer) and (a) make it a **learned Transformer interaction**, (b) **fuse it with the behavior sequence** (link to note 05), and (c) tune it to follow **scaling laws** by raising MFU and pruning redundant compute. The live disagreements are *how unified* to go — one merged stream (OneTrans) vs. independent sequences with cross-attention/decoding (HyFormer, MixFormer, EST, HeMix) — and *where to spend FLOPs* — full attention vs. cross-only/sparse attention, dense vs. sparse-MoE expansion.
