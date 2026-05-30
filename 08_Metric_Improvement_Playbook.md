# Part 08 — Methods to Improve Metrics (涨指标的方法)

> A synthesis / playbook chapter. It does not introduce many brand-new models; instead it ties together everything from Parts 02–07 (retrieval, pre-rank, fine-rank, multi-objective fusion, behavior-sequence modeling, online A/B testing, re-ranking) and frames it all around one practical question every recommendation algorithm engineer faces: **how do I actually move the core metrics?** Most of this material — especially diversity, special populations, and interaction behaviors — is industry experience rarely found in public sources.

---

## Table of Contents

1. [What Metrics Matter and the General Strategy](#1-what-metrics-matter-and-the-general-strategy)
   - [1.1 Core metrics: DAU and retention (LT7 / LT30)](#11-core-metrics-dau-and-retention-lt7--lt30)
   - [1.2 Secondary core metrics and non-core metrics](#12-secondary-core-metrics-and-non-core-metrics)
   - [1.3 The five-part playbook](#13-the-five-part-playbook)
2. [Retrieval Improvements (召回)](#2-retrieval-improvements-召回)
   - [2.1 Retrieval channels and the fixed quota](#21-retrieval-channels-and-the-fixed-quota)
   - [2.2 Improving the two-tower model](#22-improving-the-two-tower-model)
   - [2.3 Item-to-Item (I2I) and its relatives](#23-item-to-item-i2i-and-its-relatives)
   - [2.4 Niche retrieval models](#24-niche-retrieval-models)
3. [Ranking-Model Improvements (排序模型)](#3-ranking-model-improvements-排序模型)
   - [3.1 Improving the fine-rank model](#31-improving-the-fine-rank-model)
   - [3.2 Improving the pre-rank model and rank-consistency distillation](#32-improving-the-pre-rank-model-and-rank-consistency-distillation)
   - [3.3 User behavior-sequence modeling](#33-user-behavior-sequence-modeling)
   - [3.4 Online learning](#34-online-learning)
   - [3.5 The "old-soup" model problem (老汤模型)](#35-the-old-soup-model-problem-老汤模型)
4. [Diversity Improvements (多样性)](#4-diversity-improvements-多样性)
   - [4.1 Fine-rank diversity](#41-fine-rank-diversity)
   - [4.2 Pre-rank diversity](#42-pre-rank-diversity)
   - [4.3 Retrieval diversity](#43-retrieval-diversity)
   - [4.4 Exploration traffic](#44-exploration-traffic)
5. [Special Treatment of Special User Populations (特殊人群)](#5-special-treatment-of-special-user-populations-特殊人群)
   - [5.1 Why treat them specially](#51-why-treat-them-specially)
   - [5.2 Special content pools](#52-special-content-pools)
   - [5.3 Special ranking strategies](#53-special-ranking-strategies)
   - [5.4 Differentiated ranking models (and a cautionary tale)](#54-differentiated-ranking-models-and-a-cautionary-tale)
6. [Leveraging Interaction Behaviors (利用交互行为)](#6-leveraging-interaction-behaviors-利用交互行为)
   - [6.1 Follow](#61-follow)
   - [6.2 Forward / Share](#62-forward--share)
   - [6.3 Comment](#63-comment)
7. [Key Insights](#7-key-insights)
8. [Frontier Directions (2026): The Generative-Recommendation Era](#frontier-directions-2026-the-generative-recommendation-era)
   - [1. Generative Recommendation & semantic IDs](#1-generative-recommendation--semantic-ids)
   - [2. Scaling laws & unified Transformers](#2-scaling-laws--unified-transformers)
   - [3. Lifelong / ultra-long sequence modeling](#3-lifelong--ultra-long-sequence-modeling)
   - [4. LLM reasoning & agentic recsys](#4-llm-reasoning--agentic-recsys)
   - [5. Generative retrieval & serving efficiency](#5-generative-retrieval--serving-efficiency)
   - [6. Industrial foundation-model lessons](#6-industrial-foundation-model-lessons)
   - [Cross-references](#cross-references)
9. [Pinterest in Practice (2024–2026)](#pinterest-in-practice-20242026)
   - [LLM Relevance Judge + Stratified Sampling — when the win is the *measurement*, not the model](#llm-relevance-judge--stratified-sampling--when-the-win-is-the-measurement-not-the-model)

---

## 1. What Metrics Matter and the General Strategy

Before discussing how to raise metrics, we must define **which** metrics we care about. This chapter is about **feed/information-stream recommendation** (抖音/Douyin, 快手/Kuaishou, 小红书/RED), not e-commerce — for e-commerce the dominant metric is revenue (营收), which behaves differently.

### 1.1 Core metrics: DAU and retention (LT7 / LT30)

- **Daily Active Users (日活用户数, DAU)** and **retention (留存)** are the two most core metrics. The job of a feed recsys engineer is essentially "raise LT."
- Retention is most commonly measured by **LT7** and **LT30** (LT = "lifetime" within a window).

**Definition of LT7.** For a user who logs in today ($t_0$), LT7 counts how many days *within the next 7 days, inclusive of today* ($t_0$ through $t_6$) the user logs in.

- Example: a user logs in today, and over $t_0 \dots t_6$ logs in on 4 of those days ⟹ that user's LT7 = 4.
- Bounds: $1 \le \text{LT7} \le 7$ and $1 \le \text{LT30} \le 30$ (the user logging in today guarantees ≥1).
- The APP-level LT7 today = the average LT7 over all users who logged in today.

**The DAU/LT trap (important interview point).** LT growth *usually* means improved user experience — **but not always**. If LT goes **up** while DAU goes **down**, experience did **not** improve; the strategy merely drove away low-activity users.

> *Intuition / worked example:* LT = (total active-days of today's active users) / (number of today's active users). If you ban or churn out low-activity users, the **numerator** shrinks but the **denominator** shrinks *more*, so LT mechanically increases while DAU drops. **Rule:** whenever a model/strategy raises LT, always check that DAU did not fall.

### 1.2 Secondary core metrics and non-core metrics

- **Other core metrics:** user **dwell time (使用时长)**, total reads / total clicks (总阅读数 = 总点击数), total impressions (总曝光数). These rank **below** DAU and retention.
  - Dwell time is positively correlated with retention: more time ⟹ usually higher LT.
  - Dwell time **trades off** against reads/impressions. E.g., if Douyin shows more long videos and fewer short ones, total dwell time rises but impression count falls (a user spends long on one video without scrolling to more). Dwell time relates to DAU/retention; impressions relate to ad revenue — so the two must be balanced.
- **Non-core metrics:** click-through rate (点击率), interaction rate (交互率), etc. A dip here is acceptable **as long as core metrics rise**.
- **UGC platforms** (Douyin/Kuaishou/RED, where content is created by ordinary users) additionally treat **publish volume (发布量)** and **publish penetration rate (发布渗透率)** as core metrics, to keep the content ecosystem rich.

### 1.3 The five-part playbook

The methods to raise metrics are organized into five categories (the structure of this whole chapter):

1. **Improve retrieval models / add new retrieval models** (改进召回模型，添加新召回模型).
2. **Improve pre-rank and fine-rank models** (改进粗排和精排模型).
3. **Increase diversity** in retrieval, pre-rank, and fine-rank (提升多样性).
4. **Treat special populations specially** — new users, low-activity users (特殊对待特殊人群).
5. **Leverage three interaction behaviors** — follow, forward, comment (利用关注、转发、评论).

Parts 1–2 are mostly "model" work and were covered in detail earlier (Parts 02–06) with public literature available. Parts 3–5 are largely undocumented industry experience.

---

## 2. Retrieval Improvements (召回)

### 2.1 Retrieval channels and the fixed quota

- A production recsys has **dozens of retrieval channels (召回通道)**, but their **total retrieval budget is fixed** (e.g., 5000 items go to pre-rank). 
  - Larger total ⟹ better metrics, **but** larger pre-rank compute. Beyond a point the marginal benefit shrinks and the ROI turns unfavorable.
- **Two-tower (双塔, two-tower)** and **Item-to-Item (I2I)** are the two most important model families, together occupying **the majority of the budget**.
- Many **niche models** each take a tiny quota but are still useful. **Key lever:** with total budget *fixed*, adding a niche channel re-allocates quota away from others — it raises core metrics **without** increasing pre-rank compute.
- **Content pools (内容池):** there are many — e.g., items <30 days old, <1 day, <6 hours, a high-quality new-user pool, per-population pools.
  - One model can serve **multiple pools**, producing multiple channels. Training **one** two-tower model and applying it to the 30-day / 1-day / 6-hour pools yields **three** channels, each with its own quota — **no extra training cost**, only one extra ANN index + a small amount of online ANN retrieval cost per pool.

### 2.2 Improving the two-tower model

The standard two-tower has a **user tower** and an **item tower** (each a neural net); each outputs one vector (the user / item representation); similarity (inner product or cosine) drives retrieval via ANN. Three directions to improve it:

**Direction 1 — Optimize positive / negative samples (优化正负样本).** (The single biggest lever.)
- **Easy positive (简单正样本):** a (user, item) pair with a click.
- **Easy negative (简单负样本):** a randomly paired (user, item) — uniformly sample an item from the full corpus; the user almost surely dislikes it.
- **Hard negative (困难负样本):** a (user, item) pair that was retrieved but ranked low in ranking.
- (Best practice from Part 02: mix easy + hard negatives; do **not** use items rejected at fine-rank as negatives.)

**Direction 2 — Improve the network architecture (改进神经网络结构).**
- Replace the fully-connected tower with a stronger structure, e.g., **DCN (Deep & Cross Network, 深度交叉网络)** — better than plain FC.
- Use the **user behavior sequence (last-n)** in the user tower (embed last-n items, average the vectors, feed as a user feature).
- Use a **multi-vector model** instead of the **single-vector model** (the standard two-tower is a single-vector model):
  - **Item tower still outputs one vector** (so the vector DB / ANN index stays single — critical for cost).
  - **User tower outputs *multiple* vectors**, each the same shape as the item vector. Vector $k$'s similarity with the item vector predicts target $k$ (CTR, like-rate, favorite-rate, finish-rate, …). For 10 targets, the user tower emits 10 vectors. This makes retrieval **multi-objective**, analogous to multi-task ranking.
  - *Interview Q: why multiple user vectors but only one item vector?* Item vectors are precomputed and stored in a vector DB; 10 item vectors would mean 10 vector DBs + 10 ANN indexes — too expensive. Keeping the item tower single-vector keeps storage/serving cheap.

**Direction 3 — Improve the training method (改进训练方法).**
- Baseline: binary classification separating positives from negatives.
- **Combine binary classification with in-batch negative sampling (batch 内负采样)** — requires **debiasing (纠偏)**, because popular items are over-represented as both positives and negatives.
- **Self-supervised learning (自监督学习)** so that **cold/long-tail items (冷门物品)** learn better embeddings (they have too few clicks to learn well otherwise), improving overall metrics.

### 2.3 Item-to-Item (I2I) and its relatives

I2I is a large family that retrieves via **similar items**. The most common usage is **U2I2I** (user → item → item):
1. User $u$ likes item $i_1$ (from $u$'s interaction history).
2. Find $i_1$'s similar item $i_2$ (this step is the "I2I").
3. Recommend $i_2$ to $u$.

**How to compute item similarity:**
- **Method 1 — ItemCF and variants.** If many users like both $i_1$ and $i_2$, then $i_1 \approx i_2$. **ItemCF, Online ItemCF, Swing, Online Swing** all share this idea but differ in detail.
  - **Run all four simultaneously**, each with its own quota. Their retrieved results differ enough that combining beats any single one. E.g., letting all four retrieve 250 items each beats letting Swing alone retrieve 1000.
- **Method 2 — Item vector representations.** Compute item vectors (via a two-tower model or a **graph neural network**) and use vector inner-product / cosine as similarity. (A two-tower item vector is essentially a representation of the item's interest point.)

> Each similarity method spawns its own I2I channel; using many of them together helps. **I2I is mature** — there is not much new to improve, but it remains roughly as important as the two-tower.

### 2.4 Niche retrieval models

**I2I-like channels (extend the path):**
- **U2U2I** (user → user → item): $u_1 \approx u_2$, $u_2$ likes item $i$ ⟹ recommend $i$ to $u_1$ (collaborative on similar **users**).
- **U2A2I** (user → author → item): $u$ likes author $a$, $a$ publishes $i$ ⟹ recommend $i$ to $u$ (based on explicit/implicit follow relationships).
- **U2A2A2I** (user → author → author → item): $u$ likes author $a_1$, $a_1 \approx a_2$, $a_2$ publishes $i$ ⟹ recommend $i$ to $u$ (one extra hop on **author** similarity).

**More complex deep models** (each a tiny quota, all useful in practice):
- **PDN** — Path-based Deep Network (Li et al., SIGIR 2021).
- **Deep Retrieval** (Gao et al., CIKM 2021).
- **SINE** — Sparse-Interest Network (Tan et al., WSDM 2021).
- **M2GRL** — Multi-task Multi-view Graph Representation Learning (Wang et al., KDD 2020).

**Summary of retrieval improvements:**
- Two-tower: optimize pos/neg samples, improve architecture, improve training.
- I2I: use ItemCF + variants **and** vector-similarity methods together.
- Add niche models (PDN, Deep Retrieval, SINE, M2GRL).
- With total budget fixed, **carefully tune per-channel quotas** — and even use **different quotas per user population** (e.g., new users vs. mainstream users).

---

## 3. Ranking-Model Improvements (排序模型)

Five sub-topics: fine-rank, pre-rank, behavior-sequence modeling, online learning, old-soup model.

### 3.1 Improving the fine-rank model

Typical fine-rank structure: discrete features → embedding layer → concatenate into a few-thousand-dim vector → a few FC layers → ~hundreds-dim green vector; continuous features (only a few hundred) → another FC net → blue vector. The concatenation of blue+green is the **base (基座)** output, which feeds **multi-objective heads** that each predict a target (CTR, like-rate, forward-rate, comment-rate, …).

**Improving the base:**
- **Improvement 1 — Widen & deepen (加宽加深).** Fine-rank FC nets are usually too small and **underfit**. Despite trillions of total params, >99% sit in the embedding layer; the FC nets are actually tiny. Widening/deepening makes predictions more accurate — at higher compute cost (ROI tradeoff). In practice the base is **1–6 FC layers** depending on compute and engineering: CPU inference ⟹ 1–2 layers; GPU inference ⟹ 3–6 layers.
- **Improvement 2 — Automatic feature crosses (自动特征交叉):** e.g., **bilinear** (FiBiNET, Huang et al., RecSys 2019) and **LHUC** (Swietojanski et al., WSDM 2016). These **do** work in practice.
- **Improvement 3 — Feature engineering (特征工程):** add **statistical features (统计特征)** and **multimodal content features (多模态内容特征)**; relies on engineer experience.

**Improving multi-objective prediction (多目标预估):**
- **Improvement 1 — Add new prediction targets and put them in the fusion formula (融合公式).** Target count grows over time (a few → 10s → 20s+).
  - Standard targets: CTR, like-rate, favorite-rate, forward-rate, comment-rate, follow-rate, finish-rate (完播率)…
  - Hunt for more: "entered comment section," "liked someone else's comment," etc. Targets correlated with user interest tend to lift retention.
- **Improvement 2 — MMoE / PLE structures (MMoE, KDD 2018; PLE, RecSys 2020).** "May work, but often does **not**." Many engineers tried MMoE/PLE with no gain — try it, but don't be surprised if it's a no-op.
- **Improvement 3 — Correct position bias (纠正 position bias / de-bias)** (Zhou et al., RecSys 2019). "May or may not work — more likely won't." Even with strong observed position bias in the data, de-bias frequently fails to deliver in practice.

### 3.2 Improving the pre-rank model and rank-consistency distillation

- The pre-rank (粗排) **scores ~10× more items** than fine-rank (e.g., 5000 vs. 500), forming a funnel; so per-item compute must be ~10× smaller — the pre-rank must be **fast**.
- **Simple pre-rank model:** **multi-vector two-tower**, predicting CTR + other targets simultaneously.
- **Complex pre-rank model:** the **three-tower model (三塔模型)** (COLD, Wang et al., arXiv 2020) — better quality but harder to implement.

**Pre-rank / fine-rank consistency modeling (粗精排一致性建模).** Distill the fine-rank into the pre-rank so their scores/orders agree. Lifts core metrics significantly.

- **Method 1 — Pointwise distillation.** Let $y$ = user's true behavior, $p$ = fine-rank's prediction. Train the pre-rank to fit the soft target

$$
\text{target} = \frac{y + p}{2}.
$$

  Example (CTR): user clicked ($y = 1$), fine-rank predicted $p = 0.6$ ⟹ pre-rank fits $\frac{y+p}{2} = 0.8$. Using $\frac{y+p}{2}$ beats using $y$ alone.

- **Method 2 — Pairwise / listwise distillation.** Given $k$ candidates, sort by fine-rank's predicted target; do **learning-to-rank (LTR)** so the pre-rank fits the **order** (not the values).
  - Example: fine-rank says $p_i > p_j$; LTR encourages pre-rank scores to satisfy $q_i > q_j$, else penalize. Typically **pairwise logistic loss**.
  - Simple pointwise distillation already helps; large companies use pairwise/listwise.

- **Drawback (important):** if the fine-rank has a **bug** (e.g., a feature service breaks), its predictions $p$ become biased, **poisoning the pre-rank's training data**. Such issues can degrade metrics *slowly* and be hard to detect.

### 3.3 User behavior-sequence modeling

Once ranking is well-optimized, behavior-sequence modeling becomes the **main remaining lever**.

- **Baseline:** embed the last-n item IDs into n vectors, **average** them → one user feature (what the user historically liked) (Covington et al., YouTube DNN, RecSys 2016).
- **DIN** (Deep Interest Network, Zhou et al., KDD 2018): use an **attention mechanism** for a **weighted** average of item vectors (weight by relevance to the candidate).
- Industry now follows the **SIM** direction (Qi et al., CIKM 2020): first **filter** items by attributes (e.g., category), then apply DIN's weighted average over the survivors.

**Improvements:**
- **Improvement 1 — Increase sequence length.** More accurate predictions, but higher compute / inference latency. The hardest part is the **engineering architecture**, not the algorithm. State of the art (Kuaishou) reaches **>1,000,000 items** (covering nearly all of a user's history); many companies max out around 1,000.
- **Improvement 2 — Filtering (筛选)** to cut length before the attention layer:
  - Offline: extract multimodal content features (BERT / CLIP) → item vectors.
  - Offline: cluster item vectors into ~1000 clusters (often hierarchical clustering); each item gets a cluster ID.
  - Online ranking: with $n = 1{,}000{,}000$ behavior items, if the candidate's cluster ID is 70, keep only items with cluster ID 70 → only a few thousand survive.
  - Run **several filters** (candidate category + vector clustering) in parallel and take the **union** (then possibly filter once more before attention).
- **Improvement 3 — Use features beyond item ID** for sequence items (a few extra item features help; too many break online communication/compute).

> **Summary (the SIM recipe):** make the raw sequence as long as possible → filter (by category and vector clustering) to drop length by orders of magnitude → feed survivors into DIN (attention-weighted average).

### 3.4 Online learning

Two update modes (recall from earlier):
- **Full update (全量更新):** at dawn, take **yesterday's** full data, *random-shuffle* it, and do **1 epoch** of SGD starting from the **previous day's full checkpoint** (not random init).
- **Incremental update (增量更新) = online learning:** after the dawn full update, do **minute-level** incremental updates continuously all day, publishing the latest model periodically for serving.
- Each new day's full update restarts from the **previous full model**, *not* from the incremental chain — and the incremental chain is then discarded.

**Resource cost (a favorite interview/system-design question).** Online learning needs **both** the dawn full update **and** all-day incremental updates ⟹ extra compute.
- Suppose incrementally updating one fine-rank model needs **10,000 CPU cores**.
- For A/B testing, several models run online simultaneously, **each needing its own online-learning machine set**. If there are $m$ models online, you need **$m$ sets**.
- Of the $m$ models: **1 is the holdout**, **1 is the fully-rolled-out (推全) model**, and **$m - 2$ are new test models**. (Recall the A/B testing traffic split: ~10% holdout, ~90% for experiments.)
- Example: $m = 6$, each set ≈ 10,000 CPU cores ⟹ you can only test **4** new models at once. Since LT7/LT30 require ≥7/≥30 days online, iteration is slow.
- Retrieval and pre-rank also need online learning, but their models are smaller, so they cost less than fine-rank.

**Tradeoff:** online learning gives **huge** metric gains but **severely slows model iteration**. Best adopted **after** the model is relatively mature — adopting it too early risks locking the model into a weak version.

### 3.5 The "old-soup" model problem (老汤模型)

**How it arises:** every day (with or without online learning) the model gets 1 epoch of training on freshly generated data. Over time the **old model becomes extremely well-trained and very hard to beat** — it has "seen" enormous data. A newly proposed (better-structured) model, trained from scratch, **struggles to even catch up**, making iteration painful. (Real anecdote: a known sequence-modeling bug was fixed; after chasing 100 days of data the new model was *still* slightly worse than the old one.)

Two problems:

**Problem 1 — Quickly judge whether the new structure is better than the old one** (without actually catching up to the production old model).
- **Fairly initialize both** new and old structures: random-init the FC layers for both. Embedding layers may be random-init **or** reuse the old model's trained embeddings — but apply the *same* treatment to both, so the only difference is **structure**, and the old model has no "trained-longer" advantage.
- Train both on $n$ days of data (1 epoch, oldest→newest), with $n$ small (~10 days) ⟹ the comparison is **fast**.
- If the new model is **significantly better** in this fair contest, it's *likely* genuinely better — worth continuing. (This only compares **structures**; 10 days cannot catch a 100-day old model.)

**Problem 2 — Catch up to / surpass the production old model faster** (only tens of days of data vs. the old model's hundreds). Needs tricks:
- **Trick 1 — Reuse the old model's trained embeddings (复用 embedding 层); avoid random-init of embeddings.** Embeddings are a slow-learning "memory" of user/item interest points (slower than FC layers). Random-init FC is fine, but random-init embeddings makes catch-up very hard.
- **Trick 2 — Distill from the old model as teacher.** With true behavior $y$ and old-model prediction $p$, train the new model toward $\frac{y+p}{2}$. Distilling **early in training** greatly speeds convergence.

**Summary of ranking improvements:** fine-rank (base: widen/deepen, crosses, feature engineering; heads: new targets, MMoE, position bias) · pre-rank (three-tower; consistency distillation) · behavior sequences (iterate along SIM; longer sequences, better filtering) · online learning (huge gains, slows iteration) · old-soup model (special tricks needed).

---

## 4. Diversity Improvements (多样性)

Increasing diversity in retrieval, pre-rank, and fine-rank meaningfully lifts core metrics.

### 4.1 Fine-rank diversity

At fine-rank, sort items by a **combination of an interest score and a diversity score**:
- $s_i$ — **interest score (兴趣分数):** the fused multi-objective prediction (CTR etc.).
- $d_i$ — **diversity score (多样性分数):** how different item $i$ is from already-selected items.
- Sort by $s_i + d_i$. This sort largely determines what the user finally sees.

**Computing diversity scores:** commonly **MMR** or **DPP** (from Part 07). **Fine-rank uses a sliding window (滑动窗口); pre-rank does not.**
- *Why:* fine-rank determines final exposure; **adjacent items on the exposed page should be dissimilar** (an item should differ from, say, the previous 5). A sliding window enforces "within any window, items are sufficiently different."
- Pre-rank does **not** decide final exposure — it only supplies candidates — so it should consider **overall** diversity, not within-window diversity.

**Rule-based scattering (打散策略)** — additional to diversity scores:
- **Category:** after selecting item $i$, the next **5** positions may not share $i$'s **second-level category (二级类目)**.
- **Multimodal:** precompute multimodal content vectors, cluster the whole corpus into ~1000 clusters; after selecting $i$, the next **10** positions may not be in $i$'s cluster (similar images/text get scattered).

### 4.2 Pre-rank diversity

Pre-rank scores 5000 items and selects 500 for fine-rank. A two-phase selection that injects diversity:
1. Sort all 5000 by **interest score $s_i$ only**; send the top **200** to fine-rank (guarantee the user's most-interesting items get through).
2. For the remaining 4800, compute both $s_i$ and **$d_i$** (difference from the already-selected 200).
3. Sort those 4800 by $s_i + d_i$; send the top **300** (interesting *and* diverse relative to the first 200).
- Total = 200 + 300 = 500 items into fine-rank.

### 4.3 Retrieval diversity

**Two-tower — add noise to the user vector (添加噪声).**
- After computing the user vector but **before** ANN retrieval, add **random Gaussian noise**.
- **Noise strength depends on interest breadth:** the **narrower** the user's interest (e.g., last-n items cover only a few categories), the **stronger** the noise.
- Counterintuitively, noise makes retrieval "less accurate" yet **raises** metrics by improving diversity. Because users refresh dozens of times/day (each triggering retrieval), per-refresh noise also makes successive retrievals differ.

**Two-tower & U2I2I — non-uniform sampling of the behavior sequence (抽样用户行为序列).**
- The last-n behavior items feed the user tower (and are the **seed items (种子物品)** for U2I2I).
- Keep the most recent $r$ items ($r \ll n$); from the remaining $n - r$, **randomly sample $t$ items** ($t \ll n$). Sampling may be uniform, or **non-uniform to balance categories**. Use the $r + t$ items as the sequence instead of all $n$.
- *Why it raises metrics:*
  1. **Injects randomness** → more diverse retrieval (different even across two simultaneous retrievals).
  2. **$n$ can be very large** → captures interests from long ago at unchanged retrieval cost. (E.g., compute limits the sequence to ~100 items normally; set $n = 1000$, sample 100 → cover much older interests for the same cost.)
- **U2I2I specifics:** the n behavior items cover few, **imbalanced** categories (e.g., of 200 system categories, a user covers only 15; soccer = 0.4n items, TV-drama = 0.2n, all others <0.05n each). Using them directly as seeds ⟹ retrieval clusters on soccer/TV. **Non-uniform sampling** to balance categories ⟹ more diverse U2I2I retrieval; also lets $n$ be large to cover more categories.

### 4.4 Exploration traffic

- Reserve ~**2%** of each user's exposures for **non-personalized** items, used for **interest exploration (兴趣探索)**.
- Maintain a **curated content pool (精选内容池)** of items with **high interaction-rate** quality (can be one global pool, or **per-population**, e.g., a 30–40-y-o male pool).
- Randomly sample a few items from the pool, **skip ranking**, and **insert directly** into the final result (these wouldn't survive fair ranking since they don't match past interest — so use boosting / forced insertion).
- Exploration **hurts core metrics short-term but helps long-term.** (Logic: without personalization, compensate with **high content quality**.)

**Summary of diversity:** fine-rank (interest + diversity scores; rule scattering) · pre-rank (interest-only selection + interest+diversity selection) · retrieval (noise on user vector; non-uniform sequence sampling for two-tower & U2I2I) · exploration (reserve a little non-personalized traffic).

> *No audio existed for sub-lectures 05–06; the sections below rely on the slides only.*

---

## 5. Special Treatment of Special User Populations (特殊人群)

### 5.1 Why treat them specially

New users and low-activity users (新用户、低活用户) need special handling because:
1. **Few behaviors ⟹ inaccurate personalization** (个性化推荐不准确).
2. **High churn risk** — must actively work to retain them.
3. **Behavior differs from the mainstream** (CTR, interaction rate differ), so a model trained on *all* users is **biased on these populations**.

Three approaches: (1) special content pools for retrieval, (2) special ranking strategies to protect them, (3) special ranking models to remove the prediction bias.

### 5.2 Special content pools

**Why:** since personalization is poor for new/low-activity users, **guarantee content quality** instead; and tailor pools to a population's traits to raise satisfaction (e.g., a "promote-comment" pool for middle-aged women who like to comment, satisfying their interaction need).

**How to build a special pool:**
- **Method 1 — Select quality items by interaction count / interaction rate.**
  - **Define the population** (e.g., 18–25-y-o males in tier-1/2 cities).
  - **Build the pool:** score items by **that population's** interaction count/rate; take the top-scoring items. This gives a **weak-personalization** effect.
  - **Refresh periodically:** add new items, drop low-interaction or stale ones. The pool **only applies to that population.**
- **Method 2 — Causal inference (因果推断):** estimate each item's contribution to the population's **retention rate** and select by that contribution.

**Retrieval from special pools** — typically via the **two-tower model** (personalized; for new users it can't personalize well, so rely on high quality + weak personalization).
- **Extra training cost?** For normal users, **one** two-tower model regardless of pool count. For **new users** (very sparse history), a **separately trained** model is needed.
- **Extra inference cost?** Each pool needs an **ANN index** (refreshed when the pool updates) and online ANN retrieval. But special pools are **small (10–100× smaller than the full corpus)**, so the extra compute is modest.

### 5.3 Special ranking strategies

**Exclude low-quality items / protect special users.** Business-wise, for new/low-activity users we care **only about retention**, not consumption (total impressions, ad revenue, e-commerce revenue):
- **Show fewer / no ads** to new and low-activity users.
- **Don't explore newly-published items on new/low-activity users** — fresh items rank poorly and hurt experience. **Only explore (and boost new items) on active veteran users**, avoiding harm to fragile populations.

**Differentiated fusion formula (差异化的融分公式).**
- New/low-activity users' click & interaction behavior differs; their **per-capita click count is small**, and **no click ⟹ no further interaction**.
- For low-activity users, **raise the weight on predicted CTR** in the fusion formula (relative to normal users).
- **Reserve a few exposure slots for the highest-CTR items.** Example: fine-rank picks 50 of 500 — **3 slots** go to the highest-CTR items, the other 47 are decided by the fusion formula. You can even **rank the top-CTR item first** to guarantee the user sees it.

### 5.4 Differentiated ranking models (and a cautionary tale)

**Problem:** the ranking model is **dominated by mainstream users**, so it predicts poorly for special users (a model trained on all users is badly biased for new users; if an APP is 90% female, predictions for male users are biased). How to make accurate predictions for special users?

- **Method 1 — Big model + small model (大模型 + 小模型).**
  - Train the **big model** on all users; its prediction $p$ fits behavior $y$.
  - Train a **small model** on the special population; its prediction $q$ fits the **residual** $y - p$.
  - Mainstream users: use **$p$** only. Special users: use **$p + q$**.
- **Method 2 — Fuse multiple experts, MMoE-style.** One model with multiple experts, each outputting a vector; take a **weighted average** of expert outputs, where the **weights are computed from user features** (e.g., new-ness, activity level). 
- **Method 3 — Big model predict, then small model calibrate (校准).** The big model predicts CTR/interaction rates; feed **(user features + big-model predictions)** into a **small model (e.g., GBDT)** trained on the special population's data so its output fits true behavior.

**The wrong way (错误的做法) — a maintained cautionary tale.** Giving **each population its own full ranking model** (system maintains many big models; each refreshed nightly from the main model + 1 epoch on that population's data):
- Short-term it raises metrics, **but the maintenance cost is high and it's harmful long-term.**
- Example: at first, the low-activity-male model beats the main model by +0.2% AUC. After the **main model** iterates a few versions, its AUC cumulatively rises +0.5%. The special-population models become **too numerous to maintain**, go stale, and **fall behind**. Eventually, **retiring the low-activity-male model and switching to the main model raises AUC on that very population by +0.3%.** ⟹ Don't fork full models per population; prefer big+small / experts / calibration.

**Summary:** retrieval (special pools + extra channels) · ranking strategy (exclude low-quality items, protect new/low-activity users, differentiated fusion formula) · ranking model (big+small residual; multi-expert; big-then-small calibration).

---

## 6. Leveraging Interaction Behaviors (利用交互行为)

Interaction behaviors: like (点赞), favorite (收藏), forward (转发), follow (关注), comment (评论). The simplest use is to predict their rates and put them in the fusion formula. But three behaviors — **follow, forward, comment** — have **additional value** beyond ranking targets.

### 6.1 Follow

**Retention value of follow count.** The more authors a user follows, the stickier the platform: retention rate $r$ is **positively correlated** with follow count $f$ — a **concave** curve (rises fast at small $f$, saturates at large $f$). So if a user's $f$ is small, the system should **push them to follow more authors**.

- **Method 1 — Ranking strategy to boost follows.** For user $u$, model predicts candidate $i$'s follow-rate $p_i$; let $u$ already follow $f$ authors. Define a **monotonically decreasing** $w(f)$ (more follows ⟹ smaller $w(f)$). Add a term

$$
w(f)\cdot p_i
$$

  to the fusion formula. If $f$ is small **and** $p_i$ is large, this gives item $i$ a big boost (promote-follow, 促关注).
- **Method 2 — Build a "promote-follow" content pool + retrieval channel.** Items in it have high follow-rate; apply it to users with small $f$. Quota can be fixed or **negatively correlated with $f$**.

**Publish value of follower count (粉丝数对促发布).** UGC platforms treat publish volume/rate as core metrics. Interactions (especially follow & comment) raise an author's **motivation to publish**; the **fewer** an author's followers, the **larger** the marginal motivation per new follower. So help low-follower new authors gain followers:
- Author $a$ has follower count $f_a$. Item $i$ (by $a$) may be recommended to $u$ with predicted follow-rate $p_{ui}$. Define decreasing $w(f_a)$ (more followers ⟹ smaller $w(f_a)$). Add $w(f_a)\cdot p_{ui}$ to the fusion formula to help low-follower authors grow.

**Implicit follow relationships (隐式关注关系).** Channel **U2A2I** (user → author → item):
- **Explicit follow:** $u$ follows $a$ → recommend $a$'s items (CTR/interaction usually higher than other channels).
- **Implicit follow:** $u$ likes $a$'s items but hasn't followed $a$. Implicit-follow authors **far outnumber** explicit ones — mining them to build a U2A2I channel raises core metrics.

### 6.2 Forward / Share

When a user forwards an item from platform A to platform B, A gains **off-site traffic**; this **share-back (分享回流)** raises DAU and consumption metrics.

**Why naive boosting fails.** With predicted forward-rate $p$ and a fusion term $w\cdot p$, raising $w$ does promote forwarding and pull in off-site traffic — **but hurts CTR and other interaction rates.**

**KOL modeling.** Goal: pull in maximum off-site traffic **without** harming clicks/other interactions. **Whose forward brings lots of off-site traffic? Other platforms' Key Opinion Leaders (KOLs).**
- A user $u$ who is an in-site KOL but not an off-site KOL has limited forward value; a user $v$ with no in-site followers but who **is** an off-site KOL has **high** forward value.
- **Detect off-site KOLs** by how much off-site traffic their historical forwards brought.

**Promote-forward strategy:**
- **Method 1 — Add fusion term $k_u \cdot p_{ui}$.** $k_u$ is large if $u$ is an off-site KOL; $p_{ui}$ is the predicted forward-rate of recommending $i$ to $u$. So off-site KOLs get more exposure to items they're likely to forward.
- **Method 2 — Build a promote-forward content pool + retrieval channel** that activates for off-site KOLs.

### 6.3 Comment

**Comment promotes publishing (评论促发布).** Comments (like follows) raise authors' publish motivation. If a newly-published item hasn't gotten many comments, **boost its predicted comment-rate** so it gets comments fast. Add fusion term

$$
w_i \cdot p_i,
$$

where $w_i$ is a weight **negatively correlated** with item $i$'s existing comment count, and $p_i$ is the predicted comment-rate.

**Other value of comments:**
- Some users **enjoy commenting** and interacting with authors / other commenters. Give them a **promote-comment content pool** for more discussion opportunities → improves **their** retention.
- Some users routinely leave **high-quality comments** (high comment-like counts). High-quality comments help **authors' and other users'** retention (interesting/helpful). Use ranking & retrieval strategies to **encourage these users to comment more.**

**Summary of interaction behaviors:**
- **Follow:** retention value (get new users to follow more authors) + publish value (help new authors gain followers) + mine **implicit** follow for U2A2I retrieval.
- **Forward:** detect off-site KOLs and exploit their forward value to attract off-site traffic.
- **Comment:** publish value (get new items commented) + retention value (more discussion for discussion-lovers) + encourage high-quality commenters.

---

## 7. Key Insights

| Theme | Key levers | Gotchas / nuances |
|---|---|---|
| **Metrics** | DAU & retention (LT7/LT30) are core; dwell time, reads, impressions secondary | LT can rise while DAU falls (churning low-activity users) — always check DAU; dwell time ⟷ impressions trade off |
| **Retrieval** | Add channels under a **fixed budget**; improve two-tower (pos/neg, DCN, last-n, multi-vector, in-batch neg + debias, self-supervised for cold items); I2I (ItemCF/Swing × 4 + vector sim); niche (U2U2I, U2A2I, PDN, SINE…) | Adding channels under fixed budget doesn't raise pre-rank compute; multi-vector keeps **item** tower single-vector to keep ANN cheap |
| **Fine-rank** | Base: widen/deepen, bilinear/LHUC crosses, feature engineering; heads: more targets in fusion, MMoE/PLE, position bias | FC nets underfit (>99% params are embeddings); MMoE/PLE & de-bias **often don't work** |
| **Pre-rank** | Three-tower (vs. multi-vector two-tower); pointwise $\frac{y+p}{2}$ + pairwise/listwise LTR distillation | Fine-rank bugs poison pre-rank training data |
| **Sequences** | SIM recipe: long sequence → cluster/category filter → DIN attention | Bottleneck is engineering, not algorithm; SOTA >1M items |
| **Online learning** | Dawn full update + all-day minute-level incremental | $m$ online models = $m$ machine sets (1 holdout + 1 rollout + $m{-}2$ tests); huge gains but throttles iteration |
| **Old-soup** | Fair init to compare structures fast; reuse embeddings + teacher distillation $\frac{y+p}{2}$ to catch up | Embeddings learn slowly; never random-init them when chasing |
| **Diversity** | Fine-rank $s_i+d_i$ + sliding-window MMR/DPP + rule scatter; pre-rank two-phase; user-vector noise; non-uniform seq sampling; 2% exploration | Sliding window for fine-rank only; exploration hurts short-term, helps long-term |
| **Special pops** | Special pools (quality/causal); fewer ads, no new-item exploration, CTR-weighted fusion + reserved slots; big+small / multi-expert / calibration | **Never** fork a full big model per population — stale, harmful long-term |
| **Interactions** | Follow: $w(f)p_i$, U2A2I incl. implicit; Forward: off-site KOL $k_u p_{ui}$; Comment: $w_i p_i$ for under-commented items | Naive forward-boost hurts CTR; target **off-site** KOLs specifically |

## Frontier Directions (2026): The Generative-Recommendation Era

The five-part playbook above is the *operating system* of a modern feed recsys, and it is not going away. But the 2026 research wave reshapes **how each lever is implemented under the hood**. The throughline is **generative recommendation (GR, 生成式推荐)** and **scaling laws (规模定律)** — like LLMs, recsys models get monotonically better with parameters/compute *if* the architecture and serving stack are designed for it. Below is a strategic overlay mapping the six dominant themes onto "how do I actually move the metrics?", with the key trade-off for each. **This is intentionally a pointer-level synthesis — the conceptual depth and full bibliographies live in the dedicated frontier Notes 09–12, and the per-paper Expanded Reading sections of Notes 01–07.**

### 1. Generative Recommendation & semantic IDs

Replace the item-ID embedding with a short sequence of shared semantic tokens (typically via RQ-VAE / RQ-Kmeans / FSQ residual quantization), so two items can share a prefix and the model can generalize to (user, item) transitions it never literally saw in training. **So what for metrics:** the biggest unlock is **cold-start and long-tail coverage** — content tokens give new items a warm start no ID embedding can. **Trade-off:** denser tokenization buys generalization at the cost of head-item memorization, so the win lands in tail CTR / coverage rather than head-traffic AUC; naive ID→semantic swaps *degrade* unless tokens are made CF-aware (ByteDance TRM). → Full treatment in **[Note 09 — Generative Recommendation & Semantic IDs](09_Generative_Recommendation_and_Semantic_IDs.md)**.

### 2. Scaling laws & unified Transformers

The DCN-on-concatenated-embeddings fine-rank "base" of §3.1 is being replaced by Transformer backbones that **jointly scale feature interaction and sequence modeling** — the OneTrans / Kunlun / HyFormer / MixFormer / TokenMixer-Large / InterFormer / EST / HeMix wave. The headline debate is *encode-then-interaction vs. unified modeling*, but the real bottleneck is efficiency, not accuracy: **MFU**, **KV-cache** reuse for request-invariant user-side tokens, and **sliding-window / semi-local attention** decide what actually ships (Kunlun took MFU from 17%→37%, ULTRA-HSTU shipped where vanilla HSTU couldn't). **So what for metrics:** the modern version of §3.1's "widen & deepen" — billions of dense parameters that finally scale because the architecture is built for it. **Trade-off:** every AUC gain is paid in serving FLOPs/latency. → **[Note 10 — Scaling Laws & Large-Model Architectures](10_Scaling_Laws_and_Large_Model_Architectures.md)** for the architectures; **[Note 12](12_Serving_and_Inference_Efficiency.md)** for the serving stack.

### 3. Lifelong / ultra-long sequence modeling

Direct successor to SIM (Note 05 §5). The design space cleanly partitions into **search-based** (retrieve only the candidate-relevant subset from the lifelong history), **compression-based** (summarize the whole history into fixed-size state — pooling / clustering / learnable query tokens), and **hybrid**. **So what for metrics:** once short-window sequence modeling matures, lifelong modeling is the main remaining lever — it recovers stale-but-real interests short windows drop. **Trade-off:** search-based is accurate but adds per-candidate online latency; compression-based is candidate-invariant (cacheable) but lossy. → **[Note 05 §5 (SIM)](05_Sequence_Modeling.md)** for the foundation; **[Note 10 §7](10_Scaling_Laws_and_Large_Model_Architectures.md)** for the lifelong taxonomy and QARM V2.

### 4. LLM reasoning & agentic recsys

The least-proven frontier imports LLM **reasoning** and **agents** into the pipeline: reasoning re-rankers that generate an explicit rationale before re-ordering (Meta GR2); simulated user environments to train and evaluate agentic recommenders without burning real traffic (Meta RecoWorld); and self-evolving / AutoML pipelines where LLM agents autonomously propose, test, and ship model changes (Google Self-Evolving) — a potential answer to §3.4's online-learning resource pain and §3.5's old-soup catch-up. **So what for metrics:** reasoning can lift quality on hard, sparse, or explanation-sensitive surfaces; agentic AutoML compresses the LT7/LT30 experiment cycle. **Trade-off:** token-by-token LLM inference is orders of magnitude more expensive than a forward pass, so reasoning is realistic mainly at the final re-rank stage (tens of items) or offline, not in retrieval / pre-rank. → **[Note 11 — LLM-Augmented & Agentic Recommenders](11_LLM_Augmented_and_Agentic_Recommenders.md)**.

### 5. Generative retrieval & serving efficiency

If items are semantic-token sequences (Theme 1), retrieval becomes **constrained decoding** — generate a valid token sequence that must correspond to a real item, enforced by a **trie** of all valid item codes. The 2026 work is overwhelmingly about making this run on accelerators: vectorizing the trie so constrained decoding is GPU/TPU-friendly (Google STATIC), MoE load balancing so sparse experts don't bottleneck (Meta replicate-and-quantize), and custom attention kernels for recsys workloads (PyTorch GDPA). **So what for metrics:** the GR-native replacement for the two-tower + ANN channels of §2.1–§2.2; serving efficiency decides whether it is deployable at all under the fixed retrieval budget. **Trade-off:** unifies the funnel and lifts tail recall, but constrained decoding is far heavier than an ANN lookup — the entire thrust is clawing back the latency. → **[Note 09 §3](09_Generative_Recommendation_and_Semantic_IDs.md)** for the decoding mechanism; **[Note 12 — Serving & Inference Efficiency](12_Serving_and_Inference_Efficiency.md)** for the kernels and trie systems.

### 6. Industrial foundation-model lessons

How non-frontier-lab companies actually adopt GR: **user foundation models** that learn one reusable user representation across many downstream tasks (Coinbase); **multimodal content foundation models** for understanding items and powering cold-start (Netflix MediaFM — the production version of the BERT/CLIP content features in §3.3 and §4.1); large-scale **sequential recommenders** for feed ranking (LinkedIn); and **Hetero-MMoE** extending §3.1's multi-task structure with heterogeneous experts (Uber). **So what for metrics:** a shared foundation model amortizes representation-learning cost across teams and gives cold users/items a strong prior — directly attacking §5's special-population sparsity and §3.5's old-soup catch-up. **Trade-off:** foundation models centralize a dependency (one backbone many teams must trust) and reintroduce a sharper version of the staleness risk — when the backbone lags, *everything* downstream lags. → **[Note 11 §1 (LLM-as-encoder spectrum)](11_LLM_Augmented_and_Agentic_Recommenders.md)** for framing; **[Note 07 — Item Cold-Start](07_Item_Cold_Start.md)** for the MediaFM cold-start angle.

### Cross-references

Each theme is covered in depth in the dedicated frontier Notes 09–12; the per-paper write-ups stay in the **Expanded Reading** sections of the relevant fundamentals notes (no bibliography is duplicated here).

| Theme | Dedicated frontier note(s) | Per-paper Expanded Reading |
|---|---|---|
| 1. Generative Recommendation & semantic IDs | [Note 09](09_Generative_Recommendation_and_Semantic_IDs.md) | [Note 02](02_Candidate_Retrieval.md) — TRM, STATIC, TrieRec |
| 2. Scaling laws & unified Transformers | [Note 10](10_Scaling_Laws_and_Large_Model_Architectures.md) | [Note 04](04_Feature_Interaction.md) — InterFormer, OneTrans, HyFormer, MixFormer, TokenMixer-Large, Kunlun, EST, HeMix |
| 3. Lifelong / ultra-long sequence modeling | [Note 10 §7](10_Scaling_Laws_and_Large_Model_Architectures.md) | [Note 05](05_Sequence_Modeling.md) — ULTRA-HSTU, ULBM survey, QARM V2 |
| 4. LLM reasoning & agentic recsys | [Note 11](11_LLM_Augmented_and_Agentic_Recommenders.md) | [Note 06](06_Reranking_and_Diversity.md) — GR2, RecoWorld, Self-Evolving |
| 5. Generative retrieval & serving efficiency | [Note 09 §3](09_Generative_Recommendation_and_Semantic_IDs.md), [Note 12](12_Serving_and_Inference_Efficiency.md) | [Note 02](02_Candidate_Retrieval.md), [Note 03](03_Ranking_Models.md) — GDPA, R&Q MoE |
| 6. Industrial foundation-model lessons | [Note 11 §1](11_LLM_Augmented_and_Agentic_Recommenders.md), [Note 07](07_Item_Cold_Start.md) | [Note 03](03_Ranking_Models.md) — Uber Hetero-MMoE; [Note 05](05_Sequence_Modeling.md) — Coinbase UFM; [Note 07](07_Item_Cold_Start.md) — Netflix MediaFM |

## Pinterest in Practice (2024–2026)

### LLM Relevance Judge + Stratified Sampling — when the win is the *measurement*, not the model

> Wang, H., Whitworth, A., Cheung, P. M., Zhang, Z., & Kamath, K. (2025). *LLM-based relevance assessment for web-scale search evaluation at Pinterest*. arXiv. https://arxiv.org/abs/2509.03764

This chapter is about **measuring and moving metrics**, and Pinterest Search offers a sharp reminder that the cheap-to-compute *judge* is only half the story — the other half is **how you sample**. They fine-tuned an XLM-RoBERTa$_{large}$ cross-encoder relevance judge to replace human labels for whole-page relevance evaluation in A/B experiments (73.7% exact-match, 91.7% within 1 point), but the LLM judge **alone** did not deliver the headline gain. The real payoff came from exploiting the now-cheap labels to redesign the sampling: **stratified sampling** with strata = (query interest × popularity segment), allocated via **Neyman optimal allocation**, with **paired control/treatment samples** that block between-query variance. Because **between-query variance dominates** relevance at Pinterest, stratification removes that variance term and cut the **minimum detectable effect (MDE, 最小可检测效应)** from roughly **1.3–1.5% down to ≤0.25%** — about an order of magnitude — with most of the gain from variance reduction, not raw sample size ($\hat\sigma$ drops ~52% from stratification alone). The interview takeaway ties straight to §3.4's A/B-sensitivity pain: a 10× lower MDE means you can detect smaller effects and ship more, smaller experiments — **measurement infrastructure as a velocity multiplier**, not just a better ranker. (One integrity guardrail: they keep humans in the loop for experiments that change the relevance model itself, to avoid the judge "grading its own homework.")
