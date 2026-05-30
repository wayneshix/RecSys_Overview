# Part 03 — Rank (排序)

Recommendation systems course notes — Part 03: Ranking Models.

Ranking is the second major stage of the recommendation pipeline. Retrieval (召回) returns a few thousand candidate items; ranking scores them and keeps the best. Ranking is itself split into two sub-stages:

- **Pre-rank (粗排)**: scores the few-thousand candidates with a *lightweight* model, keeping the top few hundred.
- **Fine-rank (精排)**: scores those few hundred with a *heavy, accurate* model.
- The fine-rank scores then flow into **re-rank (重排)**, which does diversity sampling and finally shows a few dozen items to the user.

Pre-rank and fine-rank share the **same principles** — pre-rank is just a smaller model with fewer features and lower accuracy, used for fast preliminary filtering. So most of this part describes a generic "ranking model" without distinguishing the two; the differences are made explicit in the final section.

The central question of ranking: **how interested is this user in this item?** We answer it by predicting multiple engagement probabilities (click, like, collect, share, play-time, completion…), fusing them into one score, sorting, and truncating.

---

## Table of Contents

1. [Multi-Objective Ranking Model](#1-multi-objective-ranking-model)
   - [1.1 Why multiple objectives?](#11-why-multiple-objectives)
   - [1.2 Architecture: Shared Bottom + per-task heads](#12-architecture-shared-bottom--per-task-heads)
   - [1.3 Training](#13-training)
2. [Class Imbalance, Negative Down-Sampling & Calibration](#2-class-imbalance-negative-down-sampling--calibration)
   - [2.1 The problem: class imbalance (类别不平衡)](#21-the-problem-class-imbalance-类别不平衡)
   - [2.2 Solution: negative down-sampling (负样本降采样)](#22-solution-negative-down-sampling-负样本降采样)
   - [2.3 The side effect: prediction is biased high](#23-the-side-effect-prediction-is-biased-high)
   - [2.4 Calibration formula (预估值校准)](#24-calibration-formula-预估值校准)
3. [Multi-gate Mixture-of-Experts (MMoE)](#3-multi-gate-mixture-of-experts-mmoe)
   - [3.1 Structure](#31-structure)
   - [3.2 The polarization problem (极化现象 / Polarization)](#32-the-polarization-problem-极化现象--polarization)
   - [3.3 Fix: Dropout on the Softmax outputs](#33-fix-dropout-on-the-softmax-outputs)
4. [Score Fusion](#4-score-fusion)
   - [4.1 Simple weighted sum (加权和)](#41-simple-weighted-sum-加权和)
   - [4.2 Click rate × weighted sum of the rest](#42-click-rate--weighted-sum-of-the-rest)
   - [4.3 Overseas short-video app — product of powered terms](#43-overseas-short-video-app--product-of-powered-terms)
   - [4.4 Domestic short-video app — rank-based fusion](#44-domestic-short-video-app--rank-based-fusion)
   - [4.5 E-commerce — product over the conversion funnel](#45-e-commerce--product-over-the-conversion-funnel)
5. [Video Play Modeling](#5-video-play-modeling)
   - [5.1 Image-text notes vs. video](#51-image-text-notes-vs-video)
   - [5.2 Play-time modeling (播放时长建模)](#52-play-time-modeling-播放时长建模)
   - [5.3 Completion rate (视频完播)](#53-completion-rate-视频完播)
   - [5.4 Why you can't use predicted completion rate directly (important)](#54-why-you-cant-use-predicted-completion-rate-directly-important)
6. [Ranking Model Features](#6-ranking-model-features)
   - [6.1 User profile (用户画像 / User Profile)](#61-user-profile-用户画像--user-profile)
   - [6.2 Item profile (物品画像 / Item Profile)](#62-item-profile-物品画像--item-profile)
   - [6.3 User statistical features (用户统计特征)](#63-user-statistical-features-用户统计特征)
   - [6.4 Item statistical features (笔记统计特征)](#64-item-statistical-features-笔记统计特征)
   - [6.5 Context features (场景特征 / Context)](#65-context-features-场景特征--context)
   - [6.6 Feature processing (特征处理)](#66-feature-processing-特征处理)
   - [6.7 Feature coverage (特征覆盖率)](#67-feature-coverage-特征覆盖率)
   - [6.8 Data service architecture (数据服务)](#68-data-service-architecture-数据服务)
7. [Pre-Rank Model](#7-pre-rank-model)
   - [7.1 Pre-rank vs. Fine-rank](#71-pre-rank-vs-fine-rank)
   - [7.2 Two reference architectures](#72-two-reference-architectures)
   - [7.3 The pre-rank three-tower model (粗排的三塔模型)](#73-the-pre-rank-three-tower-model-粗排的三塔模型)
   - [7.4 Distillation (蒸馏)](#74-distillation-蒸馏)
8. [Summary](#summary)
9. [Expanded Reading — Recent Advances (2026)](#expanded-reading--recent-advances-2026)
   - [(a) Pre-ranking advances](#a-pre-ranking-advances)
   - [(b) Multi-task & generative ranking](#b-multi-task--generative-ranking)
   - [(c) Scaling & serving systems](#c-scaling--serving-systems)
10. [Pinterest in Practice (2024–2026)](#pinterest-in-practice-20242026)
   - [InteractRank — Search Pre-Ranking with Cross-Interaction Features](#interactrank--search-pre-ranking-with-cross-interaction-features)
   - [DRL-PUT — Reinforcement Learning for Ranking-Utility Tuning (score fusion)](#drl-put--reinforcement-learning-for-ranking-utility-tuning-score-fusion)
   - [Clean-Room CVR — Privacy-Preserving Multi-Task Conversion Modeling (calibration, CVR)](#clean-room-cvr--privacy-preserving-multi-task-conversion-modeling-calibration-cvr)

---

## 1. Multi-Objective Ranking Model

### 1.1 Why multiple objectives?

For each item (a "note" on Xiaohongshu), the system logs counts of:

- **Impressions** (曝光次数) — how many times the item was shown.
- **Clicks** (点击次数).
- **Likes** (点赞次数).
- **Collects / saves** (收藏次数).
- **Shares / forwards** (转发次数).

From these we define rates. Note the **denominators differ** — clicks happen on impressions, but likes/collects/shares only happen *after* a click:

$$
\text{CTR (click rate)} = \frac{\text{clicks}}{\text{impressions}}, \qquad
\text{like rate} = \frac{\text{likes}}{\text{clicks}},
$$
$$
\text{collect rate} = \frac{\text{collects}}{\text{clicks}}, \qquad
\text{share rate} = \frac{\text{shares}}{\text{clicks}}.
$$

> Intuition for the denominators: a user can only like/collect/share a note *after* opening it, so the natural conditioning event is "clicked", not "impressed". Sharing is rare (far less than likes/collects) but valuable — a share to WeChat brings external traffic.

The ranking model **predicts** these rates: $p_1 = $ pCTR, $p_2 = $ p(like), $p_3 = $ p(collect), $p_4 = $ p(share). We then fuse them (Section 4), sort, and truncate.

### 1.2 Architecture: Shared Bottom + per-task heads

```
       pCTR        p(like)     p(collect)   p(share)     ← 4 predicted scores ∈ (0,1)
        ↑             ↑            ↑            ↑
  [FC + Sigmoid] [FC + Sigmoid] [FC + Sigmoid] [FC + Sigmoid]   ← 4 task-specific heads
        └─────────────┴────────────┴────────────┘
                          ↑
                  [ Neural Network ]                ← "shared bottom" (共享底座)
                          ↑
                   Concatenation
        ┌──────────┬──────────┬──────────┬──────────┐
   user feats   item feats  statistical  context        ← input features
   (用户特征)    (物品特征)   (统计特征)    (场景特征)
```

- All raw features (user / item / statistical / context — see Section 6) are **concatenated** and fed into one shared neural network, the **shared bottom**. It outputs a single representation vector.
- That vector feeds **four separate task heads**, each a few fully-connected layers ending in **Sigmoid**, producing one probability in $(0,1)$ per objective.
- The shared bottom can be a plain MLP, Wide & Deep, or any more complex structure.

> **Interview point — why share the bottom?** The four tasks (click/like/collect/share) are correlated; sharing the lower layers lets them learn common feature representations and pool data, while the small per-task heads specialize. This is classic multi-task learning.

### 1.3 Training

Each objective is a **binary classification**. Labels $y_i \in \lbrace0,1\rbrace$ are the user's true behaviors (e.g. $(y_1,y_2,y_3,y_4) = (1,0,0,1)$ = "clicked, not liked, not collected, shared").

Per-task loss is cross-entropy:

$$
\text{CrossEntropy}(y_i, p_i) = -\big[ y_i \ln p_i + (1-y_i)\ln(1-p_i)\big].
$$

The total loss is a **weighted sum** over the four tasks:

$$
\mathcal{L} = \sum_{i=1}^{4} \alpha_i \cdot \text{CrossEntropy}(y_i, p_i),
$$

where the weights $\alpha_i$ are set empirically. Train by gradient descent on the model parameters.

---

## 2. Class Imbalance, Negative Down-Sampling & Calibration

### 2.1 The problem: class imbalance (类别不平衡)

Positives are rare, negatives dominate (illustrative numbers, *not* real Xiaohongshu data):

- Per 100 impressions: ~10 clicks (positive), ~90 no-click (negative).
- Per 100 clicks: ~10 collects (positive), ~90 no-collect (negative).

Keeping all the negatives wastes compute for little benefit.

### 2.2 Solution: negative down-sampling (负样本降采样)

Keep only a small fraction of the negatives so positives and negatives are roughly balanced. This shrinks the dataset and speeds up training (e.g. 10 h → 3 h of cluster time).

### 2.3 The side effect: prediction is biased high

Let $n_+$ = #positives, $n_-$ = #negatives, and let $\alpha \in (0,1)$ be the **sampling rate** — we keep $\alpha \cdot n_-$ negatives. Because we threw away negatives, the model **over-estimates** the click rate. The smaller $\alpha$, the more severe the over-estimation.

- True click rate (expectation):
$$
p_{\text{true}} = \frac{n_+}{n_+ + n_-}.
$$
- Predicted click rate (what the down-sampled model learns):
$$
p_{\text{pred}} = \frac{n_+}{n_+ + \alpha \cdot n_-}.
$$

### 2.4 Calibration formula (预估值校准)

Eliminating $n_+, n_-$ between the two equations gives the calibration formula (He et al., Facebook 2014):

$$
\boxed{p_{\text{true}} = \frac{\alpha \cdot p_{\text{pred}}}{(1 - p_{\text{pred}}) + \alpha \cdot p_{\text{pred}}}}
$$

**Online flow:** the model outputs $p_{\text{pred}}$ → apply this formula → use the calibrated $p_{\text{true}}$ for ranking.

> Sanity checks: if $\alpha = 1$ (no down-sampling) then $p_{\text{true}} = p_{\text{pred}}$; as $\alpha \to 0$, $p_{\text{true}} \to 0$ (it scales the inflated prediction back down).

**Reference:** Xinran He et al. *Practical lessons from predicting clicks on ads at Facebook.* ADKDD 2014.

---

## 3. Multi-gate Mixture-of-Experts (MMoE)

MMoE is a refinement of the shared-bottom multi-task model. Instead of one shared bottom, it uses **several "expert" networks** plus a **gating network per task** that decides how to mix the experts.

### 3.1 Structure

1. **Experts (专家):** Take the concatenated input features and feed them into $n$ independent neural networks (the slides use $n = 3$). Their outputs are vectors $\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3$.

2. **Gating networks (门控):** Each *task* has its own small gating network: a neural net + Softmax that outputs a probability vector summing to 1.
   - Task 1 (e.g. CTR) → weights $\mathbf{p} = (p_1, p_2, p_3)$.
   - Task 2 (e.g. like) → weights $\mathbf{q} = (q_1, q_2, q_3)$.

3. **Weighted combination:** Each task takes a **weighted average of the experts** using its own gate's weights:
   - CTR tower input: $p_1\mathbf{x}_1 + p_2\mathbf{x}_2 + p_3\mathbf{x}_3$.
   - Like tower input: $q_1\mathbf{x}_1 + q_2\mathbf{x}_2 + q_3\mathbf{x}_3$.

4. Each combined vector goes into its task's head network → the predicted score (CTR, like-rate, …).

```
      pCTR                          p(like)
        ↑                              ↑
   [task head]                    [task head]
        ↑                              ↑
  p1·x1 + p2·x2 + p3·x3        q1·x1 + q2·x2 + q3·x3   ← weighted average of experts
        ↑          ↖   x1  x2  x3  ↗         ↑
   gate p ───┐    [Expert 1][Expert 2][Expert 3]   ┌─── gate q
   (Softmax) │              ↑   ↑   ↑               │ (Softmax)
             └──────────────┴───┴───┴──────────────┘
                       concatenated features
```

> **Why "multi-gate"?** Each task gets its *own* gate, so different tasks can lean on different experts — e.g. the click task may weight expert 1 heavily while the like task weights expert 3 heavily. The experts are shared (sample efficiency), but the mixing is task-specific (specialization).

**Reference:** Jiaqi Ma et al. *Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts.* KDD 2018 (Google).

### 3.2 The polarization problem (极化现象 / Polarization)

With $n$ experts, each gate's Softmax produces $n$ values. **Polarization** = the Softmax output collapses so that **one value ≈ 1 and the rest ≈ 0**.

- If a gate puts weight $\approx 1$ on a single expert and $\approx 0$ on the others, the task effectively uses only **one** expert — defeating the purpose of mixing.
- Worse, if *no* task ever assigns weight to a given expert, that expert receives no gradient and becomes **"dead"** (never trained).

### 3.3 Fix: Dropout on the Softmax outputs

During training, apply **dropout to each gate's Softmax output vector**:

- Each of the $n$ Softmax values is masked (set to 0) with probability **10%**.
- Equivalently, each expert is randomly dropped for a task with probability 10%.

**Intuition:** if a task relied entirely on one expert and that expert gets dropped, the prediction is terrible and the loss is huge — so the model is *forced* to spread weight across experts and avoid polarization.

**Reference (polarization fix):** Zhe Zhao et al. *Recommending What Video to Watch Next: A Multitask Ranking System.* RecSys 2019 (YouTube).

---

## 4. Score Fusion

After predicting $p_{\text{click}}, p_{\text{like}}, p_{\text{collect}}, \dots$, we must **fuse** them into a single score for sorting. Several formulas are used in industry; weights are tuned via A/B testing.

### 4.1 Simple weighted sum (加权和)

$$
\text{score} = p_{\text{click}} + w_1 \cdot p_{\text{like}} + w_2 \cdot p_{\text{collect}} + \cdots
$$

### 4.2 Click rate × weighted sum of the rest

Exploits the fact that the other rates are *conditioned on a click*:

$$
\text{score} = p_{\text{click}} \cdot \big(1 + w_1 \cdot p_{\text{like}} + w_2 \cdot p_{\text{collect}} + \cdots\big)
$$

> Why this is natural: $p_{\text{click}} = \frac{\text{click}}{\text{impression}}$ and $p_{\text{like}} = \frac{\text{like}}{\text{click}}$. Multiplying $p_{\text{click}} \cdot p_{\text{like}}$ gives $\frac{\text{like}}{\text{impression}}$ — an *unconditional* like rate. The structure respects the funnel: nothing matters if the user never clicks.

### 4.3 Overseas short-video app — product of powered terms

$$
\text{score} = (1 + w_1 \cdot p_{\text{time}})^{\alpha_1} \cdot (1 + w_2 \cdot p_{\text{like}})^{\alpha_2} \cdots
$$

The $\alpha$ exponents control the relative influence of each factor.

### 4.4 Domestic short-video app — rank-based fusion

Rather than using the predicted values directly, it uses each item's **rank** under each prediction:

- Sort the $n$ candidate videos by predicted play-time $p_{\text{time}}$; if a video ranks $r_{\text{time}}$, it scores $\frac{1}{r_{\text{time}}^{\alpha} + \beta}$.
- Do the same for click, like, share, comment, etc.
- Final fused score:

$$
\text{score} = \frac{w_1}{r_{\text{time}}^{\alpha_1} + \beta_1}
            + \frac{w_2}{r_{\text{click}}^{\alpha_2} + \beta_2}
            + \frac{w_3}{r_{\text{like}}^{\alpha_3} + \beta_3} + \cdots
$$

> Rank-based fusion is robust to differing score scales/distributions across objectives.

### 4.5 E-commerce — product over the conversion funnel

Conversion funnel: **impression → click → add-to-cart → pay**. Model predicts $p_{\text{click}}, p_{\text{cart}}, p_{\text{pay}}$. Final score multiplies these (plus price, since revenue matters):

$$
\text{score} = p_{\text{click}}^{\alpha_1} \times p_{\text{cart}}^{\alpha_2} \times p_{\text{pay}}^{\alpha_3} \times \text{price}^{\alpha_4}.
$$

---

## 5. Video Play Modeling

### 5.1 Image-text notes vs. video

- **Image-text notes (图文)** are ranked mainly on click / like / collect / share / comment.
- **Videos** are *additionally* ranked on **play-time (播放时长)** and **completion (完播)**. On video platforms these two are often the *primary* signals; clicks/interactions come second. (If a user watches a video to the end, that signals interest even without any like/collect/share.)
- Play-time is continuous, so the naïve idea is regression — but **directly regressing play-time works poorly**. Use the YouTube-style modeling below instead.

**Reference:** Paul Covington, Jay Adams, Emre Sargin. *Deep Neural Networks for YouTube Recommendations.* RecSys 2016.

### 5.2 Play-time modeling (播放时长建模)

This is one task head on the shared-bottom model. The trick reframes a regression as a weighted classification so that $\exp(z)$ predicts play-time.

Let the last fully-connected layer output a scalar $z$ (can be positive or negative). Define:

$$
p = \text{sigmoid}(z) = \frac{\exp(z)}{1 + \exp(z)}.
$$

Let $t$ be the **observed** play-time ($t = 0$ if there was no click). Define the fitting target:

$$
y = \frac{t}{1+t}.
$$

**Training** minimizes cross-entropy between $p$ and $y$:

$$
\mathcal{L} = -\Big[ \frac{t}{1+t}\cdot \log p + \frac{1}{1+t}\cdot \log(1-p)\Big].
$$

**Key derivation:** at the optimum $p = y$, i.e. $\frac{\exp(z)}{1+\exp(z)} = \frac{t}{1+t}$, which forces

$$
\boxed{\exp(z) = t.}
$$

So:

- **Training** uses $p = \text{sigmoid}(z)$ in the cross-entropy loss. After training, $p$ is discarded.
- **Inference** uses $\exp(z)$ directly as the **predicted play-time**, and feeds $\exp(z)$ into the score-fusion formula.

> Practical note (from the lecture audio): you can drop the $\frac{1}{1+t}$ denominators; that just turns the loss into a **play-time-weighted** cross-entropy (weight = play-time), which also works in practice.

### 5.3 Completion rate (视频完播)

**Method A — regression.** Example: a 10-min video watched for 4 min → true completion rate $y = 0.4$. Train predicted completion rate $p$ to fit $y$ with cross-entropy:

$$
\mathcal{L} = y\cdot\log p + (1-y)\cdot\log(1-p).
$$

Online, $p = 0.73$ means "expected to play 73%".

**Method B — binary classification.** Define a completion threshold, e.g. "completed = watched > 80%". For a 10-min video, > 8 min = positive, < 8 min = negative. Train a binary classifier. Online, $p = 0.73$ means $\mathbb{P}(\text{play} > 0.8) = 0.73$.

### 5.4 Why you can't use predicted completion rate directly (important)

**Longer videos have intrinsically lower completion rates** — a 15-second clip is easy to finish, a 15-minute video is not. Using raw predicted completion rate would **unfairly favor short videos** and penalize long ones.

**Fix — normalize by video length.** Fit a function $f(\text{video length})$ to the empirical curve of completion-rate vs. length (a decreasing curve — longer ⇒ lower). Then adjust:

$$
p_{\text{finish}} = \frac{\text{predicted completion rate}}{f(\text{video length})}.
$$

Because longer videos have smaller $f$, dividing by $f$ levels the playing field. Use $p_{\text{finish}}$ — not the raw completion rate — as a term in the fusion formula, alongside play-time, CTR, like-rate, etc.

---

## 6. Ranking Model Features

The ranking model uses **every available feature**. Five families:

### 6.1 User profile (用户画像 / User Profile)
- **User ID** (embedded in both retrieval and ranking).
- Demographics: gender, age.
- Account info: new vs. old account, activity level.
- Inferred interests: categories (类目), keywords, brands.

### 6.2 Item profile (物品画像 / Item Profile)
- **Item ID** (embedded).
- Publish time / age.
- **GeoHash** (lat-long encoding), city.
- Title, category, keywords, brand.
- Word count, #images, video resolution (清晰度), #tags.
- Content richness (信息量), image aesthetics (图片美学).

### 6.3 User statistical features (用户统计特征)
- User's counts over the **last 30 days / 7 days / 1 day / 1 hour**: #impressions, #clicks, #likes, #collects…
- **Bucketed by item type:** e.g. this user's CTR on image-text notes vs. on videos over last 7 days.
- **Bucketed by category:** e.g. last 30 days, this user's CTR on beauty (美妆) notes vs. food (美食) notes vs. tech/digital (科技数码) notes.

### 6.4 Item statistical features (笔记统计特征)
- Item's counts over last 30d / 7d / 1d / 1h: #impressions, #clicks, #likes, #collects…
- **Bucketed by viewer demographics:** by user gender, by user age.
- **Author features (作者特征):** #notes published, #followers (粉丝数), consumption metrics (the author's aggregate impressions/clicks/likes/collects).

### 6.5 Context features (场景特征 / Context)
- User's current GeoHash / city (e.g. same city as item ⇒ higher interest).
- Current timestamp (bucketed and embedded).
- Whether it's a weekend / holiday.
- Phone brand, phone model, OS.

> Context features are **passed in with the user request**, unlike profile features which are looked up.

### 6.6 Feature processing (特征处理)
- **Discrete features → embedding:** user ID, item ID, author ID, category, keyword, city, phone brand.
- **Continuous features → bucketize into discrete:** age, word count, video length.
- **Continuous features → other transforms:**
  - Counts (impressions, clicks, likes) → $\log(1 + x)$ (compresses heavy tails).
  - Convert to rates (CTR, like-rate, …) and apply **smoothing**.

### 6.7 Feature coverage (特征覆盖率)
- Many features don't cover 100% of samples: users skip age (low coverage), privacy settings block geolocation (missing context).
- **Improving feature coverage makes the fine-rank model more accurate.**

### 6.8 Data service architecture (数据服务)
On a user request, the **main server (主服务器)** sends item IDs, user ID, and context features to the **ranking server (排序服务器)**, which pulls features from three stores:

| Store | Content | Volatility |
|---|---|---|
| User profile (用户画像) | user features | **fairly static** (较为静态) |
| Item profile (物品画像) | item features | **static** (静态) |
| Statistics (统计数据) | statistical features | **dynamic** (动态) |

The ranking server packs the features (特征打包) and calls model serving (e.g. **TF Serving**) to score the candidates.

> Why volatility matters: static features (item profile) and fairly-static features (user profile) can be **cached**; dynamic statistics change constantly and cannot — this drives the pre-rank design in Section 7.

---

## 7. Pre-Rank Model

### 7.1 Pre-rank vs. Fine-rank

| | Pre-rank | Fine-rank |
|---|---|---|
| #items scored | a few **thousand** | a few **hundred** |
| Cost per inference | must be **small** | can be **large** |
| Accuracy | lower | higher |

Pre-rank exists for speed: running the big fine-rank model on thousands of candidates is too expensive. Pre-rank does a fast first cut.

### 7.2 Two reference architectures

**Fine-rank model = early fusion (前期融合).** Concatenate *all* features first, then run one big neural net. If there are $n$ candidate items, the **whole big model runs $n$ times** → expensive. (This is the shared-bottom model from Section 1.)

**Two-tower model = late fusion (后期融合).** User and item features go into *separate* towers; no early feature crossing. Online:
- The **user tower** runs **once** to produce user embedding $\mathbf{a}$.
- Item embeddings $\mathbf{b}$ are **precomputed and stored** in a vector database — the item tower does **no online inference**.
- Score = $\cos(\mathbf{a}, \mathbf{b})$.
- Cheap, but **less accurate** than fine-rank (no feature crossing, only a final cosine).

### 7.3 The pre-rank three-tower model (粗排的三塔模型)

A middle ground between the two — more accurate than two-tower, cheaper than fine-rank. Three towers:

```
   pCTR      p(like)    p(collect)   p(share)
    ↑           ↑           ↑           ↑
 [FC+Sig]    [FC+Sig]    [FC+Sig]    [FC+Sig]      ← upper network: runs n times
    └───────────┴───────────┴───────────┘
              Concatenation & Cross
        ┌────────────┬───────────────┬───────────────┐
   [ User Tower ]  [ Item Tower ]   [ Cross Tower ]
     (large)         (medium)          (small)
        ↑               ↑                 ↑
  user + context   item feats        statistical +
    features        (static)         cross features
```

| Tower | Inputs | Size | Online inferences |
|---|---|---|---|
| **User tower (用户塔)** | user features + context features | **very large** | **1** (one user per request) |
| **Item tower (物品塔)** | item features (**static**) | medium | up to $n$, but **cacheable** |
| **Cross tower (交叉塔)** | statistical + cross features | small | **$n$** (cannot cache) |

The three tower outputs are combined by **Concatenation & Cross**, then the **upper network** (per-task FC + Sigmoid heads) produces the four scores.

**Inference cost analysis (the crux):**

- **User tower** runs only **once** per request (there is just one user) — so even though it's large, its total cost is small. → put the model's *capacity* here.
- **Item tower** inputs are *static* item features → outputs are **cached** in the parameter server (PS). On a cache hit, no inference; only cache misses need recompute. So in practice it runs ≪ $n$ times.
- **Cross tower** consumes *dynamic statistical features* → **cannot be cached**, so it **must run $n$ times**. → keep it **small**.
- **Upper network** also runs **$n$ times** (once per item to produce $n$ scores). **Most of the pre-rank compute is in the upper network**, so it is kept lightweight.

> **Design principle:** push capacity into the parts that run once (user tower) or can be cached (item tower); keep the parts that must run $n$ times (cross tower + upper network) small. This is why statistical features (dynamic) are isolated into a tiny cross tower — they're the one thing you can't cache.

**Reference:** Zhe Wang et al. *COLD: Towards the Next Generation of Pre-Ranking System.* DLP-KDD 2020.

### 7.4 Distillation (蒸馏)

Pre-rank is commonly trained by **distillation**: the heavy, accurate **fine-rank model acts as the teacher**, and the lightweight pre-rank model is the **student** trained to match the teacher's scores. This aligns pre-rank's ordering with fine-rank's at a fraction of the cost, reducing the inconsistency between the two stages.

> *Source note: distillation is the standard industry practice for pre-rank and is referenced across this course; the slides for this lesson focus on the three-tower architecture rather than spelling out the distillation recipe.*

---

## Summary

- **Multi-objective model:** one shared bottom + per-task Sigmoid heads predict click / like / collect / share rates; train with weighted cross-entropy.
- **Imbalance:** down-sample negatives at rate $\alpha$ → predictions inflate → calibrate with $p_{\text{true}} = \frac{\alpha p_{\text{pred}}}{(1-p_{\text{pred}}) + \alpha p_{\text{pred}}}$.
- **MMoE:** experts + per-task Softmax gates; beware **polarization**, fix with **dropout** on gate outputs.
- **Score fusion:** weighted sum, click×rest, powered products, or rank-based — tuned by A/B tests.
- **Video:** model play-time via the $\exp(z)=t$ trick; model completion rate but **normalize by video length** ($p_{\text{finish}} = \text{pred}/f(\text{length})$) for fairness.
- **Features:** user/item profiles, user/item statistics, context; embed discretes, bucketize/transform continuous; mind coverage.
- **Pre-rank:** lightweight first cut via a **three-tower model** (heavy user tower runs once, cacheable item tower, tiny cross tower for dynamic stats), often trained by **distillation** from the fine-rank model.

## Expanded Reading — Recent Advances (2026)

The 2026 ranking literature pushes three frontiers that map directly onto this chapter: **smarter pre-rank supervision and routing**, **richer multi-task and generative rankers** (extending MMoE and score fusion into one generative sequence), and the **scaling/serving systems** that make billion-parameter rankers tractable online.

### (a) Pre-ranking advances

#### Generative Pseudo-Labeling for Pre-Ranking with LLMs (GPL)

> Bi, J., Niu, X., Cheng, D., Yuan, K., Wang, T., Cao, B., Wu, J., & Jiang, Y. (2026). *Generative pseudo-labeling for pre-ranking with LLMs*. arXiv. https://arxiv.org/abs/2602.20995

**Method.** GPL attacks the **sample-selection bias (SSB, 样本选择偏差)** of pre-rank head-on: the model is trained only on *exposed* items but must score *all* recalled candidates online, including unexposed long-tail items. Instead of mislabeling unexposed items as negatives (negative sampling) or distilling from an already-biased ranker, GPL generates **unbiased, content-aware pseudo-labels** offline using an LLM. Items are tokenized into hierarchical **semantic IDs (SIDs)** by a frozen multimodal encoder + **RQ-VAE (Residual-Quantized VAE)**; a fine-tuned LLM predicts the user's next likely SIDs from history and decodes them into **interest anchors**. Relevance for an unexposed pair $(u, h)$ is the match between the user's anchors and the candidate in the frozen semantic space, calibrated by an uncertainty-aware weight (semantic coherence + historical consistency + LLM confidence). The pre-ranker then minimizes a confidence-weighted joint objective $\min_f  \mathcal{L}_{\text{al}}(\mathcal{D}_e) + \lambda\mathcal{L}_{\text{pl}}(\mathcal{D}_u)$ over actual labels $\mathcal{D}_e$ and pseudo-labels $\mathcal{D}_u$.

**Key results.** Deployed in a large-scale Alibaba production system: **+3.07% CTR** in online A/B testing, with improved recommendation diversity and long-tail item discovery.

**Trade-offs / limitations.** All LLM inference and pseudo-labeling are **offline and cached per user**, so there is zero online latency — but at the cost of an offline LLM + RQ-VAE pipeline and the storage to cache anchors. Pseudo-label quality hinges on the frozen semantic space and the calibration weighting; a bad codebook propagates bias.

**Connection to this chapter.** Section 7.4 notes pre-rank is usually trained by **distillation** from the fine-rank teacher. GPL points out that teacher *inherits exposure bias* (the teacher only saw exposed items too), so distillation propagates SSB. GPL is a complementary supervision source: it does not replace the three-tower architecture (§7.3) but supplies better *labels* for the unexposed candidates the three-tower model must score online.

#### Heterogeneity-Aware Adaptive Pre-ranking (HAP)

> Tong, P., Chen, S., Zhang, C., Wang, B., Pi, Q., Li, P., & Liu, Z. (2026). *Not all candidates are created equal: A heterogeneity-aware approach to pre-ranking in recommender systems*. arXiv. https://arxiv.org/abs/2603.03770

**Method.** HAP observes that pre-rank training samples are **heterogeneous** — they come from retrieval results, ranking signals, and exposure feedback, mixing trivially-easy negatives with near-positive hard negatives. Indiscriminately mixing them causes **gradient conflicts**: hard negatives induce disproportionately large gradients (shown theoretically for BCE and InfoNCE) and dominate training while easy samples are underused. HAP has two parts: (1) **Gradient-Harmonized Contrastive Learning (GHCL)** builds four difficulty-aware negative sets (exposed > ranking > pre-rank/random) and a tailored contrastive loss that *harmonizes* gradient magnitudes across them; (2) **Difficulty-Aware Model Routing (DAMR)** runs a lightweight model over *all* candidates for cheap coverage, then routes only the *hard* candidates to a stronger, more expressive model — allocating compute where it pays off rather than uniformly scaling the whole model.

**Key results.** Deployed in ByteDance's **Toutiao (今日头条)** for 9 months: up to **+0.4%** user app usage duration and **+0.05%** active days, **without additional compute cost**. The authors also release a large-scale industrial hybrid-sample dataset.

**Trade-offs / limitations.** DAMR adds a routing decision and a second (heavier) model path, increasing serving complexity; the gains are concentrated on hard candidates, so easy-dominated traffic sees little benefit. GHCL's four-bucket difficulty taxonomy is platform-specific and needs tuning.

**Connection to this chapter.** DAMR is a sharper take on §7.1's pre-rank/fine-rank cost trade-off and §7.3's "push capacity where it pays" principle — instead of one fixed lightweight model for *all* candidates, it spends the big model only on hard ones. GHCL directly addresses the negative-sampling and class-imbalance issues of §2, replacing a single down-sampling rate $\alpha$ with difficulty-stratified, gradient-balanced negatives.

### (b) Multi-task & generative ranking

#### Sequential Modeling and Hetero-MMoE at Uber

> Estrada, D., & Lin, L. (2026). *Transforming ads personalization with sequential modeling and Hetero-MMoE at Uber*. Uber Engineering Blog.

**Method.** Two upgrades to Uber Eats ads ranking. (1) **Target-aware transformer sequence features**: a user's engagement history (store UUID, cuisine, hour/day, engagement type) is encoded by a transformer in which the **candidate ad acts as the query**, so attention focuses on the history events most relevant to the item being scored — preserving recency/ordering that flat aggregate counts lose. They swap quadratic Multi-Head Attention for **Multi-Head Latent Attention (MLA)**, routing attention through a small set of learnable latents to cut complexity from $O(N^2)$ to $O(N \times L)$ with $L \ll N$. (2) **Hetero-MMoE (Heterogeneous MMoE)**: instead of homogeneous MLP experts, the mixture blends **MLP, DCN (Deep & Cross Network), and CIN (Compressed Interaction Network, from xDeepFM)** experts, so different experts specialize in implicit vs. explicit, low- vs. high-order feature interactions. Per-task gates and task towers still produce pCTR and pCTO (click-to-order).

**Key results.** Online gains over the previous architecture — pCTR: **+0.93% AUC, +0.70% LogLoss**; pCTO: **+0.66% AUC, +2.0% LogLoss** — translating to revenue, engagement, and advertiser-conversion improvements.

**Trade-offs / limitations.** Heterogeneous experts and a target-aware transformer are heavier than a plain shared-bottom or MLP-expert MMoE, so this is a **fine-rank**-scale model, not pre-rank. Multi-hash embeddings trade a little accuracy for memory scalability; MLA's latent bottleneck can lose detail on very long sequences.

**Connection to this chapter.** This is the most direct extension of **§3 (MMoE)**. The note's MMoE uses homogeneous experts and warns about **polarization** (§3.2); Hetero-MMoE keeps the per-task Softmax gates but diversifies the experts to raise representational capacity, and the pCTR/pCTO heads are exactly the §1.2 per-task Sigmoid heads. The target-aware sequence encoder is a richer alternative to the static **user statistical features** of §6.3.

#### One Model, Two Markets: Bid-Aware Generative Recommendation (GEM-Rec)

> Jiang, Y., Feng, Z., Mah, C. P., Mehta, A., & Wang, D. (2026). *One model, two markets: Bid-aware generative recommendation*. arXiv. https://arxiv.org/abs/2603.22231

**Method.** GEM-Rec (Google Research) unifies **organic** and **sponsored (ad)** ranking inside a single generative recommender built on TIGER-style **semantic IDs**. It augments the SID vocabulary with **control tokens** (`<ORG>` / `<AD>`) that *factorize the slot decision* (whether to show an ad) from the *content decision* (which item) — the model learns valid ad-placement patterns directly from interaction logs. At inference, a **Bid-Aware Decoding** mechanism injects real-time auction bids into the decoding search: for an `<AD>` branch, each candidate token's score is shifted by the max bid under its prefix (a precomputed lookup $B(c_k \mid c_{<k})$), with a parameter $\lambda$ controlling aggressiveness. They prove this guarantees **Allocative Monotonicity** (higher bids weakly increase exposure) and **Organic Integrity** (organic ranking stays purely relevance-based), *without retraining*.

**Key results.** Simulation/offline experiments show a tunable **revenue–relevance frontier**: $\lambda = 0$ collapses to the baseline TIGER ad density ($\sim 3.5$%), and increasing $\lambda$ trades NDCG@10 for higher total (first-price) revenue and ad rate, all controllable at serving time.

**Trade-offs / limitations.** Results are from **simulated auction environments**, not a deployed system, so production CTR/revenue lift is unproven. Bid-aware decoding requires a per-prefix max-bid lookup table and assumes a (first-price) auction abstraction; correctness of the monotonicity guarantee depends on that abstraction.

**Connection to this chapter.** GEM-Rec is a **generative replacement for §4 (score fusion)** in the ads setting: rather than predicting separate $p_{\text{click}}, p_{\text{like}}, \dots$ and fusing them with hand-tuned weights, it generates the item directly and folds the monetization objective (bids) into decoding — the bid plays the role that revenue/price terms play in the §4.5 e-commerce fusion formula, but applied to a generative model.

#### S2GR: Stepwise Semantic-Guided Reasoning in Latent Space

> Guo, Z., Wang, J., Zhou, R., Liu, Y., Guo, J., Zhao, J., Xu, X., Liu, Y., & Zhan, K. (2026). *S2GR: Stepwise semantic-guided reasoning in latent space for generative recommendation*. arXiv. https://arxiv.org/abs/2601.18664

**Method.** S2GR (Kuaishou) adds **LLM-style reasoning** to generative recommendation over hierarchical SIDs. Prior reasoning-enhanced GR puts all reasoning *before* generation (one unidirectional thinking block), which starves the later (fine-grained) SID codes of compute. S2GR instead inserts a **thinking token before each SID code generation step** — each thinking token explicitly represents the **coarse-grained semantic category** for that step and is supervised by **contrastive learning against the ground-truth codebook cluster distribution**, so the latent reasoning path is interpretable and physically grounded (not free-floating latent vectors). It also optimizes the codebook with item-co-occurrence, load-balancing, and uniformity objectives to maximize codebook utilization and enforce a coarse-to-fine hierarchy.

**Key results.** Average improvements of **+9.47%** (public) and **+33.75%** (industrial) over baselines across HR@K / NDCG@K; codebook (CoBa RQ-VAE) utilization reaches **99.30%** vs. ~95.86% for plain RQ-VAE. Online A/B test on Kuaishou's short-video platform (5.25% of users, 7 days) showed statistically significant gains (e.g. **+0.09%** on core metrics, CI excluding zero).

**Trade-offs / limitations.** Inserting thinking tokens before every SID step increases decoding length and inference cost — a concern under the strict low-latency budget of online serving; the contrastive supervision needs a well-clustered codebook to be meaningful.

**Connection to this chapter.** Like GEM-Rec, S2GR is part of the **generative-ranking** shift away from the discriminative shared-bottom + per-task-head model of §1. Its **completion/watch-time** orientation on a short-video platform ties to **§5 (video play modeling)**: the goal is the same (rank by genuine interest captured via play behavior), but the mechanism is autoregressive SID generation with explicit semantic reasoning rather than the $\exp(z)=t$ play-time head.

### (c) Scaling & serving systems

#### Generalized Dot-Product Attention (GDPA)

> Xu, J., Chen, C., Yang, S., Hoehnerbach, M., Liu, X., Zadouri, T., Shankar, D., Zhou, J., Yu, H., Ren, M., Xu, H., Yang, C., Nie, J., Zhang, H., Li, H., Shu, M., Sultan, M., Leung, M., Bocharov, J., & Dao, T. (2026). *Generalized Dot-Product Attention: Tackling real-world challenges in GPU training kernels*. PyTorch / Meta Engineering Blog.

**Method.** GDPA generalizes standard dot-product (softmax) attention by **replacing softmax with arbitrary element-wise activations** (GELU, SiLU, …), the pattern used inside Meta's RecSys foundation models — **GEM** (Meta's largest RecSys training model), **HSTU**, **InterFormer**, and **Kunlun**. The contribution is a **GPU training kernel** built on **FlashAttention-4 (FA4)** and specialized for RecSys traffic, which (unlike LLM workloads) is dominated by **short, asymmetric, jagged sequences and very large batches**. The kernel unifies self-attention, PMA, and PFFN modules — all "two matmuls with an optional activation in between" — into one implementation, with workload-driven optimizations for pipeline occupancy and compute–memory overlap.

**Key results.** On NVIDIA B200 GPUs, up to **2× forward-pass speedup** (1,145 BF16 TFLOPs, ~97% tensor-core utilization) and up to **1.6× backward** vs. the original Triton kernel; up to **3.5× forward / 1.6× backward** vs. FA4 under some production traffic; **>30% end-to-end training throughput** when applied across the full model.

**Trade-offs / limitations.** Hardware- and shape-specific (tuned for B200 + Meta's irregular RecSys shapes); benefits depend on the non-softmax attention pattern, so it is most relevant to foundation-model-scale sequence rankers, not classic shared-bottom MLPs.

**Connection to this chapter.** This is the **serving/training-infrastructure** layer beneath modern sequence-based rankers. The §1 shared-bottom and §3 MMoE are MLP-era; as fine-rank moves to transformer sequence encoders (cf. the Uber target-aware transformer above), the attention kernel becomes the bottleneck, and GDPA is what makes such heavy rankers trainable at scale.

#### Replicate-and-Quantize (R&Q) for MoE load balancing

> Liu, Z., Peng, J., Duan, J., Liu, Z., Zhou, K., Liang, M., Simon, L., Liu, X., Xu, Z., & Chen, T. (2026). *A replicate-and-quantize strategy for plug-and-play load balancing of sparse mixture-of-experts LLMs*. arXiv. https://arxiv.org/abs/2602.19938

**Method.** A **training-free, inference-time** fix for **load imbalance** in Sparse Mixture-of-Experts (a minority of "heavy-hitter" experts get most tokens). R&Q **replicates** heavy-hitter experts for parallel capacity and **quantizes** the less-important experts (and the replicas) to stay within the original memory budget; imbalance is measured by a per-layer **Load-Imbalance Score (LIS)** (= 1 when perfectly balanced). It needs no router changes or retraining — only a small calibration set.

**Key results.** Up to **1.4× reduction in load imbalance** with accuracy within **±0.6%**. (General-LLM paper; Meta AI + UNC.)

**Connection to this chapter.** Although evaluated on LLMs, the problem is exactly the **polarization / dead-expert** failure mode of MMoE in §3.2 — there, the fix is dropout on gate outputs at *training* time; R&Q is the *inference-time* analogue for MoE-based rankers, balancing expert workload at serving so a few experts don't bottleneck latency.

## Pinterest in Practice (2024–2026)

Three recent Pinterest production papers map cleanly onto this chapter's core ideas — search **pre-ranking** with cheap cross features, **utility-weight tuning** at the score-fusion layer, and **calibration-aware multi-task CVR** modeling under privacy constraints.

### InteractRank — Search Pre-Ranking with Cross-Interaction Features

> Khandagale, S., Juneja, B., Agarwal, P., Subramanian, A., Yang, J., & Wang, Y. (2025). *InteractRank: Personalized web-scale search pre-ranking with cross interaction features*. arXiv. https://arxiv.org/abs/2504.06609

InteractRank is Pinterest Search's production **pre-ranking** model, scoring $O(10^4{-}10^5)$ candidates per query in milliseconds — the same speed-vs-accuracy tension as the three-tower model of §7. Its key move solves the two-tower limitation noted in §7.2: instead of relying only on a late dot product (which can't capture query–item interaction), it injects explicit **cross-interaction features** — engagement-based priors $IQP_p(q)=P_h(p\mid q)=\frac{C(p,q)}{C(q)}$ over multiple time windows — into a small affine projection layer *after* the towers, so item embeddings stay precomputable. To fix the impression-only training bias (missing easy negatives, cf. §2), it adds a sampled-softmax contrastive loss with logQ correction. Online it lifted Search Intent Fulfillment Rate **+6.5% vs BM25** (+3.7% vs two-tower) at **~21× fewer serving FLOPs** than an early-interaction model — a canonical "cheap explicit cross features beat expensive implicit cross architectures in pre-ranking" result.

### DRL-PUT — Reinforcement Learning for Ranking-Utility Tuning (score fusion)

> Yang, X., Ben Ayed, M., Zhao, L., Zhou, F., Shen, Y., Engle, A., Zhuang, J., Leng, L., Xu, J., Rosenberg, C., & Deshikachar, P. (2025). *Deep reinforcement learning for ranking utility tuning in the ad recommender system at Pinterest*. arXiv. https://arxiv.org/abs/2509.05292

This paper automates the **score-fusion (§4)** weights in Pinterest's ads ranker. The utility $U = p(\text{action})\cdot bid + \sum_i p(\text{engagement}_i)\cdot w_i$ is exactly the weighted-sum fusion of §4.1, but its weights $\lbrace w_i\rbrace$ were historically tuned by hand — unprincipled, combinatorially explosive, and *static* across users and seasons. DRL-PUT frames per-request weight selection as a one-step RL problem: a policy-based REINFORCE agent picks the discretized, semantically-grouped weight vector ($g^n = 1000$ actions) that maximizes a reward balancing revenue and user value, leaving all upstream prediction models frozen. Online A/B on the treated segment gave **CTR +9.71%** and CTR30 **+7.73%**; reward ablations expose the revenue-vs-engagement trade-off explicitly, and bucketing confirms it personalizes weights (higher $w_{click}$ for high-CTR users). The takeaway for §4: the fusion weights are themselves a learnable, personalizable decision rather than fixed A/B-tuned constants.

### Clean-Room CVR — Privacy-Preserving Multi-Task Conversion Modeling (calibration, CVR)

> Li, K., Chen, X., Leng, L., Xu, J., Sun, J., & Rezaei, B. (2024). *Privacy preserving conversion modeling in data clean room*. In *Proceedings of the 18th ACM Conference on Recommender Systems (RecSys '24)* (pp. 819–822). ACM. https://doi.org/10.1145/3640457.3688054

This addresses training a **multi-task CVR** model (the §1-style shared-bottom-plus-towers family — main click-through CVR plus 5 auxiliary tasks including CTR) when the advertiser holds the conversion labels and won't share them. Training runs as vertical split learning through a data clean room, exchanging only **batch-level aggregated gradients** $\nabla_f L = \sum_i (p_i - y_i)\partial z_i/\partial f$ (rather than per-sample gradients that leak labels), with gated LoRA adapters to cut cross-party communication. The chapter-relevant subtlety is **calibration (§2.4)**: label differential privacy (random label flips) miscalibrates CVR badly (calibration ratio up to 4.0 at $\epsilon=3$), which matters because ad auctions need calibrated probabilities, not just ranking AUC. Folding the flip probability $q_\epsilon = \frac{e^\epsilon}{e^\epsilon+1}$ into the loss as a de-biasing correction restores calibration to 1.0 with only a 0.2–0.5% AUC drop, projecting **>10% lower cost-per-action** for advertisers who previously had no conversion data.
