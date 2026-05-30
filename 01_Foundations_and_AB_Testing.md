# Part 01 — Recommendation System Basics / Overview (概要)

*Recommendation systems course notes, based on the Xiaohongshu (小红书) recommendation system. Cross-referenced from the lecture slides, the written notes (讲义), and the audio transcripts. Slides + notes are treated as ground truth; the Chinese audio is supplementary.*

---

## Table of Contents

1. [What a Recommendation System Does](#1-what-a-recommendation-system-does)
2. [The Conversion Funnel & Consumption Metrics](#2-the-conversion-funnel--consumption-metrics)
   - [Consumption Metrics (消费指标)](#consumption-metrics-消费指标)
   - [Effective Click (有效点击) — from the written notes](#effective-click-有效点击--from-the-written-notes)
3. [Core / North-Star Metrics](#3-core--north-star-metrics)
   - [3.1 User Scale (用户规模)](#31-user-scale-用户规模)
   - [3.2 User Retention (用户留存) — from notes](#32-user-retention-用户留存--from-notes)
   - [3.3 Consumption (消费)](#33-consumption-消费)
   - [3.4 Publishing (发布)](#34-publishing-发布)
   - [3.5 Core vs. Non-core](#35-core-vs-non-core)
4. [The Recommendation Pipeline / Link](#4-the-recommendation-pipeline--link)
   - [4.1 Retrieval](#41-retrieval)
   - [4.2 Pre-Rank & Rank](#42-pre-rank--rank)
   - [4.3 Re-Rank](#43-re-rank)
   - [4.4 The Funnel Intuition](#44-the-funnel-intuition)
5. [The Experiment Workflow](#5-the-experiment-workflow)
6. [A/B Testing](#6-ab-testing)
   - [6.1 Treatment vs. Control & Random Bucketing](#61-treatment-vs-control--random-bucketing)
   - [6.2 Layered Experiments](#62-layered-experiments)
   - [6.3 Mutual Exclusion vs. Orthogonality](#63-mutual-exclusion-vs-orthogonality)
   - [6.4 Holdout Mechanism](#64-holdout-mechanism)
   - [6.5 Full Rollout & Reverse Experiments](#65-full-rollout--reverse-experiments)
7. [Key Insights](#7-key-insights)
8. [Expanded Reading — Recent Advances (2026)](#expanded-reading--recent-advances-2026)
   - [Ding et al. (2026) — How Well Does Generative Recommendation Generalize?](#ding-et-al-2026--how-well-does-generative-recommendation-generalize)

---

## 1. What a Recommendation System Does

The job of an information-feed recommendation system is to **select a few dozen items (物品) out of a database of hundreds of millions** and display them to the user. On Xiaohongshu the "item" is a *note* (笔记), and there are on the order of **a few hundred million notes** in the corpus.

A recommendation algorithm engineer's daily job is to **improve models and strategies (策略)** so as to lift the system's **business metrics (业务指标)**. Every model/strategy change must pass an online **A/B test** before it can be rolled out — experiment data, not intuition, decides whether a change is good.

> All the numbers in this course ("a few thousand", "a few hundred", "80 items") are illustrative; Wang explicitly says he cannot disclose Xiaohongshu's real production numbers.

---

## 2. The Conversion Funnel & Consumption Metrics

When a note is shown to a user, the user can flow through a **conversion funnel (转化流程)**:

```
Impression (曝光) → Click (点击) → { ScrollToEnd (滑动到底) → Comment (评论) }
                                    { Like (点赞) }
                                    { Collect (收藏) }
                                    { Share (转发) }
```

- **Impression**: the note is rendered to the user.
- **Click**: the user taps in to open the note's detail page.
- After clicking, four parallel actions can happen: **Like**, **Collect**, **Share**, and **ScrollToEnd**; reaching the end of a note can then lead to a **Comment**.

### Consumption Metrics (消费指标)

These rates measure how users engage with each note. Note carefully which event is in the **denominator** — only CTR is normalized by impressions; all post-click rates are normalized by clicks.

$$
\text{CTR (点击率)} = \frac{\text{clicks (点击次数)}}{\text{impressions (曝光次数)}}
$$

$$
\text{Like rate (点赞率)} = \frac{\text{likes}}{\text{clicks}}
\qquad
\text{Collect rate (收藏率)} = \frac{\text{collects}}{\text{clicks}}
$$

$$
\text{Share rate (转发率)} = \frac{\text{shares}}{\text{clicks}}
$$

$$
\text{Read-completion rate (阅读完成率)} = \frac{\text{scroll-to-end (滑动到底次数)}}{\text{clicks}} \times f(\text{note length, 笔记长度})
$$

> **Why the $f(\text{note length})$ factor?** Reaching the bottom of a short note is easy; reaching the bottom of a long note signals much stronger engagement. The factor $f(\cdot)$ corrects for length so that completion rate is comparable across notes of different sizes.

These rates are the prediction targets the ranking neural network learns to estimate (see §4.2).

### Effective Click (有效点击) — from the written notes

Users sometimes mis-tap ("手滑"). The notes define an **effective click**: a click only counts as effective if, after opening the detail page, the user either **dwells longer than $\tau$ seconds** OR performs any interaction (like/collect/follow/etc.). Therefore:

$$
\text{effective CTR} \le \text{CTR}.
$$

---

## 3. Core / North-Star Metrics

The slides group the north-star metrics into three families. The written notes go deeper, so both are merged here.

### 3.1 User Scale (用户规模)
- **DAU (日活用户数, daily active users)** — primary measure of stickiness/activeness. A user who opens the app 1 time or 10 times today both contribute **1** to DAU.
- **MAU (月活用户数, monthly active users)** — a user who uses the app at least once in the next 28 days contributes 1. (**WAU**, weekly active users, is analogous.)
- **FDAU / SDAU** (from notes): for an app with both feed and search, use **feed DAU (FDAU)** to judge the recommendation product and **search DAU (SDAU)** to judge search, rather than the whole-app DAU. Since many users use both, $\text{SDAU} < \text{DAU}$ and $\text{FDAU} < \text{DAU}$, but $\text{SDAU} + \text{FDAU} > \text{DAU}$.
- **Recommendation penetration (推荐渗透率)** $= \text{FDAU}/\text{DAU}$ — fraction of app users who used the recommendation product today; higher is better.

### 3.2 User Retention (用户留存) — from notes
Retention rate (留存率) measures the app's ability to keep users. Two definitions:
- **次 n 留 (cumulative "next-n" retention)**: of today's ($t{+}0$) users, the fraction who use the app **at least once within the next $n$ days** ($t{+}1 \dots t{+}n$).
- **第 n 留 ("day-n" retention)**: of today's users, the fraction who use the app **on exactly day $t{+}n$**.

Most used in industry: **次1留 and 次7留**. The 次-$n$ metric lags by $n$ days (you can only compute it $n$ days after launch). 次28留 is a long-term-only metric.

> **Worked examples from the notes:**
> - If 100M users use the app on Feb 1 and 60M use it on Feb 2, then 次1留 = 第1留 = **60%**. (When $n=1$ the two definitions coincide.)
> - If 80M of the 100M use it at least once during Feb 2–8, then 次1留 = **80%**; if 30M use it on Feb 8 specifically, then 第7留 = **30%**.
>
> Always: $\text{次}n\text{留} \ge \text{第}n\text{留}$, 次-$n$ is monotonically increasing in $n$, while 第-$n$ usually decreases in $n$.

**User Lifetime (用户生命周期, LT)** — expected number of active days in a window. With day-$n$ retention $r_n$ (and $r_0 = 1$ for today):

$$
\text{LT}_n = 1 + \sum_{i=1}^{n} r_i, \qquad 1 \le \text{LT}_n \le n+1.
$$

Common: **LT7** and **LT28**.

> **Retention caveat (important for interviews):** retention can rise *for the wrong reason*. If a strategy drives away low-activity users, the remaining users are more active, so retention goes *up* (false prosperity) while DAU drops. Conversely, reviving a churned user into a low-activity user is good but can *lower* retention. If retention rises while DAU falls, investigate whether the strategy chased away inactive users. Retention is the **highest-level goal**; a strategy that lifts retention but hurts consumption is still worth rolling out.

### 3.3 Consumption (消费)
- **Per-capita time spent on recommendation (人均使用推荐时长)**.
- **Per-capita notes read (人均阅读量)** — for double-column feeds, only a *click into* the detail page counts as "read" (an impression does not); for the video inner-stream, each swipe to a new video counts.

> **Consumption metrics trade off against each other.** Across scenes (double-column, video inner-stream, sub-channels) reading volume is a zero-sum see-saw — boosting video-stream reads cuts double-column reads. Sum/weighted-sum across all scenes to get the system-level metric. Likewise per-capita read count and per-capita time often move in opposite directions; a data-analysis expert must define a fair **exchange rate (兑换关系)** (e.g. "+1 read = −1.2 minutes"), and a new strategy is only rolled out if it beats that exchange rate.

### 3.4 Publishing (发布)
- **Publishing penetration (发布渗透率)** and **per-capita publish volume (人均发布量)**.
- Relevant only to **UGC** platforms (Xiaohongshu, Douyin). **PGC** platforms (e.g. Tencent Video) buy content and ignore publishing metrics. (This metric is handled by item cold-start, covered in a later chapter.)

### 3.5 Core vs. Non-core
Core (north-star) metrics = user scale, retention, consumption (+ publishing for UGC). **Non-core** metrics — CTR, interaction rate (交互率) — are **observation-only** and do **not** by themselves decide rollout. But if a strategy *severely* harms a non-core metric, you must examine the exchange relationship before rolling out.

---

## 4. The Recommendation Pipeline / Link

The pipeline has **four stages**, each acting as a funnel that shrinks the candidate set:

```
hundreds of millions
   │
   ▼
[ 召回 Retrieval ] ── thousands ──►
[ 粗排 Pre-Rank  ] ── hundreds ──►
[ 精排 Rank      ] ── hundreds ──►   (no truncation at Xiaohongshu)
[ 重排 Re-Rank   ] ── tens, e.g. 80 items ──► shown to user
```

| Stage | Chinese | Input size | Output size | Model / Method |
|---|---|---|---|---|
| Retrieval | 召回 | ~几亿 | ~几千 | Many channels (CF, two-tower, followed authors…) |
| Pre-Rank | 粗排 | ~几千 | ~几百 | Small neural network |
| Rank | 精排 | ~几百 | ~几百 | Large deep neural network |
| Re-Rank | 重排 | ~几百 | ~几十 | Diversity sampling (MMR/DPP) + rules + ads |

### 4.1 Retrieval

Goal: **quickly fetch a candidate set** from the huge item database.

- The system runs **many retrieval channels (召回通道) in parallel** — on Xiaohongshu, on the order of **dozens** of channels.
- Common channels: **collaborative filtering (协同过滤)**, **two-tower model (双塔模型)**, **followed authors (关注的作者)**, etc.
- Each channel returns **tens to a few hundred** notes; together they return **a few thousand** candidates (slides illustrate 10 channels each returning "a few hundred").
- After retrieval, results are **merged, deduplicated, and filtered (融合、去重、过滤)** — filtering removes authors/notes/topics the user dislikes.

> **Transcription note:** the audio renders 协同过滤 (collaborative filtering) as "系统过滤" / "照回" due to AI mis-transcription. The slides clearly say **协同过滤** and **召回** — trust the slides.

### 4.2 Pre-Rank & Rank

Why split ranking into two stages? Scoring **a few thousand** candidates one-by-one with a *large* deep network would be far too expensive. So ranking is broken into:

- **Pre-Rank:** a **small-scale neural network** scores the ~few thousand candidates quickly, keeps the **top ~few hundred**.
- **Rank:** a **large-scale deep neural network** scores the ~few hundred. It is *much larger* and uses *more features*, so its scores are more reliable but more expensive.

This **balances compute cost vs. accuracy** — cheap model filters, expensive model refines.

> At Xiaohongshu, **精排 does not truncate (不做截断)**: all the hundreds of notes carry their 精排 score into 重排.

**Model structure (粗排 and 精排 are nearly identical; only size/feature-count differ).** Input = concatenation of three feature groups:
- **User features (用户特征)**
- **Item / candidate features (物品特征)**
- **Statistical features (统计特征)**

These feed a **neural network (神经网络)** that outputs **multiple predicted values (预估值)** — e.g. estimated **CTR, like rate, collect rate, share rate**. The larger each estimate, the more interested the user is.

Finally, the estimates are **fused (融合)** into a single **ranking score (排序分数)**, e.g. a weighted sum:

$$
\text{score} = w_1 \cdot \widehat{\text{CTR}} + w_2 \cdot \widehat{\text{like}} + w_3 \cdot \widehat{\text{collect}} + w_4 \cdot \widehat{\text{share}} + \dots
$$

This single score decides **whether and where** the note is shown. (粗排 scores ~thousands of notes; 精排 scores ~hundreds; each note gets one fused score.)

### 4.3 Re-Rank

The last stage. Even with good 精排 scores, raw score-sorted output has problems, so 重排 adjusts it:

1. **Diversity sampling (多样性抽样)** — select ~tens from ~hundreds using methods like **MMR** and **DPP**. Sampling balances two criteria: the **精排 score** and **diversity (多样性)**.
2. **Rule-based spreading (规则打散)** — push apart similar notes so they aren't adjacent. *Example:* even a basketball fan doesn't want the top 5 slots all NBA; if slot 1 is NBA, the next few slots cannot be NBA, and similar notes get pushed down.
3. **Insert ads & operational content (插入广告、运营推广内容)** and adjust ordering per **ecosystem requirements (生态要求)** (e.g. don't run many "pretty-girl" photos in a row).

The output (e.g. the **top ~80 items**, mixing notes + ads) is what the user finally sees. 重排 rules are **very complex — thousands of lines of code**.

### 4.4 The Funnel Intuition

- **Retrieval and Pre-Rank are the biggest funnels**, taking candidates from hundreds of millions → thousands → hundreds.
- Only once the set is down to **a few hundred** is it feasible to run the **large 精排 network** and **DPP-style diversity sampling**. If the candidate set were still huge, neither would be computationally possible.

---

## 5. The Experiment Workflow

A model/strategy change goes through three stages (slides 01_Basics_01, p.9):

```
离线实验 (Offline) → 小流量 A/B 测试 (Small-traffic A/B) → 全流量上线 (Full rollout)
```

1. **Offline experiment (离线实验):** collect historical data, train and test on it. The algorithm is **not deployed**, has **no interaction with real users**.
2. **Small-traffic A/B test (小流量 A/B 测试):** deploy to a *small fraction* of users (e.g. a random 10%), observe real behavior (DAU, retention, clicks, interactions). Small traffic limits risk, since a new channel could badly hurt experience.
3. **Full rollout (全流量上线):** deploy to all (well, 90%+ — see holdout) users.

A/B testing also serves **hyperparameter selection** — e.g. run three groups for GNN depth $\in \lbrace1, 2, 3\rbrace$ simultaneously and pick the best.

---

## 6. A/B Testing

Running example throughout: the retrieval team built a **GNN (graph neural network, 图神经网络) retrieval channel**, offline results are positive, and we now A/B test it online — both to measure its business-metric lift and to tune GNN depth $\in \lbrace1,2,3\rbrace$.

### 6.1 Treatment vs. Control & Random Bucketing

**Random bucketing (随机分桶):**
- Split users into $b$ buckets (e.g. $b = 10$), each holding $10$% of users.
- Method: hash the user ID into an integer in some range, then **uniformly randomly** partition those integers into $b$ buckets.
- With enough users, every bucket's DAU, retention, CTR, etc. are (approximately) equal.

With $n$ total users split into $b$ buckets, each bucket has $\tfrac{n}{b}$ users.

**Treatment / control assignment:** take 1 bucket as **control (对照组)**, one or more as **treatment (实验组)**. In the example: buckets 1/2/3 = treatment groups using 1/2/3-layer GNN, bucket 4 = control (no GNN). Compute each treatment's **diff** vs. control across business metrics; if a group is **significantly positive (显著正向)**, expand its traffic.

> **Bucketing must be uniform enough.** Splitting into equal-count buckets does **not** guarantee every metric (DAU, mid/low-activity counts, retention, consumption, impressions, clicks, interactions) is equal to within ~1/10000. If two buckets differ by a few parts-per-10000 on a core metric, the measured diff is meaningless. Balance buckets across activity levels and demographics.

### 6.2 Layered Experiments

**The traffic-shortage problem:** an info-feed company has many departments/teams (recommendation = retrieval/pre-rank/rank/re-rank teams; plus UI, ads) all needing A/B tests — **dozens to hundreds simultaneously**. With a single split of 10 buckets (1 control + 9 treatment), you can run **at most 9** experiments at once. Not enough.

**Solution — layered experiments:** divide experiments into **layers**, e.g. *retrieval, pre-rank, rank, re-rank, UI, ads*. (The GNN retrieval channel belongs to the **retrieval layer**.)

- **Same-layer mutual exclusion (同层互斥):** experiments in the same layer use disjoint buckets. If the GNN experiment takes 4 of the 10 retrieval-layer buckets, other retrieval experiments use only the remaining 6.
- **Cross-layer orthogonality (不同层正交):** each layer **independently and randomly** re-buckets users, so each layer can use **100% of users**.

**Reference:** Tang et al., *Overlapping experiment infrastructure: more, better, faster experimentation*, KDD 2010.

**The math (slides 01_Basics_03 p.8; notes p.8–9):** Let retrieval-layer buckets be $\mathcal{U}_1, \dots, \mathcal{U}_{10}$ and rank-layer buckets be $\mathcal{V}_1, \dots, \mathcal{V}_{10}$, with $n$ total users:

$$
|\mathcal{U}_i| = |\mathcal{V}_j| = \frac{n}{10}.
$$

- **Same layer (exclusion):** $\mathcal{U}_i \cap \mathcal{U}_j = \varnothing$ for $i \ne j$ — two retrieval experiments never hit the same user.
- **Cross layer (orthogonal):** $|\mathcal{U}_i \cap \mathcal{V}_j| = \dfrac{n}{100}$ — a retrieval bucket and a rank bucket share $\tfrac{n}{100}$ users.

So a user can be in **one retrieval experiment AND one rank experiment** simultaneously, but never **two retrieval experiments**. Intuitively, retrieval-layer bucket $\mathcal{U}_3$'s 10% of users get **uniformly spread (均匀打散)** across all 10 rank-layer buckets — so if $\mathcal{U}_3$'s GNN lifts metrics, it lifts all 10 rank buckets *equally*, keeping rank-bucket comparisons fair.

### 6.3 Mutual Exclusion vs. Orthogonality

If everything were orthogonal you could run *infinitely many* experiments. So why ever use exclusion?

1. **Some same-type strategies are naturally exclusive (天然互斥):** e.g. two different rank-model *structures* — a single user can only use one. Orthogonality is impossible.
2. **Same-type strategies interfere:** two added retrieval channels can **reinforce ($1+1>2$)** or **cancel ($1+1<2$)** each other. Exclusion avoids this interference.
3. **Different-type strategies usually don't interfere ($1+1=2$):** e.g. "add a retrieval channel" vs. "optimize the pre-rank model" — make them orthogonal layers, each using 100% of users.

> **Why orthogonality corrupts interfering experiments — worked Example 2.3 (notes p.9–10):**
> Add retrieval channels A and B, each *alone* gives +1 min per-capita time. But A and B are very similar, so together they **cancel**: $1+1 = 1$ (only +1 min total, not +2).
> - **If run mutually exclusive** (disjoint users): each measures diff = **+1 min**. Correct.
> - **If run orthogonal** (each 50% treatment / 50% control): each experiment's control group is "**not clean**" — it is contaminated by the *other* experiment's effect. The control sits $1 \times 0.25/0.5 = 0.5$ min high, so the measured diff = $1 - 0.5 = $ **+0.5 min** for each of A and B. **Wrong** (real effect is +1 each).
>
> Correct approach: keep them **mutually exclusive**, roll out the higher-gain one (say A), then test B **on top of** the rolled-out A.

> This is essentially the **interference / SUTVA-violation** problem from causal inference: the stable-unit-treatment-value assumption (a unit's outcome depends only on its own treatment) breaks when correlated experiments overlap on the same users. Same-layer exclusion enforces clean, non-interfering comparisons; cross-layer orthogonality is only safe when treatments are genuinely independent.

### 6.4 Holdout Mechanism

**The problem:** each experiment reports its own diff as the engineer's individual credit. But a company wants the **total** contribution of a whole *department* (e.g. recommendation) over a review period (e.g. 2 months). You **cannot just sum** all the individual A/B diffs — stacked experiments usually **discount (折损)** each other, so the naive sum diverges badly from reality.

**The holdout solution:**
- Treat the whole recommendation system as one big layer, **orthogonal** to the UI layer and ads layer.
- Inside it, split users **10% vs. 90%, mutually exclusive**:
  - **10% = holdout bucket (holdout 桶)** — a **clean** bucket that **no new experiment** is allowed to touch.
  - **90% = experiment buckets** — sub-divided into the orthogonal retrieval/pre-rank/rank/re-rank layers, each able to use the full 90%.
- All effective experiments in those 4 layers create a **diff vs. the holdout**, and those diffs **stack** (with discount).
- At review-period end (e.g. bi-monthly), compute **diff between the 90% experiment traffic and the 10% holdout** (normalized) = the department's total business-metric contribution.

**Resetting each cycle:**
- At cycle end, **clear the holdout** — rolled-out experiments expand from 90% → 100% of users.
- **Re-split users randomly** into a fresh 10% holdout vs. 90% experiment for the next cycle.
- Because the split is random, the new holdout-vs-experiment diff is **≈ 0** at the start, giving a fair baseline.
- As retrieval/pre-rank/rank/re-rank experiments launch and roll out, the diff **gradually grows** again.

### 6.5 Full Rollout & Reverse Experiments

**Full rollout (推全):** all experiments start small; if the diff is **significantly positive**, roll out.
- *Example (from audio):* a re-rank experiment using a treatment + control bucket touches 20% of users. If significantly positive, roll it out and free its buckets for others.
- **Rollout creates a NEW layer (新层)** covering 90% of users, orthogonal to all other layers.
- **Diff scaling intuition:** a strategy giving a true +9 (units, e.g. ten-thousandths "万分点") on CTR, while at 10% small traffic, only lifts the holdout-diff by ~1 (since it touches 1/10 of users). After rollout to 90% it lifts the diff ~9× — matching the A/B-measured effect.
  - The notes phrase it as: during the 10% phase the channel contributes $+\tfrac{1}{9}$ min vs. holdout; after rollout to 90% it contributes the full +1 min.

**Metric lag (指标滞后) — why reverse experiments are needed:**
- **Fast metrics:** CTR, like rate, **completion rate (完播率)** react *immediately*; results at day 1 ≈ day 10.
- **Medium-lag metrics:** time spent, items-impressed — stabilize after a few days.
- **Heavily-lagged metrics:** **retention (留存)** — may show *no* significant short-term change but keep improving for *months*. Reason: better experience takes time for users to feel, and only then does stickiness rise.

**The tension:** observing longer → more accurate metrics; but engineers want to **roll out fast** (within 1–2 weeks of seeing significant gains) to free buckets and to build follow-on work on top of the new strategy.

**Reverse experiment (反转实验):** resolves the tension — roll out fast *and* keep observing long-term.
- In the **new rollout layer**, open a small **reverse bucket (反转桶)** that keeps the **old strategy (旧策略)**.
- The rest of the layer runs the **new strategy (新策略)**.
- Keep the reverse bucket for a long time to **track new-vs-old diff** on lagged metrics.
- Clearing the holdout at cycle-end does **not** affect the reverse bucket; it applies the new strategy to the holdout's users only.
- When the reverse experiment finishes, **close it** — the new strategy applies to the reverse-bucket users too; now the rollout is truly 100%.

> **Source disagreement on bucket sizes:** the **audio** describes the rollout-layer split loosely (e.g. "85% new / a small reverse bucket"; one segment mentions an example 20% experiment in re-rank). The **written notes (figure 2.4)** give concrete numbers: in the new rollout layer, **5% reverse bucket (old strategy)** and **85% new strategy** (within the 90% experiment side). The **slides** just label "推全新策略 / 旧策略" without percentages. Use the notes' **5% / 85%** as the canonical figures.

---

## 7. Key Insights

- **Pipeline order:** 召回 → 粗排 → 精排 → 重排. Sizes: hundreds of millions → thousands → hundreds → hundreds → tens.
- **召回:** many parallel channels (CF, two-tower, followed authors), then merge + dedup + filter.
- **粗排 vs 精排:** same architecture (user + item + statistical features → NN → predicted CTR/like/collect/share → fused score); 精排 is bigger, more features, more reliable, more expensive. 粗排 is the cheap filter. Two stages balance **compute vs. accuracy**.
- **重排:** diversity sampling (**MMR / DPP**) + rule-based spreading + ad/operational insertion; thousands of lines of rules.
- **Biggest funnels:** 召回 and 粗排.
- **Metric denominators:** CTR / impressions; all post-click rates / clicks.
- **North-star metrics:** user scale (DAU/MAU, FDAU/SDAU, penetration), retention (次n留 vs 第n留, LT), consumption (per-capita time & reads). Publishing for UGC only. CTR/interaction are non-core (observation).
- **Retention is highest goal** but laggy and gameable (can rise by chasing away inactive users).
- **A/B layering:** same-layer **互斥** (disjoint buckets, avoid interference / SUTVA violation), cross-layer **正交** (independent re-bucketing, 100% per layer). $|\mathcal{U}_i \cap \mathcal{U}_j| = 0$; $|\mathcal{U}_i \cap \mathcal{V}_j| = n/100$.
- **Use exclusion when** strategies are naturally exclusive or same-type (reinforce/cancel); **use orthogonality when** strategies are different-type and non-interfering.
- **Holdout:** clean 10% never touched by experiments → measures whole-department contribution = diff(90% experiment, 10% holdout); reset & re-randomize each review cycle.
- **Reverse experiment:** small reverse bucket (old strategy, ~5%) inside the rollout layer to observe lagged metrics (esp. retention) long-term while still rolling out fast.

---

## Expanded Reading — Recent Advances (2026)

This section connects the conceptual foundations above to a recent theory paper that reframes *why* the newer **generative recommendation (生成式推荐, GR)** paradigm — increasingly used inside the 召回 / 精排 stages — beats the classic item-ID models.

### Ding et al. (2026) — How Well Does Generative Recommendation Generalize?

> Ding, Y., Guo, Z., Li, J., Peng, L., Shao, S., Shao, W., Luo, X., Simon, L., Shang, J., McAuley, J., & Hou, Y. (2026). *How well does generative recommendation generalize?*. arXiv. https://arxiv.org/abs/2603.19809

**Method.** The paper builds an analytical framework that labels each test instance $(u, i_t)$ (a user history $u = [i_1, \dots, i_{t-1}]$ and its ground-truth next item $i_t$) by the *capability* a model needs to predict it correctly. The unit of analysis is the **item transition (物品转移)** — a directed pair $[i_s \to i_t]$ with $s < t$ and a *hop count* $t - s$ — rather than the target item alone. An instance is **memorization-related (记忆)** if the exact 1-hop transition $[i_{t-1} \to i_t]$ already appears in *any* training user's history: $(u, i_t) \in \mathcal{D}_{\text{mem}} \iff \exists u' \in \mathcal{D}_{\text{train}}\ \text{s.t.}\ [i_{t-1} \to i_t] \subseteq u'$. Otherwise it is **generalization-related (泛化)** if the transition can be *composed* from observed ones via **transitivity** ($[i_{t-1} \to x]$ and $[x \to i_t]$ both seen), **symmetry** (the reverse $[i_t \to i_{t-1}]$ seen), **2nd-order symmetry** (common cause / common effect / reverse path), or **substitutability** (a multi-hop skip transition $[i_{t-k} \to \cdots \to i_t]$ seen). Anything covered by neither (up to a max hop of 4) is **uncategorized**. They benchmark **TIGER** (a semantic-ID GR model) against **SASRec** (an item-ID model) on seven datasets, partition the test set by these categories, and report NDCG@10 per subset. The key explanatory move is a **token-level lens**: because GR tokenizes each item into a sequence of shared semantic-ID tokens $\text{tok}(i) = [z_1, \dots, z_L]$, they define **prefix $n$-gram memorization** — an instance is token-memorizable if the $n$-token prefixes $\text{pref}_n(\cdot)$ of *both* endpoints of the transition co-occur in training, even if the exact items differ. Finally they propose a training-free **memorization-aware adaptive ensemble**: use the item-ID model's max-softmax-probability $s_{\text{Conf}}(u) = \max_{j} P_{\text{ID}}(i_t = j \mid u)$ as a confidence proxy for "this instance is memorization-like", and blend the two models with weight $\alpha(u) = \mathrm{sigmoid}(-q(s_{\text{Conf}}(u) - \tau))$.

**Key results.** TIGER consistently *wins on generalization* subsets (e.g. +58.8% on Office, +56.7% on Beauty, +39.8% on Sports in NDCG@10) but consistently *loses on memorization* subsets (e.g. −43.6% on Yelp, −41.2% on Sports). The token-level lens explains the divergence: $>99$% of "item-level generalization" instances are actually **1-gram prefix-memorizable**, so what looks like generalization at the item level is largely *memorization at the token level* — and more prefix-transition support correlates with larger TIGER gains. The flip side is a **dilution effect**: GR spreads probability mass across all items sharing a prefix, so when an item transition is frequent ($\phi$ high) but its prefix transition is not distinctive ($\psi$ low), GR underperforms on memorization. A controlled study confirms the mechanism causally — shrinking the semantic-ID **codebook size** (denser tokens → more prefix sharing) improves generalization by +10.24% relative while degrading memorization by −7.62% relative. The adaptive ensemble beats both base models and a fixed-weight ensemble, confirming the two paradigms are **complementary**.

**Trade-offs / limitations.** The headline finding is a genuine **memorization-vs-generalization trade-off**, not a free lunch: you buy generalization with denser tokenization at the cost of memorizing specific high-frequency transitions. The framework is also bounded — categorization caps at a max hop of 4, leaving an "uncategorized" tail (always $<10$%) on which *both* models score near-zero, and it studies only two representative models (TIGER, SASRec) on small-to-mid public datasets with short average sequence lengths ($\approx 8$–13). The semantic-ID quantization is fixed ($256 \times 3$ in most experiments), so conclusions are tied to that tokenizer family rather than arbitrary GR designs.

**Connection to this chapter.** This paper is **foundational/conceptual** for understanding the modeling stages this chapter introduced. Recall §4.1–§4.2: the corpus is hundreds of millions of items, but any one user has touched only a handful, so the vast majority of *useful* candidate→target transitions a 召回 or 精排 model must score were **never observed exactly** — they live in the generalization regime this paper formalizes. That directly explains the industry shift toward GR: representing items by shared semantic-ID tokens turns item-level generalization into token-level memorization, which is precisely the capability needed to surface relevant-but-unseen items and thus widen the top of the funnel (the §4.4 intuition that 召回 is the biggest funnel). The trade-off framing also sharpens the metric discussion in §2–§3: a GR change might *lift* novel-discovery (generalization-heavy) consumption while *hurting* the easy, high-frequency memorized recommendations, so the diff you read off an A/B test (§6) is a blend of two opposing effects — exactly the kind of see-saw the chapter warns about with consumption exchange rates and the holdout. The adaptive ensemble is the practical takeaway: rather than picking one paradigm globally, route per-instance, which in pipeline terms means GR and item-ID retrieval channels are *complementary 召回 channels* (cf. §4.1's "many channels in parallel") rather than competitors.
