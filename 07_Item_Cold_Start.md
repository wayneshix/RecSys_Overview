# Part 07 — Item Cold-Start (物品冷启动)

Recommendation systems course notes — Part 07: Item Cold-Start.

**Item cold-start** is the problem of how to recommend *newly published items* to suitable users. On Xiaohongshu (小红书) the items are user-published notes (笔记); on Bilibili they are newly uploaded videos; on Toutiao they are newly published articles. This part focuses on **UGC** (User Generated Content) cold-start — platforms where users themselves upload the content (Xiaohongshu, Bilibili, YouTube, Zhihu). UGC cold-start is *harder* than PGC (Platform Generated Content, e.g. Netflix, Disney+, Tencent Video), because UGC content is enormous in volume and uneven in quality, so it cannot be hand-curated or manually traffic-managed by an operations team. The flip side is that the headroom for algorithmic improvement on UGC cold-start is large — most platforms never come close to the ceiling here.

Cold-start is widely regarded as the hardest, most complex part of a recommender system. It cannot reuse the ordinary recommendation pipeline and evaluation system, because new items are special in two structural ways (see §1).

---

## Table of Contents

1. [Why New Items Need Special Treatment](#1-why-new-items-need-special-treatment)
2. [Optimization Goals of Cold-Start](#2-optimization-goals-of-cold-start)
3. [Evaluation Metrics](#3-evaluation-metrics)
   - [3.1 Author-Side Metrics](#31-author-side-metrics)
   - [3.2 User-Side Metrics](#32-user-side-metrics)
   - [3.3 Content-Side Metrics](#33-content-side-metrics)
4. [Cold-Start Retrieval: The Difficulty](#4-cold-start-retrieval-the-difficulty)
5. [Simple Retrieval Channels for New Items](#5-simple-retrieval-channels-for-new-items)
   - [5.1 Adapting the Two-Tower Model](#51-adapting-the-two-tower-model)
   - [5.2 Category & Keyword Retrieval](#52-category--keyword-retrieval)
6. [Clustering-Based Retrieval](#6-clustering-based-retrieval)
   - [Content-similarity model (multimodal network)](#content-similarity-model-multimodal-network)
   - [Training the model](#training-the-model)
   - [Clustering index (聚类索引)](#clustering-index-聚类索引)
   - [Online retrieval](#online-retrieval)
7. [Look-Alike Retrieval](#7-look-alike-retrieval)
   - [Applying Look-Alike to new-note retrieval](#applying-look-alike-to-new-note-retrieval)
8. [Traffic / Flow Control](#8-traffic--flow-control)
   - [8.1 Why Tilt Traffic Toward New Items](#81-why-tilt-traffic-toward-new-items)
   - [8.2 Evolution of Flow-Control Techniques](#82-evolution-of-flow-control-techniques)
   - [8.3 Boosting](#83-boosting)
   - [8.4 Quota Guarantee](#84-quota-guarantee)
   - [8.5 Differentiated Quota Guarantee](#85-differentiated-quota-guarantee)
9. [Cold-Start A/B Testing](#9-cold-start-ab-testing)
   - [Standard A/B test (recap)](#standard-ab-test-recap)
   - [User-Side Experiment](#user-side-experiment)
   - [Author-Side Experiment](#author-side-experiment)
   - [Design checklist (ask yourself)](#design-checklist-ask-yourself)
10. [Key Insights](#10-key-insights)
   - [Source cross-reference notes / disagreements](#source-cross-reference-notes--disagreements)
11. [Expanded Reading — Recent Advances (2026)](#expanded-reading--recent-advances-2026)
   - [MediaFM: A Tri-Modal Content Foundation Model for Cold-Start (Netflix, 2026)](#mediafm-a-tri-modal-content-foundation-model-for-cold-start-netflix-2026)
12. [Pinterest in Practice (2024–2026)](#pinterest-in-practice-20242026)
   - [PinCLIP — content-only multimodal embedding with a collaborative-structure injection](#pinclip--content-only-multimodal-embedding-with-a-collaborative-structure-injection)
   - [Pinterest Canvas — generative content enhancement as a cold-start lever](#pinterest-canvas--generative-content-enhancement-as-a-cold-start-lever)

---

## 1. Why New Items Need Special Treatment

A recommender mainly uses three kinds of item features:

1. **Content features (内容特征)** — image, text, location, algorithm/human-assigned tags (category, keywords). New items **have** these.
2. **Statistical features (统计特征)** — CTR, like-rate, etc. from accumulated exposures. New items **lack** these (or the values are unreliable).
3. **ID embedding** — a vector per item ID, learned from user–item interaction. New items **lack** a usable one (it is freshly initialized, not yet updated by backprop).

So a new item is missing **two of the three** feature families. Consequences:

- Recommendation accuracy is poor → if routed through the normal pipeline, new items rarely get exposed, and when they do the consumption metrics are bad.
- **ItemCF / UserCF do not apply** (need item–user interaction history; a new item has none).
- The **two-tower model** is the single most important retrieval channel, but it depends on the item-ID embedding, which is not learned yet → effect is poor without modification.

There is also a **business reason** (specific to UGC): cold-start strategy is accountable for **author-side publishing metrics**, whereas the ordinary pipeline only owns user-side consumption metrics. Xiaohongshu's experiments show: **the more and the earlier** a new note gets exposure and interaction, the higher the author's willingness to keep publishing. Intuition: if your notes get no views for hours, you stop posting; if your note gets likes/comments within an hour, you post more. Supporting low-exposure new notes is the key to incentivizing publishing.

Finally, the cold-start **evaluation system and A/B-test design also differ** from ordinary recommendation (see §3, §9).

---

## 2. Optimization Goals of Cold-Start

Industry consensus — three goals:

1. **Accurate recommendation (精准推荐):** overcome the difficulty and push new items to the *right* users, at minimum not provoking dislike (反感).
2. **Incentivize publishing (激励发布):** tilt traffic toward low-exposure new notes so authors are motivated to publish, enlarging the content pool (内容池). Note the leverage: lifting a note from 10k → 20k exposures barely affects author motivation, but lifting from 20 → 100 exposures helps a lot — so the support targets *low-exposure* notes.
3. **Mine high-potential items (挖掘高潜):** give every new note a small "spread-the-dew-evenly" (雨露均沾) exploratory exposure (e.g. 100–200 impressions), find the high-quality ones, then tilt more traffic to grow them into hits.

These three goals map onto the three metric families below.

---

## 3. Evaluation Metrics

Ordinary recommendation only watches **user-side** metrics; cold-start watches **three families**, which makes its evaluation noticeably more complex.

| Family | Metrics | Reflects |
|---|---|---|
| Author-side (作者侧) | publish penetration rate, per-capita publish count | author publishing willingness (goal 2) |
| User-side (用户侧) | new-note CTR/interaction-rate; DAU/MAU/usage-time (大盘) | recommendation accuracy + not hurting experience (goal 1) |
| Content-side (内容侧) | high-heat note ratio; high-exposure-low-conversion ratio | ability to mine quality + ecosystem health (goal 3) |

Author-side & user-side metrics are standard at all good big firms. Content-side metrics are used only by a few.

### 3.1 Author-Side Metrics

On Xiaohongshu every user is both a consumer (reads) and a creator (publishes).

**Publish penetration rate (发布渗透率, penetration rate):**

$$
\text{penetration rate} = \frac{\text{users who published today}}{\text{DAU}}
$$

A user who publishes **one or more** notes counts as exactly **one** publisher (publishing 1 vs. 10 does not change penetration). Example: 1,000,000 publishers / 20,000,000 DAU = **5%**.

**Per-capita publish count (人均发布量):**

$$
\text{per-capita publish} = \frac{\text{notes published today}}{\text{DAU}}
$$

Example: 2,000,000 notes / 20,000,000 DAU = **0.1**. This is *larger* than penetration (per fixed DAU) because some authors publish several notes a day.

> **Source note:** slides/audio use the example DAU = 20,000,000 (penetration 5%, per-capita 0.1); the written Notes use DAU = 50,000,000 (penetration 2%, per-capita 0.04). Numbers are illustrative only — neither is real Xiaohongshu data. The *formulas* agree across all three sources.

These two metrics reflect publishing enthusiasm. To lift them: make the new item's first exposure and interaction occur **as early and as often as possible** — e.g. shorten time-to-first-exposure from 5 min to 30 s (an engineering optimization on the retrieval indexing path), and lift the fraction of notes that reach 100 exposures within 24h (e.g. 50% → 80%) via flow control.

### 3.2 User-Side Metrics

Two sub-types.

**(a) New-note consumption metrics — new-note CTR / interaction-rate.** If new notes are matched to interested users, these go up. But a naive whole-population number is **misleading** because of a severe **head effect (头部效应)**: the Gini coefficient (基尼系数) of exposure is large (often > 0.9), i.e. a few head notes grab most of the exposure, so the aggregate metric is dominated by the head. Even if almost all new notes are recommended badly, a few well-recommended head notes can make the aggregate look fine.

> **Two cautionary examples (from the Notes).**
> - *Fake prosperity:* without changing any model, shift traffic from low- to high-exposure new notes → aggregate new-note CTR rises, yet recommendation did not actually improve.
> - *Misleading A/B:* a model that slightly improves head items but badly hurts low-exposure items can still show a *higher* aggregate new-note CTR.

**Fix: bucket by exposure and look at the buckets separately**, e.g. $[0,99]$, $[100,999]$, $[1000,+\infty)$ (a common threshold split is high vs. low exposure at **1000**). We mostly care about the **low-exposure** buckets because (i) they are the vast majority of new notes, and (ii) they are the hard case (few interactions → imprecise → need dedicated techniques). High-exposure new notes are easy and need no special handling.

**(b) Big-board / overall consumption metrics (大盘指标)** — usage time, DAU, MAU. These do **not** distinguish new vs. old items. Cold-start's job is **not** to grow these, but its strategy **must not significantly hurt** them — keep them roughly flat.

**Seesaw effect (跷跷板效应):** aggressively boosting low-exposure new notes improves author-side publishing metrics but **degrades** user-side big-board metrics. Reason: low-exposure notes lack interaction features → imprecise recommendation → showing more of them lowers experience → lower usage time / DAU / MAU.

### 3.3 Content-Side Metrics

Cold-start influence is not confined to the cold-start window (e.g. first 24h). A good strategy mines quality items and helps them "climb the slope" (爬坡) into hits, and avoids wasting traffic on junk. Author- and user-side metrics can't measure this longer-term ecosystem effect; content-side metrics can.

**High-heat note ratio (高热笔记占比).** A high-heat note = a note that gets **1000+ clicks within its first 30 days** (note lifecycle = 30 days, because Xiaohongshu Home Feed only distributes notes younger than 30 days). Example: of 2,000,000 notes published today, if 200,000 reach 1000+ clicks within 30 days, ratio = 10%. Higher is better → stronger ability to mine quality.

> *Subtlety (Notes):* does optimizing high-heat ratio worsen the head effect? It can concentrate "dew" traffic onto good items (worsening head); but it can also *reduce* head effect — splitting one note's 1M exposures across 100 notes lets all 100 hit the high-heat threshold, increasing the *number* of high-heat items.

**High-exposure-low-conversion ratio (高曝低转占比)** (Notes only). E.g. "high-exposure" = >1000 exposures, "low-conversion" = CTR < 5%. Among notes that are high-exposure, the fraction that are also low-conversion. **Lower is better** — high-exposure-low-conversion means imprecise recommendation / wasted traffic (over-boosting, or pushing junk).

**Two optimization levers for cold-start overall:** (1) optimize the **full pipeline** (retrieval + ranking) for new items; (2) **flow control** — how traffic is split between new and old items.

---

## 4. Cold-Start Retrieval: The Difficulty

What we **have** for a new note: image, text, location; algorithm/human tags (category, keywords).
What we **lack**: user click/like statistics; a learned note-ID embedding.

Channel applicability:

| Channel | Cold-start? |
|---|---|
| ItemCF retrieval | ✗ not applicable |
| Two-tower model (双塔) | ⚠ applicable **after modification** |
| Category / keyword (类目、关键词) | ✓ |
| Clustering (聚类) | ✓ |
| Look-Alike | ✓ |

**Why ItemCF fails:** ItemCF measures item similarity from the overlap of users who interacted with both items. Let $U_1, U_2$ be the users who interacted with item 1, item 2; similarity grows with $|U_1 \cap U_2|$. For a new item, $U_2 \approx \varnothing$, so the overlap is undefined → no similarity → no retrieval. Collaborative-filtering channels generally fail for the same reason.

**Lifecycle of channels** (zero → high exposure):

$$
\text{category, clustering} \longrightarrow \text{Look-Alike} \longrightarrow \text{two-tower, collaborative filtering}
$$

Content-based channels (category, clustering) fire first (need no interaction); once a few interactions appear, Look-Alike can retrieve more accurately; as exposure/interaction grow, two-tower and CF take over.

---

## 5. Simple Retrieval Channels for New Items

### 5.1 Adapting the Two-Tower Model

Standard two-tower: user tower (user ID + features) and item tower (note ID + features) each output a vector; cosine similarity = predicted interest. The **note-ID embedding** is the item tower's most important feature, but for a new note it is unlearned. Two fixes:

**Fix 1 — Default embedding (default embedding).** Make **all** new notes share one **default ID**, instead of using their own real IDs. The default ID's embedding is *learned* (not random/zero init), so it carries real signal and beats random/all-zero init. At the next model-training cycle, each new note then gets its own learned ID embedding. This yields a measurable gain.

**Fix 2 — Borrow similar items' embeddings.** When a note is published, find the **top-$k$ content-most-similar high-exposure notes** (similarity via image/text/category using a multimodal network) and **average their ID embeddings** as the new note's embedding. High-exposure notes are used because their ID embeddings are well-learned.

**Multiple vector retrieval pools (多个向量召回池).** Maintain several pools — e.g. **1h new notes, 6h, 24h, 30-day**. This gives new notes more chances to be retrieved (in a single 30-day pool a new note is easily drowned out). All pools **share the same two-tower model parameters**, so multiple pools add **no extra training cost**. Smaller pools also rebuild their ANN index faster (the ≤1h pool is ~1/24 the size of the ≤24h pool), reducing indexing latency.

> Beyond the slides, the Notes also describe **real-time incremental updates** of the two-tower model: build a real-time data stream that packages fresh user–item interactions into training data and updates the model continuously; then periodically publish the model, recompute item vectors, and refresh the vector-DB index. Both stages add latency; their sum = time from interaction to the item vector reflecting it.

### 5.2 Category & Keyword Retrieval

Every feed/social/e-commerce platform maintains a **user profile (用户画像)** recording demographics and interest tags — interested **second-level categories (二级类目)** (e.g. food/探店, tech & digital, movies) and **keywords (关键词)** (e.g. New York, workplace, comedy, programmers, college), inferred from the user's clicks/interactions (or self-filled). At publish time an NLP model auto-tags each note with category/keyword labels.

**Category index (类目索引):** `category → note-ID list (sorted by publish time, newest first)`.

**Category retrieval logic:** `user → categories → note list`.
On a request: read the user profile → get interested categories → for each category, take the **top-$k$ newest** notes. If the user has 3 interested categories, this returns $3k$ notes.

**Keyword retrieval** is identical but keyed on keywords instead of categories. Keywords vastly outnumber categories, so the granularity is finer.

**Self-defined targeting / fastest path.** Category retrieval is the **fastest channel to first exposure**: as soon as a note enters the `category → note` index it can be retrieved. If the index refresh interval is $t$, the minimum time-to-first-exposure is $t$; shorter $t$ → higher creator motivation. Xiaohongshu changed this index from a 5-minute batch update to a **real-time stream → second-level (秒级) updates**, which materially lifted author-side publishing metrics.

**Two drawbacks:**
1. **Effective only for *just-published* notes.** Because the list is publish-time-descending, after a few (tens of) minutes a note falls to rank hundreds/thousands and is never retrieved by this channel again — a very short window.
2. **Weak personalization (弱个性化), imprecise.** It only matches profile-category to note-category, which is coarse. Example: "I like aquarium fish; it belongs to the *pets* category, but the 100 newest pet notes are all cats/dogs → 100 irrelevant results." CTR/like-rate from this channel is noticeably below the big-board.

Despite the drawbacks these channels are important: they give a just-published note **instant** exposure, helping author motivation.

---

## 6. Clustering-Based Retrieval

**Core idea:** if a user likes a note, they will like **content-similar** notes. (e.g. you saved a "buying a house in Shanghai" note → recommend more Shanghai-housing notes.) Retrieval is based on the note's **image-text content**, not on interaction.

### Content-similarity model (multimodal network)

Train a multimodal neural network mapping a note's image+text to a feature vector:

- **CNN** extracts an image feature vector (CNN pretrained, e.g. on ImageNet; fine-tune optional).
- **BERT** extracts a text feature vector (pretrained; fine-tune optional).
- **Concatenate** the two → feed a **fully-connected (FC) layer** → final note feature vector. (FC is randomly initialized, learned from data.)

Two notes' similarity = cosine of their feature vectors:

$$
\cos(\mathbf{a}, \mathbf{b}) = \frac{\langle \mathbf{a}, \mathbf{b} \rangle}{\lVert \mathbf{a} \rVert_2 \cdot \lVert \mathbf{b} \rVert_2}
$$

### Training the model

Each training sample is a **triplet**: a **seed note (种子笔记)** $\mathbf{a}$, a **positive (正样本)** $\mathbf{b}^+$ (content-similar), a **negative (负样本)** $\mathbf{b}^-$ (content-dissimilar). The three towers share identical parameters (CNN+BERT+FC). Goal: encourage $\cos(\mathbf{a},\mathbf{b}^+) > \cos(\mathbf{a},\mathbf{b}^-)$.

**Triplet hinge loss:**

$$
L(\mathbf{a}, \mathbf{b}^+, \mathbf{b}^-) = \max\Big\lbrace0, \cos(\mathbf{a},\mathbf{b}^-) + m - \cos(\mathbf{a},\mathbf{b}^+)\Big\rbrace
$$

where $m$ is the **margin** hyperparameter (e.g. $m=1$).

**Triplet logistic loss:**

$$
L(\mathbf{a}, \mathbf{b}^+, \mathbf{b}^-) = \log\Big(1 + \alpha \cdot \exp\big(\cos(\mathbf{a},\mathbf{b}^-) - \cos(\mathbf{a},\mathbf{b}^+)\big)\Big)
$$

where $\alpha > 0$ controls the logistic shape (slides write it without the leading $\alpha$; the Notes include $\alpha$).

**Choosing positives** — two methods: (1) **human annotation** of pair similarity (accurate but expensive, limited data); (2) **algorithmic selection** with filters: only use **high-exposure** notes as the pair (they have rich interaction info → cleaner labels); the two notes must share the **same second-level category** (e.g. both "recipe tutorials"); then pick positives via **ItemCF item similarity** (more co-interested users → more similar). Caveat: *interest-point similarity ≠ image-text content similarity*. The current best approach is **CLIP-style pretraining** (Radford et al., ICML 2021), which the Notes recommend over this older triplet approach.

**Choosing negatives** — randomly from all notes, satisfying: (i) enough words (so the network extracts meaningful text), and (ii) high quality (avoid image-text-irrelevant junk).

### Clustering index (聚类索引)

After training, map **all** items to feature vectors and **cluster by cosine similarity** (k-means with cosine) into $k$ clusters. (Slides say **1000** clusters; Notes say **2048** — both illustrative.) Record each cluster's **center direction (中心方向)**. Build:

- forward index: `cluster → note list (publish-time-descending)`
- inverted index: `note → cluster`

When a new note is published: compute its vector → compare to the $k$ cluster centers → pick the nearest → append to that cluster's list (at the front).

### Online retrieval

`user → interacted notes → cluster → top-m newest notes`. Given a user ID, take the **last-$n$** interacted notes (e.g. last-$n$ liked + saved + shared) as **seed notes**, map each to its cluster (via the inverted index), and pull back the newest $m$ notes per cluster → at most $m \cdot n$ new notes (slides use $n$ seeds → $mn$; Notes use $3n$ seeds → $3mn$).

> **Why not just ANN-retrieve the $m$ most-similar items to each seed vector?** Because we want notes that are *similar but not too similar*. If a retrieved note is nearly identical to one already consumed (e.g. almost the same cover image), it should be filtered out as a duplicate. Cluster-level retrieval gives "same neighborhood, not same point."

---

## 7. Look-Alike Retrieval

**Origin: internet advertising.** An advertiser (e.g. Tesla) knows its Model 3 **typical users**: age 25–35, college+ degree, follows tech & digital, likes Apple products. Users matching all conditions = **seed users (种子用户)** — but only a few thousand can be identified (most users don't disclose age/education), while the advertiser wants to reach, say, 1,000,000. **Look-alike audience expansion (人群扩散)** solves this: find users *similar to the seed users* — the **look-alike users** — and target them too.

Look-alike is a **framework, not a specific algorithm**; the engineer must define user–user similarity. Two ways:
- **UserCF:** similarity = number of items both users like (shared interest points).
- **Embedding:** large cosine between the two users' (e.g. two-tower) embedding vectors.

### Applying Look-Alike to new-note retrieval

When a new note is published, the system shows it to many users; a few **click / like / save / share** → those interactions signal interest → treat those users as the note's **seed users**. (The Notes add a threshold, e.g. $k=3$: if interaction users exceed $k$, ignore the weaker *click* signal and use only interactions, since interaction is a stronger interest signal than click.) Then **expand** to similar users via look-alike.

**Mechanics (near-line update of the note vector):**

1. Each seed user has an embedding (reuse the two-tower user vector).
2. The note's **feature vector = average of its seed users' vectors** — it represents *what kind of user is interested in this note*.
3. **Near-line update (近线更新):** every time a *new* user interacts with the note, recompute the average and update the note vector. "Near-line" = minute-level refresh suffices (no need for real-time second-level). E.g. for ≤1h notes refresh every 5 min, for 1–24h notes every 1h.
4. Store `⟨vector, note-ID⟩` in a **vector database** (e.g. Milvus, Faiss) supporting ANN.

**Online serving:** when a user opens the app, compute their two-tower vector → use it as the **query** for **ANN nearest-neighbor search** in the vector DB → retrieve dozens of new notes.

**Why this *is* look-alike:** the requesting user is matched to new notes whose *seed users* are similar to them. If the user resembles a note's seed users, the system judges they'll be interested → audience expansion, identical in essence to ad look-alike.

**Main challenge = engineering real-time-ness.** New-item interactions are scarce, so each one is precious and must be exploited fast to make the next retrieval more accurate. Let $t_1$ = interaction → note-vector update latency, $t_2$ = vector-DB index refresh interval; total delay from interaction to retrieval impact $\le t_1 + t_2$. Minimizing $t_1 + t_2$ improves look-alike performance.

---

## 8. Traffic / Flow Control

Flow control = how to allocate traffic (exposure opportunities) between **new and old items**. Setup: Xiaohongshu Home Feed distributes notes younger than 30 days. Under **natural distribution (自然分发)** (new and old compete fairly), a note younger than 24h would get exposure share $\approx \tfrac{1}{30}$. Flow control deliberately makes new-note exposure share **far exceed $\tfrac{1}{30}$**.

### 8.1 Why Tilt Traffic Toward New Items

- **Goal 1 — promote publishing, enlarge content pool:** more exposure → higher creator motivation → higher penetration & per-capita publish.
- **Goal 2 — mine quality items:** an item with almost no exposure can't be discovered as good. Give every new note enough early exploratory exposure (e.g. 100) so the system can detect quality → reflected in high-heat ratio.

### 8.2 Evolution of Flow-Control Techniques

$$
\text{forced insertion (强插)} \rightarrow \text{simple boost} \rightarrow \text{quota guarantee (保量)} \rightarrow \text{differentiated quota}
$$

- **Forced insertion:** in re-ranking, force-insert new notes into the exposure queue (e.g. "insert 1 every 3"). Used because ranking can't yet score new items reliably enough to co-rank them with old items. Crude/outdated; still used on backward business lines.
- **Boost:** multiply (or add to) the new note's ranking score by a coefficient so it competes better. Good ROI, easy to build (Douyin/Xiaohongshu used it early).
- **Quota guarantee:** ensure every new note gets ≥ N exposures within T hours (e.g. 100 in 24h). Implemented *via* boosting, but with a more refined boost policy. Borrowed from ad "guaranteed delivery."
- **Differentiated quota:** set the quota target per-item by predicted quality.

### 8.3 Boosting

**Where to intervene:** the two **funnels** in the pipeline — **pre-ranking (粗排)** (thousands → hundreds) and **re-ranking (重排)** (hundreds → dozens). Multiply the new note's score by a coefficient > 1 so it has a higher chance to pass each funnel; then let new and old items **compete freely** (higher score wins).

Pipeline: `retrieval (hundreds of millions) → pre-rank + truncation (thousands → hundreds) → rank (精排) → re-rank + sampling (hundreds → tens)`.

**Pros:** easy, low investment, decent output.
**Cons:** exposure is **very sensitive** to the boost coefficient; **hard to precisely control** exposure volume → over-exposure and under-exposure both happen. Tuning helps but never gives precise control.

### 8.4 Quota Guarantee

**Definition:** regardless of quality, try to give every new note (e.g.) **100 exposures in its first 24h**.

**Simplest table-based boost** — multiply an *extra* boost coefficient (on top of the base boost) depending on **(publish age, current exposure count)**. The further behind the 100-in-24h target, the larger the coefficient. (Illustrative table; not real data.)

| publish age \ current exposures | 0–24 | 25–49 | 50–74 | 75–100 |
|---|---|---|---|---|
| 0–5h | 1.0 | 1.0 | 1.0 | 1.0 |
| 6–11h | 1.1 | 1.0 | 1.0 | 1.0 |
| 12–17h | 1.2 | 1.1 | 1.0 | 1.0 |
| 18–24h | 1.3 | 1.2 | 1.1 | 1.0 |

**Dynamic boosted quota (动态提权保量).** Compute the coefficient from **four values**: target time, target exposure, publish time, existing exposure. Define two normalized variables:

$$
t = \frac{\text{publish time}}{\text{target time}}, \qquad x = \frac{\text{existing exposure}}{\text{target exposure}}
$$

$$
\text{boost coefficient} = f\left(\frac{\text{publish time}}{\text{target time}}, \frac{\text{existing exposure}}{\text{target exposure}}\right) = f(t, x)
$$

Example: published 12h ago of a 24h target → $t = 0.5$; has 20 of 100 target exposures → $x = 0.2$ → $f(0.5, 0.2)$. **Larger $t$, smaller $x$ ⇒ larger boost** (behind schedule on time, short on exposure → push harder).

**The boost formula (from the Notes — interview-grade detail).** Under uniform delivery the ideal is $x = t$; if $x < t$, accelerate. But industry found uniform delivery is *not* optimal — **earlier exposure has higher utility** for incentivizing publishing (100 exposures in hour 1 ≫ 100 exposures in hour 24), so delivery should be **front-loaded** (fast early, decaying). A simple boost formula depending only on $t, x$:

$$
\text{boost} = \max\left(\frac{\alpha}{\sqrt{1 + \beta t + \gamma x}}, 1\right), \quad \alpha,\beta,\gamma > 0
$$

Boost decreases with $t$ (support decays over age) and with $x$ (more exposure → less support); floored at $1$ so a new item is never *suppressed* below natural distribution.

**Delivery curve / front-load (投放曲线 / frontload, Notes).** Let $y(t)$ be the target delivery curve (we want actual $x \approx y(t)$), satisfying

$$
\frac{dy}{dt} = \big[1 + \theta(1-t)\big]\cdot\frac{1 - y(t)}{1 - t}, \qquad y = 1 - (1-t)\exp(-\theta t)
$$

$\theta$ controls front-load speed (initial slope $1+\theta$; $\theta = 0$ = uniform). Then `boost = max{ g(y(t) − x), 1 }` with $g$ monotone increasing (e.g. $g(\cdot)=\alpha\exp(\cdot)$): if actual $x$ lags the curve, boost more.
The Notes also cover **multi-stage quota** (e.g. 100 in 24h, then 200 in 72h) and **time-of-day normalized delivery** — because nighttime traffic is small, replace raw $t$ by a flow-weighted $t'$ using a normalized hourly-exposure density $f(k)$ (with $\int_0^{24} f(k)dk = 1$); the "clock slows down" at night.

**Difficulty: success rate ≪ 100%** — many notes don't reach 100 exposures in 24h. Causes: (i) retrieval/ranking weaknesses (some note types are hard to retrieve); (ii) badly-tuned boost coefficients; (iii) **online environment changes** (new retrieval channel, upgraded ranking model, changed re-rank diversification rules) that perturb exposure and force re-tuning.

**Thought question — why not just give every new note a huge boost (e.g. ×4) until it hits 100, then stop?** Success rate would be high and it's easy to build. But **more boost is not always better for the note**:
- **Pro:** higher score → more exposure.
- **Con:** the note is shown to *unsuitable* audiences → low CTR/like-rate → after cold-start the recommender **suppresses (打压)** it long-term → a potentially-hot note never grows. (When you *don't* artificially inflate the score, a high score only occurs for matched audiences, which is healthy.)

Hence careful, fine-grained boost tuning rather than brute force.

### 8.5 Differentiated Quota Guarantee

Different items get **different quota targets**. Base quota: 24h / 100 exposures. Add extra quota by:

- **Content quality (内容质量):** a multimodal model scores image/text/video quality (e.g. predicts probability of reaching top-10% click count/CTR; reported as having decent AUC) → up to **+200** extra exposures.
- **Author quality (作者质量):** if the author's past notes were often high-quality, their new notes likely are too → up to **+200** extra exposures.

So a note's quota ranges from **100 (min) to 500 (max)** exposures (illustrative). After hitting the target, support stops — the note reverts to natural distribution, competing fairly with old notes. Differentiated quota tilts traffic to quality items → improves content-side metrics.

> The Notes refine the staged version: the **first-stage** target is set *at publish time* from predicted high-heat probability (content) + author quality (no observed metrics yet). Once enough exposure accumulates, observed CTR/interaction/completion become trustworthy, and a **next-stage** quota and delivery curve are set from actual performance.

---

## 9. Cold-Start A/B Testing

This is the trickiest topic — far harder than ordinary A/B testing — because cold-start has **two kinds of goals** (incentivize authors *and* satisfy users), so we must run **both a user-side experiment (consumption metrics)** and an **author-side experiment (publishing metrics)**. Every known author-side design has flaws.

### Standard A/B test (recap)

Split **users** 50/50 into experiment (新策略) and control (旧策略); both draw from **100% of items**. Compare the **diff** of consumption metrics (e.g. usage time +1%). This is the standard user-side test.

### User-Side Experiment

The standard split works for cold-start's user-side consumption metrics, with **one flaw**. Setup:
- Quota of 100 exposures is in effect.
- Assumption: **more new-note exposure → lower app usage time** (new notes are less precise; too many hurts experience).
- New strategy: **double (×2) the new-note ranking weight** in the experiment group.

Result (consumption only): the A/B **diff is negative** (experiment worse than control). **But on full rollout (推全) the diff shrinks** (e.g. −2% → −1%).

**Why?** With ×2 boost, experiment-group users see *more* new notes → their consumption worsens. But the **"100 exposures" quota is a fixed quantity** — if a note gets more exposure from the experiment group, it needs *less* from the control group, so control-group users see *fewer* new notes → control consumption *improves*. Experiment ↓ + control ↑ ⇒ the observed diff is **inflated**. After full rollout there is no "control group lifting the baseline," so the real drop is smaller. (Still, this flaw is mild and the user-side result is fairly trustworthy.)

### Author-Side Experiment

The hard part: we must split **authors** (the new notes they publish use new vs. old strategy) to compare publishing metrics.

**Scheme 1 — bucket only new items (by author).** 100% users, 100% old notes (un-bucketed). New notes split by author into experiment (boost $2\alpha$) and control (boost $\alpha$). Compare publishing-metric diff.

- **Flaw A — the two new-note groups steal traffic from each other (抢流量).** Consider **independent queues**: new and old items run separate queues with a fixed exposure split (e.g. new 1/3, old 2/3), no new-vs-old competition. Then boosting *all* new notes by the same factor changes nothing (they still compete fairly among themselves) — even a ×10000 boost has no effect, so **full rollout would not change publishing metrics at all**. Yet in the A/B test, the experiment group's larger boost lets it **steal exposure from the control group** → experiment publishing ↑, control ↓ → a **spurious positive diff** (e.g. +2%) that **vanishes on rollout** (→ 0). The diff is not credible.
- **Flaw B — new notes steal from old notes (free competition).** If new and old compete freely: in the A/B test, **50% new notes (boosted) compete with 100% old notes** → one new note steals ~two old notes' exposure. On rollout, **100% new notes compete with 100% old notes** → one new note steals ~one old note's exposure. The competition regime *changes* between test and rollout → the test **over-estimates** the gain (e.g. test diff +2%, rollout +1.5%).

**Scheme 2 — bucket authors *and* users (jointly isolate).** Experiment-group users can see only experiment-group new notes; control users only control new notes.
- **Advantage:** the two new-note buckets no longer steal from each other → author-side result is **more credible** (if a diff is seen, it's real; survives rollout).
- **Still shares Flaw B:** new vs. old still compete differently in test vs. rollout.
- **New drawback — shrinks the content pool, hurting the big board.** Each user now sees only 50% of new notes. Without isolation, retrieval/ranking pick the $n$ best new notes; with isolation only $n/2$ are available, so $n/2$ worse ones fill in → worse experience → big-board consumption drops. **Doing an A/B test damages the business.**

**Scheme 3 — bucket new items, old items, *and* users (full isolation).** Xiaohongshu is effectively split into two apps. **Most accurate** if you want precise results (rollout diff = test diff). But it **severely** hurts experience (content pool halved on *both* new and old) → big board drops a lot → too costly; not practical.

**Scheme 4 (Notes only — the author's team's design, better than Scheme 2).** Old items un-bucketed; **users split into 2 buckets**; **authors split into 3 buckets** where the middle (50%) author bucket serves as *both* experiment and control. Experiment users draw from the first two new-note buckets (new strategy); control users from the last two (old strategy). Analysis compares only the **two 25% author buckets**' publishing diff.
- Hurts experience **less** than Scheme 2: the new-note pool shrinks by only **25%** (vs. 50% in Scheme 2). If the middle 50% author bucket shrinks to 0, Scheme 4 degenerates to Scheme 2.
- **Bonus — protects high-follower authors (高粉作者):** put high-follower authors in the middle bucket so their new notes can reach *all* users (good for their publishing experience); authors in bucket 1 or 3 reach only 50% of users.

> Even this is imperfect. As Wang says: *there is probably no perfect experiment design.*

### Design checklist (ask yourself)

1. Will the **experiment- and control-group new notes steal traffic** from each other?
2. **How do new vs. old notes steal traffic** — is the stealing mechanism the same in the A/B test as after full rollout? (If it changes, the test may be inaccurate.)
3. Does **jointly isolating items and users shrink the content pool** (hurting the big board / business)?
4. If you do **quota guarantee** on new notes, what happens? (E.g. if a note gets 80 of its 100 exposures from experiment users, it barely needs the control group — only 20 more — so quota distorts the comparison.)

---

## 10. Key Insights

**Three goals → three metric families.** Incentivize publishing → author-side (penetration rate, per-capita publish). Accurate recommendation / don't hurt experience → user-side (bucketed new-note CTR/interaction; big-board DAU/MAU/usage-time). Mine high-potential → content-side (high-heat ratio; high-exposure-low-conversion ratio).

**Retrieval.** ItemCF/CF and raw two-tower **fail** for new items (no interaction, no learned ID embedding). Channels that work: **category/keyword** (fastest to first exposure, but short window + weak personalization), **clustering** (content embedding via CNN+BERT+FC or CLIP → k-means → retrieve by nearest cluster from the user's seed notes), **Look-Alike** (seed users with interactions → average their vectors as the note vector → ANN-match requesting users). Two-tower becomes usable via **default embedding** / **borrowed similar-item embeddings** + **multiple content pools** (shared params → no extra training cost).

**Flow control.** `强插 → 提权 → 保量 → 差异化保量`. Boost the new note's ranking score at the pre-rank/re-rank funnels; quota-guarantee = dynamic boost from $f(t, x)$ with $t=$ publish/target time, $x=$ existing/target exposure; front-loaded (early exposure has higher author-incentive utility); never suppress below natural distribution (boost floored at 1). Brute-force huge boost is bad: over-boosted notes hit unsuitable audiences → low CTR → long-term suppression. Differentiated quota sets per-item targets (100–500) by content & author quality.

**A/B testing — the load-bearing intuitions:**
- **Quota is a conserved quantity** → if the experiment group consumes more of a note's exposure, the control group consumes less → control becomes a *moving baseline* → diffs are inflated and shrink on rollout.
- **Stealing-traffic regimes differ between test and rollout:** 50%-new-vs-100%-old (test) vs. 100%-vs-100% (rollout) → over-estimation. Independent queues can make a real-rollout-zero effect show a spurious positive A/B diff.
- **Isolating items shrinks each user's content pool** → hurts the big board → the experiment itself damages the business. More isolation = more credible author-side result *but* more user-side harm (Scheme 1 < 2 < 4 < 3 on credibility; opposite on harm, roughly).

---

### Source cross-reference notes / disagreements

All three sources (six slide decks, the Notes PDF, six Chinese audio transcripts) agree on concepts and formulas. Minor numeric/illustrative differences (all explicitly "not real Xiaohongshu data"):

- **Author-side example:** slides/audio use DAU 20,000,000 (penetration 5%, per-capita 0.1); the Notes use DAU 50,000,000 (penetration 2%, per-capita 0.04). Formulas identical.
- **Cluster count:** slides/audio say **1000** clusters; the Notes say **2048**.
- **Seed count in clustering retrieval:** slides say $n$ seeds → $mn$ retrieved; the Notes say $3n$ seeds (last-$n$ liked + saved + shared) → $3mn$.
- **Triplet logistic loss:** slides omit the leading $\alpha$ inside the log; the Notes include $\alpha$ (shape hyperparameter).
- **Depth:** the **Notes are the richest source** and uniquely contain — the high-exposure-low-conversion content-side metric; the explicit boost formula $\max(\alpha/\sqrt{1+\beta t+\gamma x},1)$; the delivery-curve ODE and front-load solution $y=1-(1-t)e^{-\theta t}$; multi-stage quota; time-of-day flow normalization $t'$; the CLIP recommendation over triplet training; and **A/B-test Scheme 4** (the team's design + high-follower-author protection), which is absent from the slides. The audio transcripts (AI-transcribed Chinese) had expected ASR garbles — e.g. "ItemSafe" for ItemCF, "Muvis"/"Face" for Milvus/Faiss, "进线" for 近线 (near-line), "粗牌/重牌" for 粗排/重排 — all resolved against the slides/Notes as ground truth.

## Expanded Reading — Recent Advances (2026)

This section connects the chapter's content-embedding / clustering-retrieval material (§6) to a 2026 production system that produces exactly those embeddings at scale.

### MediaFM: A Tri-Modal Content Foundation Model for Cold-Start (Netflix, 2026)

> Saluja, A., Castro, S., Yan, B., & Rastogi, A. (2026). *MediaFM: The multimodal AI foundation for media understanding at Netflix*. Netflix Technology Blog. https://netflixtechblog.com/mediafm-the-multimodal-ai-foundation-for-media-understanding-at-netflix-e8c28df82e2d

**Method (how the tri-modal foundation model is built/trained).** MediaFM is an in-house, self-supervised, tri-modal (audio / video / text) content-embedding model — the production-grade descendant of the CNN+BERT+FC content model in §6, scaled up and made multimodal.

- **Unit of input = a *shot***, obtained by segmenting a title (movie/episode) with a shot-boundary-detection algorithm. (Compare §6, where the unit is a *note*.)
- **Per-shot frozen encoders** produce three modality embeddings: **video** via *SeqCLIP* (a CLIP-style model fine-tuned on video retrieval — note the direct lineage from the CLIP recommendation in §6); **audio** via Meta FAIR's *wav2vec2*; **timed text** (captions / audio descriptions / subtitles) via OpenAI's *text-embedding-3-large*. The three are concatenated and unit-normed into one **2304-dim fused embedding** per shot ($768 \times 3$).
- **Sequence model = a BERT-like Transformer encoder** over a temporally-ordered sequence of fused shot embeddings (up to **512 shots**). A learnable `[CLS]` token and a `[GLOBAL]` token (title-level metadata — synopses/tags — encoded by text-embedding-3-large) are prepended, so every shot attends to global title context. The fused embeddings are first projected to the hidden dim, contextualized through the Transformer stack with positional embeddings, then projected back up to the 2304-dim space.
- **Training objective = Masked Shot Modeling (MSM)**, a continuous analogue of BERT's masked-language-modeling. Randomly mask **20%** of shots with a learnable `[MASK]` embedding and predict the original fused embedding, minimizing **cosine distance** to the ground truth:
  $$
  L_{\text{MSM}} = \sum_{i \in \mathcal{M}} \Big(1 - \cos\big(\hat{\mathbf{e}}_i, \mathbf{e}_i\big)\Big)
  $$
  where $\mathcal{M}$ is the masked-shot set, $\mathbf{e}_i$ the ground-truth fused embedding, $\hat{\mathbf{e}}_i$ the prediction. Optimizer: **Muon** for hidden parameters, **AdamW** for the rest (the switch to Muon gave noticeable gains). The output is a content embedding learned *with no interaction data* — the property that makes it usable for cold-start.

**Key results / use cases.** Evaluated via **linear probes** on frozen embeddings (a fixed representation, task-specific linear head on top), MediaFM beats SeqCLIP, Google VertexAI multimodal embeddings, and TwelveLabs Marengo 2.7 on *all* tasks: **ad relevancy** (retrieval-stage candidate identification), **clip popularity ranking** (predicting relative CTR, by Kendall's $\tau$), **clip tone** (100 categories), **clip genre** (11 genres), and **clip retrieval** ("clip-worthy" classification, ~15% gain per improvement). Gains are largest on tasks needing **narrative/context understanding**. Ablations show the dominant driver is **contextualization** (the Transformer over the shot sequence), not merely concatenating modalities — "embedding in context" (extracting a clip's embedding from within its surrounding episode) beats embedding the clip alone.

**Trade-offs / limitations.** (1) **Embeddings over generative outputs** — chosen for a clean, reusable abstraction (generate once, consume across services) at the cost of end-to-end task optimization; outputs feed human/team decisions rather than fully end-to-end pipelines, and many uses are still mid-deployment. (2) **Frozen upstream encoders** — SeqCLIP/wav2vec2/text-embedding-3-large are not jointly trained, so quality is upper-bounded by them. (3) **Missing modalities** — timed text is often absent (silent shots), handled by zero-padding. (4) **Sequence-length limits** broke title-level evaluation for some external baselines (API video-length caps). (5) Next steps explicitly look to pretrained multimodal LLMs (e.g. Qwen3-Omni) where fusion is already learned — implying current per-modality fusion is a stopgap.

**Connection to this chapter.** MediaFM is, at production scale, the **content-similarity model of §6** ("Clustering-Based Retrieval"): both map a new item's raw image/video/audio/text to a feature vector *using only content, no user interaction*, which is precisely the signal a fresh item still has (§1) when it lacks statistical features and a learned ID embedding. The blog frames MediaFM as the **"critical backbone for effective cold start of newly launching titles."** A new Netflix title has no watch history, so collaborative / two-tower signals are unavailable (§4) — but MediaFM yields a rich embedding immediately at launch. These embeddings are exactly what feeds the §6 pipeline: k-means **clustering** into a cluster index for clustering-retrieval, cosine-similarity **content matching** (the §5.1 Fix 2 "borrow similar high-exposure items' ID embeddings" trick), and candidate-set **retrieval** (their ad-relevancy use case). The lineage is direct — §6's CNN+BERT+FC + triplet loss → CLIP-style pretraining → MediaFM's tri-modal Transformer with masked-shot self-supervision. Two upgrades worth flagging in an interview: MediaFM adds **audio** (§6 used only image+text) and replaces triplet-supervised similarity with **self-supervised masked modeling** (no curated positive/negative pairs, sidestepping §6's hard "choosing positives" problem), while the `[GLOBAL]` token injects title-level context analogous to a category/keyword prior (§5.2).

## Pinterest in Practice (2024–2026)

Two recent Pinterest systems map directly onto this chapter's cold-start levers — a content-only multimodal embedding that gives fresh items distribution without engagement history (§6), and generative content enhancement that strengthens weak creatives before they ever compete (§1's "content features" are *all* a new item has).

### PinCLIP — content-only multimodal embedding with a collaborative-structure injection

> Beal, J., Kim, E., Rao, J., Wu, R., Kislyuk, D., & Rosenberg, C. (2026). *PinCLIP: Large-scale foundational multimodal representation at Pinterest*. arXiv. https://arxiv.org/abs/2603.03544

PinCLIP is a foundational fused image+text embedding (hybrid Funnel-ViT image tower + multilingual SigLIP text tower, pooled to one $\ell_2$-normalized Pin vector) consumed by dozens of downstream retrieval and ranking models across Homefeed, Search, Related Pins, and Ads. It is trained with two sigmoid (pairwise) contrastive objectives: a standard **Image-to-Text** alignment plus the key novelty, a **Pin-to-Pin neighbor-alignment loss** that pulls together Pins that are graph neighbors on the Pin-Board bipartite graph (an edge = a user saved that Pin to that Board), with neighbors mined by offline random walks — this injects collaborative/engagement structure into what is otherwise a pure-content embedding. Because the vector is computed from content alone, a freshly published Pin gets a high-quality representation *immediately, with zero interaction history*, which is exactly the cold-start property of §6's content-similarity model — just scaled up and made multimodal. The payoff is concentrated on fresh content: online A/B as a ranking feature lifted **fresh-Pin (<28 days) repins by +5.02% on Homefeed, +14.15% on Related Pins, and +15.35% on Search**, far above the organic-surface gains, and it doubles as an ANN candidate generator (one embedding serving both retrieval and ranking).

> **Callout.** PinCLIP is the §6/§5.1 idea in production: content embeddings let fresh items inherit distribution without engagement history. The neighbor (Pin-to-Pin) loss is the interview-worthy twist — it bridges pure content semantics and collaborative signal, addressing why a content-only embedding under-serves Related Pins/Search.

### Pinterest Canvas — generative content enhancement as a cold-start lever

> Wang, Y., Tzeng, E., Shiau, R., Yang, J., Kislyuk, D., & Rosenberg, C. (2026). *Pinterest Canvas: Large-scale image generation at Pinterest*. arXiv. https://arxiv.org/abs/2603.06453

Canvas is Pinterest's in-house image-generation/editing system (FLUX.1 Kontext DiT backbone, flow matching, with a foundational base model rapidly fine-tuned into task-specific variants), shipped in the Performance+ ads suite. Its first production use cases *enhance* weak creatives: **background outpainting** (turning plain white-background catalog shots into engaging lifestyle scenes) and **aspect-ratio outpainting** (extending images taller, which performs better in Pinterest's columnar grid), with a non-generative cut-and-paste safeguard guaranteeing the product is never altered. This is content enrichment as a cold-start lever — a new ad or Pin whose only assets are its content features (§1) starts with a stronger, more engaging creative rather than a bare catalog image, lifting its odds in early exploratory exposure. Online A/B showed **background-enhanced ads +18.0% CTR and aspect-ratio-enhanced ads +12.5% CTR** (offline overall no-defect rate 47.2%, beating Nano Banana, FLUX.1 Kontext, and GPT-Image).

> **Callout.** Most of this chapter improves *how* a new item is matched and exposed; Canvas improves *the item itself*. Strengthening the only signal a new item has — its content — is an orthogonal cold-start lever, especially where the creative (not the targeting) is the bottleneck.
