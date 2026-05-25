# Part 02 — Retrieval (召回)

Recommendation systems course notes — Part 02: Candidate Retrieval.
This is the largest part of the course. Retrieval is the first stage of the recommendation pipeline: from a corpus of hundreds of millions of items, each of several dozen retrieval channels (召回通道) quickly pulls back a few hundred candidates, totaling a few thousand items, which are then passed to ranking (排序). The goal of retrieval is **speed and recall**: find items the user *might* be interested in, cheaply, so ranking can do fine-grained scoring later.

---

## Table of Contents

1. [Item-Based Collaborative Filtering (ItemCF, 基于物品的协同过滤)](#1-item-based-collaborative-filtering-itemcf)
2. [Swing Model (Swing 模型)](#2-swing-model)
3. [User-Based Collaborative Filtering (UserCF, 基于用户的协同过滤)](#3-user-based-collaborative-filtering-usercf)
4. [Discrete Feature Processing (离散特征处理): One-Hot & Embedding](#4-discrete-feature-processing-one-hot--embedding)
5. [Matrix Completion (矩阵补充) & Approximate Nearest Neighbor (ANN) Search](#5-matrix-completion--approximate-nearest-neighbor-ann-search)
6. [Two-Tower Model (双塔模型): Architecture & Training Methods](#6-two-tower-model-architecture--training-methods)
7. [Two-Tower Model: Positive & Negative Samples (正负样本)](#7-two-tower-model-positive--negative-samples)
8. [Two-Tower Model: Online Serving & Model Updates (线上服务、模型更新)](#8-two-tower-model-online-serving--model-updates)
9. [Two-Tower Model + Self-Supervised Learning (自监督学习)](#9-two-tower-model--self-supervised-learning)
10. [Deep Retrieval (Deep Retrieval 召回)](#10-deep-retrieval)
11. [Other Retrieval Channels (其他召回通道): Geo, Author, Cache](#11-other-retrieval-channels-geo-author-cache)
12. [Exposure Filtering & Bloom Filter (曝光过滤 & Bloom Filter)](#12-exposure-filtering--bloom-filter)

---

## 1. Item-Based Collaborative Filtering (ItemCF)

### 1.1 Core Idea

> If a user likes item $i_1$, and item $i_1$ is similar to item $i_2$, then the user probably also likes item $i_2$.

Concrete example: "I like reading *The Smiling, Proud Wanderer* (《笑傲江湖》); *The Smiling, Proud Wanderer* is similar to *The Deer and the Cauldron* (《鹿鼎记》); I have not read *The Deer and the Cauldron* → recommend it to me."

The non-trivial part is **how the system knows two items are similar**. ItemCF infers similarity from collective user behavior (rather than from a knowledge graph or content):
- Users who watched item A also watched item B.
- Users who gave A a good review also gave B a good review.

### 1.2 Interest-Score Prediction

A user has interacted with $n$ items (clicked, liked, saved, shared, etc.). We quantify the user's interest in each interacted item $\text{item}_j$ as $\text{like}(\text{user}, \text{item}_j)$ (e.g., click/like/save/share each count for 1 point, so the max is 4).

To estimate the user's interest in a **candidate item** $\text{item}$:

$$
\text{score}(\text{user}, \text{item}) = \sum_{j} \text{like}(\text{user}, \text{item}_j) \times \text{sim}(\text{item}_j, \text{item}).
$$

Worked example: interest scores $(2,1,4,3)$ for the four interacted items, similarities $(0.1,0.4,0.2,0.6)$ to the candidate:

$$
2\times0.1 + 1\times0.4 + 4\times0.2 + 3\times0.6 = 3.2.
$$

### 1.3 Item Similarity

Intuition: **the more two items' audiences overlap, the more similar they are.** If readers of *Legend of the Condor Heroes* and *Return of the Condor Heroes* overlap heavily, the two books are similar.

Let:
- $\mathcal{W}_1$ = set of users who like item $i_1$.
- $\mathcal{W}_2$ = set of users who like item $i_2$.
- $\mathcal{V} = \mathcal{W}_1 \cap \mathcal{W}_2$ = users who like both.

**Binary (like/no-like) similarity** — represents each item as a sparse vector over users, similarity = cosine of the angle:

$$
\text{sim}(i_1, i_2) = \frac{|\mathcal{V}|}{\sqrt{|\mathcal{W}_1| \cdot |\mathcal{W}_2|}}.
$$

This lies in $[0,1]$ because $|\mathcal{V}| \le |\mathcal{W}_1|, |\mathcal{W}_2|$. It ignores the *degree* of liking.

**Weighted (cosine) similarity** — uses the interest scores $\text{like}(v, i)$:

$$
\text{sim}(i_1, i_2) = \frac{\sum_{v \in \mathcal{V}} \text{like}(v, i_1)\cdot \text{like}(v, i_2)}{\sqrt{\sum_{u_1 \in \mathcal{W}_1} \text{like}^2(u_1, i_1)} \cdot \sqrt{\sum_{u_2 \in \mathcal{W}_2} \text{like}^2(u_2, i_2)}}.
$$

This is exactly cosine similarity, where each item is a sparse vector whose components correspond to users.

### 1.4 The Full ItemCF Retrieval Pipeline

**Offline precomputation — build two indices:**

1. **"User → Item" index** (用户→物品索引): for each user, record the last-$n$ items they recently clicked/interacted with, as `(item ID, interest score)` pairs. Given a user ID, return the user's recent interest list (last-n).
2. **"Item → Item" index** (物品→物品索引): compute pairwise item similarities; for each item, store its top-$k$ most similar items as `(item ID, similarity)` pairs. Given an item ID, quickly find its $k$ most similar items.

**Online retrieval (线上召回):**

1. Given the user ID, use the "User→Item" index to find the user's recent last-$n$ items.
2. For each of those last-$n$ items, use the "Item→Item" index to retrieve its top-$k$ similar items.
3. Score the (at most $nk$) retrieved items with the formula in §1.2.
4. Return the top-100 highest-scoring items as the ItemCF channel output.

Typical scale: $n = 200$, $k = 10$, so $nk = 2000$ scored items → return top 100.

**Why indices?** They avoid enumerating all items. Indices push heavy work offline (large offline compute) so that online compute is small.

---

## 2. Swing Model

Swing differs from ItemCF in **exactly one place: the item-similarity formula**. Everything else (the index pipeline, the prediction formula) is identical.

### 2.1 Problem ItemCF Has

ItemCF says: if the overlap of two items' audiences is large, the items are similar. But what if the overlapping users come from **one small clique (小圈子)** — e.g., one WeChat group where everyone shared the same two unrelated posts (one about a skincare discount, one about layoffs at ByteDance)? Then the high overlap is spurious; the two items are not genuinely similar. ItemCF would wrongly judge them similar.

### 2.2 Swing's Fix: Down-Weight Cliques

Swing additionally checks whether the overlapping users form a small clique, and **down-weights** such users.

Let:
- $\mathcal{J}_1$ = set of items user $u_1$ likes; $\mathcal{J}_2$ = set of items user $u_2$ likes.
- **User overlap**: $\text{overlap}(u_1, u_2) = |\mathcal{J}_1 \cap \mathcal{J}_2|$.

If $u_1$ and $u_2$ have high overlap, they likely come from one small clique → reduce their weight.

Item similarity in Swing (where $\mathcal{V} = \mathcal{W}_1 \cap \mathcal{W}_2$ = users who like both items):

$$
\text{sim}(i_1, i_2) = \sum_{u_1 \in \mathcal{V}} \sum_{u_2 \in \mathcal{V}} \frac{1}{\alpha + \text{overlap}(u_1, u_2)}.
$$

$\alpha$ is a hyperparameter. The intuition: when a pair of users $(u_1, u_2)$ who both like the two items have **high** overlap (likely a clique), the term $\tfrac{1}{\alpha + \text{overlap}}$ is small, so their contribution to "these two items are similar" is suppressed.

### 2.3 Summary

- ItemCF: two items are similar if a high *fraction* of users overlap.
- Swing: same idea, but additionally penalize the case where the overlapping users are a small clique (because clique behavior is a weak signal of true item similarity).

---

## 3. User-Based Collaborative Filtering (UserCF)

### 3.1 Core Idea

> If user₁ is similar to user₂, and user₂ likes some item, then user₁ probably also likes that item.

Concrete: "There are many netizens with interests very similar to mine; one of them liked/shared a note; I haven't seen that note → recommend it to me."

How does the system find users similar to me?
- **Method 1**: they clicked/liked/saved/shared a heavily overlapping set of notes.
- **Method 2**: they follow a heavily overlapping set of authors.

### 3.2 Interest-Score Prediction

Let $\text{sim}(\text{user}, \text{user}_j)$ be the similarity between the target user and a similar user $j$, and $\text{like}(\text{user}_j, \text{item})$ be user $j$'s interest in the candidate item.

$$
\text{score}(\text{user}, \text{item}) = \sum_j \text{sim}(\text{user}, \text{user}_j) \times \text{like}(\text{user}_j, \text{item}).
$$

Worked example: similarities $(0.9, 0.7, 0.7, 0.4)$ to four similar users; their interest in the candidate $(0,1,3,0)$:

$$
0.9\times0 + 0.7\times1 + 0.7\times3 + 0.4\times0 = 2.8.
$$

### 3.3 User Similarity

Represent each user as a sparse vector whose components correspond to items. Similarity = cosine.

Let $\mathcal{J}_1$, $\mathcal{J}_2$ = sets of items liked by $u_1$, $u_2$; $I = \mathcal{J}_1 \cap \mathcal{J}_2$.

**Basic formula:**

$$
\text{sim}(u_1, u_2) = \frac{|I|}{\sqrt{|\mathcal{J}_1| \cdot |\mathcal{J}_2|}} = \frac{\sum_{l \in I} 1}{\sqrt{|\mathcal{J}_1| \cdot |\mathcal{J}_2|}}.
$$

Here every shared item — popular or niche — has weight 1.

### 3.4 Down-Weighting Popular Items (降低热门物品权重)

Problem: if two users both liked a **mega-popular** item (e.g., *Harry Potter*), that overlap says little about their similarity — almost everyone likes popular items. Niche shared interests are far more informative.

Fix: weight each shared item $l$ by $\frac{1}{\log(1 + n_l)}$, where $n_l$ is the number of users who like item $l$ (its popularity):

$$
\text{sim}(u_1, u_2) = \frac{\sum_{l \in I} \dfrac{1}{\log(1 + n_l)}}{\sqrt{|\mathcal{J}_1| \cdot |\mathcal{J}_2|}}.
$$

A popular item (large $n_l$) contributes less to the user-similarity score.

### 3.5 The Full UserCF Retrieval Pipeline

**Offline — build two indices:**

1. **"User → Item" index**: each user's recent last-$n$ interacted items with interest scores (same as ItemCF).
2. **"User → User" index**: for each user, store the top-$k$ most similar users with similarity scores.

**Online retrieval:**

1. Given the user ID, use "User→User" index to find top-$k$ similar users.
2. For each similar user, use "User→Item" index to find their recent last-$n$ items.
3. Score the (up to $nk$) retrieved items with the formula in §3.2.
4. Return the top-100 highest-scoring items.

---

## 4. Discrete Feature Processing (One-Hot & Embedding)

Discrete (categorical) features (离散特征) examples:
- Gender: 2 categories.
- Nationality: ~200 categories.
- English words: tens of thousands of common words.
- Item ID: hundreds of millions of notes (小红书).
- User ID: hundreds of millions of users.

Two processing steps:

1. **Build a dictionary** (建立字典): map each category to an integer index. e.g., China→1, USA→2, India→3.
2. **Vectorize** (向量化): map the index to a vector, via either one-hot encoding or embedding.

### 4.1 One-Hot Encoding

Map an index to a high-dimensional **sparse** vector. The dimension equals the number of categories.

- Gender (reserve 0 for "unknown"): unknown→$[0,0]$, male→$[1,0]$, female→$[0,1]$ (2-dim).
- Nationality (200 categories): unknown→$[0,\dots,0]$, China→$[1,0,\dots]$, USA→$[0,1,0,\dots]$, India→$[0,0,1,0,\dots]$ (200-dim).

**Limitation**: when the number of categories is huge (tens of thousands of words, hundreds of millions of item IDs), the one-hot dimension is enormous and impractical. → **For large category counts, do not use one-hot; use embedding.**

### 4.2 Embedding (嵌入)

Map an index to a low-dimensional **dense** vector.

- **Parameter count** = (vector dimension) × (number of categories). E.g., 200 nationalities × 4-dim embeddings = 800 parameters. E.g., 10,000 movies × 16-dim = 160,000 parameters.
- **Implementation**: TensorFlow / PyTorch provide an embedding layer. Parameters are stored as a matrix of size (vector dimension) × (number of categories). Input = index; output = the corresponding **column** of the parameter matrix (e.g., "USA" with index 2 → column 2).
- Learned embeddings cluster semantically related items (e.g., animated films cluster together; spy films cluster together).

**Key identity**:

$$
\text{Embedding} = (\text{parameter matrix}) \times (\text{one-hot vector}).
$$

Multiplying the parameter matrix by a one-hot vector simply selects the column indexed by the "1" — i.e., embedding lookup is a matrix-vector product with a one-hot, but implemented as a fast lookup.

### 4.3 Summary

One-hot and embedding are the two ways to vectorize discrete features. When the category count is large, use embedding (word embeddings, user-ID embeddings, item-ID embeddings).

---

## 5. Matrix Completion & Approximate Nearest Neighbor (ANN) Search

### 5.1 Matrix Completion Model

A simple model that is the conceptual predecessor of the two-tower model.

Architecture:
- User ID → Embedding Layer → user vector $\mathbf{a}$.
- Item ID → a **separate** Embedding Layer (**parameters not shared**, 不共享参数) → item vector $\mathbf{b}$.
- Predicted interest = inner product $\langle \mathbf{a}, \mathbf{b} \rangle$.

Notation: let $\mathbf{A}$ = user embedding matrix (column $u$ = vector $\mathbf{a}_u$ for user $u$); $\mathbf{B}$ = item embedding matrix (column $i$ = vector $\mathbf{b}_i$ for item $i$). The inner product $\langle \mathbf{a}_u, \mathbf{b}_i \rangle$ is the predicted interest of user $u$ in item $i$.

### 5.2 Training

Dataset $\Omega = \lbrace(u, i, y)\rbrace$ of (user ID, item ID, interest score) triples, where scores are recorded by the system:
- Exposed but no click → 0 points.
- Click, like, save, share → +1 each.
- Score range: min 0, max 4.

Optimization (least-squares / regression):

$$
\min_{\mathbf{A}, \mathbf{B}} \sum_{(u,i,y) \in \Omega} \left( y - \langle \mathbf{a}_u, \mathbf{b}_i \rangle \right)^2.
$$

The name "matrix completion" comes from filling in the unobserved (gray) cells of the user×item interest matrix: each row = a user, each column = an item; green cells = observed (exposed) entries, gray = unexposed. Training learns $\mathbf{A}, \mathbf{B}$ so their product reconstructs the observed entries.

### 5.3 Why Matrix Completion Works Poorly in Practice

Three flaws — the two-tower model is essentially an upgraded matrix completion that fixes all three:

1. **Uses only ID embeddings**; ignores item attributes (category, keywords, geo, author) and user attributes (gender, age, geo, interested categories).
2. **Wrong negative-sample selection**: it uses exposed-but-not-clicked items as negatives (wrong for retrieval), and treats samples as (user, item) pairs where positive = exposed+clicked, negative = exposed+no-click. (Correct retrieval negatives differ — see §7.)
3. **Bad training method**: inner product $\langle \mathbf{a}, \mathbf{b}\rangle$ is inferior to **cosine similarity**; squared (regression) loss is inferior to **cross-entropy (classification) loss**.

### 5.4 Online Serving with Matrix Completion

1. Store columns of $\mathbf{A}$ in a key-value table: key = user ID, value = user vector $\mathbf{a}$. Given a user ID, return their embedding.
2. **Nearest-neighbor lookup**: find the $k$ items most likely to interest the user. Item $i$'s embedding is $\mathbf{b}_i$; predicted interest = $\langle \mathbf{a}, \mathbf{b}_i \rangle$; return the top-$k$ items by inner product.

(Matrix $\mathbf{B}$ storage and indexing are more complex.)

**Problem**: brute-force enumeration over all items has time complexity proportional to the number of items — too slow for billions of items. → use **Approximate Nearest Neighbor (ANN) search**.

### 5.5 Approximate Nearest Neighbor (ANN) Search

Treat the user vector $\mathbf{a}$ as a query; find the item $i$ maximizing $\langle \mathbf{a}, \mathbf{b}_i \rangle$ (or cosine, or minimizing L2 distance) **without enumerating everything**.

- **Vector databases / libraries** supporting ANN: Milvus, Faiss, HnswLib, etc.
- **Distance/similarity measures** supported:
  - Euclidean (L2) distance — minimize.
  - Inner product — maximize.
  - Cosine similarity — maximize (angle between vectors smallest).
- **Idea (intuition from the slides)**: pre-partition the vector space into regions (e.g., angular sectors, each with a representative direction vector). Index every item by which region it belongs to. At query time, find the region whose representative is closest to query $\mathbf{a}$, then only search items in (and near) that region. This is **approximate** — it may miss the true nearest neighbor occasionally — but is dramatically faster than brute force.

---

## 6. Two-Tower Model: Architecture & Training Methods

### 6.1 Architecture

The two-tower model (a.k.a. DSSM-style) is matrix completion upgraded to use rich features.

**User tower (用户塔):** takes user ID, discrete features, and continuous features.
- User ID → Embedding Layer.
- Discrete features → Embedding Layers.
- Continuous features → normalization, bucketing (分桶), etc.
- Concatenate all of the above → a neural network → **user representation vector $\mathbf{a}$**.

**Item tower (物品塔):** symmetric — takes item ID, discrete features, continuous features; concatenate → neural network → **item representation vector $\mathbf{b}$**.

**Output:** cosine similarity of the two vectors:

$$
\cos(\mathbf{a}, \mathbf{b}) = \frac{\langle \mathbf{a}, \mathbf{b}\rangle}{\Vert\mathbf{a}\Vert_2 \cdot \Vert\mathbf{b}\Vert_2}.
$$

This cosine is the estimated user interest. (Cosine > raw inner product, per the matrix-completion lesson.)

### 6.2 A Structure That Is NOT a Two-Tower (and why it can't be used for retrieval)

If instead you **concatenate** $\mathbf{a}$ and $\mathbf{b}$ and feed them jointly into a neural network that outputs the interest prediction, the model is no longer separable into two independent towers. Such a model is fine for **ranking** but **cannot be used for retrieval**, because retrieval requires precomputing item vectors offline and doing ANN — which is only possible if the user and item sides are computed independently and combined by a simple cosine/inner product at the very end.

### 6.3 Three Training Paradigms

| Paradigm | Each training sample contains | Loss |
|---|---|---|
| **Pointwise** | one user + one item (positive or negative) | binary classification |
| **Pairwise** | one user + one positive + one negative [Huang et al., KDD 2020] | triplet loss |
| **Listwise** | one user + one positive + multiple negatives [Yi et al., RecSys 2019] | softmax + cross-entropy |

#### Pointwise Training

Treat retrieval as **binary classification**.
- For positive samples, push $\cos(\mathbf{a}, \mathbf{b}) \to +1$.
- For negative samples, push $\cos(\mathbf{a}, \mathbf{b}) \to -1$.
- Control the positive:negative ratio to about $1:2$ or $1:3$.

#### Pairwise Training

A sample = (user $\mathbf{a}$, positive item $\mathbf{b}^+$, negative item $\mathbf{b}^-$). The positive tower and negative tower **share parameters** (共享参数). The model computes $\cos(\mathbf{a}, \mathbf{b}^+)$ and $\cos(\mathbf{a}, \mathbf{b}^-)$.

**Goal:** encourage $\cos(\mathbf{a}, \mathbf{b}^+)$ to exceed $\cos(\mathbf{a}, \mathbf{b}^-)$ by at least a margin $m$.
- If $\cos(\mathbf{a}, \mathbf{b}^+) > \cos(\mathbf{a}, \mathbf{b}^-) + m$, no loss.
- Otherwise, loss $= \cos(\mathbf{a}, \mathbf{b}^-) + m - \cos(\mathbf{a}, \mathbf{b}^+)$.

**Triplet hinge loss:**

$$
L(\mathbf{a}, \mathbf{b}^+, \mathbf{b}^-) = \max\lbrace0, \cos(\mathbf{a}, \mathbf{b}^-) + m - \cos(\mathbf{a}, \mathbf{b}^+)\rbrace.
$$

**Triplet logistic loss** ($\sigma$ a scaling hyperparameter):

$$
L(\mathbf{a}, \mathbf{b}^+, \mathbf{b}^-) = \log\Big(1 + \exp\big[\sigma \cdot (\cos(\mathbf{a}, \mathbf{b}^-) - \cos(\mathbf{a}, \mathbf{b}^+))\big]\Big).
$$

#### Listwise Training

A sample = one user $\mathbf{a}$, one positive $\mathbf{b}^+$, and $n$ negatives $\mathbf{b}_1^-, \dots, \mathbf{b}_n^-$.
- Encourage $\cos(\mathbf{a}, \mathbf{b}^+)$ large.
- Encourage all $\cos(\mathbf{a}, \mathbf{b}_i^-)$ small.

Compute cosine of the positive and all negatives, pass through **softmax** to get a probability vector $\mathbf{s}$, and compare against the one-hot label $\mathbf{y}$ (1 at the positive position, 0 elsewhere). Loss = cross-entropy, which simplifies to:

$$
\text{CrossEntropyLoss}(\mathbf{y}, \mathbf{s}) = -\log s^+ = -\log \frac{\exp(\cos(\mathbf{a}, \mathbf{b}^+))}{\sum \exp(\cos(\mathbf{a}, \mathbf{b}_\bullet))}.
$$

### 6.4 Summary

- User tower and item tower each output a vector; their cosine is the predicted interest.
- Three training modes: Pointwise (1 user + 1 item), Pairwise (1 user + 1 pos + 1 neg), Listwise (1 user + 1 pos + many negs).

---

## 7. Two-Tower Model: Positive & Negative Samples

> Choosing the right positive/negative samples matters more than improving the model architecture.

### 7.1 Positive Samples (正样本)

Positive = (user, item) pairs that were **exposed and clicked** (用户对物品感兴趣).

**Problem (80/20 rule, 二八法则):** a few items get most of the clicks, so positives are dominated by **popular** items. Training on too many popular positives is unfair to niche items and makes "the popular get more popular, the niche get more niche."

**Fix — re-balance by item popularity:**
- **Up-sampling (过采样)** niche items: let a niche sample appear multiple times.
- **Down-sampling (降采样)** popular items: drop some samples; drop probability is positively correlated with the item's click count.

### 7.2 Where Negatives Come From in the Pipeline

The pipeline: corpus → retrieval (a few thousand) → coarse+fine ranking (粗排/精排, → a few hundred) → re-ranking (重排, → ~a few dozen exposed). Candidate negatives at each stage:
- **Not retrieved** (没有被召回).
- **Retrieved but cut by ranking** (被召回, 但被粗排/精排淘汰).
- **Exposed but not clicked** (被曝光, 但未点击).

### 7.3 Simple Negatives (简单负样本)

Items the user is (almost certainly) not interested in.

**(a) Whole-corpus sampling:** Since only a few thousand of billions of items are retrieved, "not retrieved" ≈ "the whole corpus." So just sample from the entire corpus.
- **Uniform sampling** is unfair to niche items: positives are mostly popular, so uniform negatives would be mostly niche → over-suppresses niche items.
- **Non-uniform sampling** (to suppress popular items): sample probability positively correlated with popularity. A common empirical rule:

$$
\text{sampling probability} \propto (\text{click count})^{0.75}.
$$

**(b) Batch negatives (Batch内负样本):** Within a training batch of $n$ positive (user, item) pairs, pair user $i$ with item $j$ ($j \ne i$) to form negatives. A batch yields $n(n-1)$ negatives — all "simple."

**Sampling-bias issue with batch negatives [Yi et al., RecSys 2019]:** an item appears in a batch with probability $\propto$ its click count, so as a batch-negative its probability is $\propto (\text{click count})^{1}$ — but it *should* be $\propto (\text{click count})^{0.75}$. Hence popular items are over-sampled as negatives → over-suppressed.

**Bias correction:** let $p_i \propto$ item $i$'s click count = probability it is sampled. During **training**, replace the score $\cos(\mathbf{a}, \mathbf{b}_i)$ with:

$$
\cos(\mathbf{a}, \mathbf{b}_i) - \log p_i.
$$

This counters the over-suppression of popular items. **At online serving time, do NOT subtract $\log p_i$ — use the plain $\cos(\mathbf{a}, \mathbf{b}_i)$.**

### 7.4 Hard Negatives (困难负样本)

Items that the user has *some* affinity for, but not enough:
- **Cut by coarse ranking (粗排)** — moderately hard. (Were retrieved → some relevance, but the score wasn't high enough.)
- **Ranked low by fine ranking (精排)** — very hard. (Passed coarse ranking, so fairly relevant, but ranked near the bottom.)

In the binary-classification view: whole-corpus negatives are easy to classify correctly; coarse-ranking-cut negatives are easy to misclassify; fine-ranking-low negatives are even easier to misclassify (most similar to positives).

**Training data mix (工业界做法):** mix simple + hard negatives, e.g., 50% from the whole corpus (simple), 50% from items cut by ranking (hard).

### 7.5 A Common Mistake (常见的错误)

**Do NOT use exposed-but-not-clicked items as retrieval negatives.** This is a well-known industry trap; it degrades the two-tower model.

**Reasoning — the principle for choosing negatives:**
- **Whole-corpus (easy):** the vast majority are items the user fundamentally has no interest in. → valid retrieval negative.
- **Cut by ranking (hard):** the user *might* be interested but not interested enough. → valid retrieval negative.
- **Exposed but not clicked (useless for retrieval):** an item that survived fine ranking and got exposed is already a strong match to the user; not clicking ≠ not interested (the user may have clicked something else, or just happened not to click). These can even serve as retrieval *positives*. → **must not be used as retrieval negatives.**

The key distinction:
- **Retrieval's job:** distinguish "not interested" from "possibly interested."
- **Ranking's job:** distinguish "somewhat interested" from "very interested." — exposed-but-not-clicked items ARE valid **ranking** negatives.

### 7.6 Summary

- **Positive:** exposed + clicked.
- **Simple negatives:** whole corpus (non-uniform sampling) + batch negatives.
- **Hard negatives:** retrieved but cut by ranking.
- **Wrong:** exposed-but-not-clicked as retrieval negatives (valid only for ranking).

---

## 8. Two-Tower Model: Online Serving & Model Updates

### 8.1 Online Retrieval

**Offline storage:** After training, run the **item tower** over every item to compute its vector $\mathbf{b}$. Store all (hundreds of millions of) item vectors `<vector b, item ID>` in a **vector database** (Milvus / Faiss / HnswLib), and build an index for ANN.

**Online retrieval:**
1. Given user ID + user profile (画像), run the **user tower online** to compute user vector $\mathbf{a}$.
2. Use $\mathbf{a}$ as the ANN query against the vector database; return the top-$k$ items by cosine similarity (return their item IDs).

**Why precompute item vectors $\mathbf{b}$ but compute user vector $\mathbf{a}$ online?**
- Each retrieval uses **one** user vector but **hundreds of millions** of item vectors — computing all item vectors online would be far too expensive.
- **User interest changes dynamically; item features are relatively stable.** You *could* precompute user vectors too, but that would hurt recommendation quality (the user vector would be stale).

### 8.2 Full Update vs. Incremental Update (全量更新 vs. 增量更新)

**Full update (全量更新):** "Tonight, train on all of yesterday's data."
- Start from **yesterday's model parameters** (not random init).
- Random-shuffle one day's data, train **1 epoch** (each example seen once).
- Publish the new **user-tower neural network** and the new **item vectors** for online retrieval.
- Has relatively low requirements on the data stream / system.

**Incremental update (增量更新):** online learning to track user interest in real time.
- User interest changes continuously.
- Stream online data in real time, process it (e.g., generate TFRecord files).
- Do **online learning**, updating **only the ID embedding parameters** (the full-connection / neural-network layers are **frozen / locked**, 锁住全连接层).
- Publish the updated user ID embeddings so the user tower can compute up-to-date user vectors online.

**Can we do only incremental updates (skip full updates)?** No — full updates are better:
- Hour-level data is biased; minute-level data even more so.
- Full update = random-shuffle a full day + 1 epoch.
- Incremental update = process data in chronological (early→late) order, 1 epoch.
- **Random shuffling beats sequential ordering; full training beats incremental training.**

**Real systems combine both:** each day a full update (built on yesterday's full model + yesterday's data); throughout the day, incremental updates every few tens of minutes publishing the latest user ID embeddings.

> Note (chronology subtlety, from the slides): the next full update must be built on the previous **full** model (using that day's data), **not** chained off the incremental updates — the incremental chain is discarded and restarted from each fresh full update.

### 8.3 Summary

- **Train:** two towers output vectors; cosine = predicted interest; pointwise/pairwise/listwise; positives = clicks; negatives = whole-corpus (simple) + ranking-cut (hard).
- **Retrieve:** store item vectors in a vector DB for ANN; compute the user vector online; query for top-$k$ by cosine.
- **Update:** full update nightly (whole network, 1-epoch SGD) + incremental online learning (only ID embeddings, FC layers frozen), combined; republish user ID embeddings every few tens of minutes.

---

## 9. Two-Tower Model + Self-Supervised Learning

Reference: Tiansheng Yao et al., *Self-supervised Learning for Large-scale Item Recommendations*, CIKM 2021.

### 9.1 Motivation

The recommendation system has a severe **head effect (头部效应)**: a few items get most of the clicks; most items have low click counts. High-click items' representations are learned well; **long-tail (长尾) items' representations are learned poorly.** Self-supervised learning does **data augmentation** to learn long-tail item vectors better.

### 9.2 Recap: Listwise Training of the Main Two-Tower Loss

A batch has $n$ positive (clicked) pairs $(\mathbf{a}_1, \mathbf{b}_1), \dots, (\mathbf{a}_n, \mathbf{b}_n)$. Negatives are the cross pairs $(\mathbf{a}_i, \mathbf{b}_j)$ for $i \ne j$. Encourage $\cos(\mathbf{a}_i, \mathbf{b}_i)$ large and $\cos(\mathbf{a}_i, \mathbf{b}_j)$ small.

With the sampling-bias correction (subtract $\log p_j$ inside the softmax), the main loss for user $i$ is:

$$
L_{\text{main}}[i] = -\log\left( \frac{\exp\big(\cos(\mathbf{a}_i, \mathbf{b}_i) - \log p_i\big)}{\sum_{j=1}^{n} \exp\big(\cos(\mathbf{a}_i, \mathbf{b}_j) - \log p_j\big)} \right),
$$

and we minimize $\frac{1}{n}\sum_{i=1}^{n} L_{\text{main}}[i]$.

### 9.3 The Self-Supervised Idea

For each item, create **two different augmented views** by feature transformations, pass both through the item tower (towers **share parameters**), and require:
- The two views of the **same** item $i$ ($\mathbf{b}_i'$ and $\mathbf{b}_i''$) have **high** cosine similarity.
- Views of **different** items $i, j$ ($\mathbf{b}_i'$ and $\mathbf{b}_j''$) have **low** cosine similarity.

This is contrastive learning over items — no user clicks needed, hence "self-supervised."

### 9.4 Four Feature Transformations (特征变换)

1. **Random Mask:** randomly pick some discrete features and mask them. e.g., category $\mathcal{U}=\lbrace\text{digital}, \text{photography}\rbrace$ → masked $\mathcal{U}'=\lbrace\text{default}\rbrace$.

2. **Dropout (multi-valued discrete features only):** an item can have multiple categories (a multi-valued discrete feature). Randomly drop 50% of the values. e.g., $\mathcal{U}=\lbrace\text{beauty}, \text{photography}\rbrace$ → $\mathcal{U}'=\lbrace\text{beauty}\rbrace$.

3. **Complementary features (互补特征):** suppose 4 features {ID, category, keyword, city}. Randomly split into two groups, e.g., {ID, keyword} and {category, city}. View 1 = {ID, default, keyword, default}; view 2 = {default, category, default, city}. Encourage the two resulting vectors to be similar.

4. **Mask a group of correlated features:** mask features that tend to co-occur, measured by **mutual information (互信息)**.
   - $p(u)$ = probability a feature takes value $u$ (e.g., $p(\text{male})=20$%, $p(\text{female})=30$%, $p(\text{neutral})=50$%).
   - $p(u,v)$ = joint probability that one feature = $u$ and another = $v$ (e.g., $p(\text{female}, \text{beauty})=3$% vs $p(\text{female}, \text{digital})=0.1$%).
   - **Mutual information** between feature sets $\mathcal{U}, \mathcal{V}$:
     $$
     MI(\mathcal{U}, \mathcal{V}) = \sum_{u \in \mathcal{U}} \sum_{v \in \mathcal{V}} p(u,v) \cdot \log \frac{p(u,v)}{p(u)\cdot p(v)}.
     $$
   - **Procedure:** with $k$ features, offline-compute the $k\times k$ MI matrix. Pick a random seed feature, find the $k/2$ features most correlated with it, mask the seed + those $k/2$, keep the other $k/2$.
   - **Pro:** outperforms random mask / dropout / complementary features. **Con:** complex, hard to implement, hard to maintain.

### 9.5 Self-Supervised Loss

Uniformly sample $m$ items from the **whole corpus** as a batch (uniform, *not* click-weighted). Apply two transformations → two vector groups $\mathbf{b}_1', \dots, \mathbf{b}_m'$ and $\mathbf{b}_1'', \dots, \mathbf{b}_m''$. Self-supervised loss for item $i$ (softmax + cross-entropy, positive = same item's other view):

$$
L_{\text{self}}[i] = -\log\left( \frac{\exp\big(\cos(\mathbf{b}_i', \mathbf{b}_i'')\big)}{\sum_{j=1}^{m} \exp\big(\cos(\mathbf{b}_i', \mathbf{b}_j'')\big)} \right).
$$

### 9.6 Combined Training

Sample $n$ click pairs (for $L_{\text{main}}$) and $m$ uniform items (for $L_{\text{self}}$), and minimize the combined objective:

$$
\frac{1}{n}\sum_{i=1}^{n} L_{\text{main}}[i] + \alpha \cdot \frac{1}{m}\sum_{j=1}^{m} L_{\text{self}}[j],
$$

where $\alpha$ weights the self-supervised term. **Note the two batches differ:** $L_{\text{main}}$ uses click data (popularity-weighted); $L_{\text{self}}$ uses uniformly-sampled items (so long-tail items get equal representation).

### 9.7 Summary

The two-tower model learns low-exposure item vectors poorly. Self-supervised learning applies random feature transformations to each item, encouraging same-item views to be similar and different-item views dissimilar. Experimental result: recommendations for low-exposure and new items become more accurate.

---

## 10. Deep Retrieval

References: Weihao Gao et al., *Learning A Retrievable Structure for Large-Scale Recommendations*, CIKM 2021; analogous to Alibaba's TDM (Han Zhu et al., *Learning Tree-based Deep Model for Recommender Systems*, KDD 2018).

### 10.1 Core Idea

- Classic two-tower: represent user & item as vectors, do ANN at serving.
- **Deep Retrieval:** represent each **item as a set of paths (路径)**; at serving, find the paths the user best matches, then map paths back to items.

### 10.2 Item ↔ Path Structure & Indices

Imagine a structure of `depth = 3` layers (L1, L2, L3), each of `width = K` nodes. A **path** is one node per layer, e.g., $[2, 4, 1]$. One item can map to **multiple** paths, e.g., $\lbrace[2,4,1], [4,1,1]\rbrace$.

Two indices (analogous to ItemCF's two indices):
- **item → List⟨path⟩**: each item maps to several paths (a path is 3 nodes $[a,b,c]$).
- **path → List⟨item⟩**: each path maps to several items.

### 10.3 Prediction Model: User's Interest in a Path

A neural network predicts, autoregressively, the user's interest in a path $[a,b,c]$ given user features $\mathbf{x}$:
- $p_1(a \mid \mathbf{x})$ — interest in the L1 node $a$ (network → softmax over $K$ nodes → pick $a$).
- $p_2(b \mid a; \mathbf{x})$ — interest in L2 node $b$ given $a$. Input = concatenation of $\mathbf{x}$ and the embedding $\text{emb}(a)$ → network → softmax.
- $p_3(c \mid a,b; \mathbf{x})$ — interest in L3 node $c$ given $a,b$. Input = $\mathbf{x} \oplus \text{emb}(a) \oplus \text{emb}(b)$.

The interest in the whole path:

$$
p(a,b,c \mid \mathbf{x}) = p_1(a \mid \mathbf{x}) \cdot p_2(b \mid a; \mathbf{x}) \cdot p_3(c \mid a,b; \mathbf{x}).
$$

### 10.4 Online Retrieval: User → Path → Item

1. Given user features, use the neural network + **beam search** to retrieve a batch of top paths.
2. Use the **path → List⟨item⟩** index to retrieve items from those paths.
3. Score and rank the items, selecting a subset.

### 10.5 Beam Search

With 3 layers of $K$ nodes each, there are $K^3$ paths — too many to score exhaustively. Beam search reduces this, controlled by a **beam size** hyperparameter.

**Beam size = 1 (greedy):**
- L1: compute $p_1(\cdot \mid \mathbf{x})$ for all $K$ nodes; keep the single best, say node 5.
- L2: compute $p_2(\cdot \mid 5; \mathbf{x})$ for all $K$ nodes; keep the best, say node 4.
- L3: compute $p_3(\cdot \mid 5,4; \mathbf{x})$; keep the best, say node 1.
- Selected path = $[5,4,1]$. (Greedy is not guaranteed to find the globally optimal path $\arg\max_{a,b,c} p(a,b,c\mid\mathbf{x})$.)

**Beam size = 4:**
- L1: keep the top-4 nodes by $p_1$.
- L2: for each of the 4 kept nodes $a$, compute $p_1(a\mid\mathbf{x})\cdot p_2(b\mid a;\mathbf{x})$ for all $K$ nodes $b$ → $4K$ scores (each = a length-2 path); keep the top-4 paths.
- L3: similarly expand each of the 4 length-2 paths over $K$ L3 nodes → $4K$ scores; keep the top-4 full paths.
- Output: 4 paths. Larger beam size → better paths, more compute.

### 10.6 Training: Jointly Learn the Network AND the Item Representations

Training data: (1) the **item → path** index, (2) users' clicked items. Positive (user, item): $\text{click}(\text{user}, \text{item}) = 1$.

**(a) Learn neural-network parameters.** An item is represented by $J$ paths $[a_1,b_1,c_1], \dots, [a_J, b_J, c_J]$. If the user clicked the item, the user is interested in all $J$ of its paths. So we increase $\sum_{j=1}^{J} p(a_j, b_j, c_j \mid \mathbf{x})$:

$$
\text{loss} = -\log\left( \sum_{j=1}^{J} p(a_j, b_j, c_j \mid \mathbf{x}) \right).
$$

**(b) Learn item representations (item → paths).** Define the user's interest in a path as $p(\text{path}\mid\text{user}) = p(a,b,c\mid\mathbf{x})$. The **relevance** of item and path:

$$
\text{score}(\text{item}, \text{path}) = \sum_{\text{user}} p(\text{path} \mid \text{user}) \times \text{click}(\text{user}, \text{item}),
$$

i.e., sum over users of (user's interest in the path) × (did the user click the item, 0/1). Select $J$ paths $\Pi = \lbrace\text{path}_1, \dots, \text{path}_J\rbrace$ as the item's representation.

- **Loss** (pick paths highly relevant to the item):
  $$
  \text{loss}(\text{item}, \Pi) = -\log\left( \sum_{j=1}^{J} \text{score}(\text{item}, \text{path}_j) \right).
  $$
- **Regularization** (avoid too many items piling onto one path):
  $$
  \text{reg}(\text{path}_j) = \big(\text{number of items on } \text{path}_j\big)^4.
  $$
- **Greedy path update:** holding the other paths $\lbrace\text{path}_i\rbrace_{i\ne l}$ fixed, pick a new $\text{path}_l$ from the unused paths:
  $$
  \text{path}_l \leftarrow \arg\min_{\text{path}_l}  \text{loss}(\text{item}, \Pi) + \alpha \cdot \text{reg}(\text{path}_l).
  $$
  This selects a path with high $\text{score}(\text{item}, \text{path}_l)$ that isn't overcrowded.

### 10.7 Summary

- Given user features $\mathbf{x}$, the neural network scores paths $p(\text{path}\mid\mathbf{x})$.
- Beam search finds the top-$s$ paths.
- The **path → List⟨item⟩** index returns $n$ items per path → $s \times n$ candidates → preliminary ranking → top results.
- Training **simultaneously** learns the user→path relationship (neural-network params) and the item→path relationship (item representations): if a user clicked an item, increase $\sum_j p(\text{path}_j \mid \mathbf{x})$; item-path relevance is propagated through "item ← user → path," with each item linked to $J$ highly-relevant paths and no path overcrowded.

---

## 11. Other Retrieval Channels (Geo, Author, Cache)

A production system uses dozens of retrieval channels in parallel. Beyond CF, two-tower, and Deep Retrieval, here are several simple but useful ones.

### 11.1 Geolocation Retrieval (地理位置召回)

**GeoHash retrieval:**
- Users may be interested in things happening nearby.
- **GeoHash** = an encoding of (longitude, latitude) that corresponds to a rectangular region on the map.
- Index: **GeoHash → list of high-quality notes** (sorted by time, newest first).
- By the user's GeoHash, retrieve the $k$ newest high-quality notes from that location.
- **No personalization** — purely location + recency.

**Same-city retrieval (同城召回):**
- Users may be interested in same-city events.
- Index: **city → high-quality note list** (time-sorted).
- Also **not personalized.**

### 11.2 Author Retrieval (作者召回)

**Followed-author retrieval (关注作者召回):** users are interested in notes from authors they follow.
- Indices: **user → followed authors**, **author → published notes**.
- Retrieval: user → followed authors → newest notes.

**Interacted-author retrieval (有交互的作者召回):** if a user liked/saved/shared a note, they may like the author's other notes.
- Index: **user → interacted authors**.
- Retrieval: user → interacted authors → newest notes.

**Similar-author retrieval (相似作者召回):** if a user likes an author, they may like similar authors.
- Index: **author → similar authors** ($k$ authors).
- Retrieval: user → interested authors ($n$) → similar authors ($nk$) → newest notes ($nk$).

### 11.3 Cache Retrieval (缓存召回)

> Idea: **reuse fine-ranking results from the previous $n$ recommendation rounds.**

**Background:** Fine ranking outputs a few hundred notes for re-ranking; re-ranking does diversity sampling and selects only a few dozen for exposure. Over half of the fine-ranking results are never exposed — wasted.

**Mechanism:** cache the top-50 fine-ranked notes that were **not exposed**, and use the cache as a retrieval channel.

**Eviction policy (the cache has fixed size, so it needs eviction rules):**
- Once a note is successfully exposed, remove it from the cache.
- If the cache exceeds its size limit, evict the oldest entry (FIFO).
- A note can be retrieved at most 10 times — evict after 10.
- A note is kept at most 3 days — evict after 3 days.

### 11.4 Summary

- **Geo channels:** GeoHash retrieval, same-city retrieval.
- **Author channels:** followed authors, interacted authors, similar authors.
- **Cache retrieval.**

---

## 12. Exposure Filtering & Bloom Filter

### 12.1 The Exposure-Filtering Problem (曝光过滤问题)

If a user has already seen an item, don't show it again (repeated exposure hurts user experience — done by 小红书, 抖音; *not* done by long-video platforms like YouTube, which may re-recommend watched videos).

- For each user, record items already exposed to them. You only need recent history: 小红书 only retrieves notes < 1 month old, so only the last month of exposure history matters.
- After retrieval, check each retrieved item against the user's exposure history and remove items already exposed.
- **Brute force** comparison: a user has seen $n$ items, this round retrieves $r$ items → $O(nr)$ time. With $n, r$ both on the order of a few thousand, brute force is too slow. → use a **Bloom filter**.

### 12.2 Bloom Filter Basics

A Bloom filter (Burton H. Bloom, 1970) tests whether an item ID is in the set of already-exposed items:
- **If it says "no," the item is definitely not in the set** (no false negatives).
- **If it says "yes," the item is *probably* in the set** — there's a chance of a **false positive (误伤)**: a never-exposed item wrongly judged as exposed and filtered out.

So exposure filtering with a Bloom filter **never repeats an exposed item** (no false negatives), but **occasionally over-filters** (false positives drop some fresh items).

**Structure:**
- Represent each user's exposed-item set as an **$m$-dimensional binary vector** (so each user needs $m$ bits of storage).
- Use $k$ **hash functions**, each mapping an item ID to an integer in $[0, m-1]$.

### 12.3 How It Works

**$k = 1$ (one hash function):**
- Start with an all-zero vector.
- **Add** an exposed item: hash its ID to a position, set that bit to 1 (if already 1, leave it).
- **Query** a retrieved item: hash its ID; if the bit is 0 → definitely not exposed (correct, never wrong on negatives); if the bit is 1 → judged exposed. This may be a **false positive** (two distinct IDs hashing to the same position — a collision).

**$k = 3$ (three hash functions):**
- **Add** an item: 3 hash functions map the ID to 3 positions; set all 3 bits to 1.
- **Query** an item: hash to 3 positions; if **any** of the 3 is 0 → definitely not exposed (no false negative); if **all 3** are 1 → judged exposed (could be a false positive — all 3 happen to be set by other items).

Using more hash functions makes accidental "all bits set" less likely per item, reducing some collisions — but setting more bits per item raises the fraction of 1s. There is a trade-off (see below).

### 12.4 False-Positive Probability & Optimal Parameters

Let $n$ = size of the exposed-item set, $m$ = binary-vector dimension, $k$ = number of hash functions. The false-positive probability:

$$
\delta \approx \left( 1 - \exp\left( -\frac{kn}{m} \right) \right)^{k}.
$$

- Larger $n$ → more 1s in the vector → higher false-positive rate (a fresh item's $k$ positions are more likely all-1).
- Larger $m$ → longer vector → fewer hash collisions → lower false positives, but more storage.
- $k$ has an optimum (too large or too small is bad).

Given a tolerable false-positive rate $\delta$ (e.g., 1%), the optimal parameters are:

$$
k = 1.44 \cdot \ln\left( \frac{1}{\delta} \right), \qquad m = 2n \cdot \ln\left( \frac{1}{\delta} \right).
$$

(Roughly: $m \approx 10n$ bits per user suffices to push $\delta$ below 1%.)

### 12.5 The Exposure-Filtering Pipeline

```
retrieval ──▶ ranking ──▶ exposed items (物品1..q) ──┐
   ▲                                                  │ (write exposed items)
   │ (send user's binary vector)                      ▼
exposure-filter service  ◀── (write) ── real-time stream processing
(Bloom Filter)                            (Kafka + Flink)
```

- The app front-end logs every exposed item via tracking (买点/埋点).
- Exposed items are written to a **Kafka** message queue; **Flink** does real-time computation: reads Kafka, computes hashes of exposed items, updates the Bloom-filter binary vector.
- This must be fast: refreshes can be seconds to a couple minutes apart, so the previous exposure must hit the Bloom filter before the next refresh, or duplicates appear. (The real-time stream is the most failure-prone part — if it lags or dies, just-seen items reappear.)
- **At retrieval time:** the retrieval server requests the exposure-filter service, which sends back the user's binary vector; the retrieval server runs the Bloom filter on retrieved items, drops already-exposed ones, and forwards the rest to ranking.

### 12.6 Bloom Filter Drawbacks

- A Bloom filter represents a set as a binary vector; adding an item just sets $k$ bits to 1.
- **It supports adding but NOT deleting items.** You can't simply flip an item's $k$ bits back to 0, because bits are **shared** across items — zeroing them would effectively remove many other items too.
- In practice, you must daily remove items older than 1 month from the set (over-age items can't be retrieved anyway, and a smaller $n$ lowers the false-positive rate $\delta$). But since Bloom filters can't delete, you must **recompute the entire vector** for the set to drop aged-out items.

---

## Source Notes & Disagreements

- **Sources used:** the twelve slide PDFs `02_Retrieval_01..12.pdf` (treated as ground truth) plus the twelve AI-transcribed Chinese audio files (supplementary). The slides and audio are consistent throughout; no substantive contradictions were found.
- **Audio transcription artifacts** (corrected against the slides): the AI transcription frequently garbles Chinese book titles and terms — e.g., 《鹿鼎记》 ("The Deer and the Cauldron") appears as "鹿岭记/露顶记/露顶记," and "买点" (tracking/埋点) is a homophone error for "埋点." These do not affect the technical content. The math (similarity formulas, triplet/softmax losses, sampling-bias correction $-\log p_i$, Bloom-filter $\delta$ and optimal $k, m$) is taken verbatim from the slides.
- **Minor numbers** that are illustrative (not load-bearing): ItemCF scale ($n=200$, $k=10$, $nk=2000$); pointwise positive:negative ratio $1:2$–$1:3$; sampling exponent $0.75$ (empirical); cache eviction (≤10 retrievals, ≤3 days, top-50 cached); Bloom-filter $m\approx10n$ for $\delta<1$% — all stated identically in slides and audio.

---

## Expanded Reading — Recent Advances (2026)

This section surveys five 2026 industrial papers that push candidate retrieval in two directions: **semantic-ID / generative retrieval** (语义 ID 与生成式召回) — replacing item-ID embeddings with discrete semantic tokens and casting retrieval as autoregressive token generation (TRM, STATIC, TrieRec) — and **large-scale embedding-based retrieval (EBR, 嵌入召回)** — productionizing the two-tower / dense-retrieval idea of §6–§8 at LinkedIn and Airbnb scale.

### Farewell to Item IDs (TRM, ByteDance)

> Zhao, Z., Zhang, T., Xu, J., Cai, Q., Zhang, Q., Yang, L., Xiao, D., & Chang, X. (2026). *Farewell to item IDs: Unlocking the scaling potential of large ranking models via semantic tokens*. arXiv. https://arxiv.org/abs/2601.22694

**Method.** TRM (Token-based Recommendation Model) replaces the item-ID embedding table of §4–§5 with **semantic tokens** (语义 token). It works in three stages. (1) *Collaborative-aware representation*: a multi-modal LLM is first adapted to short videos with a caption next-token loss $L_{\text{cap}}$, then aligned to user behavior with an in-batch contrastive loss over query–item and co-click item–item pairs, $L_{\text{align}} = -\mathbb{E}_{(a,b)\in\mathcal P}\log\frac{\exp(\text{sim}(h_a,h_b)/\tau)}{\sum_{b'\in B}\exp(\text{sim}(h_a,h_{b'})/\tau)}$ — directly importing the §6–§7 cosine + in-batch-negative recipe into representation learning. (2) *Hybrid tokenization*: RQ-Kmeans residual quantization yields coarse-grained **gen-tokens** (5 layers × 4096 codebook = 20,480 tokens) for generalization; high-frequency $k$-grams are merged via **BPE** into fine-grained **mem-tokens** for memorization, fused in a Wide&Deep network. (3) *Joint optimization*: a discriminative BCE loss $L_d$ on user actions plus a generative next-token loss $L_g$ over the item's tokens, $L = L_d + \lambda L_g$ (with $\lambda=0.1$).

**Key results.** On a large video-search dataset, TRM-RankMixer gains **+0.65% CTR AUC** and **+0.85% Real-Play AUC** over the ID baseline while cutting **sparse parameters from 7.52T to 5.07T (≈32.6% reduction)**. Token-based scaling beats ID-based scaling as dense capacity grows: at ~1.9B dense params TRM reaches **+0.75% qAUC** vs. +0.60% for ID-based RankMixer. Online A/B: **+0.26% user active days, −0.75% query-change ratio**.

**Trade-offs / limitations.** Naively swapping IDs for existing semantic tokens (TIGER/OneRec/SemID) *hurts* high-frequency old items (coarse clustering loses item-specific "combinative" knowledge) — TRM's mem-tokens and joint loss are what recover it, at the cost of a heavier offline tokenization + MLLM-alignment pipeline. The token codebook is a fixed closed set, so genuinely novel semantics still depend on the upstream content model.

**Connection to this chapter.** Directly extends the **embedding / semantic-ID** ideas of §4 and the matrix-completion/two-tower foundation of §5–§6: instead of a per-item learnable vector (which suffers cold-start and "knowledge erasing" on ID churn, the head effect of §9), items are described by stable discrete tokens — the generative successor to Deep Retrieval's path representation (§10).

### Vectorizing the Trie — STATIC (Google DeepMind / YouTube)

> Su, Z., Katsman, I., Wang, Y., He, R., Heldt, L., Keshavan, R., Wang, S.-C., Yi, X., Gao, M., Dalal, O., Hong, L., Chi, E. H., & Han, N. (2026). *Vectorizing the trie: Efficient constrained decoding for LLM-based generative retrieval on accelerators*. arXiv. https://arxiv.org/abs/2602.22647

**Method.** In generative retrieval, items are sequences of semantic IDs and valid outputs form a **prefix tree (trie, 前缀树)**; constrained decoding must mask invalid next-tokens at each step. Pointer-based trie traversal is memory-bound and cannot be statically shaped, so it stalls TPUs/GPUs. STATIC (Sparse Transition matrix-Accelerated Trie Index for Constrained decoding) **flattens the trie into a static Compressed Sparse Row (CSR) sparse matrix**, turning the per-step "which children are valid" lookup into a fully vectorized sparse matrix–vector operation that XLA can compile to fixed shapes.

**Key results.** Adds only **0.033 ms per decoding step (~0.25% of inference time)** while staying under the ≤10 ms/step target. Reports a **948× speedup over a CPU trie** and **47–1033× over baseline accelerator methods**, scaling to vocabularies of **~20 million fresh items** and improving online metrics by enabling business-logic constraints (freshness, in-stock, category).

**Trade-offs / limitations.** The CSR matrix is *static* — it must be rebuilt when the valid item set changes, so very high-churn constraints add index-maintenance cost; it targets accelerator throughput, not retrieval quality (orthogonal to recall).

**Connection to this chapter.** A serving-side counterpart to the **ANN index** of §5.5 and §8.1: where two-tower retrieval needs Faiss/HNSW to make dense lookup tractable, generative retrieval needs an accelerator-friendly trie index to make constrained decoding tractable. It is the infra that makes semantic-ID retrieval (TRM, TrieRec) viable at production latency.

### Trie-Aware Transformers — TrieRec (Ant Group)

> Xu, Z., Chen, J., Chen, S., He, Y., Yang, J., Yuan, C., Ding, K., & Wang, C. (2026). *Trie-aware transformers for generative recommendation*. arXiv. https://arxiv.org/abs/2602.21677

**Method.** Generative recommenders tokenize each item into a hierarchical code sequence, inducing a trie over items, but standard Transformers flatten tokens into a linear stream and ignore that topology. TrieRec injects structural inductive bias via two positional encodings: (1) a **trie-aware absolute positional encoding** that folds a node's local structure (depth, ancestors, descendants) into its token representation; and (2) a **topology-aware relative positional encoding** that adds pairwise structural relations into self-attention — using lowest-common-ancestor (LCA) features such as $d_{\text{LCA}}$ (distance from each token to the LCA) to capture topology-induced semantic relatedness. It is model-agnostic and **hyperparameter-free** (no new tunables).

**Key results.** Dropped into three GR backbones (**TIGER, CoST, LETTER**), TrieRec gives an **average +8.83% relative improvement across four real-world datasets**, with gains exceeding 10% in some settings, at negligible added cost.

**Trade-offs / limitations.** It improves the *modeling* of an existing semantic-ID tokenization rather than the tokenizer itself, so its ceiling is bounded by codebook quality; the benefit is largest when the trie is genuinely hierarchical/deep.

**Connection to this chapter.** Refines the autoregressive path-scoring view of **Deep Retrieval (§10)** and semantic-ID retrieval: where §10 scores paths $p(a,b,c\mid\mathbf{x})$ with plain concatenated embeddings, TrieRec makes the Transformer explicitly aware that those tokens live on a tree — the same prefix tree that STATIC vectorizes for serving.

### Semantic Search at LinkedIn

> Borisyuk, F., Vasudevan, S., Wu, M., Li, G., Le, B., Zhang, S., Shen, Q. K., Juan, Y., Behdin, K., et al. (2026). *Semantic search at LinkedIn*. arXiv. https://arxiv.org/abs/2602.07309

**Method.** A production LLM-based semantic-search stack for AI Job Search and AI People Search. Retrieval uses a **contrastive LLM bi-encoder** (a two-tower architecture, §6) with a margin loss and **GPU retrieval-as-ranking** (exhaustive in-GPU scoring with attribute pre-filtering) instead of classic ANN. The ranker is a compact **Small Language Model (SLM)** trained via **multi-teacher, multi-task distillation (MTD)**: specialized relevance and engagement teachers distill calibrated soft labels into one student that jointly predicts relevance and engagement. A prefill-oriented inference design adds model **pruning**, **context/history summarization**, and **text–embedding hybrid (mixed-input)** scoring.

**Key results.** Throughput rises **>75×** under a fixed latency budget while preserving near-teacher NDCG. Concrete steps: a 0.6B summarized+pruned ranker hits **NDCG@10 0.9218 at 2,200 items/s/GPU (<500 ms)** on one H100; pruning a 375M ranker lifts throughput **290→2,200 items/s/GPU (7.5×)**; mixed-input embedding scoring adds ~25%; MixLM gives ~10× over summarized text. Quality: Job Search **+18.03% NDCG@10 / −46.88% Poor-Match-Rate@10**, People Search **>10% NDCG@10**; distilled relevance judge reaches **0.81** vs. the frontier teacher.

**Trade-offs / limitations.** GPU exhaustive retrieval-as-ranking trades the sublinear cost of ANN (§5.5) for brute-force GPU throughput — feasible at LinkedIn's catalog size but not at billion-item web scale; the multi-teacher distillation pipeline is heavy to build and maintain.

**Connection to this chapter.** A modern, LLM-powered realization of the **two-tower model (§6–§8)**: bi-encoder embeddings with contrastive in-batch negatives (§7) for retrieval, then a learned ranker — but with the candidate generator and ranker both distilled from LLM teachers, and ANN replaced by GPU-exhaustive scoring.

### Embedding-Based Retrieval at Airbnb Search

> Abdool, M., Banerjee, S., Paul, M., Kim, D., Liu, X., Xu, B., Yu, T., Gao, H., Ouyang, K., Gao, H., He, L., Moyerman, S., & Katariya, S. (2026). *Applying embedding-based retrieval to Airbnb search*. arXiv. https://arxiv.org/abs/2601.06873

**Method.** Airbnb's first EBR system for a **two-sided marketplace** (双边市场). It is a textbook **two-tower model** (§6): a query tower $Q_\theta(q)$ and a listing tower $L_\theta(l)$ scored by $F_\theta(l,q)=\text{sim}(Q_\theta(q),L_\theta(l))$, with the listing tower computed in a **daily offline batch** so only the query tower runs at serving (exactly the §8.1 precompute-items / online-user split). The key innovation is **trip-based sampling**: instead of search-based sampling (which discards searches where the booked listing wasn't shown and is biased to the last ~30% of the journey), it groups a user's searches by query parameters and builds **hard negatives** (§7.4) from intentful actions (clicks, wishlists). Serving uses **IVF** (inverted file) ANN rather than HNSW, to coexist with ~10,000 listing updates/sec and geo filters.

**Key results.** Recall for the booked listing climbs across versions — offline recall@10 **53.3% → 93.4%**, replayed-traffic recall@100 **40.5% → 87.0%** (baseline → EBR V3). Online A/B: combined **+0.31% booking conversion**, **+2.3% bookings from promotional emails**, plus statistically significant gains in new-listing and wishlisted-listing bookings; switching HNSW→IVF cut retrieval **compute ~16%**.

**Trade-offs / limitations.** Offline listing embeddings can't use real-time price/availability (refreshed later in the funnel); HNSW gave better raw ANN quality but IVF won on real-time-update + filter integration and ops cost — a concrete instance of the §5.5 approximate-vs-exact trade-off. Offline recall@K from shown-only logs doesn't generalize, forcing a custom traffic-replay evaluation.

**Connection to this chapter.** A direct industrial application of the **two-tower model + ANN** of §5–§8, validating the chapter's central lessons in a marketplace setting: the precompute-item / online-user serving split (§8.1), the primacy of positive/negative-sample design with **hard negatives** (§7.4–§7.6), and the approximate-nearest-neighbor latency trade-off (§5.5).

---

## Pinterest in Practice (2024–2026)

Pinterest is a visual-discovery platform, so retrieval there is inherently **multimodal and composed** (多模态/组合式召回) — a query is rarely just text or just an item ID, but a reference image *plus* a modifying text intent, which stretches the two-tower / ANN / embedding ideas of §5–§8 into the image+text setting.

### Multi-Modal Search at Pinterest

> Li, M. (Irene), Rezazadeh Kalehbasti, P., Chen, S., Islam SK, M., Hegde, K., Sobhani, K., Hazra, K. S., & Zhang, Z. (2025). *Introducing multi-modal search at Pinterest: A new user experience*. In Proceedings of the 4th Workshop on End-to-End Customer Journey Optimization (KDD CJ'25), Toronto, Canada. ACM.

This is a deployed Search surface that accepts **hybrid queries** (reference image or a selected object + a text query), letting Pinners either explore an *aspect* of an image (descriptors, e.g. "vivid tones") or *modify* one (pivots, e.g. change color/occasion). The retrieval stage uses **three diverse candidate generators (CGs)** — SearchSage (engagement-trained), an in-house SigLIP2 reproduction (late-fusion dual encoder), and an in-house MagicLens reproduction (early-fusion image↔text) — each producing embeddings with a different relevance profile, exactly the **multi-channel two-tower retrieval** pattern of §6 and §11 extended to image+text embeddings and queried via ANN. Candidates are fused by a heuristic **composite reranking score** that multiplicatively blends image-similarity cosine, text-relevance cosine, predicted engagement, and a CG-diversity boost, with the top 8 slots reserved for attribute-matching results. Key metric: the best config beat a SearchSage-only baseline by **+18% combined relevance@5** and **+54% overall engagement**, and vs. plain text search delivered **+200% CTR** on descriptor queries. It is a clean industrial illustration of how heterogeneous embedding retrievers (the §6 two-tower idea) are ensembled and reranked when the query itself is multimodal.

### PinPoint — Composed Image Retrieval Benchmark

> Mahadev, R., Yuan, J., Poirson, P., Xue, D., Wu, H.-Y., & Kislyuk, D. (2026). *PinPoint: Evaluation of composed image retrieval with explicit negatives, multi-image queries, and paraphrase testing*. arXiv. https://arxiv.org/abs/2603.04598

PinPoint is an **evaluation benchmark** (not a deployed model) for **Composed Image Retrieval (CIR, 组合图像检索)** — the query model where a user anchors on a reference Pin and modifies it with text ("this dress in red"), the natural retrieval task for a visual-discovery platform. It argues that the standard **Recall@K** objective (the metric this chapter optimizes retrieval for) is the *wrong* target for visual search, because real queries have many valid positives (avg 9.1) and many visually-similar hard negatives (avg 32.8), so it introduces precision-aware metrics — $\Delta\text{mAP@10}$ (sensitivity to injecting hard negatives) and Negative Recall@10 — that expose false-positive behavior leaderboards hide. Methodologically it pairs a first-stage embedding retriever (the §6 two-tower / dense-retrieval setup applied to image+text) with a **training-free MLLM pointwise reranker** that scores each candidate via yes/no token logits, pushing the best system to **mAP@10 = 0.290** while cutting Negative Recall@10 from 0.091 to 0.056. The key lesson for this chapter: the **retrieve-then-rerank** funnel and the choice of objective/negatives (§7) matter as much in composed visual retrieval as in classic ID-based retrieval — and multi-image compositional queries still collapse all current retrievers by 48–72%.
