# CSE 572 – Project Progress Checkpoint 1 (Group)

**Topic:** Amazon Reviews — Polarity Classification + RAG for Opinion Summaries  
**Team:** (Fill names / ASU IDs)

---

## 1) Recap of Pitch (Problem & Motivation)
- Reviews are **noisy, biased, and inconsistent**; customers and analysts need **trustworthy, structured insights**.
- Proposed **RAG** system to:
  - Retrieve relevant reviews for a query (e.g., *battery life*).
  - Summarize sentiment & extract salient opinions.
- Connection to data mining: **retrieval, ranking, classification**.

*(From project pitch slides: RAG overview, pipeline, and evaluation ideas.)*

---

## 2) Dataset
- **Amazon Reviews Polarity** (train/test CSVs; columns: `polarity`, `title`, `text`)
- Polarity label: `1 = negative`, `2 = positive`
- We ignore score 3 by definition in this dataset.
- Size we used for Checkpoint 1 (sampling for compute): *(fill exact N)*

---

## 3) EDA — What the data looks like
- Class balance: (insert percentages) and plots `eda_class_balance.png`
- Review lengths (tokens/chars): (insert medians) and `eda_review_length_hist.png`
- Top class-distinctive tokens & bigrams (positive vs negative): `eda_top_pos_tokens.png`, `eda_top_neg_tokens.png`
- Data quality checks: % missing, duplicates, near-duplicates

Key observations:
- (Bullet 1)
- (Bullet 2)
- (Bullet 3)

---

## 4) Data Mining Pipeline (Preliminary System)
- Text cleaning: lowercasing, HTML + punctuation stripping, de-duplication
- Features: **TF–IDF** (uni+bi-grams, `max_features=...`)
- Baselines:
  - Logistic Regression
  - Linear SVM
  - Complement Naive Bayes
- Evaluation on held-out set (Kaggle test):
  - Accuracy, Precision, Recall, F1, ROC-AUC
  - Confusion matrices

Results snapshot (fill from `baseline_metrics.json`):
- LR: Acc=..., F1=...
- Linear SVM: Acc=..., F1=...
- C-NB: Acc=..., F1=...

---

## 5) RAG (Classical) — Retrieval + Extractive Summaries
- Retrieval: TF–IDF cosine similarity over the corpus (sample)
- Answer: top-K sentences ranked by query relevance (greedy de-dup)
- Example queries and outputs (see `artifacts/rag_demo.txt`)
- Proxy retrieval metric: **Precision@k** using keyword relevance

Observations:
- (Bullet on what queries worked well)
- (Bullet on failure modes, e.g., generic queries)

---

## 6) Issues & Risks (Checkpoint 1)
- Domain constraints: dataset lacks product IDs → topic-level, not product-level RAG.
- Noisy/short titles; inconsistent formatting; escaped newlines.
- Class imbalance is modest but present; long-tail vocabulary.
- Compute limits → used sampling; accuracy improves with more data/features.

---

## 7) Plan for Checkpoint 2
- Scale training (more data, bigger `max_features`).
- **Modeling:** Calibrated Linear SVM, Logistic Regression with class weights, Light-weight neural baselines (e.g., fastText-like, DistilBERT if resources permit).
- **RAG:** Switch to **dense embeddings** (e.g., SBERT) + vector DB; add **query expansion**.
- **Summarization:** Abstractive summaries with LLM, evaluated with ROUGE.
- **Evaluation:** Fairer retrieval ground truth (e.g., synthetic labels via keyword sets).

---

## 8) Takeaways
- Baselines set a strong reference; RAG adds **qualitative insights**.
- We have a solid, scalable pipeline to enhance for Checkpoint 2.
