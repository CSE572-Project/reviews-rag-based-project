# CSE 572 — Checkpoint 1 (Amazon Reviews Polarity)

This repository contains a complete, **reproducible** pipeline for Checkpoint 1,
focusing on the *reviews* portion of the original RAG project pitch.

It includes:
- **EDA**: class balance, review-length stats, token/bigram analysis.
- **Baselines**: TF–IDF + (Logistic Regression, Linear SVM, Complement Naive Bayes).
- **RAG (classical)**: TF–IDF retrieval + extractive summary (sentence ranking).
- **Artifacts**: figures, metrics, and sample RAG outputs saved to `artifacts/`.
- **Slide outline** in `slides_outline.md` for quick presentation prep.

> Dataset: [Amazon Reviews Polarity (Kaggle)](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews/data).
> Files needed: `train.csv` and `test.csv` (columns: `polarity`, `title`, `text`).

## Quick start

1) Create a Python env and install the requirements:
```bash
pip install -r requirements.txt
```

2) Run EDA, baselines, and RAG on a *sample* (adjust sizes as needed):
```bash
python -m src.run_all \  --data_dir /path/to/folder/with/train_and_test \  --sample_train 300000 \  --sample_test 100000 \  --max_features 200000 \  --seed 42
```

This will generate:
- `artifacts/eda_*` plots (PNG)
- `artifacts/baseline_metrics.json` with per-model metrics
- `artifacts/confusion_matrix_*.png` for each model
- `artifacts/rag_demo.txt` with a few example queries/answers
- `artifacts/retrieval_eval.json` with proxy Precision@k

> Note: The dataset is very large (millions of rows). Use sampling values that fit your
> machine. You can increase them later for stronger accuracy.

## Folder structure

```
cse572_checkpoint1/
├── artifacts/                # generated plots, metrics, RAG outputs
├── src/
│   ├── utils.py              # cleaning, helpers, plotting utilities
│   ├── eda.py                # EDA pipeline
│   ├── baselines.py          # ML baselines
│   ├── rag.py                # classical RAG: retrieval + extractive summary
│   └── run_all.py            # orchestration
├── slides_outline.md         # copy into your PPT
├── requirements.txt
└── README.md
```

## Reproducibility & Time
- All random processes fixed by `--seed`.
- Designed to run on a laptop with sampling; scale up incrementally.

## Citations
- Dataset constructed by Xiang Zhang (NYU). See Kaggle page and the paper:  
  *X. Zhang, J. Zhao, Y. LeCun (2015). Character-level CNNs for Text Classification (NIPS 2015).*

