# News-Based Stock Market Prediction

---

# 1. Project Information

- **Project Title:** News-Based Stock Market Prediction
- **Group Name:** Group AI Finance
- **Group Members:**
  - Student 1 – Akram Boudebouze
  - Student 2 – Alexandre Besson
  - Student 3 – Ferdinand Carsoux
  - Student 4 – Raphaël Essengue

- **Course Name:** AI In Finance
- **Instructor:** Nicolas De Roux & Mohamed EL FAKIR
- **Submission Date:** April 2026

---

# 2. Project Description

Financial markets are highly sensitive to news and global events. Each day, thousands of headlines influence investor sentiment and, consequently, market direction. This project explores whether the content of daily news headlines can be used to predict the direction of the Dow Jones Industrial Average (DJIA) — one of the most followed stock market indices in the world.

Using a dataset of the 25 most upvoted Reddit headlines from the WorldNews subreddit for each trading day between 2008 and 2016, we train machine learning models to classify whether the DJIA closed higher or lower than the previous day. This problem is interesting because it sits at the intersection of Natural Language Processing (NLP) and quantitative finance, and directly challenges the Efficient Market Hypothesis by testing whether publicly available information carries predictive power.

Financial institutions, quantitative analysts, and algorithmic trading firms could benefit from such a system as part of a broader market signal framework.

---

# 3. Project Goal

The goal of this project is to build a binary classification system that predicts the daily direction of the Dow Jones Industrial Average (up or down) from news headlines published on the same day.

A successful solution achieves an AUC-ROC score meaningfully above 0.5 (random baseline), demonstrating that textual news signals carry at least partial predictive information about market movements. We test three progressively complex approaches: classical ML with TF-IDF features, Word2Vec embeddings, and a pre-trained financial language model (FinBERT).

---

# 4. Task Definition

- **Task Type:** Binary Classification
- **Input Variables:** 25 Reddit news headlines per trading day (text), combined with financial features (returns, volatility, RSI) and sentiment scores (VADER)
- **Target Variable:** `Label` — whether the DJIA closed higher (1) or lower (0) than the previous day
- **Evaluation Metric(s):** AUC-ROC (primary), Accuracy, F1-score

---

# 5. Dataset Description

## Dataset Overview

- **Number of samples:** 1,989 trading days
- **Number of features:** 27 raw columns (Date, Label, Top1 to Top25), expanded to 5,010 after feature engineering
- **Target variable:** `Label` (0 = DJIA down, 1 = DJIA up)
- **Data source:** [Daily News for Stock Market Prediction — Kaggle](https://www.kaggle.com/datasets/aaron7sun/stocknews)

---

## Feature Description

| Feature | Description | Type |
|---|---|---|
| Date | Trading day date | Date |
| Label | DJIA direction: 1 (up) or 0 (down) | Binary |
| Top1 to Top25 | Top 25 upvoted Reddit WorldNews headlines of the day | Text |
| ret_1d | Log return of DJIA on day J-1 | Numerical |
| ret_5d | Cumulative log return over 5 days | Numerical |
| vol_5d | Rolling volatility over 5 days | Numerical |
| vol_20d | Rolling volatility over 20 days | Numerical |
| rsi_14 | Relative Strength Index over 14 days | Numerical |
| day_of_week | Day of the week (0=Monday to 4=Friday) | Categorical |
| sent_pos | Average positive VADER sentiment across 25 headlines | Numerical |
| sent_neg | Average negative VADER sentiment across 25 headlines | Numerical |
| sent_neu | Average neutral VADER sentiment across 25 headlines | Numerical |
| sent_compound | Average compound VADER sentiment across 25 headlines | Numerical |
| TF-IDF features | 5,000 sparse text features from bigram TF-IDF | Numerical (sparse) |

---

## Target Variable

- **Variable name:** `Label`
- **Meaning:** Whether the DJIA closed higher or lower than the previous trading day
- **Possible values:** 1 (market up), 0 (market down)

The dataset contains 1,065 days labeled 1 (53.5%) and 924 days labeled 0 (46.5%), making it nearly balanced.

---

## Data Types

- **Text:** Top1 to Top25 (news headlines — raw strings, converted to TF-IDF or embeddings)
- **Numerical:** Financial features (returns, volatility, RSI) and sentiment scores
- **Categorical:** day_of_week
- **Binary:** Label (target variable)
- **Date:** Trading day timestamp

---

## Data Distribution

- The target variable is nearly balanced: 53.5% up days vs 46.5% down days — no resampling was required.
- Financial returns (ret_1d) show slight negative skew consistent with fat-tailed market distributions.
- Volatility features (vol_5d, vol_20d) are right-skewed with occasional spikes during the 2008–2009 financial crisis.
- VADER compound sentiment scores are consistently negative (mean ≈ -0.22), reflecting the predominantly negative tone of news headlines — this is consistent across all years in the dataset.
- The dataset covers a full market cycle including the 2008 crisis (DJIA ~12k → 6.5k) and the subsequent recovery.

---

## Data Quality

- **Byte-string prefix bug:** Headlines were scraped using Python 2 and stored with a `b'...'` prefix (e.g., `b'Russia invades Georgia'`). This was cleaned using a regex in preprocessing. Without this fix, TF-IDF bigrams such as `"north korea"` or `"stock market"` would have been incorrectly tokenized.
- **No missing values** in the Label column.
- **Minor missing values** in a small number of Top headline columns, replaced with empty strings.
- **No duplicate rows** detected.
- **Class balance** is acceptable (53.5% / 46.5%) — no oversampling was applied.

---

# 6. Data Preprocessing

### 1. Byte-string cleaning
Headlines were stored with a Python 2 `b'...'` artifact. A regex was applied to strip this prefix from all 25 headline columns. This was necessary to ensure correct tokenization for TF-IDF and embedding models.

### 2. Temporal train/validation/test split
The data was split **chronologically** (not randomly) to avoid data leakage:
- Train: 2008–2013 (60%)
- Validation: 2014 (20%)
- Test: 2015–2016 (20%)

A random split would allow future news to leak into the training set, which would artificially inflate performance.

### 3. Text concatenation
The 25 daily headlines were concatenated into a single string per day, forming the text input for TF-IDF and Word2Vec models.

### 4. TF-IDF vectorization
Applied on the training corpus only (`fit_transform` on train, `transform` on val/test). Parameters: `max_features=5000`, `ngram_range=(1,2)`, `sublinear_tf=True`, `stop_words='english'`.

### 5. Financial feature engineering
Downloaded DJIA price data via `yfinance`. Computed: log returns (1d, 5d), rolling volatility (5d, 20d), RSI (14 days), and day of week. All features were **shifted by 1 day** to ensure no look-ahead bias — only information available before market open was used.

### 6. Sentiment scoring (VADER)
Applied VADER sentiment analyzer to each of the 25 headlines per day. Averaged the 4 sentiment scores (positive, negative, neutral, compound) across headlines to produce 4 daily sentiment features.

### 7. Feature scaling
Financial and sentiment features were standardized using `StandardScaler` fitted on the training set only.

### 8. Feature matrix combination
The final feature matrix `X_all` was constructed by concatenating the sparse TF-IDF matrix with the dense financial and sentiment features, yielding 5,010 dimensions per day.

---

# 7. Modeling Approach

## Chosen Models

Three levels of increasing complexity were tested:

**Level 1 — Classical ML (baseline)**
- Logistic Regression (C=1.0)
- Random Forest (200 trees)
- Gradient Boosting

**Level 2 — Word Embeddings**
- Word2Vec (Skip-gram, 200 dimensions, window=5) trained on the training corpus only. Each day is represented by the average of its word vectors. Combined with financial and sentiment features, passed to a Gradient Boosting classifier.

**Level 3 — Pre-trained Language Model**
- FinBERT (`ProsusAI/finbert`): a BERT model pre-trained on financial text (Bloomberg, Reuters, SEC filings). Fine-tuned for 3 epochs with batch size 16, learning rate 2e-5, max sequence length 256 tokens.

---

## Modeling Strategy

We selected these models to test a natural progression from simple to complex representations of text:

- **Logistic Regression** serves as a linear baseline to assess whether any linear signal exists in the TF-IDF features.
- **Random Forest and Gradient Boosting** capture non-linear interactions between features.
- **Word2Vec** captures semantic similarity between words, going beyond bag-of-words.
- **FinBERT** captures contextual meaning and was pre-trained on domain-specific financial text.

**Walk-Forward Validation** was used instead of standard cross-validation: the model is trained on all available past data and tested on the next time window (3 rolling windows). This simulates real-world deployment conditions.

No hyperparameter tuning was performed beyond selecting `max_features=5000` for TF-IDF based on a systematic search.

---

## Evaluation Metrics

- **AUC-ROC (primary):** Measures the model's ability to rank positive predictions above negative ones, regardless of threshold. This is the most appropriate metric for this task as it is threshold-independent and robust to class imbalance.
- **Accuracy:** Proportion of correct predictions. Reported for interpretability but less reliable due to the slight class imbalance.
- **F1-score:** Harmonic mean of precision and recall. Useful to detect models biased toward predicting one class — as was the case for FinBERT (high F1 but low AUC revealed a systematic bias toward predicting "up").

### Results Summary

| Level | Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|---|
| N1 — Classique | Logistic Regression | 0.487 | 0.592 | 0.481 |
| N1 — Classique | Random Forest | 0.508 | 0.652 | 0.481 |
| N1 — Classique | Gradient Boosting | 0.511 | 0.588 | 0.537 |
| N2 — Embeddings | **Word2Vec + GB** 🏆 | **0.534** | **0.627** | **0.550** |
| N3 — Transformer | FinBERT | 0.503 | 0.665 | 0.482 |

**Best model: Word2Vec + Gradient Boosting (AUC 0.550)**

Notably, FinBERT (109M parameters) underperformed Word2Vec despite its complexity. With only ~1,359 training days (255 fine-tuning steps), the model memorized rather than generalized. Its high F1 (0.665) masked a systematic bias toward predicting "up" — revealed by its near-random AUC (0.482).

---

# 8. Project Structure

```
ai_in_finance_project/
│
├── data/
│   └── Combined_News_DJIA.csv       # Raw dataset from Kaggle
│
├── notebooks/
│   └── DJIA_NewsPrediction_VFinale.ipynb  # Main notebook (all steps)
│
├── PROJECT_GUIDELINES.md            # Course guidelines
└── README.md                        # This file
```

---

# 9. Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install gensim nltk vaderSentiment yfinance
pip install torch transformers  # For FinBERT (Level 3)
```

The dataset must be downloaded from Kaggle and placed in the `data/` folder:
👉 [https://www.kaggle.com/datasets/aaron7sun/stocknews](https://www.kaggle.com/datasets/aaron7sun/stocknews)
