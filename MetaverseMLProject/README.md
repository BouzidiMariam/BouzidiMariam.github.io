# Metaverse Transaction Risk Analysis

A Streamlit application that classifies blockchain transactions in the open metaverse as **low**, **moderate** or **high** risk, built around a leakage-aware preprocessing pipeline and an honest model comparison.

**Authors:** Bouzidi Mariam · Lajmi Sahar · Eya Bouhouch
**Full report (French):** [`Ml_project.pdf`](./Ml_project.pdf)

## Why version 2 exists

The first version reported near-perfect accuracy for several models. An audit of the data showed those scores measured **label leakage**, not model quality:

* `risk_score` alone predicts the label with **100 %** accuracy (its ranges never overlap between classes).
* Even without it, `transaction_type` + night-time almost determine the label. The dataset is **synthetic and rule-generated**.
* 80.8 % of transactions are low risk, so "always predict low risk" already scores 0.808 accuracy. **Accuracy is a misleading metric here; macro-F1 is used instead.**

![Leakage scan](docs/f4_leakscan.png)

## Feature scenarios

| Scenario | Removes | Decision tree macro-F1 | Purpose |
|---|---|---|---|
| `full` | nothing | 1.000 | shows the leakage (do not draw conclusions) |
| `no_score` | `risk_score` | 1.000 | realistic but still near-trivial |
| **`behavioural`** (default) | `risk_score`, `transaction_type` | **0.663** | the honest benchmark where models differ |

![Leakage lab](docs/f7_lab.png)

## Results (behavioural scenario, stratified 70/30 split, default hyperparameters)

| Model | Accuracy | Macro-F1 | Macro recall | ROC-AUC |
|---|---|---|---|---|
| **Decision Tree** | 0.710 | **0.663** | 0.862 | 0.908 |
| Random Forest | 0.713 | 0.662 | 0.857 | 0.911 |
| Gradient Boosting | 0.711 | 0.660 | 0.860 | 0.911 |
| SVM (RBF) | 0.693 | 0.642 | 0.858 | 0.888 |
| Logistic Regression | 0.664 | 0.609 | 0.843 | 0.892 |
| Neural Network | 0.866 | 0.572 | 0.623 | 0.912 |
| KNN (k = 15) | 0.858 | 0.570 | 0.607 | 0.904 |
| Gaussian Naive Bayes | 0.543 | 0.501 | 0.793 | 0.841 |
| *Baseline: always "low risk"* | *0.808* | *0.298* | – | – |

Take-aways:

* The neural network and KNN look best on accuracy but mostly predict the majority class (macro recall ≈ 0.61). **Ranking by accuracy would pick the wrong models.**
* Tree-based, class-weighted models plateau at **macro-F1 ≈ 0.66**; randomized-search tuning adds at most ~0.02. The limit is the information in the behavioural features, not the algorithm.
* A temporal split (train on the past, test on the future) gives the same picture (decision tree: 0.709 accuracy, 0.656 macro-F1).
* High-risk **recall is 0.94 but precision only 0.25**: behaviour alone cannot separate high-risk accounts from many look-alike low-risk "new" accounts.

![Random forest evaluation](docs/f8_rf_eval.png)

## Preprocessing (`preprocessing.py`)

One module shared by the notebook, the app and inference. Every decision is measured by an automatic audit:

* **Cleaning:** text normalisation, range checks, `hour_of_day` rebuilt from the timestamp, duplicate removal, chronological sort.
* **Audit:** single-feature leakage scan, Cramér's V (redundancy and noise), mutual information, depth-4 rule probe.
* **Dropped automatically:** addresses (identifiers), `purchase_pattern` (1:1 duplicate of `age_group`), `location_region` and `ip_prefix` (no association with the target).
* **Engineered:** `is_night`, `sin/cos(hour)`, `session_per_login`, `amount_per_minute`. Weekday/month/weekend and sender history are off by default (no measurable signal).
* **No train/test leakage:** the split happens first; imputation, 0.1–99.9 % clipping, scaling and encoding are fitted on the train set only inside one `ColumnTransformer`.
* **Fixed target order** (low = 0 < moderate = 1 < high = 2), fixing the v1 label-order bug.
* Stratified or **temporal** split, balanced class weights, `transform_new()` for scoring single records.

## App features

| Tab | What it does |
|---|---|
| Data & preprocessing | KPIs, pipeline steps, dropped columns and reasons, leakage scan, rule probe, downloads |
| Explore | class-aware distributions, class mix by hour / category, Spearman correlations |
| Train a model | 8 classifiers, manual or **auto-tuned** (randomized search on macro-F1), confusion matrix, per-class report, ROC / PR from scores, permutation importance |
| Compare models | macro-F1 leaderboard with a naïve baseline, CSV export, **leakage lab** across the 3 scenarios |
| Predict one transaction | form → predicted class and probabilities |

## Project structure

```
app.py                 Streamlit interface
preprocessing.py       cleaning, audit, features, pipeline, inference helpers
preprocessing.ipynb    annotated walkthrough with plots and sanity checks
requirements.txt
.streamlit/config.toml theme
Ml_project.pdf         full report (French)
metaverse_transactions_dataset.csv
```

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py                                   # the app
python preprocessing.py --scenario behavioural --out artifacts   # preprocessing only
```

Place `metaverse_transactions_dataset.csv` next to `app.py`.

## Limitations

* The data is synthetic; labels follow simple rules, so results do not transfer to real fraud detection.
* One dataset only; user-uploaded datasets with schema validation are the natural next step.
* A decision-threshold slider and cost matrix would let users trade false alarms against missed risks.
