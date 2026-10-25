"""
app.py : Metaverse Transaction Risk Analysis (v2)

Run:  streamlit run app.py

Tabs
  1. Data & preprocessing : audit, leakage scan, what was dropped and why
  2. Explore              : class-aware charts
  3. Train a model        : 8 classifiers, manual or auto-tuned (randomized search, macro-F1)
  4. Compare models       : leaderboard + leakage demo across feature scenarios
  5. Predict              : score one new transaction
"""
import ast
import itertools
import time
import warnings
from datetime import date, datetime, time as dtime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, average_precision_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve, precision_score,
                             recall_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocessing import (AGE_ORDER, CLASS_NAMES, SCENARIOS, TARGET_ORDER, PreprocessConfig,
                           load_raw, prepare_dataset, render_report, transform_new)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Page + theme
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Metaverse Transaction Risk", page_icon="🛡️", layout="wide")

COLORS = {"low_risk": "#2a9d8f", "moderate_risk": "#e9c46a", "high_risk": "#e63946"}
CLASS_COLORS = [COLORS[c] for c in TARGET_ORDER]
sns.set_theme(style="whitegrid", context="notebook")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg,#0b3c49 0%,#2a9d8f 100%); padding: 1.3rem 1.7rem;
       border-radius: 16px; color: #fff; margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 1.85rem; color: #fff; padding: 0;}
.hero p {margin: .35rem 0 0; opacity: .92; font-size: .98rem;}
.pred {border-radius: 14px; padding: 1.1rem 1.4rem; color: #fff; font-size: 1.5rem; font-weight: 700;}
.pred small {display:block; font-size: .85rem; font-weight: 400; opacity: .92;}
button[data-baseweb="tab"] {font-weight: 600;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero"><h1>🛡️ Metaverse Transaction Risk Analysis</h1>
<p>Classify blockchain transactions in the open metaverse as low, moderate or high risk,
with leakage-aware preprocessing and honest model evaluation.</p></div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Sidebar: data setup
# --------------------------------------------------------------------------- #
SCENARIO_LABELS = {"full": "Full (leaky, demo only)", "no_score": "No risk score",
                   "behavioural": "Behavioural (recommended)"}
with st.sidebar:
    st.markdown("### ⚙️ Data setup")
    scenario = st.selectbox("Feature scenario", list(SCENARIOS), index=2,
                            format_func=SCENARIO_LABELS.get, key="scenario")
    st.caption(SCENARIOS[scenario]["description"])
    split = st.radio("Split strategy", ["stratified", "temporal"], horizontal=True, key="split",
                     help="Stratified: random split keeping class proportions. Temporal: train on "
                          "the earlier transactions, test on the later ones.")
    test_size = st.slider("Test size", 0.10, 0.40, 0.30, 0.05, key="test_size")
    with st.expander("Advanced preprocessing"):
        scaler = st.radio("Scaler", ["standard", "minmax"], horizontal=True, key="scaler")
        interactions = st.checkbox("Ratio features (amount/min, session/login)", True, key="inter")
        seed = int(st.number_input("Random seed", 0, 9999, 42, key="seed"))
    st.divider()
    st.caption("ℹ️ The dataset is synthetic: labels follow simple rules. See the *Data* tab "
               "for the evidence and why the default scenario hides the shortcut features.")

CFG_KEY = (scenario, split, float(test_size), scaler, bool(interactions), seed)


@st.cache_resource(show_spinner="Loading data...")
def get_raw():
    return load_raw()


@st.cache_resource(show_spinner="Preprocessing (audit, leakage scan, pipeline)...")
def get_data(scenario, split, test_size, scaler, interactions, seed):
    cfg = PreprocessConfig(scenario=scenario, split=split, test_size=test_size, scaler=scaler,
                           interactions=interactions, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return prepare_dataset(raw=get_raw(), config=cfg)


try:
    data = get_data(*CFG_KEY)
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()

Xtr, Xte, ytr, yte = data.X_train, data.X_test, data.y_train, data.y_test
baseline_acc = float((yte == 0).mean())
p0 = baseline_acc
baseline_f1 = (2 * p0 / (1 + p0)) / 3 if p0 > 0 else 0.0     # always-"low risk" predictor

if scenario == "full":
    st.error("**Target leakage active.** `risk_score` is the risk assessment itself: its ranges never "
             "overlap between classes, so every model will score ≈100%. Use this scenario only to see "
             "the problem.")
elif scenario == "no_score":
    st.warning("`transaction_type` (scam/phishing → high, sale at night → moderate) still almost "
               "determines the label, so most models will score ≈100%. Switch to *Behavioural* for a "
               "meaningful comparison.")
else:
    st.success("**Behavioural scenario:** only who/when/how-long signals are used. This is where "
               "models genuinely differ.")

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
MODEL_NAMES = ["Random Forest", "Decision Tree", "Gradient Boosting", "Logistic Regression",
               "K-Nearest Neighbors", "Support Vector Machine", "Neural Network",
               "Gaussian Naive Bayes"]
# (key, kind, label, options/range, default)
SPECS = {
    "Logistic Regression": [
        ("C", "log", "C (inverse regularisation)", [0.001, 0.01, 0.1, 1, 10, 100], 1),
        ("max_iter", "int", "Max iterations", (100, 1000, 50), 500)],
    "Decision Tree": [
        ("max_depth", "int", "Max depth", (1, 30, 1), 8),
        ("min_samples_split", "int", "Min samples to split", (2, 50, 1), 2),
        ("min_samples_leaf", "int", "Min samples per leaf", (1, 50, 1), 1),
        ("criterion", "choice", "Criterion", ["gini", "entropy"], "gini")],
    "Random Forest": [
        ("n_estimators", "int", "Number of trees", (50, 500, 50), 100),
        ("max_depth", "int", "Max depth", (2, 40, 1), 12),
        ("min_samples_leaf", "int", "Min samples per leaf", (1, 20, 1), 1),
        ("max_features", "choice", "Features per split", ["sqrt", "log2"], "sqrt")],
    "K-Nearest Neighbors": [
        ("n_neighbors", "int", "Neighbours (k)", (1, 50, 1), 15),
        ("weights", "choice", "Weights", ["uniform", "distance"], "uniform"),
        ("p", "choice", "Distance (1 = Manhattan, 2 = Euclidean)", [1, 2], 2)],
    "Gaussian Naive Bayes": [
        ("var_smoothing", "log", "Variance smoothing", [1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1], 1e-9)],
    "Support Vector Machine": [
        ("C", "log", "C (regularisation)", [0.01, 0.1, 1, 10, 100], 1),
        ("kernel", "choice", "Kernel", ["rbf", "linear"], "rbf"),
        ("gamma", "choice", "Gamma", ["scale", "auto"], "scale"),
        ("_sample", "int", "Training sample (SVM is O(n²))", (2000, 15000, 1000), 6000)],
    "Neural Network": [
        ("hidden_layer_sizes", "choice", "Hidden layers", ["(32,)", "(64,)", "(64, 32)", "(128, 64)"], "(64,)"),
        ("activation", "choice", "Activation", ["relu", "tanh", "logistic"], "relu"),
        ("alpha", "log", "L2 penalty (alpha)", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], 1e-4),
        ("learning_rate_init", "log", "Learning rate", [1e-3, 3e-3, 1e-2], 1e-3),
        ("max_iter", "int", "Max epochs", (50, 500, 50), 200)],
    "Gradient Boosting": [
        ("max_iter", "int", "Boosting rounds", (50, 500, 50), 150),
        ("learning_rate", "log", "Learning rate", [0.01, 0.03, 0.06, 0.1, 0.2, 0.3], 0.1),
        ("max_depth", "int", "Max depth", (2, 12, 1), 6),
        ("l2_regularization", "log", "L2 regularisation", [0.0, 0.01, 0.1, 1, 10], 0.0)],
}
SEARCH = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10, 100]},
    "Decision Tree": {"max_depth": [3, 5, 8, 12, 16], "min_samples_leaf": [1, 5, 10, 20, 50],
                      "criterion": ["gini", "entropy"]},
    "Random Forest": {"n_estimators": [100, 200], "max_depth": [6, 10, 14, 20],
                      "min_samples_leaf": [1, 3, 5, 10]},
    "K-Nearest Neighbors": {"n_neighbors": [3, 5, 9, 15, 25, 41], "weights": ["uniform", "distance"],
                            "p": [1, 2]},
    "Gaussian Naive Bayes": {"var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1]},
    "Support Vector Machine": {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"]},
    "Neural Network": {"hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                       "alpha": [1e-5, 1e-4, 1e-3, 1e-2], "learning_rate_init": [1e-3, 3e-3, 1e-2],
                       "activation": ["relu", "tanh"]},
    "Gradient Boosting": {"learning_rate": [0.03, 0.06, 0.1, 0.2], "max_depth": [3, 5, 8],
                          "max_iter": [100, 200, 300], "l2_regularization": [0.0, 0.1, 1.0]},
}
TUNE_ROWS = {"Support Vector Machine": 3000, "Random Forest": 8000, "Neural Network": 8000}
SUPPORTS_BALANCE = {"Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine",
                    "Gradient Boosting", "Gaussian Naive Bayes"}


def default_params(name):
    return {k: d for k, _, _, _, d in SPECS[name]}


def render_params(name, disabled=False):
    out = {}
    for key, kind, label, opt, default in SPECS[name]:
        wk = f"{name}:{key}"
        if kind == "int":
            out[key] = st.slider(label, opt[0], opt[1], default, opt[2], key=wk, disabled=disabled)
        elif kind == "log":
            out[key] = st.select_slider(label, options=opt, value=default, key=wk,
                                        format_func=lambda v: f"{v:g}", disabled=disabled)
        else:
            out[key] = st.selectbox(label, opt, index=opt.index(default), key=wk, disabled=disabled)
    return out


def make_model(name, p, balance, seed=42):
    p = {k: v for k, v in p.items() if not k.startswith("_")}
    cw = "balanced" if balance else None
    if name == "Logistic Regression":
        return LogisticRegression(C=p["C"], max_iter=p["max_iter"], class_weight=cw)
    if name == "Decision Tree":
        return DecisionTreeClassifier(**p, class_weight=cw, random_state=seed)
    if name == "Random Forest":
        return RandomForestClassifier(**p, class_weight=cw, n_jobs=-1, random_state=seed)
    if name == "K-Nearest Neighbors":
        return KNeighborsClassifier(**p, n_jobs=-1)
    if name == "Gaussian Naive Bayes":
        return GaussianNB(var_smoothing=p["var_smoothing"], priors=[1 / 3] * 3 if balance else None)
    if name == "Support Vector Machine":
        return SVC(**p, class_weight=cw, random_state=seed)
    if name == "Neural Network":
        hl = p["hidden_layer_sizes"]
        hl = ast.literal_eval(hl) if isinstance(hl, str) else hl
        return MLPClassifier(**{**p, "hidden_layer_sizes": hl}, early_stopping=True, random_state=seed)
    if name == "Gradient Boosting":
        return HistGradientBoostingClassifier(**p, class_weight=cw, random_state=seed)
    raise ValueError(name)


def subsample(X, y, n, temporal, seed):
    if len(X) <= n:
        return X, y
    if temporal:                                   # keep the most recent rows, in order
        return X.iloc[-n:], y.iloc[-n:]
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return Xs, ys


def tune(name, balance, n_iter, seed):
    space = SEARCH[name]
    size = int(np.prod([len(v) for v in space.values()]))
    Xs, ys = subsample(Xtr, ytr, TUNE_ROWS.get(name, 10000), split == "temporal", seed)
    rs = RandomizedSearchCV(make_model(name, default_params(name), balance, seed), space,
                            n_iter=min(n_iter, size), cv=data.cv(3), scoring="f1_macro",
                            random_state=seed, n_jobs=1, refit=False)
    rs.fit(Xs, ys)
    return rs.best_params_, float(rs.best_score_), len(Xs)


def get_scores(model, X):
    return model.predict_proba(X) if hasattr(model, "predict_proba") else model.decision_function(X)


def evaluate(model, X, y):
    pred = model.predict(X)
    scores = get_scores(model, X)
    yb = label_binarize(y, classes=[0, 1, 2])
    aucs, aps, roc, pr = {}, {}, {}, {}
    for i in range(3):
        if yb[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(yb[:, i], scores[:, i])
        pre, rec, _ = precision_recall_curve(yb[:, i], scores[:, i])
        roc[i], pr[i] = (fpr, tpr), (rec, pre)
        aucs[i], aps[i] = auc(fpr, tpr), average_precision_score(yb[:, i], scores[:, i])
    return {
        "pred": pred, "scores": scores, "roc": roc, "pr": pr, "auc_by_class": aucs, "ap_by_class": aps,
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "macro_auc": float(np.mean(list(aucs.values()))) if aucs else float("nan"),
    }


def fit_and_eval(name, params, balance, seed=42):
    """Fit on the (optionally sub-sampled) train split, evaluate on the full test split."""
    model = make_model(name, params, balance, seed)
    X, y = Xtr, ytr
    if name == "Support Vector Machine":
        X, y = subsample(Xtr, ytr, int(params.get("_sample", 6000)), split == "temporal", seed)
    t = time.time()
    model.fit(X, y)
    fit_time = time.time() - t
    return model, evaluate(model, Xte, yte), fit_time


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def fig_confusion(y_true, y_pred, normalize):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    sns.heatmap(cmn if normalize else cm, annot=True, fmt=".0%" if normalize else "d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Row-normalised (recall per class)" if normalize else "Counts")
    fig.tight_layout()
    return fig


def fig_curves(res, kind):
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for i, name in enumerate(CLASS_NAMES):
        if i not in res["roc"]:
            continue
        if kind == "roc":
            x, y = res["roc"][i]
            ax.plot(x, y, color=CLASS_COLORS[i], lw=2, label=f"{name} (AUC {res['auc_by_class'][i]:.3f})")
        else:
            x, y = res["pr"][i]
            ax.plot(x, y, color=CLASS_COLORS[i], lw=2, label=f"{name} (AP {res['ap_by_class'][i]:.3f})")
            ax.axhline((yte == i).mean(), color=CLASS_COLORS[i], ls=":", lw=1)
    if kind == "roc":
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title("ROC (one-vs-rest, from model scores)")
    else:
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall (dotted = no-skill level)")
    ax.legend(loc="lower right" if kind == "roc" else "center left", fontsize=8)
    fig.tight_layout()
    return fig


def metric_row(res, base_acc):
    c = st.columns(5)
    c[0].metric("Accuracy", f"{res['accuracy']:.3f}", f"{res['accuracy'] - base_acc:+.3f} vs baseline")
    c[1].metric("Macro-F1", f"{res['macro_f1']:.3f}", f"{res['macro_f1'] - baseline_f1:+.3f} vs baseline")
    c[2].metric("Macro precision", f"{res['macro_precision']:.3f}")
    c[3].metric("Macro recall", f"{res['macro_recall']:.3f}")
    c[4].metric("ROC-AUC (macro)", f"{res['macro_auc']:.3f}")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_data, tab_explore, tab_train, tab_compare, tab_predict = st.tabs(
    ["📊 Data & preprocessing", "🔎 Explore", "🤖 Train a model", "🏁 Compare models",
     "🎯 Predict one transaction"])

# ============================ 1. DATA ====================================== #
with tab_data:
    raw = get_raw()
    k = st.columns(5)
    k[0].metric("Transactions", f"{len(raw):,}")
    k[1].metric("Model features", len(data.feature_names))
    k[2].metric("Train / test", f"{len(Xtr):,} / {len(Xte):,}")
    k[3].metric("Always-'low risk' accuracy", f"{baseline_acc:.3f}",
                help="The score of a model that ignores the inputs. Accuracy below or near this is useless.")
    k[4].metric("Always-'low risk' macro-F1", f"{baseline_f1:.3f}")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Pipeline applied")
        for i, s in enumerate(data.steps, 1):
            with st.expander(f"{i}. {s['step']}", expanded=i in (6, 8)):
                st.write(s["detail"])
    with c2:
        st.subheader("Class balance")
        counts = raw["anomaly"].value_counts().reindex(TARGET_ORDER)
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        ax.bar(CLASS_NAMES, counts.values, color=CLASS_COLORS)
        for i, v in enumerate(counts.values):
            ax.text(i, v, f"{v / counts.sum():.1%}", ha="center", va="bottom")
        ax.set_ylim(0, counts.max() * 1.15); ax.set_ylabel("transactions")
        fig.tight_layout(); show(fig)
        st.caption(f"Training uses balanced class weights where supported: {data.class_weights}")

    st.subheader("Columns dropped and why")
    st.dataframe(pd.DataFrame({"column": list(data.dropped), "reason": list(data.dropped.values())}),
                 hide_index=True)

    st.subheader("Is any single feature a shortcut?")
    scan = pd.DataFrame(data.audit["leakage_scan"])
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
        col = scan.verdict.map({"LEAK": "#e63946", "strong": "#e9c46a", "weak": "#adb5bd"})
        ax.barh(scan.feature[::-1], scan.single_feature_accuracy[::-1], color=col[::-1])
        ax.axvline(scan.majority_baseline.iloc[0], color="k", ls="--", lw=1)
        ax.set_xlim(0.75, 1.01); ax.set_xlabel("accuracy using this one feature")
        ax.set_title("Single-feature leakage scan")
        fig.tight_layout(); show(fig)
    with c2:
        mi = pd.DataFrame(data.audit["mutual_information"])
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
        ax.barh(mi.feature[::-1], mi.mutual_information[::-1], color="#264653")
        ax.set_title("Mutual information with the label"); fig.tight_layout(); show(fig)
    st.caption("Dashed line = always predicting 'low risk'. `risk_score` reaches 100% on its own (leak); "
               "`transaction_type` is a strong shortcut. Features at the baseline can still matter in "
               "combination (see the rule probe).")
    with st.expander("Rule probe: a depth-4 tree without risk_score"):
        st.code(data.audit["rule_probe_without_risk_score"])
        st.write("Scam/phishing → high risk; a sale at night → moderate; everything else → low. "
                 "The labels are rule-generated, which is why the behavioural scenario is the honest benchmark.")

    c1, c2, c3 = st.columns(3)
    show_raw = c1.checkbox("Preview raw data")
    show_proc = c2.checkbox("Preview processed data")
    if show_raw:
        st.dataframe(raw.head(200))
    if show_proc:
        st.dataframe(Xtr.head(200).round(3))

    @st.cache_data(show_spinner=False)
    def _csv(key, _data):
        return pd.concat([_data.X_train, _data.y_train], axis=1).to_csv(index=False).encode()

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Processed training set (CSV)", _csv(CFG_KEY, data), "train_processed.csv", "text/csv")
    d2.download_button("⬇️ Preprocessing report (Markdown)", render_report(data), "preprocessing_report.md")

# ============================ 2. EXPLORE =================================== #
with tab_explore:
    df = data.clean_df
    view = st.radio("Chart", ["Feature distribution by class", "Class mix by hour of day",
                              "Class mix by category", "Correlation matrix"], horizontal=True, key="view")
    if view == "Feature distribution by class":
        feats = [c for c in ["login_frequency", "session_duration", "amount", "hour_of_day",
                             "session_per_login", "amount_per_minute", "risk_score"] if c in df]
        f = st.selectbox("Feature", feats, key="dist_feat")
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(data=df, x=f, hue="anomaly", hue_order=TARGET_ORDER, palette=COLORS, stat="density",
                     common_norm=False, element="step", discrete=f in ("login_frequency", "hour_of_day"),
                     bins=45, ax=ax)
        ax.set_title(f"{f} by risk class (density)"); fig.tight_layout(); show(fig)
        st.dataframe(df.groupby("anomaly")[f].describe().reindex(TARGET_ORDER).round(2))
    elif view == "Class mix by hour of day":
        ct = pd.crosstab(df.hour_of_day, df.anomaly, normalize="index")[TARGET_ORDER]
        fig, ax = plt.subplots(figsize=(9, 4))
        ct.plot(kind="bar", stacked=True, color=CLASS_COLORS, ax=ax, width=.85)
        ax.set_ylabel("share of transactions"); ax.legend(title="", ncol=3, loc="upper center")
        ax.set_title("Moderate risk concentrates at night (0-6h)"); fig.tight_layout(); show(fig)
    elif view == "Class mix by category":
        c = st.selectbox("Category", [c for c in ["transaction_type", "age_group", "location_region",
                                                   "ip_prefix", "purchase_pattern"] if c in df], key="cat_feat")
        ct = pd.crosstab(df[c], df.anomaly, normalize="index")[TARGET_ORDER]
        fig, ax = plt.subplots(figsize=(8, 4))
        ct.plot(kind="bar", stacked=True, color=CLASS_COLORS, ax=ax, rot=0)
        ax.set_ylabel("share"); fig.tight_layout(); show(fig)
        st.caption("Flat bars (same mix for every value) mean the column carries no signal, e.g. "
                   "`location_region` and `ip_prefix`.")
    else:
        cols = [c for c in ["amount", "login_frequency", "session_duration", "hour_of_day",
                            "session_per_login", "amount_per_minute", "is_night", "risk_score"] if c in df]
        m = df[cols].assign(risk_level=data.clean_df["anomaly"].map({k: i for i, k in enumerate(TARGET_ORDER)}))
        fig, ax = plt.subplots(figsize=(7.5, 5.8))
        sns.heatmap(m.corr(method="spearman"), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
        ax.set_title("Spearman correlation"); fig.tight_layout(); show(fig)

# ============================ 3. TRAIN ===================================== #
with tab_train:
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Model")
        name = st.selectbox("Classifier", MODEL_NAMES, key="train_model")
        can_bal = name in SUPPORTS_BALANCE
        balance = st.checkbox("Balance classes", value=can_bal, disabled=not can_bal, key=f"balance:{name}",
                              help="class_weight='balanced' (uniform priors for Naive Bayes). "
                                   "KNN and the neural network do not support it.")
        do_tune = st.checkbox("Auto-tune (randomized search, macro-F1)", False, key="tune")
        n_iter = st.slider("Search iterations", 5, 30, 12, disabled=not do_tune, key="n_iter")
        st.markdown("**Hyperparameters**" + (" *(set by the search)*" if do_tune else ""))
        params = render_params(name, disabled=do_tune)
        go = st.button("🚀 Train & evaluate", type="primary", key="go_train")

    if go:
        with right, st.spinner(f"Training {name}..." + (" (tuning first)" if do_tune else "")):
            info = {}
            use = dict(params)
            if do_tune:
                t0 = time.time()
                best, cv_score, n_rows = tune(name, balance, n_iter, seed)
                use = {**default_params(name), **best}
                info = {"cv_macro_f1": cv_score, "tune_rows": n_rows, "tune_time": time.time() - t0}
            model, res, fit_time = fit_and_eval(name, use, balance, seed)
        st.session_state["train_result"] = dict(key=CFG_KEY, name=name, params=use, balance=bool(balance and can_bal),
                                                model=model, res=res, fit_time=fit_time, info=info)

    with right:
        R = st.session_state.get("train_result")
        if not R:
            st.info("Pick a model on the left and press **Train & evaluate**.")
        elif R["key"] != CFG_KEY:
            st.warning("The data setup changed since this model was trained. Train again to refresh.")
        else:
            res = R["res"]
            st.subheader(f"{R['name']} results")
            shown = {k: v for k, v in R["params"].items() if not k.startswith("_")}
            tag = "tuned" if R["info"] else "manual"
            st.caption(f"Parameters ({tag}): {shown} · balanced: {R['balance']} · fit {R['fit_time']:.1f}s")
            if R["info"]:
                st.caption(f"Search: cross-validated macro-F1 {R['info']['cv_macro_f1']:.3f} on "
                           f"{R['info']['tune_rows']:,} train rows ({R['info']['tune_time']:.0f}s).")
            if R["name"] == "Support Vector Machine":
                st.caption(f"Trained on a sample of {int(R['params'].get('_sample', 6000)):,} rows; evaluated on the full test set.")
            metric_row(res, baseline_acc)
            if res["accuracy"] > baseline_acc and res["macro_f1"] < baseline_f1 + 0.15 and scenario == "behavioural":
                st.caption("⚠️ Accuracy beats the baseline mostly by favouring the majority class; "
                           "judge the model on macro-F1 and per-class recall.")
            t1, t2, t3, t4, t5 = st.tabs(["Confusion matrix", "Per-class report", "ROC", "Precision-Recall",
                                          "Feature importance"])
            with t1:
                norm = st.toggle("Normalise by true class", True, key="cm_norm")
                show(fig_confusion(yte, res["pred"], norm))
            with t2:
                rep = pd.DataFrame(classification_report(yte, res["pred"], target_names=CLASS_NAMES,
                                                         output_dict=True, zero_division=0)).T
                st.dataframe(rep.round(3))
            with t3:
                show(fig_curves(res, "roc"))
            with t4:
                show(fig_curves(res, "pr"))
            with t5:
                if st.button("Compute permutation importance", key="perm_btn"):
                    with st.spinner("Shuffling features..."):
                        Xs, ys = subsample(Xte, yte, 3000, False, seed)
                        pi = permutation_importance(R["model"], Xs, ys, scoring="f1_macro", n_repeats=3,
                                                    random_state=seed, n_jobs=1)
                        R["perm"] = pd.DataFrame({"feature": Xs.columns, "mean": pi.importances_mean,
                                                  "std": pi.importances_std}).sort_values("mean")
                if "perm" in R:
                    fig, ax = plt.subplots(figsize=(6.2, 4.2))
                    ax.barh(R["perm"].feature, R["perm"]["mean"], xerr=R["perm"]["std"], color="#2a9d8f")
                    ax.set_xlabel("drop in macro-F1 when the feature is shuffled")
                    fig.tight_layout(); show(fig)
                else:
                    st.caption("Model-agnostic importance on a 3,000-row test sample.")

# ============================ 4. COMPARE =================================== #
with tab_compare:
    st.subheader("Leaderboard")
    st.caption("All models use default hyperparameters on the same split. Ranked by **macro-F1**, because "
               "accuracy rewards ignoring the minority classes.")
    c1, c2 = st.columns([1, 3])
    with c1:
        cmp_bal = st.checkbox("Balance classes where supported", True, key="cmp_bal")
        run_cmp = st.button("▶ Run comparison", type="primary", key="run_cmp")
    if run_cmp:
        rows, prog = [], c2.progress(0.0, text="Starting...")
        for i, nm in enumerate(MODEL_NAMES):
            prog.progress(i / len(MODEL_NAMES), text=f"Training {nm}...")
            try:
                _, r, ft = fit_and_eval(nm, default_params(nm), cmp_bal and nm in SUPPORTS_BALANCE, seed)
                rows.append({"model": nm, "accuracy": r["accuracy"], "macro_F1": r["macro_f1"],
                             "macro_precision": r["macro_precision"], "macro_recall": r["macro_recall"],
                             "balanced_acc": r["balanced_accuracy"], "ROC_AUC": r["macro_auc"],
                             "fit_s": ft})
            except Exception as e:                                   # keep the table going
                rows.append({"model": nm, "accuracy": np.nan, "macro_F1": np.nan, "fit_s": np.nan})
                st.warning(f"{nm} failed: {e}")
        prog.empty()
        st.session_state["cmp"] = dict(key=CFG_KEY, table=pd.DataFrame(rows))

    C = st.session_state.get("cmp")
    if C and C["key"] == CFG_KEY:
        tbl = C["table"].sort_values("macro_F1", ascending=False, ignore_index=True)
        base = pd.DataFrame([{"model": "Baseline: always 'low risk'", "accuracy": baseline_acc,
                              "macro_F1": baseline_f1}])
        full_tbl = pd.concat([tbl, base], ignore_index=True)
        best = tbl.iloc[0]
        st.success(f"Best by macro-F1: **{best['model']}** ({best['macro_F1']:.3f}); "
                   f"best by accuracy: **{tbl.sort_values('accuracy').iloc[-1]['model']}** "
                   f"({tbl['accuracy'].max():.3f}).")
        st.dataframe(full_tbl.style.format({c: "{:.3f}" for c in full_tbl.columns if c not in ("model", "fit_s")}
                                           | {"fit_s": "{:.1f}"}, na_rep="-")
                     .background_gradient(subset=["macro_F1"], cmap="Greens"), hide_index=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        order = tbl.sort_values("macro_F1")
        y = np.arange(len(order))
        ax.barh(y - .2, order["macro_F1"], .4, color="#2a9d8f", label="macro-F1")
        ax.barh(y + .2, order["accuracy"], .4, color="#adb5bd", label="accuracy")
        ax.axvline(baseline_f1, color="#e63946", ls="--", lw=1, label="baseline macro-F1")
        ax.axvline(baseline_acc, color="k", ls=":", lw=1, label="baseline accuracy")
        ax.set_yticks(y, order["model"]); ax.legend(fontsize=8, loc="lower right"); ax.set_xlim(0, 1.02)
        fig.tight_layout(); show(fig)
        st.download_button("⬇️ Results (CSV)", full_tbl.to_csv(index=False).encode(), "model_comparison.csv")
    elif C:
        st.warning("The data setup changed. Run the comparison again.")
    else:
        st.info("Press **Run comparison** to train all 8 models (about 30 s).")

    st.divider()
    st.subheader("🧪 Leakage lab")
    st.caption("Same models, three feature scenarios. If accuracy collapses when a shortcut is removed, "
               "the earlier score measured the shortcut, not the model.")
    if st.button("Run leakage lab", key="run_lab"):
        rows = []
        with st.spinner("Preprocessing the three scenarios and training two fast models..."):
            for sc in SCENARIOS:
                d = get_data(sc, split, float(test_size), scaler, bool(interactions), seed)
                for nm, mdl in {"Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=seed),
                                "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced")}.items():
                    mdl.fit(d.X_train, d.y_train); p = mdl.predict(d.X_test)
                    rows.append({"scenario": SCENARIO_LABELS[sc], "model": nm,
                                 "accuracy": (p == d.y_test).mean(),
                                 "macro_F1": f1_score(d.y_test, p, average="macro")})
        st.session_state["lab"] = pd.DataFrame(rows)
    if "lab" in st.session_state:
        lab = st.session_state["lab"]
        piv = lab.pivot(index="scenario", columns="model", values="macro_F1").loc[
            [SCENARIO_LABELS[s] for s in SCENARIOS]]
        fig, ax = plt.subplots(figsize=(8, 3.4))
        piv.plot(kind="bar", ax=ax, rot=0, color=["#264653", "#e9c46a"])
        ax.set_ylabel("macro-F1"); ax.set_xlabel(""); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
        fig.tight_layout(); show(fig)
        st.dataframe(lab.round(3), hide_index=True)

# ============================ 5. PREDICT =================================== #
with tab_predict:
    st.subheader("Score a single transaction")
    st.caption("In this synthetic dataset the inputs are correlated (e.g. scams only occur on 'new' accounts, "
               "'veteran' accounts are never high risk), so unrealistic combinations can give surprising answers.")
    R = st.session_state.get("train_result")
    options = (["Model from the 'Train' tab"] if R and R["key"] == CFG_KEY else []) + MODEL_NAMES
    choice = st.selectbox("Model", options, key="pred_model")

    @st.cache_resource(show_spinner="Training model with default parameters...")
    def default_model(key, nm, _data):
        mdl = make_model(nm, default_params(nm), nm in SUPPORTS_BALANCE, key[-1])
        X, y = _data.X_train, _data.y_train
        if nm == "Support Vector Machine":
            X, y = subsample(X, y, 6000, key[1] == "temporal", key[-1])
        return mdl.fit(X, y)

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        d_in = c1.date_input("Date", date(2022, 6, 14))
        t_in = c1.time_input("Time", dtime(3, 30))
        amount = c2.number_input("Amount", 0.0, 5000.0, 500.0, 10.0)
        login = c2.slider("Login frequency", 1, 8, 7)
        session = c3.slider("Session duration (min)", 1, 180, 120)
        age = c3.selectbox("Account profile (age group)", AGE_ORDER, index=2)
        extra = {}
        if "transaction_type" in data.feature_groups["nominal"]:
            extra["transaction_type"] = c1.selectbox("Transaction type", ["sale", "purchase", "transfer", "scam", "phishing"])
        if "risk_score" in data.feature_groups["numeric"]:
            extra["risk_score"] = c2.slider("Risk score", 0.0, 100.0, 40.0)
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        ts = datetime.combine(d_in, t_in)
        rec = pd.DataFrame([{"timestamp": ts, "hour_of_day": ts.hour, "amount": amount,
                             "login_frequency": login, "session_duration": session,
                             "age_group": age, **extra}])
        X1 = transform_new(data, rec)
        mdl = R["model"] if choice.startswith("Model from") else default_model(CFG_KEY, choice, data)
        probs = get_scores(mdl, X1)[0]
        if not hasattr(mdl, "predict_proba"):
            e = np.exp(probs - probs.max()); probs = e / e.sum()
        k = int(np.argmax(probs))
        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.markdown(f'<div class="pred" style="background:{CLASS_COLORS[k]}">{CLASS_NAMES[k].title()}'
                        f'<small>confidence {probs[k]:.0%}</small></div>', unsafe_allow_html=True)
            st.caption(f"Night-time (0-6h): {'yes' if ts.hour <= 6 else 'no'} · "
                       f"session per login: {session / (login + 1):.1f} min")
        with c2:
            fig, ax = plt.subplots(figsize=(5, 2.2))
            ax.barh(CLASS_NAMES[::-1], probs[::-1], color=CLASS_COLORS[::-1])
            ax.set_xlim(0, 1); ax.set_xlabel("probability" if hasattr(mdl, "predict_proba") else "relative score")
            fig.tight_layout(); show(fig)
        st.caption("Demo on synthetic data: not a real fraud decision.")

st.divider()
st.caption("Metaverse Transaction Risk · Streamlit + scikit-learn · preprocessing in `preprocessing.py`")
