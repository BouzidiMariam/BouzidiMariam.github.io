"""
preprocessing.py
================
Reproducible, leakage-aware preprocessing for the Metaverse Transaction Risk project.

Single source of truth used by:
  * the notebook  (preprocessing.ipynb)  -> explanation + EDA
  * the Streamlit app (app.py)           -> training / evaluation / inference

Design principles
-----------------
1. Every cleaning / dropping decision is *measured* on the data (audit), not assumed.
2. Anything that learns from data (imputation, clipping, scaling, encoding) is fitted
   on the TRAIN split only and wrapped in a scikit-learn ColumnTransformer, so the
   exact same object can be reused for inference and there is no train/test leakage.
3. The target is encoded with a FIXED order (low < moderate < high), which removes the
   LabelEncoder alphabetical-order bug of the first version.
4. Feature *scenarios* make label leakage explicit instead of hiding it:
        full         every feature (includes risk_score -> trivially leaky)
        no_score     risk_score removed
        behavioural  risk_score and transaction_type removed (hardest, most honest)

Usage
-----
    from preprocessing import prepare_dataset, PreprocessConfig
    data = prepare_dataset(config=PreprocessConfig(scenario="behavioural"))
    data.X_train, data.y_train, data.preprocessor, data.audit ...

CLI
---
    python preprocessing.py --scenario behavioural --split stratified --out artifacts
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (StratifiedKFold, TimeSeriesSplit,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, OrdinalEncoder,
                                   StandardScaler)
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.utils.class_weight import compute_class_weight

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DATA_FILENAMES = [
    "metaverse_transactions_dataset.csv",
    "metaverse_transactions_dataset__1_.csv",
    "metaverse_transactions_dataset (1).csv",
]
REQUIRED_COLUMNS = [
    "timestamp", "hour_of_day", "sending_address", "receiving_address", "amount",
    "transaction_type", "location_region", "ip_prefix", "login_frequency",
    "session_duration", "purchase_pattern", "age_group", "risk_score", "anomaly",
]
ID_COLUMNS = ["sending_address", "receiving_address"]       # identifiers, never features
RAW_TARGET = "anomaly"
TARGET = "risk_level"
TARGET_ORDER = ["low_risk", "moderate_risk", "high_risk"]    # fixed ordinal order
CLASS_NAMES = ["low risk", "moderate risk", "high risk"]     # matches encoded 0/1/2
AGE_ORDER = ["new", "established", "veteran"]
KEEP_ON_REDUNDANCY = {"age_group"}      # if a redundant pair contains one of these, keep it

NUMERIC_VALID_RANGES = {                  # values outside -> set to NaN then imputed
    "amount": (0, None),
    "login_frequency": (0, None),
    "session_duration": (0, None),
    "risk_score": (0, 100),
    "hour_of_day": (0, 23),
}

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "full": dict(
        drop_raw=[],
        description="All features, including risk_score. Trivially leaky (a single-feature "
                    "tree reaches 100%); kept only to demonstrate the leakage."),
    "no_score": dict(
        drop_raw=["risk_score"],
        description="risk_score removed (it is the risk assessment itself). Still easy: "
                    "transaction_type + age_group + night-time almost determine the label."),
    "behavioural": dict(
        drop_raw=["risk_score", "transaction_type"],
        description="risk_score and transaction_type removed. Only who/when/how-long "
                    "signals remain: the hardest and most honest setting."),
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessConfig:
    scenario: str = "no_score"             # full | no_score | behavioural
    split: str = "stratified"              # stratified | temporal
    test_size: float = 0.30
    scaler: str = "standard"               # standard | minmax  (use minmax for MultinomialNB)
    random_state: int = 42
    cyclical: bool = True                  # sin/cos encoding of the hour (and calendar, if used)
    calendar_features: bool = False        # weekday / month / weekend: MI ~ 0 in this data -> off
    interactions: bool = True              # amount_per_minute, session_per_login
    use_sender_history: bool = False       # causal per-sender history features
    night_hours: Tuple[int, int] = (0, 6)  # inclusive range flagged as "night"
    clip_quantiles: Tuple[float, float] = (0.001, 0.999)
    drop_duplicates: bool = True
    auto_drop: bool = True                 # drop redundant / noise categoricals found by the audit
    noise_cramers_v: float = 0.02          # categorical is noise if V < this ...
    noise_p_value: float = 0.01            # ... and chi2 p-value > this
    verbose: bool = False

    def __post_init__(self):
        if self.scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {list(SCENARIOS)}")
        if self.split not in ("stratified", "temporal"):
            raise ValueError("split must be 'stratified' or 'temporal'")
        if self.scaler not in ("standard", "minmax"):
            raise ValueError("scaler must be 'standard' or 'minmax'")
        if not 0.05 <= self.test_size <= 0.5:
            raise ValueError("test_size must be between 0.05 and 0.5")


# --------------------------------------------------------------------------- #
# Custom transformer: quantile clipping (winsorisation) learnt on TRAIN only
# --------------------------------------------------------------------------- #
class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to [q_low, q_high] quantiles learnt during fit()."""

    def __init__(self, lower: float = 0.001, upper: float = 0.999):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(arr, self.lower, axis=0)
        self.hi_ = np.nanquantile(arr, self.upper, axis=0)
        self.n_features_in_ = arr.shape[1]
        self.feature_names_in_ = np.asarray(
            X.columns if hasattr(X, "columns") else [f"x{i}" for i in range(arr.shape[1])],
            dtype=object)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_ if input_features is None else input_features,
                          dtype=object)


# --------------------------------------------------------------------------- #
# Loading & validation
# --------------------------------------------------------------------------- #
def find_data_file(path: Optional[str | Path] = None) -> Path:
    """Locate the CSV (explicit path, else next to this file / cwd / uploads)."""
    if path is not None:
        p = Path(path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Dataset not found: {p}")
    here = Path(__file__).resolve().parent
    for folder in (here, Path.cwd(), here / "data", Path("/mnt/user-data/uploads")):
        for name in DATA_FILENAMES:
            if (folder / name).exists():
                return folder / name
    raise FileNotFoundError(f"Put one of {DATA_FILENAMES} next to preprocessing.py")


def load_raw(path: Optional[str | Path] = None) -> pd.DataFrame:
    return pd.read_csv(find_data_file(path))


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Dataset is empty")


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def _fmt_ip(v):
    if pd.isna(v):
        return np.nan
    try:
        return f"{float(v):g}"           # 192.0 -> '192', 172.16 -> '172.16'
    except (TypeError, ValueError):
        return str(v).strip()


def clean(df: pd.DataFrame, has_target: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Type fixing, text normalisation, range checks, de-duplication, time ordering.
    Returns (clean_df, log). Never learns anything from the data."""
    log: Dict[str, Any] = {"rows_in": int(len(df))}
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]

    # --- text normalisation ------------------------------------------------
    for c in ["transaction_type", "purchase_pattern", "age_group", RAW_TARGET]:
        if c in out:
            out[c] = (out[c].astype(object).map(
                lambda v: str(v).strip().lower().replace("-", "_").replace(" ", "_")
                if pd.notna(v) else np.nan))
    if "location_region" in out:
        out["location_region"] = out["location_region"].astype(object).map(
            lambda v: str(v).strip() if pd.notna(v) else np.nan)
    if "ip_prefix" in out:       # numeric-looking but really a categorical network class
        out["ip_prefix"] = out["ip_prefix"].map(_fmt_ip).astype(object)

    # --- timestamp ---------------------------------------------------------
    if "timestamp" in out:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        log["unparseable_timestamps"] = int(out["timestamp"].isna().sum())

    # --- numeric coercion + range checks ------------------------------------
    invalid: Dict[str, int] = {}
    for c, (lo, hi) in NUMERIC_VALID_RANGES.items():
        if c not in out:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
        bad = pd.Series(False, index=out.index)
        if lo is not None:
            bad |= out[c] < lo
        if hi is not None:
            bad |= out[c] > hi
        invalid[c] = int(bad.sum())
        out.loc[bad, c] = np.nan
    log["out_of_range_set_to_nan"] = invalid

    # --- hour_of_day must agree with timestamp ------------------------------
    if "timestamp" in out and "hour_of_day" in out:
        ts_hour = out["timestamp"].dt.hour
        mismatch = (out["hour_of_day"] != ts_hour) & ts_hour.notna()
        log["hour_mismatch_fixed"] = int(mismatch.sum())
        out["hour_of_day"] = out["hour_of_day"].where(~mismatch, ts_hour)
        out["hour_of_day"] = out["hour_of_day"].fillna(ts_hour)

    # --- target ------------------------------------------------------------
    if has_target:
        no_target = out[RAW_TARGET].isna() | ~out[RAW_TARGET].isin(TARGET_ORDER)
        log["rows_without_valid_target_dropped"] = int(no_target.sum())
        out = out.loc[~no_target].copy()

    # --- duplicates + time order (training only: inference must keep rows and order) ----
    if has_target:
        n_dup = int(out.duplicated().sum())
        log["duplicate_rows"] = n_dup
        if n_dup:
            out = out.drop_duplicates()
        if "timestamp" in out:       # needed for the temporal split & causal features
            out = out.sort_values("timestamp", kind="stable")
    else:
        log["duplicate_rows"] = 0
    out = out.reset_index(drop=True)
    log["rows_out"] = int(len(out))
    return out, log


# --------------------------------------------------------------------------- #
# Feature engineering (row-wise or causal -> no information from the test set)
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    out = df.copy()
    ts = out["timestamp"]
    lo, hi = cfg.night_hours

    out["day_of_week"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(float).where(ts.notna())
    out["is_night"] = out["hour_of_day"].between(lo, hi).astype(float).where(out["hour_of_day"].notna())

    if cfg.cyclical:
        for name, col, period in [("hour", "hour_of_day", 24), ("dow", "day_of_week", 7),
                                  ("month", "month", 12)]:
            angle = 2 * np.pi * out[col] / period
            out[f"{name}_sin"] = np.sin(angle)
            out[f"{name}_cos"] = np.cos(angle)

    if cfg.interactions:
        out["amount_per_minute"] = out["amount"] / (out["session_duration"] + 1.0)
        out["session_per_login"] = out["session_duration"] / (out["login_frequency"] + 1.0)

    if cfg.use_sender_history and "sending_address" in out:
        g = out.groupby("sending_address", sort=False)
        out["sender_prior_tx"] = g.cumcount().astype(float)      # only PAST transactions
        out["sender_hours_since_last"] = g["timestamp"].diff().dt.total_seconds() / 3600.0
    return out


def encode_target(s: pd.Series) -> pd.Series:
    """Fixed ordinal encoding: low=0, moderate=1, high=2 (replaces alphabetical LabelEncoder)."""
    mapping = {k: i for i, k in enumerate(TARGET_ORDER)}
    return s.map(mapping).astype(int).rename(TARGET)


# --------------------------------------------------------------------------- #
# Data audit: measured evidence behind every decision
# --------------------------------------------------------------------------- #
def cramers_v(a: pd.Series, b: pd.Series) -> Tuple[float, float]:
    ct = pd.crosstab(a, b)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0, 1.0
    chi2, p, _, _ = chi2_contingency(ct, correction=False)
    n = ct.values.sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))), float(p)


def profile_raw(raw: pd.DataFrame) -> Dict[str, Any]:
    num = raw.select_dtypes("number")
    return {
        "shape": list(raw.shape),
        "missing_by_column": {c: int(v) for c, v in raw.isna().sum().items() if v},
        "duplicate_rows": int(raw.duplicated().sum()),
        "unique_senders": int(raw["sending_address"].nunique()),
        "unique_receivers": int(raw["receiving_address"].nunique()),
        "numeric_summary": num.describe().round(3).to_dict(),
        "class_distribution": raw[RAW_TARGET].value_counts().to_dict(),
        "categorical_levels": {c: raw[c].nunique() for c in
                               ["transaction_type", "location_region", "ip_prefix",
                                "purchase_pattern", "age_group"]},
    }


def categorical_analysis(df: pd.DataFrame, cats: List[str], y: pd.Series,
                         cfg: PreprocessConfig) -> Dict[str, Any]:
    """Association with the target (Cramér's V, chi2 p) + pairwise redundancy."""
    assoc = {}
    for c in cats:
        v, p = cramers_v(df[c], y)
        assoc[c] = {"cramers_v": round(v, 4), "p_value": round(p, 4),
                    "is_noise": bool(v < cfg.noise_cramers_v and p > cfg.noise_p_value)}
    redundant = []
    for i, a in enumerate(cats):
        for b in cats[i + 1:]:
            v, _ = cramers_v(df[a], df[b])
            if v >= 0.99:
                redundant.append({"a": a, "b": b, "cramers_v": round(v, 4)})
    return {"association_with_target": assoc, "redundant_pairs": redundant}


def leakage_scan(df: pd.DataFrame, y: pd.Series, features: List[str],
                 seed: int = 0, max_rows: int = 40000) -> pd.DataFrame:
    """Accuracy of a depth-6 tree trained on ONE feature at a time (3-fold CV).
    ~100% => that feature leaks the label; far above the majority baseline => strong."""
    idx = df.sample(min(len(df), max_rows), random_state=seed).index
    ys = y.loc[idx]
    base = float(ys.value_counts(normalize=True).max())
    rows = []
    for col in features:
        x = df.loc[idx, [col]]
        x = pd.get_dummies(x, dtype=int) if x[col].dtype == object or str(x[col].dtype) in ("str", "string") \
            else x.fillna(-1)
        acc = cross_val_score(DecisionTreeClassifier(max_depth=6, random_state=seed),
                              x, ys, cv=3).mean()
        flag = "LEAK" if acc >= 0.99 else ("strong" if acc >= base + 0.05 else "weak")
        rows.append({"feature": col, "single_feature_accuracy": round(float(acc), 4),
                     "majority_baseline": round(base, 4), "verdict": flag})
    return pd.DataFrame(rows).sort_values("single_feature_accuracy", ascending=False,
                                          ignore_index=True)


def mutual_information(df: pd.DataFrame, y: pd.Series, numeric: List[str]) -> pd.DataFrame:
    X = df[numeric].fillna(df[numeric].median())
    mi = mutual_info_classif(X, y, random_state=0)
    return (pd.DataFrame({"feature": numeric, "mutual_information": mi.round(4)})
            .sort_values("mutual_information", ascending=False, ignore_index=True))


def rule_probe(df: pd.DataFrame, y: pd.Series, features: List[str], depth: int = 4) -> str:
    """Shallow tree printed as text: reveals when the label is a simple rule."""
    X = pd.get_dummies(df[features], dtype=int).fillna(-1)
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X, y)
    return export_text(tree, feature_names=list(X.columns))


# --------------------------------------------------------------------------- #
# Split + pipeline
# --------------------------------------------------------------------------- #
def split_data(X: pd.DataFrame, y: pd.Series, cfg: PreprocessConfig):
    if cfg.split == "temporal":                       # train on the past, test on the future
        n_train = int(len(X) * (1 - cfg.test_size))
        return X.iloc[:n_train], X.iloc[n_train:], y.iloc[:n_train], y.iloc[n_train:]
    return train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_state,
                            stratify=y)


def build_preprocessor(numeric: List[str], binary: List[str], nominal: List[str],
                       ordinal: List[str], cfg: PreprocessConfig) -> ColumnTransformer:
    scaler = StandardScaler() if cfg.scaler == "standard" else MinMaxScaler()
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clip", QuantileClipper(*cfg.clip_quantiles)),
            ("scale", scaler)]), numeric))
    if binary:
        transformers.append(("bin", SimpleImputer(strategy="most_frequent"), binary))
    if nominal:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), nominal))
    if ordinal:
        transformers.append(("ord", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(categories=[AGE_ORDER] * len(ordinal),
                                      handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scale", MinMaxScaler())]), ordinal))
    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    feature_names: List[str]
    feature_groups: Dict[str, List[str]]
    class_names: List[str]
    class_weights: Dict[int, float]
    config: PreprocessConfig
    cleaning_log: Dict[str, Any]
    audit: Dict[str, Any]
    dropped: Dict[str, str]
    steps: List[Dict[str, str]] = field(default_factory=list)
    clean_df: Optional[pd.DataFrame] = None          # cleaned + engineered, before encoding

    def cv(self, n_splits: int = 5):
        """Cross-validation splitter consistent with the chosen split strategy."""
        if self.config.split == "temporal":
            return TimeSeriesSplit(n_splits=n_splits)
        return StratifiedKFold(n_splits, shuffle=True, random_state=self.config.random_state)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def prepare_dataset(path: Optional[str | Path] = None,
                    config: Optional[PreprocessConfig] = None,
                    raw: Optional[pd.DataFrame] = None) -> PreparedData:
    cfg = config or PreprocessConfig()
    steps: List[Dict[str, str]] = []

    def step(name: str, detail: str):
        steps.append({"step": name, "detail": detail})
        if cfg.verbose:
            print(f"[{len(steps):02d}] {name}: {detail}")

    # 1. load & validate
    raw = load_raw(path) if raw is None else raw
    validate_schema(raw)
    step("Load & validate", f"{raw.shape[0]:,} rows x {raw.shape[1]} columns, schema OK")
    raw_profile = profile_raw(raw)

    # 2. clean
    df, log = clean(raw)
    step("Clean", f"{log['rows_in']:,} -> {log['rows_out']:,} rows | duplicates removed: "
                  f"{log['duplicate_rows']} | out-of-range values -> NaN: "
                  f"{sum(log['out_of_range_set_to_nan'].values())} | hour/timestamp mismatches fixed: "
                  f"{log.get('hour_mismatch_fixed', 0)} | sorted by timestamp")

    # 3. features + target
    df = engineer_features(df, cfg)
    y_all = encode_target(df[RAW_TARGET])
    step("Encode target", "low_risk=0 < moderate_risk=1 < high_risk=2 (fixed order, class names "
                          "now match the encoding)")
    time_feats = ["is_night"] + (["hour_sin", "hour_cos"] if cfg.cyclical else [])
    if cfg.calendar_features:
        time_feats += ["is_weekend"] + (["dow_sin", "dow_cos", "month_sin", "month_cos"]
                                         if cfg.cyclical else ["day_of_week", "month"])
    step("Engineer features",
         f"time: {', '.join(time_feats)}; night={cfg.night_hours[0]}-{cfg.night_hours[1]}h"
         + ("" if cfg.calendar_features else " (weekday/month/weekend computed for the audit but excluded: mutual information ~ 0)")
         + ("; interactions: amount_per_minute, session_per_login" if cfg.interactions else "")
         + ("; sender history (causal)" if cfg.use_sender_history else ""))

    # 4. audit (on the full cleaned data: descriptive, drives *which columns* are kept)
    cats = ["transaction_type", "location_region", "ip_prefix", "purchase_pattern", "age_group"]
    num_base = ["amount", "login_frequency", "session_duration", "hour_of_day", "risk_score"]
    engineered_num = [c for c in df.columns if c in
                      ["amount_per_minute", "session_per_login", "is_night", "is_weekend",
                       "day_of_week", "month", "sender_prior_tx", "sender_hours_since_last"]]
    cat_info = categorical_analysis(df, cats, y_all, cfg)
    scan = leakage_scan(df, y_all, num_base + cats + engineered_num)
    mi = mutual_information(df, y_all, num_base + engineered_num)
    probe_feats = [c for c in cats + ["amount", "login_frequency", "session_duration",
                                      "hour_of_day"] if c in df]
    audit = {
        "raw_profile": raw_profile,
        "categorical_analysis": cat_info,
        "leakage_scan": scan.to_dict("records"),
        "mutual_information": mi.to_dict("records"),
        "rule_probe_without_risk_score": rule_probe(df, y_all, probe_feats),
    }
    leaks = scan.loc[scan.verdict == "LEAK", "feature"].tolist()
    step("Audit", f"leakage scan flagged {leaks or 'nothing'}; redundant pairs: "
                  f"{[(p['a'], p['b']) for p in cat_info['redundant_pairs']] or 'none'}; noise "
                  f"categoricals: {[c for c, v in cat_info['association_with_target'].items() if v['is_noise']] or 'none'}")

    # 5. decide drops
    dropped: Dict[str, str] = {c: "identifier" for c in ID_COLUMNS}
    dropped["timestamp"] = "replaced by engineered time features"
    for c in SCENARIOS[cfg.scenario]["drop_raw"]:
        dropped[c] = f"scenario '{cfg.scenario}'"
    if cfg.scenario == "full" and leaks:
        warnings.warn(f"Scenario 'full' keeps leaking feature(s) {leaks}: metrics will be "
                      f"meaninglessly high.")
    if cfg.auto_drop:
        for pair in cat_info["redundant_pairs"]:
            a, b = pair["a"], pair["b"]
            victim = b if a in KEEP_ON_REDUNDANCY or b not in KEEP_ON_REDUNDANCY else a
            dropped.setdefault(victim, f"duplicate of '{a if victim == b else b}' "
                                       f"(Cramér's V={pair['cramers_v']})")
        for c, v in cat_info["association_with_target"].items():
            if v["is_noise"]:
                dropped.setdefault(c, f"no association with target (V={v['cramers_v']}, "
                                      f"p={v['p_value']})")
    step("Select features", "dropped: " + "; ".join(f"{k} [{v}]" for k, v in dropped.items()
                                                    if k not in ID_COLUMNS + ["timestamp"])
         + f"; identifiers {ID_COLUMNS} and raw timestamp removed")

    # 6. feature groups
    cal_num = ((["dow_sin", "dow_cos", "month_sin", "month_cos"] if cfg.cyclical
                else ["day_of_week", "month"]) if cfg.calendar_features else [])
    numeric = [c for c in ["amount", "login_frequency", "session_duration", "hour_of_day",
                           "risk_score", "amount_per_minute", "session_per_login",
                           "sender_prior_tx", "sender_hours_since_last"]
               + (["hour_sin", "hour_cos"] if cfg.cyclical else []) + cal_num
               if c in df and c not in dropped]
    binary = [c for c in ["is_night"] + (["is_weekend"] if cfg.calendar_features else [])
              if c not in dropped]
    nominal = [c for c in ["transaction_type", "location_region", "ip_prefix", "purchase_pattern"]
               if c not in dropped]
    ordinal = [c for c in ["age_group"] if c not in dropped]
    X_all = df[numeric + binary + nominal + ordinal].copy()
    for c in nominal + ordinal:
        X_all[c] = X_all[c].astype(object).where(X_all[c].notna(), np.nan)

    # 7. split BEFORE fitting any learnable transformer
    X_tr, X_te, y_tr, y_te = split_data(X_all, y_all, cfg)
    d_tr = {k: round(float(v), 4) for k, v in y_tr.value_counts(normalize=True).sort_index().items()}
    d_te = {k: round(float(v), 4) for k, v in y_te.value_counts(normalize=True).sort_index().items()}
    step("Split", f"{cfg.split}, test={cfg.test_size:.0%}: train {len(X_tr):,} / test {len(X_te):,}; "
                  f"class share train {d_tr} vs test {d_te}")

    # 8. fit transformer on TRAIN only
    pre = build_preprocessor(numeric, binary, nominal, ordinal, cfg)
    Xtr_t = pre.fit_transform(X_tr)
    Xte_t = pre.transform(X_te)
    names = list(Xtr_t.columns)
    step("Fit transformer (train only)",
         f"numeric: median-impute -> clip to {cfg.clip_quantiles} quantiles -> {cfg.scaler} scale; "
         f"binary: passthrough; nominal: one-hot (unknown-safe); ordinal: age_group "
         f"{AGE_ORDER} -> 0..1 | {len(names)} final features")

    # 9. class weights for imbalance
    classes = np.array(sorted(y_tr.unique()))
    cw = {int(k): round(float(v), 4) for k, v in zip(classes, compute_class_weight("balanced", classes=classes, y=y_tr))}
    step("Imbalance", f"class_weight='balanced' weights {cw} (train class shares {d_tr})")

    return PreparedData(
        X_train=Xtr_t, X_test=Xte_t, y_train=y_tr,
        y_test=y_te, preprocessor=pre, feature_names=names,
        feature_groups={"numeric": numeric, "binary": binary, "nominal": nominal,
                        "ordinal": ordinal},
        class_names=CLASS_NAMES, class_weights={int(k): float(v) for k, v in cw.items()},
        config=cfg, cleaning_log=log, audit=audit, dropped=dropped, steps=steps, clean_df=df)


# --------------------------------------------------------------------------- #
# Inference on new raw records (used by the app's "predict one transaction" form)
# --------------------------------------------------------------------------- #
def transform_new(data: PreparedData, records: pd.DataFrame) -> pd.DataFrame:
    """Raw-schema records (no target needed) -> model-ready features."""
    rec = records.copy()
    for c in ID_COLUMNS:                               # not needed at inference
        if c not in rec:
            rec[c] = "unknown"
    if "anomaly" not in rec:
        rec["anomaly"] = TARGET_ORDER[0]
    if "risk_score" not in rec:
        rec["risk_score"] = np.nan
    cleaned, _ = clean(rec, has_target=False)
    eng = engineer_features(cleaned, data.config)
    cols = sum(data.feature_groups.values(), [])
    X = eng[cols].copy()
    for c in data.feature_groups["nominal"] + data.feature_groups["ordinal"]:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return data.preprocessor.transform(X)


# --------------------------------------------------------------------------- #
# Reporting / persistence
# --------------------------------------------------------------------------- #
def render_report(data: PreparedData) -> str:
    c = data.config
    lines = [f"# Preprocessing report", "",
             f"**Scenario:** `{c.scenario}` - {SCENARIOS[c.scenario]['description']}",
             f"**Split:** {c.split} ({1 - c.test_size:.0%}/{c.test_size:.0%}) | "
             f"**Scaler:** {c.scaler} | **Seed:** {c.random_state}", "", "## Pipeline steps", ""]
    lines += [f"{i}. **{s['step']}** - {s['detail']}" for i, s in enumerate(data.steps, 1)]
    lines += ["", "## Single-feature leakage scan", "",
              "| feature | single-feature accuracy | baseline | verdict |", "|---|---|---|---|"]
    lines += [f"| {r['feature']} | {r['single_feature_accuracy']:.3f} | "
              f"{r['majority_baseline']:.3f} | {r['verdict']} |" for r in data.audit["leakage_scan"]]
    lines += ["", "## Categorical association with the target", "",
              "| feature | Cramér's V | p-value | noise? |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['cramers_v']} | {v['p_value']} | {v['is_noise']} |"
              for k, v in data.audit["categorical_analysis"]["association_with_target"].items()]
    lines += ["", "## Dropped columns", ""] + [f"- `{k}`: {v}" for k, v in data.dropped.items()]
    lines += ["", "## Final feature groups", ""] + \
             [f"- **{k}** ({len(v)}): {', '.join(v)}" for k, v in data.feature_groups.items()]
    lines += ["", f"Final model matrix: **{len(data.feature_names)}** columns.", "",
              "## Rule probe (depth-4 tree, risk_score excluded)", "", "```",
              data.audit["rule_probe_without_risk_score"].rstrip(), "```"]
    return "\n".join(lines)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def save_artifacts(data: PreparedData, out_dir: str | Path = "artifacts") -> Path:
    import joblib
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(data.preprocessor, out / "preprocessor.joblib")
    pd.concat([data.X_train, data.y_train.rename(TARGET)], axis=1).to_csv(out / "train_processed.csv", index=False)
    pd.concat([data.X_test, data.y_test.rename(TARGET)], axis=1).to_csv(out / "test_processed.csv", index=False)
    (out / "audit.json").write_text(json.dumps(
        {"config": asdict(data.config), "cleaning_log": data.cleaning_log, "dropped": data.dropped,
         "feature_groups": data.feature_groups, "class_weights": data.class_weights,
         "audit": data.audit}, indent=2, default=_jsonable), encoding="utf-8")
    (out / "preprocessing_report.md").write_text(render_report(data), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description="Run the Metaverse preprocessing pipeline")
    ap.add_argument("--data", default=None)
    ap.add_argument("--scenario", default="no_score", choices=list(SCENARIOS))
    ap.add_argument("--split", default="stratified", choices=["stratified", "temporal"])
    ap.add_argument("--scaler", default="standard", choices=["standard", "minmax"])
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()
    cfg = PreprocessConfig(scenario=a.scenario, split=a.split, scaler=a.scaler, verbose=True)
    data = prepare_dataset(a.data, cfg)
    print(f"\nSaved to {save_artifacts(data, a.out).resolve()}")
    print(f"X_train {data.X_train.shape} | X_test {data.X_test.shape}")


if __name__ == "__main__":
    main()
