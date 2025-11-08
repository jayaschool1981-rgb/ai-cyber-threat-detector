import argparse
import yaml
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

try:
    from imblearn.over_sampling import RandomOverSampler
except Exception:
    RandomOverSampler = None

from src.utils.io import save_joblib, save_json
from src.utils.data import load_csvs

def build_model(model_cfg):
    mtype = model_cfg.get("type", "RandomForestClassifier")
    params = model_cfg.get("params", {})
    if mtype != "RandomForestClassifier":
        raise NotImplementedError(f"Only RandomForestClassifier supported in this starter. Got: {mtype}")
    return RandomForestClassifier(**params)

def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    out_cfg = cfg["output"]

    # Load data
    df = load_csvs(data_cfg.get("input_csv"), data_cfg.get("input_glob"))

    # STRIP SPACES FROM COLUMN NAMES
    df.columns = df.columns.str.strip()

    target_col = data_cfg["target_col"]
    drop_cols = data_cfg.get("drop_cols", [])

    # Defensive drops (ignore missing cols)
    for c in list(drop_cols):
        if c in df.columns:
            df = df.drop(columns=[c])

    # Basic cleaning: drop fully NA columns
    df = df.dropna(axis=1, how="all")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.fillna(0)

    # Ensure target exists
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in data. Available: {list(df.columns)[:10]} ...")

    # Separate features/target
    y = df[target_col].astype(str)  # ensure labels are strings
    X = df.drop(columns=[target_col])

    # Identify numeric vs categorical
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )

    sampler = None
    if data_cfg.get("use_class_balance", False) and RandomOverSampler is not None:
        sampler = RandomOverSampler(random_state=data_cfg.get("random_state", 42))

    model = build_model(model_cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_cfg.get("test_size", 0.2), random_state=data_cfg.get("random_state", 42), stratify=y
    )

    X_train_pre = pre.fit_transform(X_train)
    X_test_pre = pre.transform(X_test)

    # Extract transformed feature names
    try:
        feature_names = []
        if num_cols:
            feature_names += [f"num__{c}" for c in num_cols]
        if cat_cols:
            ohe = pre.named_transformers_["cat"]
            ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
            feature_names += ohe_names
    except Exception:
        feature_names = [f"f{i}" for i in range(X_train_pre.shape[1])]

    # Optional resampling
    if sampler is not None:
        X_train_pre, y_train = sampler.fit_resample(X_train_pre, y_train)

    model.fit(X_train_pre, y_train)

    y_pred = model.predict(X_test_pre)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred)[:2000])

    # Save artifacts
    save_joblib(model, out_cfg["model_path"])
    save_joblib(pre, out_cfg["scaler_path"])
    save_json(feature_names, out_cfg["feature_cols_path"])

    print("Saved:")
    print(" -", out_cfg["model_path"])
    print(" -", out_cfg["scaler_path"])
    print(" -", out_cfg["feature_cols_path"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    main(args.config)
