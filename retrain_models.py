import os
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(BASE_DIR, "training_data", "floor_plan_samples.parquet")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def train_xgb(df: pd.DataFrame):
    drop_cols = [
        # non-numeric / leakage
        "error_type",
        "plot_size_band",
        "plot_authority",
        "plot_category",
        "plot_layout_type",
        # zones
        "masterbedr_zone",
        "toiletatta_zone",
        "living_zone",
        "kitchen_zone",
        "verandah_zone",
        "bedroom2_zone",
        "bedroom3_zone",
        "dining_zone",
        "toiletcomm_zone",
        "utility_zone",
        "pooja_zone",
        "bedroom4_zone",
        "store_zone",
        # plot size dependence removal (Option B to fix ML distribution mismatch)
        "plot_w",
        "plot_d",
        "plot_area",
        "net_w",
        "net_d",
        "net_area",
        # score leakage
        "score_overall",
        "score_vastu",
        "score_nbc",
        "score_circulation",
        "score_adjacency",
    ]

    y = df["is_valid"].astype(int)
    X = df.drop(columns=["is_valid"] + drop_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number]).astype(np.float32).fillna(0.0)

    feature_cols = list(X.columns)
    print("=== XGBOOST FEATURES ===")
    print(f"Feature count: {len(feature_cols)}")
    print("Feature list:")
    print(feature_cols)
    print()

    pos = int(y.sum())
    neg = int((y == 0).sum())
    scale_pos_weight = float(neg) / float(max(pos, 1))
    print(f"Class balance: pos={pos} neg={neg} scale_pos_weight={scale_pos_weight:.6f}")
    print()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_te, y_te)],
        verbose=False,
        callbacks=[xgb.callback.EarlyStopping(rounds=20, save_best=True)],
    )

    probs = clf.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_te, probs)
    prec = precision_score(y_te, preds, zero_division=0)
    rec = recall_score(y_te, preds, zero_division=0)
    f1 = f1_score(y_te, preds, zero_division=0)

    print("=== XGBOOST METRICS (TEST) ===")
    print(f"AUC:       {auc:.6f}")
    print(f"Precision: {prec:.6f}")
    print(f"Recall:    {rec:.6f}")
    print(f"F1:        {f1:.6f}")
    print()

    booster = clf.get_booster()
    gain = booster.get_score(importance_type="gain")
    imp = pd.Series({c: float(gain.get(c, 0.0)) for c in feature_cols}).sort_values(ascending=False)
    print("=== TOP 15 FEATURE IMPORTANCES (GAIN) ===")
    for name, val in imp.head(15).items():
        print(f"  {name}: {val:.6f}")
    print()

    out_path = os.path.join(MODELS_DIR, "constraint_scorer.pkl")
    joblib.dump(clf, out_path)
    print(f"Saved: {out_path}")
    print()

    return clf, X_tr


def train_dim_model(df: pd.DataFrame):
    x_cols = [
        "plot_w",
        "plot_d",
        "plot_area",
        "net_w",
        "net_d",
        "net_area",
        "bhk",
        "facing_code",
        "climate_code",
    ]
    y_cols = [
        "masterbedr_w", "masterbedr_d",
        "toiletatta_w", "toiletatta_d",
        "living_w", "living_d",
        "kitchen_w", "kitchen_d",
        "verandah_w", "verandah_d",
        "bedroom2_w", "bedroom2_d",
        "bedroom3_w", "bedroom3_d",
        "dining_w", "dining_d",
        "toiletcomm_w", "toiletcomm_d",
        "utility_w", "utility_d",
        "pooja_w", "pooja_d",
        "bedroom4_w", "bedroom4_d",
        "store_w", "store_d",
        "masterbedr_area", "toiletatta_area", "living_area",
        "kitchen_area", "verandah_area", "bedroom2_area",
        "bedroom3_area", "dining_area", "toiletcomm_area",
        "utility_area", "pooja_area", "bedroom4_area", "store_area",
    ]

    X = df[x_cols].astype(np.float32).fillna(0.0).to_numpy()
    Y = df[y_cols].astype(np.float32).fillna(0.0).to_numpy()

    tf.random.set_seed(42)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(len(x_cols),)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(len(y_cols), activation="linear"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="mse",
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
    ]

    print("=== KERAS DIM MODEL ===")
    print(f"Inputs:  {len(x_cols)}")
    print(f"Outputs: {len(y_cols)}")
    print()

    model.fit(
        X,
        Y,
        epochs=100,
        batch_size=256,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    out_path = os.path.join(MODELS_DIR, "room_dimensions.h5")
    model.save(out_path)
    print(f"Saved: {out_path}")
    print()

    return model, x_cols, y_cols


def train_shap_explainer(clf, X_train: pd.DataFrame):
    print("=== SHAP EXPLAINER ===")
    sample = X_train.sample(n=min(5000, len(X_train)), random_state=42)
    explainer = shap.TreeExplainer(clf)
    # Touch once to ensure it can compute without error on the sample.
    _ = explainer.shap_values(sample.iloc[:50])
    out_path = os.path.join(MODELS_DIR, "shap_explainer.pkl")
    joblib.dump(explainer, out_path)
    print(f"Saved: {out_path}")
    print()
    return explainer


def verify_models():
    print("=== VERIFICATION ===")
    clf = joblib.load(os.path.join(MODELS_DIR, "constraint_scorer.pkl"))

    cols = list(getattr(clf, "feature_names_in_", []))
    if not cols:
        raise RuntimeError("constraint_scorer.pkl missing feature_names_in_.")

    row = {c: 0.0 for c in cols}
    row["plot_w"] = 12.0
    row["plot_d"] = 15.0
    row["plot_area"] = 180.0
    row["net_w"] = 10.8
    row["net_d"] = 12.0
    row["net_area"] = 129.6
    row["bhk"] = 2.0
    row["facing_code"] = 0.0
    row["climate_code"] = 1.0
    X_one = pd.DataFrame([row], columns=cols).astype(np.float32)

    prob = float(clf.predict_proba(X_one)[0][1])
    print(f"score_valid for zero-filled 12x15 test row: {prob:.6f}")
    print("Expected: > 0.1")
    print()

    paths = [
        os.path.join(MODELS_DIR, "constraint_scorer.pkl"),
        os.path.join(MODELS_DIR, "room_dimensions.h5"),
        os.path.join(MODELS_DIR, "shap_explainer.pkl"),
    ]
    print("=== MODEL FILE SIZES ===")
    for p in paths:
        size = os.path.getsize(p) if os.path.exists(p) else -1
        print(f"{os.path.basename(p)}: {size} bytes")


def main():
    df = pd.read_parquet(PARQUET_PATH)
    clf, X_train = train_xgb(df)
    # _model, _x_cols, _y_cols = train_dim_model(df) # Skipped to re-use working dim_model
    _explainer = train_shap_explainer(clf, X_train)
    verify_models()


if __name__ == "__main__":
    main()
