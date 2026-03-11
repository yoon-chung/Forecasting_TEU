"""
train_lgbm.py  —  LightGBM multi-horizon rolling-origin backtest
Usage:
    python train_lgbm.py <csv_path>

For each horizon h in [1, 3, 6, 12]:
  - Build lag features (TEU lags 1-12) + exogenous covariates
  - Train a separate LGBMRegressor on chronological train split
  - Evaluate on rolling-origin test windows (n=8, same protocol as LSTM/TCN/TFT)
"""

import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from utils import load_dataset, TARGET_COL, EXO_COLS, HORIZONS, inverse_log


RANDOM_STATE = 42
N_LAGS = 12          # TEU lag features: t-1 … t-12
N_ORIGINS = 8        # rolling-origin test windows


def make_lag_features(df: pd.DataFrame, t: int) -> np.ndarray:
    """Build flat feature vector for prediction origin t.
    Features: log(TEU) lags 1..N_LAGS + z-normalized exogenous at t.
    Uses only data up to t-1 (no look-ahead).
    """
    y_log = np.log1p(df[TARGET_COL].astype(float).values)
    lags = y_log[t - N_LAGS: t][::-1]   # [t-1, t-2, …, t-12]
    exo  = df[EXO_COLS].astype(float).values[t]
    return np.concatenate([lags, exo])


def build_train_matrix(df: pd.DataFrame, train_end: int, horizon: int):
    """Build (X_train, y_train) using all valid windows up to train_end."""
    X, y = [], []
    for t in range(N_LAGS, train_end - horizon + 1):
        X.append(make_lag_features(df, t))
        y.append(np.log1p(df[TARGET_COL].iloc[t + horizon]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def rolling_origin_eval(csv_path: str):
    np.random.seed(RANDOM_STATE)

    df = load_dataset(csv_path)
    n  = len(df)

    # z-normalize exogenous columns (fit on full series — same as utils.py)
    scaler = StandardScaler()
    df[EXO_COLS] = scaler.fit_transform(df[EXO_COLS].astype(float))

    # test origins: last N_ORIGINS usable positions (same window as LSTM/TCN)
    max_h = max(HORIZONS)
    origins = list(range(n - max_h - N_ORIGINS + 1, n - max_h + 1))

    # per-horizon collectors
    collectors = {h: {"pred": [], "true": []} for h in HORIZONS}

    for h in HORIZONS:
        for t in origins:
            if t < N_LAGS + 10:
                continue

            X_tr, y_tr = build_train_matrix(df, train_end=t, horizon=h)
            if len(X_tr) < 10:
                continue

            model = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                verbose=-1,
            )
            model.fit(X_tr, y_tr)

            if t + h >= n:
                continue
            x_test = make_lag_features(df, t).reshape(1, -1)
            y_pred = inverse_log(model.predict(x_test))[0]
            y_true = df[TARGET_COL].iloc[t + h]

            if y_true > 0:
                collectors[h]["pred"].append(y_pred)
                collectors[h]["true"].append(float(y_true))

    # ── print results table (same format as other train_*.py scripts) ──
    print(f"\n[Rolling-Origin] LightGBM (seed={RANDOM_STATE})")
    print(f"{'horizon':>8} {'n':>4} {'RMSE':>15} {'MAE':>15} {'MAPE(%)':>10}")
    for h in HORIZONS:
        p = np.array(collectors[h]["pred"])
        t_arr = np.array(collectors[h]["true"])
        if len(p) == 0:
            print(f"{h:>8} {'0':>4} {'N/A':>15} {'N/A':>15} {'N/A':>10}")
            continue
        rmse = float(np.sqrt(((p - t_arr) ** 2).mean()))
        mae  = float(np.abs(p - t_arr).mean())
        mape = float(np.abs((p - t_arr) / t_arr).mean()) * 100
        print(f"{h:>8} {len(p):>4} {rmse:>15,.6f} {mae:>15,.6f} {mape:>10.6f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_lgbm.py <csv_path>")
        sys.exit(0)
    rolling_origin_eval(sys.argv[1])
