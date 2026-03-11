import math
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd

# Horizons for this pilot; extend to [1,3,6,12,36,60] on full data
HORIZONS = [1, 3, 6, 12]

EXO_COLS = [
    "freightos_index",
    "diesel_price_usd",
    "gscpi",
    "us_manuf_pmi",
    "peak_season",
]

TARGET_COL = "total_teu"

@dataclass
class WindowConfig:
    lookback: int = 36
    horizons: List[int] = None
    log_target: bool = True
    def __post_init__(self):
        if self.horizons is None:
            self.horizons = HORIZONS

def load_dataset(csv_path: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    # detect date column
    candidates = [c for c in raw.columns if c.lower() in ("date", "month", "timestamp")]
    if not candidates:
        raise ValueError(f"No date-like column found. Columns: {list(raw.columns)}")
    dc = candidates[0]
    raw = raw.rename(columns={dc: "date"})
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    if raw["date"].isna().any():
        bad = raw.index[raw["date"].isna()].tolist()[:5]
        raise ValueError(f"Unparsed date rows: {bad}")

    # normalize column names
    raw.columns = [c.lower() for c in raw.columns]
    alias = {
        "monthly_total_teus": "total_teu",
        "total_teus": "total_teu",
        "total_teu_count": "total_teu",
        "fbx": "freightos_index",
        "diesel": "diesel_price_usd",
        "pmi": "us_manuf_pmi",
    }
    for old, new in alias.items():
        if old in raw.columns:
            raw.rename(columns={old: new}, inplace=True)

    # check required
    required = [TARGET_COL] + EXO_COLS
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Present: {list(raw.columns)}")

    return raw.sort_values("date").reset_index(drop=True)

def make_windows(df: pd.DataFrame, cfg: WindowConfig):
    from sklearn.preprocessing import StandardScaler
    y = df[TARGET_COL].astype(float).values
    y_t = np.log1p(y) if cfg.log_target else y.copy()
    X_exo = df[EXO_COLS].astype(float).values
    scaler = StandardScaler()
    X_exo = scaler.fit_transform(X_exo)

    dates = df["date"].tolist()
    L = cfg.lookback
    H = max(cfg.horizons)

    X_seq, Y_out, idx = [], [], []
    for t in range(L, len(df) - H):
        seq = np.column_stack([y_t[t-L:t], X_exo[t-L:t]]).astype(np.float32)
        X_seq.append(seq)
        Y_out.append([y_t[t+h] for h in cfg.horizons])
        idx.append(dates[t])

    return np.stack(X_seq), np.array(Y_out, np.float32), idx, scaler

def split_chronologically(n: int, train_ratio=0.7, val_ratio=0.15):
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)

def inverse_log(x):
    return np.expm1(x)

def metrics_per_horizon(y_true, y_pred):
    from math import sqrt
    rows = []
    for i, h in enumerate(HORIZONS):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        rmse = sqrt(np.mean((yp - yt) ** 2))
        mae = np.mean(np.abs(yp - yt))
        mape = np.mean(np.abs((yp - yt) / yt)) * 100
        rows.append((h, len(yt), rmse, mae, mape))
    return pd.DataFrame(rows, columns=["horizon", "n", "RMSE", "MAE", "MAPE(%)"])
