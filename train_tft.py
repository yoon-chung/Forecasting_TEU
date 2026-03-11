import sys
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from utils import EXO_COLS, TARGET_COL, HORIZONS

# HORIZONS = [1, 3, 6, 12]  →  0-indexed positions in 12-step prediction window
HORIZON_IDX = {h: h - 1 for h in HORIZONS}   # {1:0, 3:2, 6:5, 12:11}


def prepare_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    date_col = next(c for c in df.columns if c.lower() in ("date", "month", "timestamp"))
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df.columns = [c.lower() for c in df.columns]

    alias = {
        "monthly_total_teus": "total_teu",
        "total_teus": "total_teu",
        "total_teu_count": "total_teu",
    }
    for old, new in alias.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    df = df.sort_values("date").reset_index(drop=True)
    df["group"] = "pola"
    df["time_idx"] = np.arange(len(df))

    scaler = StandardScaler()
    df[EXO_COLS] = scaler.fit_transform(df[EXO_COLS].astype(float))
    df["target"] = np.log1p(df[TARGET_COL].astype(float))
    return df


def make_tft_datasets(df, train_cutoff, max_encoder_length=36, max_prediction_length=12):
    shared_params = dict(
        time_idx="time_idx",
        target="target",
        group_ids=["group"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx"] + EXO_COLS,
        time_varying_unknown_reals=["target"],
        static_categoricals=["group"],
        add_relative_time_idx=True,
        add_target_scales=True,
    )

    train_end = int(train_cutoff * 0.85)
    train_ds = TimeSeriesDataSet(df[df.time_idx < train_end], **shared_params)
    val_ds = TimeSeriesDataSet.from_dataset(
        train_ds, df[df.time_idx < train_cutoff], predict=True, stop_randomization=True
    )
    test_ds = TimeSeriesDataSet.from_dataset(
        train_ds, df[df.time_idx <= train_cutoff], predict=True, stop_randomization=True
    )
    return train_ds, val_ds, test_ds


def train_and_predict(train_ds, val_ds, test_ds):
    train_loader = train_ds.to_dataloader(train=True, batch_size=16, num_workers=0)
    val_loader   = val_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
    test_loader  = test_ds.to_dataloader(train=False, batch_size=64, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=1e-3,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=4,
        log_interval=-1,
    )

    trainer = pl.Trainer(
        max_epochs=60,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=8, min_delta=1e-4)],
        log_every_n_steps=5,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(tft, train_loader, val_loader)

    preds = tft.predict(
        test_loader, mode="prediction", return_y=True,
        trainer_kwargs={"accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                        "logger": False}
    )
    # preds.output shape: (n_sequences, max_prediction_length)
    yhat_log = preds.output.cpu().numpy()   # keep 2-D: (n_seq, 12)
    y_log    = preds.y[0].cpu().numpy()     # (n_seq, 12)
    yhat = np.expm1(yhat_log)
    y    = np.expm1(y_log)
    return yhat, y                          # both (n_seq, 12)


def rolling_origin_eval(csv_path: str, n_origins: int = 8,
                         max_encoder_length: int = 36,
                         max_prediction_length: int = 12):
    pl.seed_everything(42)
    df = prepare_df(csv_path)
    n = len(df)

    origins = list(range(n - max_prediction_length - n_origins + 1, n - max_prediction_length + 1))

    # per-horizon collectors  {horizon: {"pred": [], "true": []}}
    collectors = {h: {"pred": [], "true": []} for h in HORIZONS}

    for i, t in enumerate(origins):
        if t < max_encoder_length + 10:
            print(f"Origin {i+1}: skip (not enough history)")
            continue

        print(f"Origin {i+1}/{len(origins)} (t={t}, train up to idx {t-1})...")

        try:
            train_ds, val_ds, test_ds = make_tft_datasets(
                df, train_cutoff=t,
                max_encoder_length=max_encoder_length,
                max_prediction_length=max_prediction_length,
            )
            yhat, y = train_and_predict(train_ds, val_ds, test_ds)

            # yhat / y are (n_seq, 12); take first sequence (= forecast from t)
            yhat_seq = yhat[0]   # shape (12,)
            y_seq    = y[0]      # shape (12,)

            for h in HORIZONS:
                idx = HORIZON_IDX[h]
                if idx < len(yhat_seq) and y_seq[idx] > 0:
                    collectors[h]["pred"].append(yhat_seq[idx])
                    collectors[h]["true"].append(y_seq[idx])

            print(f"  → 1M pred={yhat_seq[0]:,.0f}  true={y_seq[0]:,.0f}")

        except Exception as e:
            print(f"  Origin {i+1} failed: {e}")
            continue

    # ── print results table (same format as train_lstm_tcn.py) ──
    print(f"\n[Rolling-Origin] TFT (seed=42)")
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
        print("Usage: python train_tft.py <csv_path>")
        sys.exit(0)
    rolling_origin_eval(sys.argv[1], n_origins=8)