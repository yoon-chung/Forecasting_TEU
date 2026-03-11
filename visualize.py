"""
visualize.py  —  POLA TEU Forecasting: Result Visualizations
Usage:
    python visualize.py <csv_path> [output_dir]

Generates 4 publication-quality figures:
    fig1_mape_comparison.png     — MAPE bar chart: all models × all horizons
    fig2_forecast_vs_actual.png  — Forecast vs Actual: SARIMAX & LightGBM
    fig3_horizon_trend.png       — MAPE by horizon (line chart)
    fig4_error_distribution.png  — Rolling-origin error boxplot
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils import load_dataset, TARGET_COL, EXO_COLS, HORIZONS, inverse_log

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})

COLORS = {
    "SARIMAX":    "#4C72B0",
    "LightGBM":   "#DD8452",
    "LSTM":       "#55A868",
    "TCN":        "#C44E52",
    "TFT":        "#8172B2",
}

# ── Stored deep-model results (seed=42 rolling-origin) ────────────────────
STORED = {
    "LSTM": {
        "mape": {1: 18.05, 3: 15.12, 6: 19.97, 12: 20.12},
        "rmse": {1: 160678, 3: 140977, 6: 174893, 12: 180693},
    },
    "TCN": {
        "mape": {1: 15.42, 3: 16.09, 6: 15.44, 12: 21.15},
        "rmse": {1: 132698, 3: 153335, 6: 135404, 12: 161665},
    },
    "TFT": {
        "mape": {1: 7.07, 3: 10.29, 6: 13.81, 12: 6.15},
        "rmse": {1: 54114,  3: 80995,  6: 104391, 12: 67164},
    },
}

RANDOM_STATE = 42
N_ORIGINS    = 8
N_LAGS       = 12


# ══════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════

def prepare(csv_path):
    df = load_dataset(csv_path)
    scaler = StandardScaler()
    df[EXO_COLS] = scaler.fit_transform(df[EXO_COLS].astype(float))
    return df


# ══════════════════════════════════════════════════════════════════════════
# SARIMAX rolling-origin
# ══════════════════════════════════════════════════════════════════════════

def run_sarimax(df):
    y = df[TARGET_COL].astype(float).values
    exog = df[EXO_COLS].astype(float).values
    n = len(df)
    dates = df["date"].tolist()

    results = {h: {"pred": [], "true": [], "date": []} for h in HORIZONS}

    for t in range(36, n):
        y_tr = y[:t]
        e_tr = exog[:t]
        for h in HORIZONS:
            if t + h >= n:
                continue
            try:
                model = SARIMAX(
                    y_tr, exog=e_tr,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 0, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                fc = model.forecast(steps=h, exog=exog[t: t + h])
                pred = float(fc.iloc[-1]) if hasattr(fc, "iloc") else float(fc[-1])
                results[h]["pred"].append(pred)
                results[h]["true"].append(y[t + h])
                results[h]["date"].append(dates[t + h])
            except Exception:
                pass

    return results


# ══════════════════════════════════════════════════════════════════════════
# LightGBM rolling-origin
# ══════════════════════════════════════════════════════════════════════════

def make_lgbm_feat(df, t):
    y_log = np.log1p(df[TARGET_COL].astype(float).values)
    lags  = y_log[t - N_LAGS: t][::-1]
    exo   = df[EXO_COLS].astype(float).values[t]
    return np.concatenate([lags, exo])


def run_lgbm(df):
    np.random.seed(RANDOM_STATE)
    n = len(df)
    max_h = max(HORIZONS)
    origins = list(range(n - max_h - N_ORIGINS + 1, n - max_h + 1))
    dates   = df["date"].tolist()

    results = {h: {"pred": [], "true": [], "date": []} for h in HORIZONS}

    for h in HORIZONS:
        for t in origins:
            if t < N_LAGS + 10 or t + h >= n:
                continue
            X_tr, y_tr = [], []
            for s in range(N_LAGS, t - h + 1):
                X_tr.append(make_lgbm_feat(df, s))
                y_tr.append(np.log1p(df[TARGET_COL].iloc[s + h]))
            if len(X_tr) < 10:
                continue
            model = lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.05, num_leaves=15,
                min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, verbose=-1,
            )
            model.fit(np.array(X_tr), np.array(y_tr))
            x_test = make_lgbm_feat(df, t).reshape(1, -1)
            pred   = inverse_log(model.predict(x_test))[0]
            true_v = float(df[TARGET_COL].iloc[t + h])
            if true_v > 0:
                results[h]["pred"].append(pred)
                results[h]["true"].append(true_v)
                results[h]["date"].append(dates[t + h])

    return results


# ══════════════════════════════════════════════════════════════════════════
# METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════════

def mape(pred, true):
    p, t = np.array(pred), np.array(true)
    mask = t > 0
    return float(np.mean(np.abs((p[mask] - t[mask]) / t[mask])) * 100)


def errors(pred, true):
    p, t = np.array(pred), np.array(true)
    return (p - t) / t * 100   # % error


# ══════════════════════════════════════════════════════════════════════════
# FIG 1 — MAPE bar chart
# ══════════════════════════════════════════════════════════════════════════

def fig1_mape_bar(sar_res, lgbm_res, out_dir):
    models = ["SARIMAX", "LightGBM", "LSTM", "TCN", "TFT"]

    mape_table = {}
    for h in HORIZONS:
        mape_table[h] = {
            "SARIMAX":  mape(sar_res[h]["pred"],  sar_res[h]["true"]),
            "LightGBM": mape(lgbm_res[h]["pred"], lgbm_res[h]["true"]),
            "LSTM":     STORED["LSTM"]["mape"][h],
            "TCN":      STORED["TCN"]["mape"][h],
            "TFT":      STORED["TFT"]["mape"][h],
        }

    x   = np.arange(len(HORIZONS))
    w   = 0.15
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, model in enumerate(models):
        vals = [mape_table[h][model] for h in HORIZONS]
        bars = ax.bar(x + (i - 2) * w, vals, w,
                      label=model, color=COLORS[model], alpha=0.88, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}-Month" for h in HORIZONS], fontsize=11)
    ax.set_ylabel("MAPE (%)", fontsize=11)
    ax.set_title("Model Comparison: MAPE by Forecast Horizon\n"
                 "(Rolling-Origin Backtest, seed=42)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax.set_ylim(0, max(v for hd in mape_table.values() for v in hd.values()) * 1.25)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_mape_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return mape_table


# ══════════════════════════════════════════════════════════════════════════
# FIG 2 — Forecast vs Actual (SARIMAX & LightGBM, 6M & 12M)
# ══════════════════════════════════════════════════════════════════════════

def fig2_forecast_vs_actual(sar_res, lgbm_res, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    configs = [
        (sar_res,  "SARIMAX",   6,  axes[0, 0]),
        (sar_res,  "SARIMAX",   12, axes[0, 1]),
        (lgbm_res, "LightGBM",  6,  axes[1, 0]),
        (lgbm_res, "LightGBM",  12, axes[1, 1]),
    ]

    for res, model, h, ax in configs:
        dates = pd.to_datetime(res[h]["date"])
        true  = np.array(res[h]["true"])
        pred  = np.array(res[h]["pred"])
        mp    = mape(pred, true)

        ax.plot(dates, true / 1e6, color="black",        lw=1.8, label="Actual",    zorder=3)
        ax.plot(dates, pred / 1e6, color=COLORS[model],  lw=1.8, label=f"{model} Forecast",
                linestyle="--", zorder=3)
        ax.fill_between(dates, true / 1e6, pred / 1e6,
                        alpha=0.12, color=COLORS[model])

        ax.set_title(f"{model}  ·  {h}-Month Ahead  (MAPE {mp:.2f}%)",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel("TEU (millions)", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.xaxis.set_major_formatter(
            matplotlib.dates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.2f}M"))

    fig.suptitle("Forecast vs Actual — SARIMAX & LightGBM\n"
                 "(Rolling-Origin Backtest on 2009–2016 POLA TEU Data)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_forecast_vs_actual.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# FIG 3 — MAPE trend by horizon
# ══════════════════════════════════════════════════════════════════════════

def fig3_horizon_trend(mape_table, out_dir):
    models = ["SARIMAX", "LightGBM", "LSTM", "TCN", "TFT"]
    markers = {"SARIMAX": "o", "LightGBM": "s", "LSTM": "^", "TCN": "D", "TFT": "*"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for model in models:
        vals = [mape_table[h][model] for h in HORIZONS]
        ax.plot(HORIZONS, vals,
                marker=markers[model], color=COLORS[model],
                lw=2, ms=8, label=model, zorder=3)
        for h, v in zip(HORIZONS, vals):
            ax.annotate(f"{v:.1f}%", (h, v),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7.5, color=COLORS[model])

    ax.set_xticks(HORIZONS)
    ax.set_xticklabels([f"{h}M" for h in HORIZONS], fontsize=11)
    ax.set_xlabel("Forecast Horizon", fontsize=11)
    ax.set_ylabel("MAPE (%)", fontsize=11)
    ax.set_title("MAPE Trend by Forecast Horizon\n"
                 "(lower = better)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # annotate insight boxes
    ax.annotate("LightGBM leads\nshort-term (1M)",
                xy=(1, mape_table[1]["LightGBM"]),
                xytext=(2.2, mape_table[1]["LightGBM"] + 4),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1),
                fontsize=8, color="gray")
    ax.annotate("TFT leads\nlong-term (12M)",
                xy=(12, mape_table[12]["TFT"]),
                xytext=(9, mape_table[12]["TFT"] + 5),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1),
                fontsize=8, color="gray")

    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_horizon_trend.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# FIG 4 — Error distribution boxplot
# ══════════════════════════════════════════════════════════════════════════

def fig4_error_boxplot(sar_res, lgbm_res, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)

    for ax, h in zip(axes, HORIZONS):
        data, labels, colors = [], [], []
        for model, res in [("SARIMAX", sar_res), ("LightGBM", lgbm_res)]:
            if res[h]["pred"]:
                errs = errors(res[h]["pred"], res[h]["true"])
                data.append(errs)
                labels.append(model)
                colors.append(COLORS[model])

        bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                        medianprops=dict(color="black", lw=2),
                        whiskerprops=dict(lw=1.3),
                        capprops=dict(lw=1.3),
                        flierprops=dict(marker="o", ms=4, alpha=0.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.axhline(0, color="black", lw=1, linestyle="--", alpha=0.6)
        ax.set_title(f"{h}-Month", fontsize=11, fontweight="bold")
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel("")

    axes[0].set_ylabel("Forecast Error (%)\n(pred − actual) / actual × 100", fontsize=9)
    legend_elements = [
        Patch(facecolor=COLORS["SARIMAX"],   alpha=0.75, label="SARIMAX"),
        Patch(facecolor=COLORS["LightGBM"],  alpha=0.75, label="LightGBM"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.85)
    fig.suptitle("Rolling-Origin Forecast Error Distribution\n"
                 "SARIMAX vs LightGBM (% error, 0 = perfect)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_error_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main(csv_path, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading data …")
    df = prepare(csv_path)

    print("Running SARIMAX rolling-origin (this takes ~2 min) …")
    sar_res = run_sarimax(df)

    print("Running LightGBM rolling-origin …")
    lgbm_res = run_lgbm(df)

    print("Generating figures …")
    mape_table = fig1_mape_bar(sar_res, lgbm_res, out_dir)
    fig2_forecast_vs_actual(sar_res, lgbm_res, out_dir)
    fig3_horizon_trend(mape_table, out_dir)
    fig4_error_boxplot(sar_res, lgbm_res, out_dir)

    print("\nAll figures saved to:", out_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <csv_path> [output_dir]")
        sys.exit(0)
    csv   = sys.argv[1]
    odir  = sys.argv[2] if len(sys.argv) > 2 else "figures"
    main(csv, odir)
