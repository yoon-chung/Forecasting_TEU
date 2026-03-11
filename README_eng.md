[한국어 README](README.md)

---

# POLA Multi-Horizon TEU Forecasting

**Port of Los Angeles · Monthly Container Throughput · 1 / 3 / 6 / 12-Month Horizons**

Georgia Tech CS 7643 Group Project (2025)

---

## 1. Problem & Motivation

Port operators, trucking companies, and warehouse managers need accurate TEU (Twenty-foot Equivalent Unit) forecasts to pre-book vessels, allocate yard capacity, and schedule labor. Current practice relies on rule-based seasonal adjustments that break down during shocks (COVID-19, labor disputes, fuel price swings).

**Research question**: Can multi-horizon deep learning models outperform classical statistical baselines for monthly port throughput forecasting, and how does model complexity interact with limited data availability?

---

## 2. Dataset

| Item | Detail |
|---|---|
| Source | Port of Los Angeles Open Data Portal |
| Period | January 2009 – September 2016 |
| Rows | 93 monthly observations |
| Target | `total_teu` (loaded imports + exports + empties) |
| Forecast horizons | 1, 3, 6, 12 months |

### 2.1 Exogenous Feature Engineering

The raw CSV contains only TEU counts — no external drivers. We manually engineered five monthly covariates from public APIs:

| Feature | Source | Rationale |
|---|---|---|
| Freightos Baltic Index | Freightos API | Ocean spot-freight rates signal demand-supply balance 1–2 months ahead of TEU moves |
| U.S. Diesel Price ($/gal) | EIA | Carrier schedule and import cost driver; spiked $3.8 in 2011–14, crashed below $2.6 in 2015 |
| GSCPI | Federal Reserve FRED | Supply-chain congestion index; peaked during 2014–15 ILWU labor dispute |
| U.S. Manufacturing PMI | ISM | Industrial demand proxy; trough 44 in 2009, >55 in expansion years |
| Peak-Season Dummy | Domain rule | Oct–Nov retail surge flag |

> **Interview note**: Exogenous variables were z-normalized for deep models (StandardScaler). For the rolling-origin backtest, we used *realized* exogenous values at t+h — a known limitation discussed in Section 6.

---

## 3. Models

### 3.1 SARIMAX (Statistical Baseline)

- Compact ARIMA orders with seasonal differencing (period=12)
- Exogenous variables passed as `exog` to `statsmodels.tsa.statespace.SARIMAX`
- Provides the primary statistical benchmark; stable and interpretable

### 3.2 LightGBM (ML Baseline)

**Design rationale**: Tree-based gradient boosting is widely used in production forecasting because it handles tabular features efficiently without requiring large datasets.

**Feature construction**:
- TEU lag features: `log(TEU)` at t−1, t−2, …, t−12 (12 features)
- Exogenous covariates at time t: 5 features
- Total: **17 features per sample**

**Direct multi-step strategy**: A separate LGBMRegressor is trained for each horizon h ∈ {1, 3, 6, 12}. This avoids error accumation from recursive forecasting.

**Hyperparameters**:
```
n_estimators=300, learning_rate=0.05, num_leaves=15,
min_child_samples=5, subsample=0.8, colsample_bytree=0.8
```
`num_leaves=15` and `min_child_samples=5` are intentionally conservative to prevent overfitting on the 93-row dataset.

### 3.3 Seq2Seq LSTM

- 2-layer encoder LSTM (128 hidden units), single FC decoder
- Input: 36-month sliding window of log-TEU + 5 exogenous features → shape `(36, 6)`
- Output: 4-dimensional vector `[h=1, h=3, h=6, h=12]`
- Loss: horizon-weighted MSE (higher weight on short horizons)
- Optimizer: AdamW (lr=1e-3), early stopping on val MSE

### 3.4 TCN (Temporal Convolutional Network)

- 2 × TemporalBlock with dilated causal Conv1D (kernel=5, channels=[64,64], dilation=1,2)
- Adaptive average pooling → linear head
- Same loss/optimizer setup as LSTM
- Causal convolutions ensure no look-ahead leakage

### 3.5 TFT (Temporal Fusion Transformer)

- `pytorch-forecasting` implementation with `lightning.pytorch`
- Attention-based variable selection + LSTM encoder + multi-head attention decoder
- Encoder length: 36 months; prediction length: 12 months
- Loss: QuantileLoss (median used for point forecast evaluation)
- EarlyStopping (patience=8) on val_loss

---

## 4. Evaluation Protocol

### 4.1 Rolling-Origin Backtest

All models use the same rolling-origin protocol:

```
for t in last 8 origins:
    train on all data up to t
    predict t+1, t+3, t+6, t+12
    advance t by 1 month
```

This mimics real-world deployment: the model is retrained each month on newly available data, with no look-ahead. SARIMAX uses n=57–68 due to its longer chronological split.

### 4.2 Metrics

RMSE, MAE, MAPE per horizon. Primary ranking metric: **MAPE** (scale-invariant, interpretable to business stakeholders).

### 4.3 Reproducibility

All PyTorch/LightGBM experiments use `seed=42` (`torch.manual_seed`, `pl.seed_everything`, `np.random.seed`, `random_state`).

---

## 5. Results

### 5.1 MAPE Summary (%, lower is better)

| Model | Type | MAPE 1M | MAPE 3M | MAPE 6M | MAPE 12M |
|---|---|---|---|---|---|
| SARIMAX | Statistical | 10.08% | 10.87% | 11.89% | 13.26% |
| LightGBM | ML | **6.26%** | 8.22% | 8.70% | 6.35% |
| LSTM | Deep Learning | 18.05% | 15.12% | 19.97% | 20.12% |
| TCN | Deep Learning | 15.42% | 16.09% | 15.44% | 21.15% |
| TFT | Deep Learning | 7.07% | 10.29% | 13.81% | **6.15%** |

![MAPE Comparison](figures/fig1_mape_comparison.png)

![MAPE Trend by Horizon](figures/fig3_horizon_trend.png)

### 5.2 Full Metrics

**SARIMAX** (rolling-origin, n=57–68)

| Horizon | n | RMSE | MAE | MAPE |
|---|---|---|---|---|
| 1M | 68 | 88,272 | 66,091 | 10.08% |
| 3M | 66 | 88,731 | 71,709 | 10.87% |
| 6M | 63 | 104,261 | 78,881 | 11.89% |
| 12M | 57 | 111,713 | 87,735 | 13.26% |

**LightGBM** (rolling-origin, n=7–8, seed=42)

| Horizon | n | RMSE | MAE | MAPE |
|---|---|---|---|---|
| 1M | 8 | 49,342 | 45,137 | 6.26% |
| 3M | 8 | 69,132 | 58,598 | 8.22% |
| 6M | 8 | 84,481 | 61,078 | 8.70% |
| 12M | 7 | 47,802 | 45,656 | 6.35% |

**LSTM** (rolling-origin, n=8, seed=42)

| Horizon | n | RMSE | MAE | MAPE |
|---|---|---|---|---|
| 1M | 8 | 160,678 | 131,146 | 18.05% |
| 3M | 8 | 140,977 | 108,419 | 15.12% |
| 6M | 8 | 174,893 | 140,691 | 19.97% |
| 12M | 8 | 180,693 | 146,945 | 20.12% |

**TCN** (rolling-origin, n=8, seed=42)

| Horizon | n | RMSE | MAE | MAPE |
|---|---|---|---|---|
| 1M | 8 | 132,698 | 112,346 | 15.42% |
| 3M | 8 | 153,335 | 113,074 | 16.09% |
| 6M | 8 | 135,404 | 108,423 | 15.44% |
| 12M | 8 | 161,665 | 151,165 | 21.15% |

**TFT** (rolling-origin, n=8, seed=42)

| Horizon | n | RMSE | MAE | MAPE |
|---|---|---|---|---|
| 1M | 8 | 54,114 | 51,476 | 7.07% |
| 3M | 8 | 80,995 | 67,703 | 10.29% |
| 6M | 8 | 104,391 | 84,240 | 13.81% |
| 12M | 8 | 67,164 | 46,950 | 6.15% |

### 5.3 Key Findings

1. **LightGBM leads at short horizons (1M: 6.26%)**: Lag-feature trees efficiently capture autocorrelation in monthly TEU data with minimal parameters, outperforming all other models at 1 month.

2. **TFT leads at long horizons (12M: 6.15%)**: Attention-based variable selection effectively leverages exogenous signals for long-range forecasting, beating SARIMAX by 54%.

3. **LSTM and TCN underperform on 93-row data**: Deep recurrent and convolutional architectures require substantially more training samples to generalize. On this pilot, they overfit despite regularization (dropout=0.1, early stopping).

4. **Data scale × model complexity tradeoff**: Classical and shallow ML models are more sample-efficient. Deep learning advantages materialize with longer history — a key motivation for the 2003–2025 extension in future work.

![Forecast vs Actual](figures/fig2_forecast_vs_actual.png)

![Error Distribution](figures/fig4_error_distribution.png)

---

## 6. Limitations

| Limitation | Detail |
|---|---|
| Small sample (n=93) | Insufficient for reliable LSTM/TCN training; deep model results should be interpreted as pilot-only |
| Realized exogenous values | Backtest used actual t+h exogenous values. In deployment, these must be forecast separately or replaced with scenario assumptions |
| Rolling-origin n=8 | Small test set limits statistical power; Diebold-Mariano significance testing conducted for SARIMAX vs. MLP only |
| Single port | Results may not generalize to other ports with different seasonal and structural characteristics |

---

## 7. Repository Structure

```
├── utils.py                # Data loading, sliding windows, metrics
├── train_lgbm.py           # LightGBM multi-horizon rolling-origin backtest
├── train_lstm_tcn.py       # Seq2Seq LSTM and TCN (PyTorch)
├── train_tft.py            # Temporal Fusion Transformer (pytorch-forecasting)
├── visualize.py            # figures
├── POLA_models_colab.ipynb  # End-to-end Colab notebook
├── requirements.txt
└── README.md
```

---

## 8. Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run LightGBM (fastest, no GPU needed)
python train_lgbm.py pola_teu.csv

# Run LSTM or TCN
python train_lstm_tcn.py pola_teu.csv lstm
python train_lstm_tcn.py pola_teu.csv tcn

# Run TFT (GPU recommended)
python train_tft.py pola_teu.csv
```

---

## 9. Future Work

- **Extend to 2003–2025 full series**: 270+ months enables reliable 36/60-month horizon evaluation and improves deep learning performance
- **Exogenous forecasting pipeline**: Integrate ARIMA-based auxiliary forecasters for exogenous variables to enable true out-of-sample deployment
- **Prophet baseline**: Facebook Prophet with regressor support as an additional statistical benchmark
- **Ensemble**: Blend LightGBM (short horizons) and TFT (long horizons) based on horizon-specific strengths observed in this pilot

---

## 10. Tech Stack

`Python 3.11` · `PyTorch 2.6` · `pytorch-forecasting` · `lightning.pytorch` · `LightGBM` · `statsmodels` · `scikit-learn` · `pandas` · `numpy`

---

## References

1. Sutskever et al. (2014). Sequence to Sequence Learning with Neural Networks. NeurIPS.
2. Bai et al. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks. ICLR.
3. Lim et al. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. IJF.
4. Diebold & Mariano (1995). Comparing Predictive Accuracy. JBES.
5. City of Los Angeles Open Data Portal — POLA TEU Counts.

