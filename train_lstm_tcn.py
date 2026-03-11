import sys, math, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import load_dataset, make_windows, WindowConfig, split_chronologically, inverse_log, metrics_per_horizon, HORIZONS

class SeqDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)  # (N,L,F)
        self.Y = torch.from_numpy(Y)  # (N,H)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]

class LSTMSeq2Seq(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=2, out_dim=len(HORIZONS), dropout=0.1):
        super().__init__()
        self.enc = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim))
    def forward(self, x):
        _, (hn, _) = self.enc(x)
        return self.fc(hn[-1])

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, dilation=1, dropout=0.1):
        super().__init__()
        pad = (k-1)*dilation
        self.c1 = nn.Conv1d(in_ch,out_ch,k,padding=pad,dilation=dilation); self.r1=nn.ReLU(); self.d1=nn.Dropout(dropout)
        self.c2 = nn.Conv1d(out_ch,out_ch,k,padding=pad,dilation=dilation); self.r2=nn.ReLU(); self.d2=nn.Dropout(dropout)
        self.down = nn.Conv1d(in_ch,out_ch,1) if in_ch!=out_ch else nn.Identity()
    def forward(self,x):
        y=self.d1(self.r1(self.c1(x))); y=self.d2(self.r2(self.c2(y)))
        if y.shape[-1]!=x.shape[-1]: y=y[...,:x.shape[-1]]
        return y + self.down(x)

class TCNHead(nn.Module):
    def __init__(self, in_dim, channels=[64,64], k=5, dropout=0.1, out_dim=len(HORIZONS)):
        super().__init__(); blocks=[]; C_in=in_dim; dil=1
        for C_out in channels:
            blocks.append(TemporalBlock(C_in,C_out,k,dil,dropout)); C_in=C_out; dil*=2
        self.net = nn.Sequential(*blocks); self.pool = nn.AdaptiveAvgPool1d(1); self.fc = nn.Linear(C_in,out_dim)
    def forward(self,x):
        x=x.transpose(1,2); z=self.net(x); z=self.pool(z).squeeze(-1); return self.fc(z)

def horizon_weighted_mse(pred, target, weights=None):
    if weights is None:
        return nn.MSELoss()(pred, target)
    return (weights * (pred - target) ** 2).mean()

def train(model, train_loader, val_loader, epochs=200, lr=1e-3, weights=None, device="cpu"):
    model.to(device); opt=optim.AdamW(model.parameters(), lr=lr)
    best=float("inf"); best_state=None
    for ep in range(1, epochs+1):
        model.train(); tr_loss=0.0
        for xb,yb in train_loader:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad(); out=model(xb); loss=horizon_weighted_mse(out,yb,weights); loss.backward(); opt.step()
            tr_loss += loss.item()*xb.size(0)
        tr_loss/=len(train_loader.dataset)
        model.eval(); va_loss=0.0
        with torch.no_grad():
            for xb,yb in val_loader:
                out=model(xb.to(device)); loss=horizon_weighted_mse(out,yb.to(device),weights); va_loss+=loss.item()*xb.size(0)
        va_loss/=len(val_loader.dataset)
        if va_loss<best: best, best_state = va_loss, model.state_dict()
        if ep%20==0: print(f"[{ep}] train={tr_loss:.5f} val={va_loss:.5f}")
    model.load_state_dict(best_state); return model

def main(csv_path, model_type="lstm"):
    torch.manual_seed(42)
    np.random.seed(42)

    df = load_dataset(csv_path)
    cfg = WindowConfig(lookback=36, horizons=HORIZONS, log_target=True)
    X,Y,idx,_ = make_windows(df, cfg)
    tr,va,te = split_chronologically(len(X))
    train_loader = DataLoader(SeqDataset(X[tr],Y[tr]), batch_size=32, shuffle=True)
    val_loader   = DataLoader(SeqDataset(X[va],Y[va]), batch_size=64)
    test_loader  = DataLoader(SeqDataset(X[te],Y[te]), batch_size=64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = torch.tensor([1.0,1.0,0.9,0.9], device=device)

    in_dim=X.shape[2]; out_dim=Y.shape[1]
    if model_type=="lstm":
        model = LSTMSeq2Seq(in_dim=in_dim, hidden=128, layers=2, out_dim=out_dim, dropout=0.1)
    elif model_type=="tcn":
        model = TCNHead(in_dim=in_dim, channels=[64,64], k=5, dropout=0.1, out_dim=out_dim)
    else:
        raise ValueError("model_type must be 'lstm' or 'tcn'")
    model = train(model, train_loader, val_loader, epochs=200, lr=1e-3, weights=weights, device=device)

    preds=[]; model.eval()
    with torch.no_grad():
        for xb,yb in test_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    yhat = np.vstack(preds)
    y_true = test_loader.dataset.Y.numpy()
    y_pred_lin = inverse_log(yhat); y_true_lin = inverse_log(y_true)
    table = metrics_per_horizon(y_true_lin, y_pred_lin)
    print("\\nTest metrics (original scale):\\n", table.to_string(index=False))


def rolling_origin_eval(csv_path, model_type="tcn", n_origins=8):
    torch.manual_seed(42)
    np.random.seed(42)

    df = load_dataset(csv_path)
    cfg = WindowConfig(lookback=36, horizons=HORIZONS, log_target=True)
    X, Y, idx, _ = make_windows(df, cfg)

    n = len(X)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = X.shape[2]
    out_dim = Y.shape[1]
    weights = torch.tensor([1.0, 1.0, 0.9, 0.9], device=device)

    # 리포트와 동일: 마지막 n_origins개를 순서대로 테스트 원점으로
    test_indices = list(range(n - n_origins, n))
    all_true, all_pred = [], []

    for i, t in enumerate(test_indices):
        # t 이전 데이터 전체로 학습 (누수 없음)
        # 단, 타겟이 t+max(H) 이내인 샘플만 사용
        train_end = t  # t 시점 이전까지만 훈련
        X_train = X[:train_end]
        Y_train = Y[:train_end]

        if len(X_train) < 20:
            print(f"Origin {i+1}: skip (not enough data)")
            continue

        # val은 훈련 데이터의 마지막 15%
        n_tr = int(len(X_train) * 0.85)
        # val이 너무 작으면 전체를 train으로
        if len(X_train) - n_tr < 3:
            n_tr = len(X_train)

        train_loader = DataLoader(
            SeqDataset(X_train[:n_tr], Y_train[:n_tr]),
            batch_size=32, shuffle=True
        )
        val_data = X_train[n_tr:] if n_tr < len(X_train) else X_train[-5:]
        val_Y    = Y_train[n_tr:] if n_tr < len(X_train) else Y_train[-5:]
        val_loader = DataLoader(SeqDataset(val_data, val_Y), batch_size=32)

        if model_type == "lstm":
            model = LSTMSeq2Seq(in_dim=in_dim, hidden=128, layers=2,
                                out_dim=out_dim, dropout=0.1)
        else:
            model = TCNHead(in_dim=in_dim, channels=[64, 64], k=5,
                           dropout=0.1, out_dim=out_dim)

        model = train(model, train_loader, val_loader,
                     epochs=200, lr=1e-3, weights=weights, device=device)

        # t 시점에서 예측
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(X[t:t+1]).to(device)).cpu().numpy()

        all_true.append(Y[t])   # shape (4,) — 각 horizon의 실제값
        all_pred.append(pred[0])
        print(f"Origin {i+1}/{len(test_indices)} (t={t}) done")

    y_true = inverse_log(np.array(all_true))  # (n_origins, 4)
    y_pred = inverse_log(np.array(all_pred))

    print(f"\n[Rolling-Origin] {model_type.upper()} (n={len(all_true)})")
    table = metrics_per_horizon(y_true, y_pred)
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_lstm_tcn.py <csv_path> [lstm|tcn] [--rolling]")
        sys.exit(0)

    csv = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "lstm"
    rolling = "--rolling" in sys.argv

    if rolling:
        rolling_origin_eval(csv, model_type=model, n_origins=8)
    else:
        main(csv, model)