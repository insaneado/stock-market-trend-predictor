"""
Module 3 of 4 - model training.

A stacked LSTM over the (Close, Volume, Sentiment) window, with a linear head
that reads the final hidden state and predicts the next Close.
"""

import numpy as np
import torch
import torch.nn as nn


class PredictionModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=2, output_dim=1, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])   # last timestep only


def train_model(
    dataset,
    hidden_dim=128,
    num_layers=2,
    epochs=150,
    lr=1e-3,
    device=None,
    val_split=0.0,
    seed=42,
    verbose=True,
):
    """Train on `dataset` and return (model, history).

    `val_split` holds out a *tail* slice of the training set for validation.
    It defaults to 0.0, and that default is deliberate: the data is
    chronological, so the tail of the training set is the window immediately
    before the test set and by far its most informative predictor. Holding it
    out costs a lot of accuracy - on AAPL over 2020-01-01..2026-01-31, a 10%
    holdout moves test RMSE from about $9.2 to about $19.

    So: leave it at 0 to reproduce the reported numbers, and set it above 0
    when you want a validation curve to check for overfitting, accepting that
    the resulting test error is not comparable.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.from_numpy(dataset.X_train).float()
    y = torch.from_numpy(dataset.y_train).float()

    n_val = int(len(X) * val_split)
    if n_val > 0:
        X_tr, y_tr = X[:-n_val].to(device), y[:-n_val].to(device)
        X_val, y_val = X[-n_val:].to(device), y[-n_val:].to(device)
    else:
        X_tr, y_tr = X.to(device), y.to(device)
        X_val = y_val = None

    model = PredictionModel(
        input_dim=dataset.n_features, hidden_dim=hidden_dim, num_layers=num_layers
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train": [], "val": []}
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr), y_tr)
        loss.backward()
        optimizer.step()
        history["train"].append(loss.item())

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                history["val"].append(criterion(model(X_val), y_val).item())

        if verbose and epoch % 25 == 0:
            v = f"  val {history['val'][-1]:.6f}" if X_val is not None else ""
            print(f"epoch {epoch:4d}  train {loss.item():.6f}{v}")

    if verbose:
        print("Training complete.")
    return model, history


@torch.no_grad()
def predict(model, dataset, split="test", device=None):
    """Predict and return (y_true, y_pred) in dollars."""
    device = device or next(model.parameters()).device
    X = dataset.X_test if split == "test" else dataset.X_train
    y = dataset.y_test if split == "test" else dataset.y_train

    model.eval()
    pred_scaled = model(torch.from_numpy(X).float().to(device)).cpu().numpy()
    return dataset.inverse_price(y), dataset.inverse_price(pred_scaled)


def rmse(y_true, y_pred) -> float:
    """Root mean squared error in dollars."""
    return float(np.sqrt(np.mean((np.asarray(y_true).flatten()
                                  - np.asarray(y_pred).flatten()) ** 2)))


def relative_error(y_true, y_pred) -> float:
    """RMSE as a fraction of the mean actual price."""
    y_true = np.asarray(y_true).flatten()
    return rmse(y_true, y_pred) / float(np.mean(y_true))
