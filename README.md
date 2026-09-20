# Multimodal Stock Market Trend Predictor

![Python](https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Tests](https://img.shields.io/badge/tests-15%20passing-success?style=for-the-badge)

Forecasts short-term stock price movement by combining **quantitative** market
data (OHLCV from yfinance) with **qualitative** signal (FinBERT sentiment over
news headlines), then backtests the resulting predictions as a paper-trading
strategy.

---

## Pipeline

Four independent modules. Each can be run, tested and replaced without touching
the others; `run.py` only wires them together.

| # | Module | Responsibility |
|---|---|---|
| 1 | `src/data_ingestion.py` | yfinance OHLCV download, sentiment merge, train/test split, scaling, sequence windowing |
| 2 | `src/sentiment.py` | FinBERT (`yiyanghkust/finbert-tone`) scoring of Google News headlines, aggregated per day |
| 3 | `src/model.py` | Stacked LSTM, training loop with a validation split, RMSE metrics |
| 4 | `src/backtest.py` | Paper-trading simulation, buy-and-hold comparison, 95% confidence bands |

```
ingestion ──┐
            ├──> sequences ──> LSTM ──> predictions ──> backtest
sentiment ──┘
```

Scalers are fit on the training split only, so test-set statistics never leak
into training. The backtest decides each action from the *predicted* move and
fills at the price already known at that step.

---

## Quickstart

```bash
git clone https://github.com/insaneado/stock-market-trend-predictor.git
cd stock-market-trend-predictor
pip install -r requirements.txt
```

Run the tests (fast, no network, no model download):

```bash
pytest tests/ -q
```

Run the full pipeline:

```bash
python run.py --ticker AAPL --plot
```

```
[1/4] scoring news sentiment for AAPL with FinBERT...
[2/4] downloading AAPL OHLCV from 2020-01-01...
[3/4] training LSTM for 150 epochs...
      test RMSE $9.00  (3.60% of mean price)
[4/4] backtesting the paper-trading strategy...
```

Useful flags: `--no-sentiment` (price-only baseline), `--epochs`, `--seq-length`,
`--threshold`, `--ticker`.

The original exploratory notebook is kept at `notebooks/trend_predictor.ipynb`.

### Web app

```bash
streamlit run streamlit_app.py
```

Pick a ticker and window, run the pipeline, and read the metrics, the
prediction band and the portfolio curve in the browser. It calls the same four
modules as `run.py`.

**Deploy it:** on [share.streamlit.io](https://share.streamlit.io), *New app* ->
pick this repository -> main file `streamlit_app.py`. `requirements.txt`
already pins the CPU-only PyTorch build, which keeps the image inside the free
tier's build limits.

---

## Results

Reproduce exactly:

```bash
python run.py --ticker AAPL --no-sentiment --end 2026-01-31
```

AAPL, daily data, 80/20 chronological split, 150 epochs. Pinning `--end` matters:
without it the window extends to today and the numbers move.

| Window | Test RMSE | Relative error |
|---|---|---|
| 2020-01-01 -> 2026-01-31 | **$9.2 - $10.1** (mean $9.6 over 4 seeds) | 3.9% - 4.3% |
| 2020-01-01 -> today | $17.5 - $18.6 | 6.5% - 6.9% |

The error roughly doubles on the extended window. Recent price action is more
volatile than the period the architecture was tuned against, and nothing about
the model adapts to that - worth knowing before quoting a single number.

**The trading strategy does not beat buy-and-hold.** Over the pinned window it
returns about -14% against +1.65% for buy-and-hold; over the full window, ~36%
against ~64%. The model tracks the trend closely enough to score a low RMSE
while still being too smooth to time entries and exits profitably - which is
the honest result, and exactly why the backtest is in the repository rather
than the RMSE alone.

`--plot` renders actual vs predicted with a 95% band, and the portfolio curve
against buy-and-hold.

### A note on the validation split

`train_model(val_split=...)` defaults to **0**. The data is chronological, so
the tail of the training set is the window immediately before the test set and
its single most informative predictor. Holding out 10% of it moves test RMSE
from ~$9.2 to ~$19 - a large, easily-missed regression. Set `--val-split 0.1`
when you want a validation curve to check overfitting, but the resulting test
error is not comparable to the numbers above.

---

## A caveat on the sentiment signal

**The sentiment feature contributes far less than the architecture suggests, and
the repository measures this rather than hiding it.**

Sentiment comes from the Google News RSS feed, which only returns roughly the
last 100 headlines. Measured on AAPL, that is **38 distinct days against ~1400
trading days of price history - about 2.7% coverage.** Every other day is
neutral-filled with 0.

Worse, those 38 days are all recent, so they land in the *test* split. The
training split sees an almost-constant sentiment column, learns to ignore it,
and then meets non-zero values at test time that it was never trained to use.

`run.py` prints the coverage ratio each run and warns when it drops below 5%:

```
      sentiment coverage: 38/1401 trading days (2.7%)
      WARNING: Google News RSS only returns recent headlines, so
               almost every day is neutral-filled. The model is
               effectively price-only over this window.
```

Compare against `python run.py --no-sentiment` to see the effect directly.

Fixing it properly needs a historical news corpus with real date coverage -
a licensed financial news API, or a dataset such as FNSPID - rather than an RSS
feed. Restricting the evaluation window to the covered dates is the cheaper
alternative, at the cost of a much smaller sample.

---

## Other limitations

- One ticker at a time; no cross-sectional or sector features.
- The backtest charges no commission, slippage or spread, so returns are
  optimistic relative to live trading.
- Confidence bands assume normally distributed residuals; real price residuals
  are fat-tailed, so the 95% band is indicative rather than calibrated.
- Predicting next-day close from a 30-day window is a smoothing-heavy task: a
  low RMSE does not by itself imply tradeable signal, which is exactly why the
  backtest is compared against buy-and-hold.

**This is a learning project, not investment advice.**

---

## Tech stack

PyTorch (LSTM) · Hugging Face Transformers (FinBERT) · yfinance · feedparser ·
pandas · NumPy · scikit-learn · Matplotlib

Built for the Consulting and Analytics Club, IIT Guwahati.
