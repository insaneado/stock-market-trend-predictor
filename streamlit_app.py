"""
Streamlit front end for the trend predictor.

Runs the same four modules as run.py - ingestion, sentiment, training,
backtest - and renders the result. Deployable as-is to Streamlit Community
Cloud or a Hugging Face Space.

    streamlit run streamlit_app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

from src import backtest as bt
from src import data_ingestion as di
from src import model as mm

st.set_page_config(page_title="Stock Trend Predictor", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def run_pipeline(ticker, start, end, epochs, seq_length, use_sentiment, seed):
    """Cached so re-rendering the page does not retrain the model."""
    sentiment_df = None
    coverage = None
    if use_sentiment:
        from src import sentiment as sent
        sentiment_df = sent.daily_sentiment(ticker)

    features = di.DEFAULT_FEATURES if use_sentiment else ["Close", "Volume"]
    ds = di.build_dataset(
        ticker=ticker, start=str(start), end=str(end) if end else None,
        sentiment_df=sentiment_df, features=features, seq_length=seq_length,
    )
    if use_sentiment:
        from src import sentiment as sent
        coverage = sent.coverage(ds.frame)

    net, _ = mm.train_model(ds, epochs=epochs, seed=seed, verbose=False)
    y_true, y_pred = mm.predict(net, ds)
    result = bt.simulate_trading_strategy(y_true, y_pred)
    lower, upper = bt.confidence_band(y_true, y_pred)

    return {
        "dates": ds.test_dates,
        "y_true": np.asarray(y_true).flatten(),
        "y_pred": np.asarray(y_pred).flatten(),
        "lower": lower, "upper": upper,
        "rmse": mm.rmse(y_true, y_pred),
        "rel": mm.relative_error(y_true, y_pred),
        "result": result,
        "coverage": coverage,
        "n_train": len(ds.X_train), "n_test": len(ds.X_test),
    }


st.title("Multimodal Stock Market Trend Predictor")
st.caption(
    "LSTM over OHLCV and FinBERT news sentiment, with a paper-trading backtest. "
    "Four independent modules: ingestion, sentiment, training, backtesting."
)

with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
    start = st.date_input("Start", pd.to_datetime("2020-01-01"))
    pin = st.checkbox("Pin the end date", value=True,
                      help="Results drift as new market data arrives. Pin the "
                           "window to reproduce a specific number.")
    end = st.date_input("End", pd.to_datetime("2026-01-31")) if pin else None
    st.divider()
    epochs = st.slider("Training epochs", 25, 300, 150, step=25)
    seq_length = st.slider("Sequence length (days)", 10, 60, 30, step=5)
    seed = st.number_input("Random seed", value=42, step=1)
    use_sentiment = st.checkbox(
        "Use FinBERT sentiment", value=False,
        help="Downloads a ~400MB model on first use. Coverage from the Google "
             "News RSS feed is only a few percent of the price history.",
    )
    go = st.button("Run pipeline", type="primary", use_container_width=True)

# Streamlit reruns the whole script on every widget interaction, and a plain
# button is True only on the run that triggered it. Persist the result so the
# output survives later reruns instead of vanishing.
if go:
    with st.spinner(f"Ingesting {ticker}, training for {epochs} epochs..."):
        try:
            st.session_state["run"] = run_pipeline(
                ticker, start, end, epochs, seq_length, use_sentiment, int(seed)
            )
            st.session_state["run_label"] = (
                f"{ticker}  |  {start} to {end or 'today'}  |  {epochs} epochs"
            )
        except Exception as exc:
            st.session_state.pop("run", None)
            st.error(f"Pipeline failed: {exc}")
            st.stop()

if "run" not in st.session_state:
    st.info("Configure the run in the sidebar, then press **Run pipeline**.")
    st.stop()

out = st.session_state["run"]
st.caption(f"Showing: {st.session_state.get('run_label', '')}")
res = out["result"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Test RMSE", f"${out['rmse']:.2f}")
c2.metric("Relative error", f"{out['rel']:.2%}")
c3.metric("Strategy return", f"{res.strategy_return_pct:.2f}%",
          delta=f"{res.strategy_return_pct - res.buy_hold_return_pct:.2f}% vs buy & hold")
c4.metric("Trades", res.trades)

if out["coverage"] is not None:
    cov = out["coverage"]
    msg = (f"Sentiment coverage: {cov['days_with_sentiment']} of "
           f"{cov['total_days']} trading days ({cov['ratio']:.1%}).")
    if cov["ratio"] < 0.05:
        st.warning(msg + " The news feed only returns recent headlines, so almost "
                         "every day is neutral-filled and the model is effectively "
                         "price-only over this window.")
    else:
        st.info(msg)

st.subheader("Prediction vs actual")
chart_df = pd.DataFrame(
    {"Actual": out["y_true"], "Predicted": out["y_pred"],
     "Lower 95%": out["lower"], "Upper 95%": out["upper"]},
    index=pd.to_datetime(out["dates"]),
)
st.line_chart(chart_df)

st.subheader("Portfolio value against buy & hold")
st.line_chart(pd.DataFrame({"Strategy portfolio": res.portfolio_values}))

if not res.beat_buy_hold:
    st.warning(
        f"The strategy returned {res.strategy_return_pct:.2f}% against "
        f"{res.buy_hold_return_pct:.2f}% for buy & hold. A low RMSE means the "
        "model tracks the trend closely; it does not mean the signal is "
        "tradeable. That distinction is why the backtest is here."
    )

with st.expander("Run details"):
    st.write({
        "ticker": ticker,
        "window": f"{start} to {end or 'today'}",
        "train sequences": out["n_train"],
        "test sequences": out["n_test"],
        "epochs": epochs, "sequence length": seq_length, "seed": int(seed),
        "features": "Close, Volume, Sentiment" if use_sentiment else "Close, Volume",
    })

st.caption("Learning project, not investment advice.")
