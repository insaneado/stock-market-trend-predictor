"""
End-to-end pipeline: ingestion -> sentiment -> training -> backtest.

    python run.py --ticker AAPL
    python run.py --ticker MSFT --epochs 200 --no-sentiment

Each stage lives in its own module under src/ and can be run or swapped
independently; this script only wires them together.
"""

import argparse


def main():
    ap = argparse.ArgumentParser(description="Stock market trend predictor.")
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None,
                    help="YYYY-MM-DD; pin the window so results reproduce")
    ap.add_argument("--seq-length", type=int, default=30)
    ap.add_argument("--val-split", type=float, default=0.0,
                    help="hold out this tail fraction of training data for a "
                         "validation curve; costs test accuracy (see src/model.py)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--threshold", type=float, default=0.005)
    ap.add_argument("--no-sentiment", action="store_true",
                    help="skip FinBERT and run on price data alone")
    ap.add_argument("--plot", action="store_true", help="show the charts")
    args = ap.parse_args()

    from src import backtest, data_ingestion, model as model_mod, sentiment

    # ── 1. sentiment ──────────────────────────────────────────────────────────
    sentiment_df = None
    if not args.no_sentiment:
        print(f"[1/4] scoring news sentiment for {args.ticker} with FinBERT...")
        sentiment_df = sentiment.daily_sentiment(args.ticker)
        print(f"      {len(sentiment_df)} day(s) of headline sentiment")
    else:
        print("[1/4] sentiment skipped (--no-sentiment)")

    # ── 2. ingestion ──────────────────────────────────────────────────────────
    window = f"{args.start} to {args.end or 'today'}"
    print(f"[2/4] downloading {args.ticker} OHLCV ({window})...")
    features = (["Close", "Volume"] if args.no_sentiment
                else data_ingestion.DEFAULT_FEATURES)
    ds = data_ingestion.build_dataset(
        ticker=args.ticker, start=args.start, end=args.end,
        sentiment_df=sentiment_df, features=features,
        seq_length=args.seq_length,
    )
    print(f"      train {ds.X_train.shape}  test {ds.X_test.shape}")

    if not args.no_sentiment:
        cov = sentiment.coverage(ds.frame)
        print(f"      sentiment coverage: {cov['days_with_sentiment']}/"
              f"{cov['total_days']} trading days ({cov['ratio']:.1%})")
        if cov["ratio"] < 0.05:
            print("      WARNING: Google News RSS only returns recent headlines, so")
            print("               almost every day is neutral-filled. The model is")
            print("               effectively price-only over this window.")

    # ── 3. training ───────────────────────────────────────────────────────────
    print(f"[3/4] training LSTM for {args.epochs} epochs...")
    net, history = model_mod.train_model(
        ds, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        epochs=args.epochs, val_split=args.val_split,
    )
    y_true, y_pred = model_mod.predict(net, ds)
    err = model_mod.rmse(y_true, y_pred)
    rel = model_mod.relative_error(y_true, y_pred)
    print(f"      test RMSE ${err:.2f}  ({rel:.2%} of mean price)")

    # ── 4. backtest ───────────────────────────────────────────────────────────
    print("[4/4] backtesting the paper-trading strategy...")
    result = backtest.simulate_trading_strategy(
        y_true, y_pred, initial_capital=args.capital, threshold=args.threshold
    )
    print(result.summary())

    if args.plot:
        backtest.plot_predictions(
            ds.test_dates, y_true, y_pred, f"{args.ticker} prediction vs actual"
        ).show()
        backtest.plot_portfolio(result).show()


if __name__ == "__main__":
    main()
