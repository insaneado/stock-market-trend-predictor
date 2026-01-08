# stock market trend predictor

# Multimodal Stock Market Trend Predictor

## Overview
This project is an AI-powered financial forecasting tool that predicts stock market trends by combining **quantitative data** (historical prices & volume) with **qualitative data** (news sentiment).

Unlike traditional models that only look at charts, this system uses a **Hybrid Architecture**:
1. **Financial Transformer (FinBERT):** Analyzes news headlines to gauge market sentiment (Positive/Negative).
2. **LSTM (Long Short-Term Memory):** Processes time-series data to learn long-term price dependencies.

## Key Features
* **Multimodal Input:** Merges OHLCV (Open, High, Low, Close, Volume) data with NLP-derived sentiment scores.
* **Sentiment Analysis:** Uses `yiyanghkust/finbert-tone`, a BERT model fine-tuned on financial text.
* **Risk Visualization:** dynamic plotting of **95% Confidence Intervals** to visualize model uncertainty.
* **Strategy Simulation:** Includes a "Paper Trading" bot that backtests the model's predictions against a "Buy & Hold" strategy.

## Tech Stack
* **Deep Learning:** PyTorch (LSTM), Hugging Face Transformers (FinBERT)
* **Data Processing:** Pandas, NumPy, Scikit-learn
* **Data Sources:** `yfinance` (Stock Data), `feedparser` (Google News RSS)
* **Visualization:** Matplotlib

## Results
* **Target Stock:** Apple Inc. (AAPL)
* **Model Accuracy:** Achieved a Test RMSE (Root Mean Squared Error) of **~$9.00** on a price range of $180-$280 (approx. 3.6% relative error).
* **Trading Performance:** The simulation demonstrates the model's ability to identify major trend reversals (e.g., the March 2025 dip and subsequent recovery).

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install torch transformers yfinance feedparser scikit-learn matplotlib pandas
