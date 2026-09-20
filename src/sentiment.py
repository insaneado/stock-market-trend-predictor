"""
Module 2 of 4 - sentiment analysis.

Scores financial news headlines with FinBERT (`yiyanghkust/finbert-tone`) and
aggregates them into one sentiment value per calendar day.

The score is `P(positive) - P(negative)`, so it lives in [-1, 1]: positive when
the headline reads bullish, negative when bearish, near zero when neutral.
"""

from urllib.parse import quote

import numpy as np
import pandas as pd

FINBERT_MODEL = "yiyanghkust/finbert-tone"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}"


class SentimentScorer:
    """Wraps FinBERT. Loading the model is slow, so reuse one instance."""

    def __init__(self, device=None, model_name=FINBERT_MODEL):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()

    def score(self, text: str) -> float:
        """Score one headline in [-1, 1]. Empty or trivial text scores 0."""
        if not text or len(text) < 5:
            return 0.0
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
        probs = self.torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
        # finbert-tone label order: 0 neutral, 1 positive, 2 negative
        return float(probs[1] - probs[2])

    def score_many(self, texts):
        return [self.score(t) for t in texts]


def fetch_headlines(query: str, limit: int = 100):
    """Pull recent headlines for `query` from the Google News RSS feed.

    NOTE: this feed only returns *recent* items (roughly the last few days).
    It cannot supply the multi-year history the price series covers - see
    `daily_sentiment` and the coverage warning in the README.
    """
    import feedparser

    feed = feedparser.parse(GOOGLE_NEWS_RSS.format(query=quote(query)))
    out = []
    for entry in feed.entries[:limit]:
        published = getattr(entry, "published", None)
        title = getattr(entry, "title", None)
        if not title or not published:
            continue
        try:
            date = pd.to_datetime(published).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
        out.append({"date": date, "title": title})
    return out


def daily_sentiment(query: str, scorer: SentimentScorer = None, limit: int = 100):
    """Return a DataFrame of [Date, Sentiment], one averaged row per day."""
    scorer = scorer or SentimentScorer()
    headlines = fetch_headlines(query, limit=limit)
    if not headlines:
        return pd.DataFrame(columns=["Date", "Sentiment"])

    buckets = {}
    for item in headlines:
        buckets.setdefault(item["date"], []).append(scorer.score(item["title"]))

    df = pd.DataFrame(
        [(d, float(np.mean(s))) for d, s in buckets.items()],
        columns=["Date", "Sentiment"],
    )
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def coverage(price_df: pd.DataFrame, sentiment_col: str = "Sentiment") -> dict:
    """How much of the price history actually carries a sentiment signal.

    Sentiment is merged with a left join and missing days are filled with 0, so
    a low ratio here means the model is effectively price-only. Reporting it
    keeps the 'multimodal' claim honest.
    """
    total = len(price_df)
    nonzero = int((price_df[sentiment_col] != 0).sum())
    return {
        "total_days": total,
        "days_with_sentiment": nonzero,
        "ratio": (nonzero / total) if total else 0.0,
    }
