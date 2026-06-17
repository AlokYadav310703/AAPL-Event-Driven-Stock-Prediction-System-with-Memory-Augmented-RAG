# AAPL Event-Driven Stock Prediction System

A memory-augmented, event-driven stock prediction pipeline for Apple Inc. (AAPL) built on a Retrieval-Augmented Generation (RAG) architecture with online learning.

> **Disclaimer:** This system is for academic research and portfolio demonstration only. It must not be used for live trading decisions.

---

## Overview

Traditional stock prediction models rely on historical price data (OHLCV) and ignore the real-world events that actually drive price movements. This system takes a different approach: it models cause-and-effect relationships between Apple-related news events and AAPL stock price direction.

Each trading day, the system:
1. Collects the previous day's Apple news
2. Extracts structured event data via LLM
3. Retrieves historically similar events from a weighted vector-database memory
4. Produces a directional prediction — **HIGH**, **LOW**, or **NEUTRAL** — with a confidence score and natural-language explanation
5. After market close, fetches the actual outcome and updates the model

---

## Key Features

- **Event-driven prediction** — forecasts based on news cause-effect patterns, not price autocorrelation
- **Importance-weighted memory** — ChromaDB stores not just what happened, but how much it moved the market
- **Online learning** — XGBoost classifier is incrementally retrained after every trading day
- **Explainability-first design** — every prediction includes the top retrieved past events, similarity scores, and an LLM-generated reasoning paragraph
- **Fully automated daily pipeline** — APScheduler drives all phases from ingestion to retraining

---

## System Architecture

The pipeline runs in six sequential phases triggered automatically each trading day:

| Phase | Time | Description |
|---|---|---|
| 1 — Data Ingestion | 6:00 AM | Collect Apple news, SEC filings, AAPL price data |
| 2 — LLM Event Extraction | 6:00 AM | Parse articles into structured JSON events via Claude Haiku |
| 3 — Vector Memory Update | 6:00 AM | Embed events with FinBERT, store in ChromaDB with importance weights |
| 4 — Prediction Engine | 7:00 AM | Retrieve similar past events, run XGBoost, generate reasoning |
| 5 — Online Learning | 5:00 PM | Fetch actual outcome, update memory, retrain model |
| 6 — Dashboard | Always on | Streamlit UI for predictions, accuracy tracking, and log browsing |

---

## Technology Stack

| Component | Tool |
|---|---|
| Scheduler | APScheduler (Python) |
| News Ingestion | NewsAPI, PRAW, requests |
| LLM Extraction | Claude Haiku (Anthropic API) |
| Embeddings | FinBERT (sentence-transformers) |
| Vector Database | ChromaDB |
| Classifier | XGBoost |
| Price Data | yfinance |
| Log Storage | SQLite |
| Dashboard | Streamlit |
| Experiment Tracking | MLflow |
| Deployment | Railway.app / Render.com |

---

## Data Sources

**Primary historical dataset (cold start):**  
[FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) — ~180,000 AAPL-specific news records (1999–2023), pre-aligned with OHLCV price data. A minimum of 500 labelled events in ChromaDB is required before reliable predictions can be made.

**Live data sources:**

| Source | Role |
|---|---|
| NewsAPI / GNews | Daily Apple news headlines |
| yfinance | AAPL daily OHLCV for outcome labelling |

---

## Event Extraction Schema

Every article is parsed by Grok into a validated JSON object:

| Field | Description |
|---|---|
| `event_type` | Category: `earnings_beat`, `product_launch`, `legal`, `regulatory`, etc. |
| `sentiment_score` | Float −1.0 (very negative) to +1.0 (very positive) |
| `severity` | Expected market relevance, 0.0–10.0 |
| `affected_product` | iPhone, Mac, Services, App Store, Vision Pro, General |
| `market_context` | `bull_market`, `bear_market`, `high_volatility`, `neutral` |
| `source_credibility` | Tier 1 (Reuters/WSJ), Tier 2 (CNBC), Tier 3 (blogs) |

---

## Importance-Weighted Memory

The vector memory stores three importance numbers alongside every event embedding:

- **`price_impact_pct`** — actual AAPL price change at T+1, T+3, T+5 days after the event
- **`impact_score`** (0.0–1.0) — composite of price movement (50%), abnormal volume (30%), and idiosyncratic return vs. S&P 500 (20%)
- **`event_weight`** (1.0–~2.6) — retrieval multiplier combining impact score with an event-type bonus (e.g. `earnings_miss` gets 1.4×, `analyst_upgrade` gets 0.9×)

All weights decay exponentially over time with a configurable half-life (default: 365 days).

---

## Prediction Engine

For each new article, the engine:
1. Embeds the article with FinBERT (768-dim vector)
2. Queries ChromaDB for the top-10 most similar past events by cosine similarity
3. Re-ranks using **60% cosine similarity + 40% normalised event_weight**
4. Feeds the top-5 retrieved events into the XGBoost classifier
5. Generates a natural-language explanation via LLM

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Directional Accuracy | % of days where predicted direction matches actual outcome |
| F1 Score (macro) | Balanced across HIGH, LOW, NEUTRAL classes |
| Precision / Recall | Reported separately for HIGH and LOW |
| Retrieval Quality (MRR) | Mean Reciprocal Rank of retrieved events |
| Impact Score Correlation | Pearson correlation between impact_score and prediction influence |
| Rolling 30-day Accuracy | Primary indicator that online learning is working |

All metrics are compared against three baselines: buy-and-hold, random direction classifier, and plain sentiment classifier. A system warning is triggered if rolling 30-day accuracy drops below 50%.

## Constraints

- Scope is limited to a single asset (AAPL); not designed for multi-stock portfolios
- Predictions are directional only — no price targets or percentage moves are forecast
- NewsAPI free tier allows ~100 requests/day with 1 month of historical archive access
- All articles are assumed to be in English; non-English content is filtered at ingestion
- System observes NYSE trading hours and the Federal Reserve bank holiday calendar

---

## Project Context

This is a Final Year / Research Project submitted to a course supervisor. It is scoped as an entry-level Data Science portfolio project demonstrating production ML skills including RAG architecture, vector databases, online learning, and LLM integration.

**Document Version:** 1.0 | **Date:** 18 May 2026
