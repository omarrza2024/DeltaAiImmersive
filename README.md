# TradeMatrix

Streamlit-based finance dashboard: price charts, volatility & correlation analysis, a risk-bucket stock recommender, personal watchlists, and a Groq-powered investment chat with SEC filings and marketaux news sentiment baked in.

## Features

- **Prices & volatility** — one-ticker view with cumulative % change and 21-day rolling volatility.
- **Compare two stocks** — annualized + rolling volatility side-by-side.
- **Correlation** — pairwise log-return heatmap and numeric matrix.
- **Recommender** — K-Means / DBSCAN / rules clustering of ETF holdings, predefined universes (Tech / Medical / Finance), and your own tickers. Risk-labels every stock Low / Medium / High (±) and ranks them. Optional features: Volatility, Returns, SMA, MACD, Fundamentals (SEC), Sentiment (marketaux). Per-feature bar charts, risk-sorted table, expandable news articles per ticker.
- **Watchlist** — per-user named lists, prefix autocomplete, inline stock detail view, and a Compare picker dialog with normalized returns, correlation heatmap, and rolling volatility.
- **Investment Assistant** — Groq LLM (`openai/gpt-oss-120b`) that asks about your goals, falls back to a full-universe search if you'd rather skip questions, pulls real SEC filings for fundamental analysis, and quotes exact marketaux sentiment scores when you ask about news. The articles behind every number are one click away.
- **Auth** — MongoDB-backed accounts, bcrypt-hashed passwords, signed JWT in the URL so refresh keeps you signed in.

## Running locally

### Prerequisites

- Python 3.13+
- A MongoDB connection (Atlas is fine)
- A Groq API key — [get one here](https://console.groq.com/)
- A marketaux API key — [get one here](https://www.marketaux.com/) (free tier returns up to 3 articles per request)
- An SEC EDGAR identity line (`"Your Name your@email.com"`)

### Install

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure secrets

Create `.streamlit/secrets.toml` (gitignored):

```toml
GROQ_API_KEY      = "gsk_..."
SEC_IDENTITY      = "Your Name your@email.com"
MONGO_URI         = "mongodb+srv://user:password@cluster.mongodb.net/?appName=TradeMatrix"
MARKETAUX_API_KEY = "your-marketaux-key"
```

### Run

```bash
streamlit run app.py
```

Opens on `http://localhost:8501`.

## Project layout

```
app.py                                     # tab wiring + global CSS
finance_dashboard/
├── analytics.py                           # volatility / returns math
├── charts.py                              # Plotly figure builders
├── config.py                              # period options, defaults
├── controller.py                          # dashboard dispatcher
├── dashboards.py                          # BaseDashboard + Price/Compare/Correlation
├── data.py                                # yfinance history fetcher + ticker names
├── models.py                              # UserInputError, parsers, dedupe
├── sentiment.py                           # marketaux client + SentimentResult
├── assistant/
│   ├── assistant.py                       # InvestmentAssistant orchestrator
│   ├── financials.py                      # SEC EDGAR via EdgarTools
│   ├── goal_parser.py                     # LLM extracts goal fields from chat
│   ├── llm.py                             # Groq client wrapper
│   ├── matcher.py                         # pure-Python match-score computation
│   ├── models.py                          # InvestmentGoal, Recommendation
│   └── ui.py                              # chat UI, cards, article expanders
├── auth/
│   ├── chats.py                           # Mongo chat persistence
│   ├── db.py                              # Mongo client w/ certifi TLS
│   ├── profiles.py                        # per-user preferences & colors
│   ├── session.py                         # JWT encode/decode
│   ├── ui.py                              # login/signup/sidebar
│   └── users.py                           # bcrypt signup/login
├── recommender/
│   ├── features.py                        # per-stock scalar features (SEC fundamentals, sentiment hook)
│   ├── models.py                          # RecommendationRequest, RiskLabel, enums
│   ├── recommender.py                     # StockRecommender orchestrator
│   ├── risk.py                            # cluster → Low/Medium/High tier mapping
│   ├── strategies.py                      # K-Means / DBSCAN / rules
│   ├── ui.py                              # recommender dashboard + sentiment panel
│   └── universe.py                        # ETF holdings + predefined sectors
└── watchlist/
    ├── store.py                           # Mongo CRUD, per-user scoping
    └── ui.py                              # watchlist tab + compare picker dialog
tests/                                     # pytest suite — >150 tests
```

## Risk score

Each recommended stock gets a 0–100 risk score computed from two parts:

```
tier_index       ∈ {0, 1, 2}        # Low / Medium / High bucket
within_tier_rank ∈ [0.0, 1.0]       # volatility percentile inside the bucket
risk_score       = (tier + rank) / 3 × 100
```

Band boundaries are non-overlapping so sorting by score is safe — a Low-Risk stock never outranks a Medium one. See [`finance_dashboard/recommender/risk.py`](finance_dashboard/recommender/risk.py).

## Tests

```bash
pytest -q
```

Covers: feature math, clustering strategies, risk labeling, recommender orchestration, matcher scoring, assistant E2E flows (sector optional, no-preference fallback, sentiment context injection, SEC context injection, follow-up chat vs re-recommend), watchlist CRUD + user scoping, auth (users, chats, profiles, JWT session), and the marketaux sentiment client.

## Security notes

- `.streamlit/secrets.toml` is in `.gitignore`. Never commit it. Never paste a key into chat.
- Mongo passwords in `MONGO_URI` are passwords — if you paste a screenshot of them anywhere, rotate the Atlas database password immediately.
- JWT session tokens are HS256-signed; the secret is derived from `MONGO_URI` via SHA-256, so compromising the token means you also had the Mongo creds.

## License

Personal project — not currently licensed for redistribution.
