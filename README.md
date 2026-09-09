# Vectorized Trade Simulator

A high-performance backtesting engine for testing stop-loss / take-profit strategies on historical OHLCV data, wrapped in an interactive Streamlit dashboard.

[![Live Demo](https://img.shields.io/badge/Live_Demo-AWS_EC2-FF9900?style=for-the-badge&logo=amazonec2)](https://d1rarx529p1tpn.cloudfront.net/trade/)

## Features

- **Vectorized trade simulation** with ATR-based SL/TP barriers and numba JIT acceleration
- **90+ technical indicators** via [Tecana](https://pypi.org/project/tecana/) integration
- **160+ trading signals** (momentum, zone, trend, volatility) with int8 encoding
- **8 scoring formulas** (A–H) for ranking strategies
- **Smart OHLCV parser** — auto-detects CSV/Parquet/Excel columns, delimiters, date formats
- **API connectors** — Yahoo Finance (equities) and ccxt (100+ crypto exchanges)
- **Interactive Streamlit app** with candlestick charts, equity curves, trade overlays, and CSV export
- **Grid search mode** — sweep SL/TP parameter ranges to find optimal settings
- Performance metrics: Sharpe ratio, max drawdown, alpha, beta, win rate, profit factor

## Quick Start

```bash
git clone https://github.com/Neogus/vectorized-trade-simulator
cd vectorized-trade-simulator
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

> **🌐 Try it online:** [d1rarx529p1tpn.cloudfront.net/trade/](https://d1rarx529p1tpn.cloudfront.net/trade/) — no installation needed!

## Architecture

```
vectorized-trade-simulator/
├── simulator/           # Core engine (pip-installable)
│   ├── returns.py       # Trade simulation (numba optional)
│   ├── metrics.py       # Sharpe, drawdown, alpha, beta, accuracy
│   ├── signals.py       # Signal enum, encode, aggregate, tecana bridge
│   ├── scoring.py       # 8 scoring formulas (A–H)
│   ├── decay.py         # Time-decay weighting
│   └── resample.py      # OHLCV resampling
├── data/                # Data handling
│   ├── parser.py        # Smart format/column detection
│   ├── sources.py       # yfinance + ccxt connectors
│   └── validator.py     # Data quality checks + auto-fix
├── app.py               # Streamlit app (local, full)
├── pages/               # Streamlit multi-page
│   ├── 1_Data_Source.py # Upload file or download from API
│   ├── 2_Configure.py   # SL/TP, signals, scoring
│   ├── 3_Run_Backtest.py# Execute + progress bar
│   └── 4_Results.py     # Charts + metrics + export
└── demo/                # Resource-capped demo (resource-capped)
```

## Streamlit App Pages

| Page | Description |
|------|-------------|
| **📂 Data Source** | Upload CSV/Parquet/Excel (auto-detects columns) or download from Yahoo Finance / crypto exchanges |
| **⚙️ Configure** | Set SL/TP (fixed or grid search), fees, slippage, select signals from 160+ tecana options, choose scoring formula |
| **▶️ Run Backtest** | Execute with progress bar, view trade log and metrics |
| **📈 Results** | Interactive equity curve, drawdown chart, return histogram, trade entries/exits on price, CSV export |

## Signal Convention

The simulator engine uses:

| Value | Meaning |
|-------|---------|
| **+1** | LONG entry (bullish) |
| **-1** | SHORT entry (bearish) |
| **0** | No signal (neutral) |

> **Tecana compatibility:** The [Tecana](https://github.com/Neogus/tecana) library uses the opposite convention (`-1 = buy`, `+1 = sell`). When using Tecana signals with this simulator, the app **automatically negates** them to align the two systems. No manual conversion needed.
## Dependencies

- **Required:** numpy, pandas, streamlit, plotly
- **Recommended:** tecana (90+ indicators), yfinance (equity data)
- **Optional:** ccxt (crypto), numba (JIT acceleration), matplotlib

## Deployment

Hosted on AWS EC2 (free tier) with Nginx reverse proxy. Auto-deploys via GitHub Actions on every push to `main`.

## Disclaimer

**This software is provided "as-is" without any express or implied warranty.**

The backtesting engine, technical indicators, and trading signals are based on mathematical formulas applied to historical price data. **There is no guarantee that the calculations are free of errors, bugs, or inaccuracies.** Results are hypothetical simulations — past performance does not predict future results.

The output is for **informational and educational purposes only** and should **not** be construed as financial advice or trading recommendations. **The author is not responsible for any financial losses, trading errors, or damages** arising from the use of this software. Trading and investing involve substantial risk of loss.

By using this software, you acknowledge and accept these risks.

## 💖 Support This Project

If you find this project useful, consider supporting its development:

<a href="https://github.com/sponsors/Neogus">
  <img src="https://img.shields.io/badge/Sponsor-❤️-ea4aaa?style=for-the-badge&logo=github" alt="GitHub Sponsors" />
</a>
<a href="https://ko-fi.com/neogus43222">
  <img src="https://img.shields.io/badge/Ko--fi-Buy_me_a_coffee-FFDD00?style=for-the-badge&logo=kofi" alt="Ko-fi" />
</a>
<a href="https://buymeacoffee.com/neogus">
  <img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black" alt="Buy Me A Coffee" />
</a>

Your support helps keep this project maintained and free for everyone. Thank you! 🙏

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).

Free to view, share, and adapt for non-commercial use. Commercial use requires explicit permission.

## Contact

Gustavo Rabino — gusrab@gmail.com

- GitHub: https://github.com/Neogus/vectorized-trade-simulator
- Tecana library: https://github.com/Neogus/tecana
- LinkedIn: https://www.linkedin.com/in/gustavo-rabino-58411238/
