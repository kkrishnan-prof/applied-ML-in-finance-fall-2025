# ML-Powered Crypto Lending Rate Optimizer

**Dynamic lending rate calculator with ML volatility prediction and risk-adjusted pricing for crypto margin trading**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-00a393.svg)](https://fastapi.tiangolo.com/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

This system calculates optimal lending rates for leveraged crypto trading by combining:
- **ML Volatility Prediction**: LightGBM model predicting 24h forward volatility
- **Risk-Adjusted Pricing**: Dynamic maintenance margin rates based on leverage and market conditions
- **Historical Backtesting**: Path-dependent simulation using real BTC price data
- **Interactive Frontend**: Next.js dashboard for visualization and testing

**Result**: Lender-profitable rates that accurately price risk while maintaining competitive trading costs.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- 384MB free disk space (for BTC price data)

### 1. Clone & Setup

```bash
git clone https://github.com/hyperiidea/crypto-lending-optimizer.git
cd crypto-lending-optimizer

# Backend setup
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd webapp
npm install
cd ..
```

### 2. Download Dataset

**Download BTC Historical Data (366MB)**:
- Source: [Kaggle - Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
- File: Download `bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv`
- Rename to: `btcusd_1-min_data.csv`
- Place in: Project root directory (`crypto-lending-optimizer/`)

**Directory structure after download**:
```
crypto-lending-optimizer/
├── btcusd_1-min_data.csv  ← Place downloaded file here
├── data/
│   ├── raw/
│   └── processed/
├── models/
└── ...
```

### 3. Process Data & Train Models

```bash
# Process 13+ years of BTC hourly data (creates 121K samples with 41 features)
python src/data_processor.py

# Train volatility prediction model (~3 seconds)
python run_fast_model_comparison.py

# Train revenue optimization model (~2 seconds)
python train_revenue_model.py
```

### 4. Start Services

```bash
# Terminal 1: Backend API
PYTHONPATH=. python api/main.py
# → http://localhost:8000 (API docs: /docs)

# Terminal 2: Frontend
cd webapp && npm run dev
# → http://localhost:3000
```

---

## 📊 Features

### 1. **Real-Time Rate Calculation**
- Predicts 24h volatility using 41 engineered features
- Calculates risk-adjusted maintenance margin rates
- Returns hourly/daily/annualized lending rates

### 2. **Historical Backtesting**
- Simulates trades using actual BTC price candles
- Path-dependent liquidation tracking (hour-by-hour)
- Proves profitability with real market data

### 3. **Interactive Dashboard**
- Visual entry point selection on price charts
- Live P&L simulation with liquidation indicators
- Equity curve tracking over trade duration

### 4. **Risk Management**
- Isolated margin liquidation formula: `Liq Price = Entry × (1 - IMR + MMR)`
- Dynamic MMR = Base (2.5%) + Volatility Premium + Leverage Scaling
- Liquidation fee (5% of position) protects lender capital

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Next.js Frontend                      │
│  - Interactive backtesting UI                            │
│  - Real-time rate calculator                             │
│  - Historical price chart visualization                  │
└────────────────┬────────────────────────────────────────┘
                 │ HTTP/REST
    ┌────────────▼─────────────┐
    │   FastAPI Backend         │
    │  /calculate_rate          │
    │  /backtest_trade          │
    │  /sequential_candles      │
    └────────┬──────────────────┘
             │
    ┌────────▼────────┐  ┌───────────────┐
    │ LightGBM Model  │  │  BTC Dataset   │
    │                 │  │                │
    │ Volatility      │  │  121K hourly   │
    │ Predictor       │  │  candles       │
    │                 │  │  2012-2025     │
    │ Test R²: 0.311  │  │  41 features   │
    └─────────────────┘  └────────────────┘
```

---

## 📁 Project Structure

```
crypto-lending-optimizer/
├── api/
│   └── main.py                        # FastAPI endpoints
├── src/
│   ├── data_processor.py              # Feature engineering pipeline
│   ├── volatility_models.py           # ML model implementations
│   ├── revenue_optimizer.py           # Revenue simulation
│   └── rate_calculator.py             # Rate calculation logic
├── webapp/                             # Next.js frontend
│   ├── app/
│   │   ├── page.tsx                   # Rate calculator page
│   │   └── backtest/page.tsx          # Backtesting interface
│   └── lib/
│       └── api.ts                     # TypeScript API client
├── models/                             # Trained ML models
│   ├── volatility_model_lightgbm.pkl
│   └── revenue_model.pkl
├── data/
│   ├── raw/                           # BTC 1-min OHLCV data
│   └── processed/                     # Processed hourly features
├── btcusd_1-min_data.csv              # 13+ years BTC price data
├── config.yaml                         # System configuration
└── requirements.txt                    # Python dependencies
```

---

## 🔧 API Endpoints

### Calculate Lending Rate

**POST** `/calculate_rate`

```json
{
  "asset": "BTC",
  "position": "long",
  "leverage": 10.0,
  "collateral": 1000.0,
  "timestamp": "2024-08-01T14:00:00Z"
}
```

**Response**:
```json
{
  "lending_rate_per_day": 0.00286,
  "lending_rate_annualized": 1.044,
  "lending_rate_per_hour": 0.000119,
  "base_rate": 0.0025,
  "volatility_premium": 0.00036,
  "predicted_volatility_24h": 0.592,
  "volatility_regime": "normal_vol",
  "expected_revenue": 28.50,
  "leverage_factor": 10.0
}
```

### Historical Backtest

**POST** `/backtest_trade`

```json
{
  "timestamp": "2024-08-01T14:00:00",
  "position": "long",
  "leverage": 10,
  "collateral": 1000,
  "duration_hours": 24
}
```

**Response**:
```json
{
  "entry_timestamp": "2024-08-01 14:00:00",
  "entry_price": 64549.0,
  "exit_price": 64709.0,
  "liquidated": false,
  "liquidation_price_threshold": 61234.0,
  "trader_pnl": -3.72,
  "lender_revenue": 28.50,
  "interest_charged": 28.50,
  "maintenance_margin": 468.0,
  "maintenance_margin_rate": 0.0468,
  "hourly_breakdown": [...]
}
```

### Get Sequential Candles

**GET** `/sequential_candles?count=200`

Returns consecutive hourly BTC candles from test set for chart visualization.

---

## 🧮 Key Formulas

### Isolated Margin Liquidation

```
For Long positions:
Liq Price = Entry Price × (1 - IMR + MMR)

For Short positions:
Liq Price = Entry Price × (1 + IMR - MMR)

Where:
  IMR (Initial Margin Rate) = 1 / leverage
  MMR (Maintenance Margin Rate) = base_mmr + vol_adjustment + leverage_scaling
```

### Risk-Adjusted MMR

```python
base_mmr = 0.025  # 2.5% base
vol_adjustment = predicted_volatility * 0.02  # 2% per volatility unit
leverage_scaling = min(0.01 * (leverage / 10), 0.05)  # Up to 5% for high leverage

MMR = base_mmr + vol_adjustment + leverage_scaling
```

**Examples**:
- 10x leverage, 0.59 vol → MMR = 4.68% → $468 maintenance margin on $10k position
- 50x leverage, 0.59 vol → MMR = 8.68% → $4,340 maintenance margin on $50k position

---

## 📈 Model Performance

### Volatility Prediction (LightGBM)

| Metric | Train | Test |
|--------|-------|------|
| RMSE | 0.414 | 0.183 |
| MAE | 0.231 | 0.136 |
| R² | 0.628 | 0.311 |
| Training Time | ~1 second | - |
| Overfitting Severity | Mild (R² gap: 0.316) | - |

**Regularization Strategy**: Aggressive anti-overfitting configuration with shallow trees (depth=4), large leaf sizes (100 samples), strong L1/L2 penalties (5.0), and reduced ensemble size (150 trees).

**Feature Categories** (41 total):
- Volatility indicators (realized, Parkinson, volume-weighted)
- Return metrics (momentum, rolling returns)
- Market regime (trend strength, range detection)
- Volume ratios and moving averages
- Cyclical time encoding

---

## 🎮 Usage Examples

### Python API Client

```python
import requests

# Calculate optimal rate
response = requests.post('http://localhost:8000/calculate_rate', json={
    'asset': 'BTC',
    'position': 'long',
    'leverage': 10.0,
    'collateral': 1000.0
})

rate_data = response.json()
print(f"Hourly rate: {rate_data['lending_rate_per_hour']*100:.4f}%")
print(f"APR: {rate_data['lending_rate_annualized']*100:.2f}%")
```

### TypeScript Frontend

```typescript
import { api } from '@/lib/api';

const result = await api.backtestTrade({
  timestamp: '2024-08-01T14:00:00',
  position: 'long',
  leverage: 10,
  collateral: 1000,
  duration_hours: 24
});

console.log(`Trader P&L: $${result.trader_pnl}`);
console.log(`Lender Revenue: $${result.lender_revenue}`);
console.log(`Liquidated: ${result.liquidated}`);
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize system behavior:

```yaml
rate_params:
  min_rate_per_day: 0.0001  # 0.01% minimum
  max_rate_per_day: 0.01    # 1.0% maximum
  vol_premium_scaling: 0.001
  leverage_exponent: 0.7

models:
  revenue:
    simulation:
      base_demand: 0.8
      elasticity: 20
      vol_factor: 1.0
      avg_hold_time_hours: 8
      liquidation_fee: 0.05
```

---

## 🧪 Testing

```bash
# Test API endpoints
python test_api.py

# Verify installation
python verify_installation.py

# Run model comparison
python run_model_comparison.py
```

---

## 📊 Dataset

- **Source**: BTC/USD 1-minute OHLCV data
- **Period**: 2012-01-01 to 2025-12-05 (13+ years)
- **Total Samples**: 121,914 hourly candles
- **Features**: 41 engineered technical indicators
- **Train/Test Split**: 80/20 chronological

---

## 🚧 Future Enhancements

- [ ] Multi-asset support (ETH, SOL, etc.)
- [ ] Real-time price feed integration
- [ ] Confidence intervals via ensemble methods
- [ ] Auto-retraining pipeline with new data
- [ ] Advanced simulations (slippage, partial liquidations)
- [ ] Mobile-responsive UI improvements

---

## 🛠️ Tech Stack

**Backend**:
- FastAPI (API framework)
- LightGBM (volatility prediction)
- pandas/numpy (data processing)
- loguru (logging)

**Frontend**:
- Next.js 14 (React framework)
- TypeScript (type safety)
- Recharts (visualization)
- Tailwind CSS (styling)

**ML & Data**:
- scikit-learn
- scipy
- PyYAML (configuration)

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📚 Citation

If you use this system in your research or production:

```bibtex
@software{crypto_lending_optimizer,
  title={ML-Powered Crypto Lending Rate Optimizer},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/crypto-lending-optimizer}
}
```

---

**Built with ❤️ for efficient, risk-aware crypto lending**
