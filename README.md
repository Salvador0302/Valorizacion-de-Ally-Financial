# Ally Financial (ALLY) Stock Valuation Project

**Comprehensive stock valuation analysis for Ally Financial using classical financial methods and AI-powered predictions.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

This project provides a comprehensive valuation analysis of Ally Financial (ALLY) stock using multiple approaches:

### 📊 Traditional Valuation Methods
- **Book Value** - Basic equity valuation
- **Adjusted Book Value** - Equity minus intangible assets
- **P/E Ratio Analysis** - Earnings-based valuation with industry comparisons
- **Dividend Discount Model (DDM)** - Present value of future dividends
- **Comparable Companies Analysis** - Peer multiples comparison
- **DCF (Free Cash Flow)** - Discounted cash flow valuation

### 🤖 AI/ML Models
- **LSTM Neural Network** - Deep learning model for price prediction
- **Monte Carlo Simulation** - Probabilistic price forecasting using Geometric Brownian Motion

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Salvador0302/Valorizacion-de-Ally-Financial.git
cd Valorizacion-de-Ally-Financial
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Jupyter Notebook

Run the comprehensive analysis notebook:
```bash
jupyter notebook notebooks/ally_valuation_analysis.ipynb
```

### Streamlit Dashboard

Launch the interactive dashboard:
```bash
streamlit run streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Python Module Usage

```python
from src.data_loader import DataLoader
from src.valuation import ValuationEngine
from src.lstm_model import LSTMPredictor
from src.monte_carlo import MonteCarloSimulation

# Load data
loader = DataLoader(ticker="ALLY")
summary = loader.get_summary()
prices = loader.get_historical_prices(period="5y")

# Run valuations
valuation = ValuationEngine(data_loader=loader)
results = valuation.get_all_valuations()
fair_value = valuation.get_fair_value_estimate()

# Monte Carlo simulation
mc = MonteCarloSimulation(n_simulations=10000, n_days=252)
mc_results = mc.run_simulation(prices['Close'])

# LSTM predictions (requires TensorFlow)
lstm = LSTMPredictor(sequence_length=60, epochs=25)
lstm.train(prices['Close'])
predictions = lstm.predict_future(prices['Close'], days_ahead=30)
```

## 📁 Project Structure

```
Valorizacion-de-Ally-Financial/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── data_loader.py        # Financial data loading (yfinance)
│   ├── valuation.py          # Valuation methods engine
│   ├── lstm_model.py         # LSTM price prediction model
│   └── monte_carlo.py        # Monte Carlo simulation
├── notebooks/
│   └── ally_valuation_analysis.ipynb  # Jupyter analysis notebook
├── data/                     # Data directory (for cached data)
├── streamlit_app.py          # Streamlit dashboard
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 📈 Valuation Methods Explained

### 1. Book Value
- **Formula**: Total Equity / Shares Outstanding
- **Use Case**: Conservative floor valuation for asset-heavy companies

### 2. Adjusted Book Value
- **Formula**: (Total Equity - Intangible Assets) / Shares Outstanding
- **Use Case**: More conservative estimate excluding goodwill and intangibles

### 3. P/E Ratio
- **Formula**: EPS × Target P/E (industry median)
- **Use Case**: Quick relative valuation against peers

### 4. Dividend Discount Model (DDM)
- **Formula**: PV of Stage 1 Dividends + PV of Terminal Value
- **Use Case**: Income-focused valuation for dividend-paying stocks

### 5. Comparable Companies
- **Approach**: Uses median P/E and P/B multiples from peer companies
- **Peers**: COF, SYF, DFS, AXP, C (financial services sector)

### 6. DCF (Free Cash Flow)
- **Formula**: Sum of discounted FCFs + PV of Terminal Value
- **Use Case**: Fundamental intrinsic value estimation

### 7. LSTM Neural Network
- **Architecture**: 2 LSTM layers with dropout, Dense output
- **Use Case**: Pattern-based price prediction

### 8. Monte Carlo Simulation
- **Model**: Geometric Brownian Motion (GBM)
- **Use Case**: Probabilistic price distribution and risk metrics (VaR, CVaR)

## 📊 Dashboard Features

The Streamlit dashboard provides:

- **Key Metrics Display**: Current price, market cap, EPS, dividend yield
- **Interactive Price Charts**: Historical OHLC candlestick charts
- **Valuation Analysis Tabs**: Detailed breakdown of each method
- **Comparison Charts**: Visual comparison of all valuation results
- **Monte Carlo Visualization**: Price paths and distribution plots
- **LSTM Predictions**: Optional AI-based forecasting
- **Investment Recommendation**: Automated buy/hold/sell signal

## 🛠️ Configuration

### Valuation Parameters (adjustable in dashboard)

| Parameter | Default | Description |
|-----------|---------|-------------|
| Required Return (WACC) | 10% | Discount rate for DCF/DDM |
| Growth Rate (Stage 1) | 5% | Near-term growth rate |
| Terminal Growth | 2% | Long-term perpetual growth |
| Forecast Years | 5 | Explicit forecast period |

### Monte Carlo Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Number of Simulations | 10,000 | Price path simulations |
| Forecast Days | 252 | Trading days (1 year) |

## 📚 Dependencies

- **yfinance**: Yahoo Finance API for financial data
- **pandas/numpy**: Data manipulation and numerical operations
- **tensorflow**: LSTM neural network (optional)
- **scikit-learn**: Data preprocessing
- **matplotlib/seaborn/plotly**: Visualization
- **streamlit**: Interactive dashboard

## ⚠️ Disclaimer

This project is for **educational purposes only** and should not be considered as financial advice. Stock valuations involve significant uncertainty and assumptions. Always:

- Conduct your own research
- Consult with a qualified financial advisor
- Understand that past performance does not guarantee future results
- Consider your own risk tolerance and investment objectives

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or suggestions, please open an issue in this repository.
