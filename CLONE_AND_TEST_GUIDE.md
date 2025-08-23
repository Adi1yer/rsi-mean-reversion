# Clone and Test Guide

This guide walks you through cloning and testing the RSI Mean Reversion Trading Bot from scratch.

## Prerequisites

- Python 3.8 or higher
- Git installed
- Internet connection (for downloading market data)

## Step 1: Clone the Repository

```bash
git clone https://github.com/Adi1yer/rsi-mean-reversion.git
cd rsi-mean-reversion
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Test Installation

```bash
python test_installation.py
```

**Expected Output:**
```
🎉 ALL TESTS PASSED!
Your installation is working correctly.
You can now run: python run_optimized_backtest.py
```

## Step 4: Run the Backtest

```bash
python run_optimized_backtest.py
```

**This will:**
- Download 10 years of market data (2014-2024)
- Execute the optimized trading strategy
- Generate performance charts
- Save results to `results/` directory

**Expected Results:**
- Total Return: ~368%
- SPY Benchmark Return: ~290%
- Excess Return: ~78%
- Multiple performance charts saved

## Step 5: Review Results

Check the `results/` directory for:
- `optimized_backtest_results_*.json` - Detailed backtest results
- `optimized_backtest_charts_*.png` - Performance visualization

## Troubleshooting

### Common Issues:

**Import Errors:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Missing Data:**
```bash
python setup.py  # Creates required directories
```

**Test Failures:**
```bash
python test_installation.py  # Identify specific issues
```

## What the Strategy Does

1. **70% Dividend Aristocrats:** Blue-chip stocks with consistent dividends
2. **30% High-Growth Unprofitable:** High-growth companies with positive cash flow
3. **RSI-Based Trading:** Uses RSI mean reversion for entry/exit signals
4. **Dynamic Rebalancing:** Adjusts allocation based on market conditions
5. **Dividend Reinvestment:** Automatically reinvests all dividends

## Performance Highlights

- **368% total return** vs 290% SPY (10-year backtest)
- **78% excess return** over benchmark
- **Dynamic rebalancing** maintains target allocation
- **Risk management** through diversification

## Next Steps

- Modify parameters in `run_optimized_backtest.py`
- Explore different time periods
- Add your own stocks to the universe
- Implement live trading (not included)

## Support

If you encounter issues:
1. Check this guide first
2. Run `python test_installation.py` to diagnose
3. Ensure all dependencies are installed
4. Review error messages for specific issues

## Repository Structure

```
rsi-mean-reversion/
├── README.md                    # Main documentation
├── requirements.txt             # Dependencies
├── setup.py                     # Setup script
├── test_installation.py         # Installation tester
├── run_optimized_backtest.py    # Main backtest script
├── src/                         # Core modules
│   ├── data/                    # Data analysis
│   ├── backtesting/             # Backtesting engine
│   ├── strategies/              # Trading strategies
│   ├── ml/                      # Machine learning
│   └── risk_management/         # Risk management
├── data/                        # Market data (created)
└── results/                     # Backtest results (created)
```

Happy backtesting! 🚀 