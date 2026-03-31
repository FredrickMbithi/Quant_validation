# FX Quant Strategy Validation Framework

A Python-based validation framework for systematically testing quantitative FX trading hypotheses with rigorous statistical analysis, backtesting, and cost modeling.

## Purpose

This framework provides a structured approach to validate FX trading strategies before live deployment, ensuring:

- **Statistical Rigor** — IC tests, stationarity checks, significance testing
- **No Look-Ahead Bias** — Enforced signal lag and proper execution timing
- **Realistic Costs** — Spread, slippage, and commission modeling
- **Reproducibility** — All tests and validations documented

## Validation Workflow

```
Hypothesis → Data Collection → Statistical Tests → Backtest → Cost Analysis → Pass/Fail Decision
```

### 4-Phase Research Process

1. **Hypothesis Formulation** (`research/01_hypothesis.md`)
   - Define quantitative hypothesis
   - Specify expected outcomes
   - Identify risk factors
   - Document data requirements

2. **Exploratory Analysis** (`research/02_exploration.ipynb`, `research/03_exploration_gbpusd.ipynb`)
   - Visual data inspection
   - Feature distribution analysis
   - Correlation studies
   - Preliminary signal testing

3. **Statistical Validation** (via `src/stats.py`, `src/validation.py`)
   - Stationarity tests (ADF, KPSS)
   - Information Coefficient (IC) analysis
   - Significance testing (t-tests, p-values)
   - Cross-pair validation

4. **Results Documentation** (`research/04_results.md`)
   - Pass/fail verdict
   - Key statistical findings
   - Confidence levels
   - Recommendations for next steps

## 🏗️ Architecture

```
Quant_validation/
├── src/
│   ├── data_loader.py          # Generic data loading interface
│   ├── fx_data_loader.py       # FX-specific loader (29KB, production-ready)
│   ├── dukascopy_fetcher.py    # Dukascopy tick data fetcher
│   ├── histdata_converter.py   # HistData.com CSV converter
│   ├── backtest.py             # Backtesting engine
│   ├── stats.py                # Statistical test utilities
│   ├── validation.py           # Validation framework
│   ├── costs.py                # Cost model (spread/slippage)
│   └── execution_costs.py      # Execution cost analysis
├── research/
│   ├── 01_hypothesis.md        # Hypothesis template
│   ├── 02_exploration.ipynb    # Initial exploratory notebook
│   ├── 03_exploration_gbpusd.ipynb  # GBPUSD analysis (567KB)
│   └── 04_results.md           # Results template
├── data/
│   ├── raw/                    # Raw OHLC data
│   └── processed/              # Cleaned/validated data
├── results/
│   ├── csv/                    # Numerical results
│   ├── figures/                # Charts and visualizations
│   └── metrics/                # Performance metrics
├── tests/
│   ├── test_stats.py           # Statistical function tests
│   ├── test_no_lookahead.py    # Look-ahead bias tests
│   └── test_costs.py           # Cost model tests
└── requirements.txt            # Python dependencies
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/FredrickMbithi/Quant_validation.git
cd Quant_validation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### 1. Load FX Data

```python
from src.fx_data_loader import FXDataLoader

# Load GBPUSD hourly data
loader = FXDataLoader()
df = loader.load_pair('GBPUSD', timeframe='H1', start='2020-01-01', end='2023-12-31')
```

#### 2. Run Statistical Tests

```python
from src.stats import calculate_ic, adf_test

# Information Coefficient test
ic = calculate_ic(feature, forward_returns)
print(f"IC: {ic:.4f}")

# Stationarity test
adf_result = adf_test(df['close'])
print(f"ADF p-value: {adf_result['p_value']:.4f}")
```

#### 3. Backtest with Costs

```python
from src.backtest import Backtest
from src.costs import CostModel

# Initialize cost model
costs = CostModel(spread_pips=1.5, commission=7.0, slippage_pips=0.5)

# Run backtest
bt = Backtest(data=df, signal=signal, cost_model=costs)
results = bt.run()

print(f"Sharpe: {results['sharpe']:.2f}")
print(f"Net Return: {results['total_return']:.2%}")
```

#### 4. Validate Strategy

```python
from src.validation import validate_strategy

# Run full validation suite
verdict = validate_strategy(
    signal=signal,
    data=df,
    ic_threshold=0.03,
    min_trades=100,
    max_drawdown=0.25
)

if verdict['passed']:
    print("✓ Strategy PASSED validation")
else:
    print("✗ Strategy FAILED:", verdict['reason'])
```

## Validation Checklist

Before strategy deployment, ensure:

- [ ] **IC Test** — Information Coefficient > 0.03, p < 0.05
- [ ] **Stationarity** — ADF test p-value < 0.05
- [ ] **No Look-Ahead** — Signal shifted +1 bar, passes `test_no_lookahead.py`
- [ ] **Minimum Trades** — At least 100 trades in backtest
- [ ] **Sharpe Ratio** — Sharpe > 0.5 (preferably > 1.0)
- [ ] **Drawdown** — Max drawdown < 25%
- [ ] **Cost Sensitivity** — Profitable after 2x cost assumptions
- [ ] **Cross-Pair** — Works on at least 2 currency pairs
- [ ] **Regime Tests** — Stable across volatility regimes

## Testing

Run unit tests to verify framework integrity:

```bash
# Run all tests
python -m pytest tests/ -v

# Test specific modules
python -m pytest tests/test_stats.py
python -m pytest tests/test_no_lookahead.py
python -m pytest tests/test_costs.py
```

### Test Coverage

- **Statistical Functions** — IC, ADF, t-tests, p-values
- **Look-Ahead Bias** — Ensures signals use only past data
- **Cost Models** — Spread, slippage, commission calculations

## Data Sources Supported

### 1. Dukascopy (via `dukascopy_fetcher.py`)
- Tick data and 1-minute bars
- High-quality institutional data
- Free access (rate-limited)

### 2. HistData.com (via `histdata_converter.py`)
- 1-minute and tick data
- Popular pairs (EURUSD, GBPUSD, etc.)
- CSV format

### 3. Custom Data
- Flexible loader interface (`data_loader.py`)
- Supports any OHLC format
- Auto-validates for gaps and duplicates

## Key Modules

### `fx_data_loader.py` (29KB - Production Ready)

Comprehensive FX data loading with:
- Gap detection and filling
- Duplicate removal
- Timezone handling (UTC enforcement)
- Multi-pair support
- Timeframe resampling (M1 → H1, H4, D1)

### `stats.py`

Statistical utilities:
- Information Coefficient (Spearman rank correlation)
- Stationarity tests (ADF, KPSS)
- Significance testing (t-stats, p-values)
- Distribution analysis

### `validation.py`

Strategy validation framework:
- Pass/fail decision logic
- Minimum threshold enforcement
- Automated validation reports

### `execution_costs.py`

Realistic cost modeling:
- Fixed spread costs
- Variable slippage simulation
- Commission per trade
- Market impact estimation

## 📈 Example Research Workflow

### Step 1: Define Hypothesis

Edit `research/01_hypothesis.md`:

```markdown
## Research Question
Can ALMA slope predict mean reversion in GBPUSD H1?

## Expected Outcome
Negative IC, 55%+ win rate on counter-trend entries

## Risk Factors
- Low volatility regimes may reduce signal quality
- Spread costs may erode edge
```

### Step 2: Explore Data

Run `research/02_exploration.ipynb`:
- Load data
- Plot price/features
- Check for patterns

### Step 3: Test Hypothesis

```python
# In notebook or script
from src.stats import calculate_ic
from src.backtest import Backtest

# Test IC
ic = calculate_ic(alma_slope, forward_returns)
print(f"IC: {ic:.4f}")

# Backtest
bt = Backtest(data, signal)
results = bt.run()
print(results)
```

### Step 4: Document Results

Edit `research/04_results.md`:

```markdown
## Pass/Fail Verdict
PASS

## Key Findings
- IC = 0.04 (p < 0.01)
- Sharpe = 1.2
- Win rate = 58%
- Max DD = 18%

## Recommendation
Proceed to walk-forward validation
```

## Development

### Adding New Tests

1. Create feature in `src/`
2. Write tests in `tests/`
3. Document in `research/`
4. Validate with full framework

### Extending Data Loaders

Implement custom loader inheriting from `data_loader.py` interface.

## Related Projects

- [fx-quant-research](https://github.com/FredrickMbithi/fx-quant-research) — Full production FX system
- [ma-hp-filter](https://github.com/FredrickMbithi/ma-hp-filter) — MA + HP filter research
- [Exhaustion-failure-to-continue-hypothesis](https://github.com/FredrickMbithi/Exhaustion-failure-to-continue-hypothesis) — 65% win rate strategy

## ⚠️ Usage Notes

This is a **research framework**, not a trading system. Use it to:
- Validate hypotheses before coding strategies
- Ensure statistical rigor
- Document research process
- Avoid common pitfalls (look-ahead bias, overfitting)

**Not included:**
- Live execution
- Position sizing
- Risk management
- Order routing

For production deployment, see [fx-quant-research](https://github.com/FredrickMbithi/fx-quant-research).

## License

MIT License

## Author

Fredrick Mbithi

---

**Framework Version:** 1.0  
**Language:** Python 3.8+  
**Focus:** Statistical Validation & Hypothesis Testing  
**Status:** Production-Ready Research Framework
