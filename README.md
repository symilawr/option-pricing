# Option Pricing & Mispricing Detector

A command-line tool that pulls live option chains from Yahoo Finance, extrapolates the risk-free rate empirically via put-call parity, and uses Black-Scholes to flag potentially mispriced options.

---

## Requirements

- Python 3.9+
- Dependencies:

```bash
pip install yfinance scipy numpy pandas
```

---

## Usage

```bash
python option_pricing_model.py <TICKER> [options]
```

### Examples

```bash
# Basic run — analyse AAPL with defaults
python option_pricing_model.py AAPL

# Higher open interest filter, tighter misprice threshold, save to CSV
python option_pricing_model.py TSLA --min-oi 500 --misprice-threshold 0.15 --export-csv

# Include IV skew table for the nearest expiration
python option_pricing_model.py SPY --skew

# Scan more expirations with a low OI bar
python option_pricing_model.py NVDA --max-expirations 6 --min-oi 100
```

---

## Parameters

| Flag | Default | Description |
|---|---|---|
| `ticker` | *(required)* | Stock ticker symbol (e.g. `AAPL`, `SPY`, `TSLA`) |
| `--min-oi` | `200` | Minimum open interest — options below this are ignored |
| `--misprice-threshold` | `0.10` | % gap between market price and BS value to flag as mispriced |
| `--max-expirations` | `4` | Number of nearest expiration dates to analyse |
| `--skew` | off | Print IV skew table for the nearest expiration |
| `--export-csv` | off | Save full results to `<TICKER>_option_analysis.csv` in the current directory |

---

## How It Works

### Step 1 — Risk-Free Rate Extraction

Rather than using a fixed Treasury rate, the model derives the risk-free rate directly from the live option chain using **put-call parity**:

```
C − P = S − K · e^(−rT)
```

For every matched call/put pair at the same strike, it solves for `r`. The median across all qualifying pairs becomes the working rate for that expiration. This self-calibrates to whatever rate the market is actually pricing in, which can differ from quoted Treasury yields due to borrow costs, dividends, or funding spreads.

### Step 2 — Implied Volatility

Using the extracted `r`, each option's market mid-price (or last price if no bid/ask is available) is inverted through **Newton-Raphson iteration** to produce an implied volatility. The solver runs up to 200 iterations with `σ` bounded between 0.01% and 2000%. Falls back to Yahoo Finance's own IV field if convergence fails.

### Step 3 — Black-Scholes Re-Pricing

With `(r, IV)` in hand, the model computes the Black-Scholes fair value and calculates:

```
pct_diff = (market_mid − BS_value) / BS_value
```

Options beyond `--misprice-threshold` are flagged:
- **▲ OVER** — market price exceeds BS fair value (potentially expensive to buy)
- **▼ UNDER** — market price is below BS fair value (potential long opportunity)

---

## Output

The tool prints four sections to the terminal:

1. **Extracted risk-free rates** — per expiration, with standard deviation across the pairs used
2. **Mispriced options table** — all options exceeding the threshold, sorted by deviation magnitude
3. **Top 5 underpriced / overpriced** — quick-glance best candidates
4. **IV skew table** *(optional, `--skew`)* — call and put IVs across all strikes for the nearest expiry

If `--export-csv` is passed, a full results file is saved as `<TICKER>_option_analysis.csv` in the current working directory, containing all analysed options with every computed field.

---

## Interpreting Results

**Most apparent mispricings are not real.** Before acting on any flag, check:

- **Bid/ask spread** — wide spreads (common in low-volume options) account for most gaps between mid-price and model value
- **Stale last price** — if volume is zero, `lastPrice` may be days old; the model uses mid-price when bid/ask are available but falls back otherwise
- **Flat IV (e.g. exactly 30%)** — Yahoo Finance sometimes returns a default value for illiquid strikes with no recent trades; these are not meaningful signals
- **Very short DTE** — Black-Scholes is least reliable at 0–2 days to expiry, especially with volatile surfaces
- **Earnings / events** — elevated put IV skew near ATM typically reflects event risk, not mispricing

---

## Limitations

- Uses European Black-Scholes pricing; does not account for early exercise premium on American options
- Dividends are not modelled; results may be less accurate for high-yield stocks
- Risk-free rate extraction requires liquid put-call pairs at the same strike — may fall back to 4.5% if insufficient pairs are found
- Yahoo Finance data quality varies; illiquid options may have stale prices or missing bid/ask

---

## License

MIT
