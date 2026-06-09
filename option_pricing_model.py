"""
Option Pricing & Mispricing Detector
=====================================
Pulls live option chains from Yahoo Finance, extrapolates the risk-free rate
from put-call parity / Newton-Raphson implied vol, then re-prices every option
with Black-Scholes to flag potential mispricings.

Usage:
    python option_pricing_model.py AAPL
    python option_pricing_model.py TSLA --min-oi 500 --misprice-threshold 0.15
"""

import argparse
import warnings
from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import sys

warnings.filterwarnings("ignore")

# ── ANSI colours ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes helpers
# ─────────────────────────────────────────────────────────────────────────────

def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "call") -> float:
    """Black-Scholes price for European call or put."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if option_type == "call" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega (dPrice/dSigma) — shared by calls and puts."""
    if T <= 0 or sigma <= 0:
        return 1e-10
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def implied_vol(market_price: float, S: float, K: float, T: float, r: float,
                option_type: str = "call") -> Optional[float]:
    """
    Newton-Raphson implied volatility.
    Returns None if convergence fails or price is below intrinsic.
    """
    if T <= 0 or market_price <= 0:
        return None
    intrinsic = max(0.0, (S - K) if option_type == "call" else (K - S))
    if market_price < intrinsic * 0.99:
        return None
    try:
        sigma = 0.3  # initial guess
        for _ in range(200):
            price  = bs_price(S, K, T, r, sigma, option_type)
            vega   = bs_vega(S, K, T, r, sigma)
            diff   = price - market_price
            if abs(diff) < 1e-8:
                break
            if vega < 1e-10:
                break
            sigma -= diff / vega
            sigma = max(1e-4, min(sigma, 20.0))
        if sigma <= 0 or sigma > 15:
            return None
        return sigma
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Risk-free rate extrapolation via put-call parity
# ─────────────────────────────────────────────────────────────────────────────

def extrapolate_risk_free_rate(calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                                S: float, T: float,
                                min_oi: int = 100) -> Optional[float]:
    """
    Use put-call parity (C - P = S - K*e^{-rT}) to back out r.

    We collect matched (call, put) pairs at the same strike, compute r for each,
    then return the median of the distribution as the best estimate.
    """
    rates = []
    merged = pd.merge(
        calls_df[["strike", "lastPrice", "openInterest"]].rename(
            columns={"lastPrice": "call_price", "openInterest": "call_oi"}),
        puts_df[["strike", "lastPrice", "openInterest"]].rename(
            columns={"lastPrice": "put_price", "openInterest": "put_oi"}),
        on="strike"
    )
    merged = merged[
        (merged["call_oi"] >= min_oi) &
        (merged["put_oi"]  >= min_oi) &
        (merged["call_price"] > 0.05) &
        (merged["put_price"]  > 0.05)
    ]
    for _, row in merged.iterrows():
        K   = row["strike"]
        C   = row["call_price"]
        P   = row["put_price"]
        # C - P = S - K*e^{-rT}  =>  K*e^{-rT} = S - C + P
        lhs = S - C + P
        if lhs <= 0 or K <= 0 or T <= 0:
            continue
        r = -np.log(lhs / K) / T
        if -0.10 < r < 0.30:          # sanity bounds
            rates.append(r)
    if len(rates) < 3:
        return None
    return float(np.median(rates))


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis engine
# ─────────────────────────────────────────────────────────────────────────────

class OptionAnalyzer:
    def __init__(self, ticker: str, min_oi: int = 200,
                 misprice_threshold: float = 0.10,
                 max_expirations: int = 4):
        self.ticker             = ticker.upper()
        self.min_oi             = min_oi
        self.misprice_threshold = misprice_threshold
        self.max_expirations    = max_expirations
        self.yf_ticker          = yf.Ticker(self.ticker)

    # ── Data fetching ─────────────────────────────────────────────────────────

    def get_spot_price(self) -> float:
        hist = self.yf_ticker.history(period="1d")
        if hist.empty:
            raise ValueError(f"Could not fetch price for {self.ticker}")
        return float(hist["Close"].iloc[-1])

    def get_expirations(self):
        exps = self.yf_ticker.options
        if not exps:
            raise ValueError(f"No options data found for {self.ticker}")
        today = date.today()
        future = [e for e in exps
                  if datetime.strptime(e, "%Y-%m-%d").date() > today]
        return future[:self.max_expirations]

    def get_option_chain(self, expiration: str):
        chain = self.yf_ticker.option_chain(expiration)
        return chain.calls, chain.puts

    # ── Per-expiration processing ─────────────────────────────────────────────

    def process_expiration(self, expiration: str, S: float) -> pd.DataFrame:
        calls_raw, puts_raw = self.get_option_chain(expiration)
        exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
        T = (exp_date - date.today()).days / 365.0
        if T <= 0:
            return pd.DataFrame()

        # Filter by open interest
        calls = calls_raw[calls_raw["openInterest"] >= self.min_oi].copy()
        puts  = puts_raw [puts_raw ["openInterest"] >= self.min_oi].copy()

        if calls.empty and puts.empty:
            return pd.DataFrame()

        # ── Step 1: Extrapolate risk-free rate ────────────────────────────────
        r_extracted = extrapolate_risk_free_rate(
            calls_raw, puts_raw, S, T, min_oi=max(50, self.min_oi // 2))
        r = r_extracted if r_extracted is not None else 0.045  # fallback

        rows = []
        for opt_type, df in [("call", calls), ("put", puts)]:
            for _, row in df.iterrows():
                K           = float(row["strike"])
                mkt_price   = float(row.get("lastPrice", 0))
                bid         = float(row.get("bid", 0))
                ask         = float(row.get("ask", 0))
                oi          = int(row.get("openInterest", 0))
                volume      = int(row.get("volume", 0) or 0)
                yf_iv       = float(row.get("impliedVolatility", 0) or 0)

                # Use mid-price when possible, fall back to lastPrice
                mid_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else mkt_price
                if mid_price <= 0:
                    continue

                # ── Step 2: Compute IV from market price using our r ──────────
                iv = implied_vol(mid_price, S, K, T, r, opt_type)
                if iv is None:
                    iv = yf_iv if yf_iv > 0 else None
                if iv is None:
                    continue

                # ── Step 3: BS fair value ─────────────────────────────────────
                bs_val   = bs_price(S, K, T, r, iv, opt_type)
                pct_diff = (mid_price - bs_val) / bs_val if bs_val > 0.01 else 0.0

                rows.append({
                    "expiration"  : expiration,
                    "type"        : opt_type,
                    "strike"      : K,
                    "days_to_exp" : int(T * 365),
                    "T"           : T,
                    "spot"        : S,
                    "bid"         : bid,
                    "ask"         : ask,
                    "mid_price"   : round(mid_price, 4),
                    "bs_value"    : round(bs_val, 4),
                    "pct_diff"    : round(pct_diff, 4),
                    "impl_vol"    : round(iv, 4),
                    "r_extracted" : round(r, 4),
                    "open_interest": oi,
                    "volume"      : volume,
                    "moneyness"   : round(S / K, 4),
                })
        return pd.DataFrame(rows)

    # ── Full run ──────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        print(f"\n{BOLD}{CYAN}━━━  Option Pricing & Mispricing Detector  ━━━{RESET}")
        print(f"{BOLD}Ticker:{RESET} {self.ticker}    "
              f"{BOLD}Min OI:{RESET} {self.min_oi}    "
              f"{BOLD}Misprice threshold:{RESET} {self.misprice_threshold:.0%}\n")

        S = self.get_spot_price()
        print(f"  Spot price : {BOLD}${S:.2f}{RESET}")

        expirations = self.get_expirations()
        print(f"  Expirations analysed : {', '.join(expirations)}\n")

        all_frames = []
        for exp in expirations:
            df = self.process_expiration(exp, S)
            if not df.empty:
                all_frames.append(df)

        if not all_frames:
            print(f"{YELLOW}No options met the minimum open-interest filter.{RESET}")
            return pd.DataFrame()

        result = pd.concat(all_frames, ignore_index=True)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def flag(pct_diff: float, threshold: float) -> str:
    if pct_diff >  threshold:   return f"{GREEN}▲ OVER{RESET} "
    if pct_diff < -threshold:   return f"{RED}▼ UNDER{RESET}"
    return f"{DIM}  fair {RESET}"


def print_summary(df: pd.DataFrame, threshold: float):
    mispriced = df[df["pct_diff"].abs() >= threshold].copy()
    if mispriced.empty:
        print(f"\n{YELLOW}No significantly mispriced options found at the "
              f"{threshold:.0%} threshold.{RESET}")
        return

    print(f"\n{BOLD}━━━  Potentially Mispriced Options  ({'>' + f'{threshold:.0%}'} deviation)  ━━━{RESET}\n")
    mispriced = mispriced.sort_values("pct_diff", key=abs, ascending=False)

    header = (f"{'Exp':<12} {'T':<5} {'Type':<5} {'Strike':>7} {'Moneyness':>10} "
              f"{'Bid':>6} {'Ask':>6} {'Mid':>7} {'BS Val':>7} "
              f"{'Diff%':>7} {'IV':>6} {'r%':>6} {'OI':>7} {'Vol':>6}  Flag")
    print(f"{BOLD}{header}{RESET}")
    print("─" * len(header))

    for _, r in mispriced.iterrows():
        diff_col = (f"{GREEN}{r.pct_diff:+.1%}{RESET}"
                    if r.pct_diff > 0 else f"{RED}{r.pct_diff:+.1%}{RESET}")
        iv_pct   = f"{r.impl_vol:.1%}"
        r_pct    = f"{r.r_extracted:.2%}"
        print(
            f"{r.expiration:<12} {r.days_to_exp:<5} {r.type:<5} "
            f"{r.strike:>7.2f} {r.moneyness:>10.3f} "
            f"{r.bid:>6.2f} {r.ask:>6.2f} {r.mid_price:>7.4f} {r.bs_value:>7.4f} "
            f"{diff_col:>15} {iv_pct:>6} {r_pct:>6} "
            f"{r.open_interest:>7,} {r.volume:>6,}  "
            f"{flag(r.pct_diff, threshold)}"
        )


def print_rf_summary(df: pd.DataFrame):
    """Show extracted risk-free rates per expiration."""
    print(f"\n{BOLD}━━━  Extracted Risk-Free Rates (put-call parity)  ━━━{RESET}\n")
    summary = (df.groupby("expiration")
                 .agg(r_mean=("r_extracted", "mean"),
                      r_std =("r_extracted", "std"),
                      n_opts=("r_extracted", "count"))
                 .reset_index())
    for _, row in summary.iterrows():
        bar = "█" * max(1, int(row.r_mean * 200))
        print(f"  {row.expiration}  r = {BOLD}{row.r_mean:.3%}{RESET}  "
              f"±{row.r_std:.3%}  ({int(row.n_opts)} options)  {CYAN}{bar}{RESET}")


def print_iv_skew(df: pd.DataFrame, expiration: str):
    """Print a simple IV skew table for one expiration."""
    sub = df[df["expiration"] == expiration].sort_values("strike")
    if sub.empty:
        return
    print(f"\n{BOLD}━━━  IV Skew — {expiration}  ━━━{RESET}\n")
    print(f"  {'Strike':>8}  {'Moneyness':>10}  {'Call IV':>8}  {'Put IV':>8}")
    print("  " + "─" * 42)
    calls = sub[sub["type"] == "call"].set_index("strike")["impl_vol"]
    puts  = sub[sub["type"] == "put" ].set_index("strike")["impl_vol"]
    strikes = sorted(set(calls.index) | set(puts.index))
    spot = sub["spot"].iloc[0]
    for K in strikes:
        c_iv = f"{calls[K]:.2%}" if K in calls.index else "   —  "
        p_iv = f"{puts[K]:.2%}"  if K in puts.index  else "   —  "
        mono = spot / K
        marker = f"{BOLD}◄ ATM{RESET}" if abs(mono - 1.0) < 0.02 else ""
        print(f"  {K:>8.2f}  {mono:>10.3f}  {c_iv:>8}  {p_iv:>8}  {marker}")


def export_csv(df: pd.DataFrame, ticker: str):
    path = f"{ticker}_option_analysis.csv"
    df.to_csv(path, index=False)
    print(f"\n{DIM}Full results saved → {path}{RESET}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Black-Scholes option mispricing detector with live Yahoo Finance data")
    parser.add_argument("ticker",
                        help="Stock ticker symbol (e.g. AAPL, TSLA, SPY)")
    parser.add_argument("--min-oi", type=int, default=200,
                        help="Minimum open interest to consider (default: 200)")
    parser.add_argument("--misprice-threshold", type=float, default=0.10,
                        help="Percentage difference to flag as mispriced (default: 0.10)")
    parser.add_argument("--max-expirations", type=int, default=4,
                        help="Number of nearest expirations to analyse (default: 4)")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export full results to CSV")
    parser.add_argument("--skew", action="store_true",
                        help="Print IV skew table for nearest expiration")
    args = parser.parse_args()

    analyzer = OptionAnalyzer(
        ticker             = args.ticker,
        min_oi             = args.min_oi,
        misprice_threshold = args.misprice_threshold,
        max_expirations    = args.max_expirations,
    )

    try:
        df = analyzer.run()
    except ValueError as e:
        print(f"\n{RED}Error: {e}{RESET}")
        sys.exit(1)

    if df.empty:
        sys.exit(0)

    # ── Stats ─────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}Total options analysed:{RESET} {len(df):,}  "
          f"(calls: {(df.type=='call').sum():,} / puts: {(df.type=='put').sum():,})")
    print(f"{BOLD}IV range:{RESET} "
          f"{df.impl_vol.min():.1%} – {df.impl_vol.max():.1%}  "
          f"(median {df.impl_vol.median():.1%})")

    print_rf_summary(df)
    print_summary(df, args.misprice_threshold)

    if args.skew:
        nearest_exp = df["expiration"].min()
        print_iv_skew(df, nearest_exp)

    if args.export_csv:
        export_csv(df, args.ticker)

    # ── Top opportunities ─────────────────────────────────────────────────────
    print(f"\n{BOLD}━━━  Top 5 Underpriced Options (market < BS value)  ━━━{RESET}\n")
    top_under = df.nsmallest(5, "pct_diff")[
        ["expiration","type","strike","mid_price","bs_value","pct_diff","impl_vol","open_interest"]]
    print(top_under.to_string(index=False))

    print(f"\n{BOLD}━━━  Top 5 Overpriced Options (market > BS value)  ━━━{RESET}\n")
    top_over = df.nlargest(5, "pct_diff")[
        ["expiration","type","strike","mid_price","bs_value","pct_diff","impl_vol","open_interest"]]
    print(top_over.to_string(index=False))

    print(f"\n{DIM}Note: Mispricing signals are relative to the Black-Scholes model with "
          f"implied risk-free rate. Always verify with bid/ask spread and market conditions "
          f"before trading.{RESET}\n")


if __name__ == "__main__":
    main()