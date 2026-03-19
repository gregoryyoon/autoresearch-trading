# strategy.py — THE FILE THE AI AGENT EDITS
#
# Strategy: EMA-SMA crossover with ROC momentum confirmation and
#           percentage trailing stop to capture momentum while protecting profits.
#
# Key changes from winning baseline:
#   - Add ROC filter to confirm trend direction before entering
#   - Simple percentage trailing stop instead of ATR (fewer params, more robust)
#   - Tighter parameter bounds around the winning configuration
#   - Keep asymmetric waits (longer on exits to let winners run)

import numpy as np
from numba import njit
from strategy_helpers import *

# ---------------------------------------------------------------------------
#  Strategy definition
# ---------------------------------------------------------------------------

def get_strategy() -> dict:
    return dict(
        name="ema_sma_momentum_trailing_v1",
        variables=["ema_period", "sma_period", "roc_period", "roc_threshold",
                   "trailing_stop_pct", "wait_buy", "wait_sell"],
        bounds=([30, 45, 10, 0.5, 1.0, 30, 100],
                [60, 80, 30, 3.0, 5.0, 100, 250]),
        simulate=simulate,
    )

# ---------------------------------------------------------------------------
#  Simulation entry point (regular Python — bridges numpy prep and numba)
# ---------------------------------------------------------------------------

def simulate(close: np.ndarray, high: np.ndarray, low: np.ndarray,
             volume: np.ndarray, x: np.ndarray) -> tuple:
    """
    Compute indicators, then delegate to the numba-compiled trading loop.
    """
    ema_period = int(x[0])
    sma_period = int(x[1])
    roc_period = max(int(x[2]), 1)
    roc_threshold = float(x[3])
    trailing_stop_pct = float(x[4])
    wait_buy = int(x[5])
    wait_sell = int(x[6])
    
    ema = ema_np(close, ema_period)
    sma = sma_np(close, sma_period)
    roc = roc_np(close, roc_period)
    
    return _execute(close, 1_000_000.0, ema, sma, roc,
                    ema_period, sma_period, roc_period, roc_threshold,
                    trailing_stop_pct, wait_buy, wait_sell)

# ---------------------------------------------------------------------------
#  Numba-compiled trading loop (the hot path)
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def _execute(close, start_cash, ema, sma, roc,
             ema_period, sma_period, roc_period, roc_threshold,
             trailing_stop_pct, wait_buy, wait_sell):
    cash = start_cash
    num_coins = 0
    last_trade = 0
    num_trades = 0
    peak_price = 0.0
    
    for i in range(len(close)):
        price = close[i]
        c_ema = ema[i]
        c_sma = sma[i]
        c_roc = roc[i]
        
        # Skip NaN values (first period-1 values are NaN)
        if np.isnan(c_ema) or np.isnan(c_sma) or np.isnan(c_roc):
            continue
        
        # Update peak for trailing stop (track highest price since entry)
        if num_coins > 0 and price > peak_price:
            peak_price = price
        
        # Check trailing stop hit
        if num_coins > 0 and peak_price > 0:
            if price <= peak_price * (1.0 - trailing_stop_pct):
                cash, num_coins = sell_all(cash, num_coins, price)
                last_trade = i
                num_trades += 1
                peak_price = 0.0
                continue
        
        # Buy signal: EMA crosses above SMA + ROC confirms upward momentum
        # ROC > 0 means price is higher than ROC_period ago
        # ROC > threshold adds confirmation we're in a real uptrend
        if num_coins == 0 and c_ema > c_sma and c_roc > 0 and c_roc > roc_threshold and i > last_trade + wait_buy:
            cash, num_coins = buy_all(cash, num_coins, price)
            last_trade = i
            num_trades += 1
            peak_price = price
        # Sell signal: EMA crosses below SMA (trend reversal)
        elif num_coins > 0 and c_ema < c_sma and i > last_trade + wait_sell:
            cash, num_coins = sell_all(cash, num_coins, price)
            last_trade = i
            num_trades += 1

    # Force-sell at end of period
    cash, num_coins = sell_all(cash, num_coins, close[-1])
    return cash / start_cash, num_trades