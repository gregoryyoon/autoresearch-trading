import numpy as np
from numba import njit
from strategy_helpers import *

# ---------------------------------------------------------------------------
#  Strategy definition
# ---------------------------------------------------------------------------

def get_strategy() -> dict:
    return dict(
        name="ema_sma_roc_volatility_v2",
        variables=["ema_period", "sma_period", "roc_period", "roc_threshold",
                   "atr_period", "volatility_threshold", "trailing_stop_pct", 
                   "wait_buy", "wait_sell"],
        bounds=([30, 40, 10, 0.2, 14, 0.01, 1.5, 20, 100],
                [80, 70, 30, 2.0, 42, 0.05, 8.0, 80, 250]),
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
    atr_period = max(int(x[4]), 1)
    volatility_threshold = float(x[5])
    trailing_stop_pct = float(x[6])
    wait_buy = int(x[7])
    wait_sell = int(x[8])
    
    ema = ema_np(close, ema_period)
    sma = sma_np(close, sma_period)
    roc = roc_np(close, roc_period)
    atr = atr_np(high, low, close, atr_period)
    
    return _execute(close, 1_000_000.0, ema, sma, roc, atr,
                    ema_period, sma_period, roc_period, roc_threshold,
                    atr_period, volatility_threshold, trailing_stop_pct,
                    wait_buy, wait_sell)

# ---------------------------------------------------------------------------
#  Numba-compiled trading loop (the hot path)
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def _execute(close, start_cash, ema, sma, roc, atr,
             ema_period, sma_period, roc_period, roc_threshold,
             atr_period, volatility_threshold, trailing_stop_pct,
             wait_buy, wait_sell):
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
        c_atr = atr[i]
        
        # Skip NaN values (first period-1 values are NaN)
        if np.isnan(c_ema) or np.isnan(c_sma) or np.isnan(c_roc):
            continue
        if np.isnan(c_atr):
            continue
        
        # VOLATILITY FILTER: Skip trading in choppy/low-volatility markets
        # This helps with AAPL-type stocks that don't trend well
        normalized_atr = c_atr / price
        if normalized_atr < volatility_threshold:
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
        
        # BUY SIGNAL: EMA crosses above SMA + ROC confirms momentum
        # ROC > threshold ensures we're in a real trend, not noise
        if num_coins == 0:
            if c_ema > c_sma and c_roc > roc_threshold and i > last_trade + wait_buy:
                cash, num_coins = buy_all(cash, num_coins, price)
                last_trade = i
                num_trades += 1
                peak_price = price
        # SELL SIGNAL: EMA crosses below SMA (trend reversal)
        elif num_coins > 0 and c_ema < c_sma and i > last_trade + wait_sell:
            cash, num_coins = sell_all(cash, num_coins, price)
            last_trade = i
            num_trades += 1

    # Force-sell at end of period
    cash, num_coins = sell_all(cash, num_coins, close[-1])
    return cash / start_cash, num_trades