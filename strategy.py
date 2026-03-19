import numpy as np
from numba import njit
from strategy_helpers import *

def get_strategy() -> dict:
    return dict(
        name="ema_sma_adx_roc_rsi_atr_v5",
        variables=["ema_period", "sma_period", "adx_period", "adx_threshold",
                   "roc_period", "rsi_period", "rsi_upper",
                   "atr_period", "atr_stop_pct", "sell_cooldown"],
        bounds=([30, 35, 10, 12, 15, 10, 55, 20, 1.0, 45],
                [50, 50, 18, 16, 28, 20, 75, 35, 3.0, 65]),
        simulate=simulate,
    )

def simulate(close: np.ndarray, high: np.ndarray, low: np.ndarray,
             volume: np.ndarray, x: np.ndarray) -> tuple:
    """
    Refined trend-following strategy:
    - EMA-SMA crossover as core entry/exit signal
    - ADX trend filter (lower threshold for AAPL)
    - ROC momentum + RSI quality for entry confirmation
    - ATR trailing stop for risk management
    - Longer sell cooldown to avoid whipsaws
    - 10 parameters for better optimization stability
    """
    ema_period = int(x[0])
    sma_period = int(x[1])
    adx_period = max(int(x[2]), 1)
    adx_threshold = float(x[3])
    roc_period = max(int(x[4]), 1)
    rsi_period = max(int(x[5]), 1)
    rsi_upper = float(x[6])
    atr_period = max(int(x[7]), 1)
    atr_stop_pct = float(x[8])
    sell_cooldown = int(x[9])
    
    # Pre-compute indicators
    ema = ema_np(close, ema_period)
    sma = sma_np(close, sma_period)
    adx_data = adx_np(high, low, close, adx_period)
    adx = adx_data[0]
    roc = roc_np(close, roc_period)
    rsi = rsi_np(close, rsi_period)
    atr = atr_np(high, low, close, atr_period)
    
    # Fixed buy cooldown (shorter than sell for faster entries)
    buy_cooldown = 8
    
    return _execute(close, 1_000_000.0, ema, sma, adx, roc, rsi, atr,
                    adx_threshold, roc_period, rsi_upper,
                    atr_stop_pct, buy_cooldown, sell_cooldown)

@njit(fastmath=True)
def _execute(close, start_cash, ema, sma, adx, roc, rsi, atr,
             adx_threshold, roc_period, rsi_upper,
             atr_stop_pct, buy_cooldown, sell_cooldown):
    cash = start_cash
    num_coins = 0
    last_trade = 0
    num_trades = 0
    peak_price = 0.0
    
    for i in range(len(close)):
        price = close[i]
        c_ema = ema[i]
        c_sma = sma[i]
        c_adx = adx[i]
        c_roc = roc[i]
        c_rsi = rsi[i]
        c_atr = atr[i]
        
        # Skip NaN values
        if np.isnan(c_ema) or np.isnan(c_sma) or np.isnan(c_adx):
            continue
        if np.isnan(c_roc) or np.isnan(c_rsi) or np.isnan(c_atr):
            continue
        
        # Update peak for trailing stop (track highest price since entry)
        if num_coins > 0 and price > peak_price:
            peak_price = price
        
        # TRAILING STOP: ATR-based percentage exit to lock in gains
        if num_coins > 0 and peak_price > 0:
            stop_price = peak_price * (1.0 - atr_stop_pct)
            if price <= stop_price:
                cash, num_coins = sell_all(cash, num_coins, price)
                last_trade = i
                num_trades += 1
                peak_price = 0.0
                continue
        
        # BUY SIGNAL: Multiple confirmations required
        if num_coins == 0:
            # 1. EMA > SMA (uptrend direction)
            # 2. ADX > threshold (trend strength)
            # 3. ROC > 0 (positive momentum)
            # 4. RSI < upper bound (not overbought)
            # 5. Buy cooldown period respected
            if (c_ema > c_sma and 
                c_adx > adx_threshold and 
                c_roc > 0.0 and
                c_rsi < rsi_upper and
                i > last_trade + buy_cooldown):
                cash, num_coins = buy_all(cash, num_coins, price)
                last_trade = i
                num_trades += 1
                peak_price = price
        
        # SELL SIGNAL: EMA crosses below SMA (trend reversal)
        # Longer cooldown to avoid being shaken out of strong trends
        elif num_coins > 0 and c_ema < c_sma and i > last_trade + sell_cooldown:
            cash, num_coins = sell_all(cash, num_coins, price)
            last_trade = i
            num_trades += 1
        
    # Force-sell at end of period
    cash, num_coins = sell_all(cash, num_coins, close[-1])
    return cash / start_cash, num_trades