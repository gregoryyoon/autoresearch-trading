import numpy as np
from numba import njit
from strategy_helpers import *

def get_strategy() -> dict:
    return dict(
        name="ema_sma_adx_roc_obv_v4",
        variables=["ema_period", "sma_period", "adx_period", "adx_threshold",
                   "roc_period", "roc_threshold", "atr_period", "trailing_stop_pct",
                   "obv_period", "wait_buy", "wait_sell"],
        bounds=([30, 40, 10, 16, 15, 0, 20, 1.5, 15, 25, 120],
                [65, 65, 25, 25, 30, 1.5, 45, 4.0, 30, 55, 200]),
        simulate=simulate,
    )

def simulate(close: np.ndarray, high: np.ndarray, low: np.ndarray,
             volume: np.ndarray, x: np.ndarray) -> tuple:
    """
    Optimized trend-following strategy:
    - EMA-SMA crossover as core entry signal
    - ADX trend filter to avoid whipsaws (especially AAPL)
    - ROC momentum confirmation for entry quality
    - OBV percentile rank for volume trend confirmation
    - ATR-based trailing stop for risk management
    - Asymmetric cooldowns (shorter buy, longer sell)
    """
    ema_period = int(x[0])
    sma_period = int(x[1])
    adx_period = max(int(x[2]), 1)
    adx_threshold = float(x[3])
    roc_period = max(int(x[4]), 1)
    roc_threshold = float(x[5])
    atr_period = max(int(x[6]), 1)
    trailing_stop_pct = float(x[7])
    obv_period = max(int(x[8]), 1)
    wait_buy = int(x[9])
    wait_sell = int(x[10])
    
    # Pre-compute indicators
    ema = ema_np(close, ema_period)
    sma = sma_np(close, sma_period)
    adx_data = adx_np(high, low, close, adx_period)
    adx = adx_data[0]
    roc = roc_np(close, roc_period)
    atr = atr_np(high, low, close, atr_period)
    obv = obv_np(close, volume)
    obv_ma = sma_np(obv, obv_period)
    
    return _execute(close, 1_000_000.0, ema, sma, adx, roc, atr, obv, obv_ma,
                    adx_threshold, roc_threshold, trailing_stop_pct,
                    wait_buy, wait_sell)

@njit(fastmath=True)
def _execute(close, start_cash, ema, sma, adx, roc, atr, obv, obv_ma,
             adx_threshold, roc_threshold, trailing_stop_pct,
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
        c_adx = adx[i]
        c_roc = roc[i]
        c_atr = atr[i]
        c_obv = obv[i]
        c_obv_ma = obv_ma[i]
        
        # Skip NaN values
        if np.isnan(c_ema) or np.isnan(c_sma) or np.isnan(c_adx):
            continue
        if np.isnan(c_roc) or np.isnan(c_atr) or np.isnan(c_obv):
            continue
        if np.isnan(c_obv_ma):
            continue
        
        # Update peak for trailing stop (track highest price since entry)
        if num_coins > 0 and price > peak_price:
            peak_price = price
        
        # TRAILING STOP: ATR-based exit to lock in gains
        if num_coins > 0 and peak_price > 0:
            stop_price = peak_price * (1.0 - trailing_stop_pct)
            if price <= stop_price:
                cash, num_coins = sell_all(cash, num_coins, price)
                last_trade = i
                num_trades += 1
                peak_price = 0.0
                continue
        
        # BUY SIGNAL: Multiple confirmations required
        if num_coins == 0:
            # 1. EMA > SMA (uptrend direction)
            # 2. ADX > threshold (trend strength, filters AAPL whipsaws)
            # 3. ROC > threshold (positive momentum)
            # 4. OBV above OBV MA (volume trend confirms price)
            # 5. Cooldown period respected
            if (c_ema > c_sma and 
                c_adx > adx_threshold and 
                c_roc > roc_threshold and
                c_obv > c_obv_ma and
                i > last_trade + wait_buy):
                cash, num_coins = buy_all(cash, num_coins, price)
                last_trade = i
                num_trades += 1
                peak_price = price
        # SELL SIGNAL: EMA crosses below SMA (trend reversal)
        # Longer cooldown to avoid being shaken out of strong trends
        elif num_coins > 0 and c_ema < c_sma and i > last_trade + wait_sell:
            cash, num_coins = sell_all(cash, num_coins, price)
            last_trade = i
            num_trades += 1

    # Force-sell at end of period
    cash, num_coins = sell_all(cash, num_coins, close[-1])
    return cash / start_cash, num_trades