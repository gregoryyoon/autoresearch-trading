import numpy as np
from numba import njit
from strategy_helpers import *

def get_strategy() -> dict:
    return dict(
        name="ema_sma_adx_roc_volume_v3",
        variables=["ema_period", "sma_period", "adx_period", "adx_threshold",
                   "roc_period", "roc_threshold", "atr_period", "trailing_stop_pct",
                   "wait_buy", "wait_sell"],
        bounds=([30, 40, 10, 15, 12, 0.3, 14, 1.5, 20, 80],
                [70, 70, 25, 35, 30, 2.0, 42, 5.0, 60, 200]),
        simulate=simulate,
    )

def simulate(close: np.ndarray, high: np.ndarray, low: np.ndarray,
             volume: np.ndarray, x: np.ndarray) -> tuple:
    """
    Hybrid strategy: EMA-SMA crossover + ADX trend filter + ROC momentum + Volume confirmation + ATR trailing stop
    """
    ema_period = int(x[0])
    sma_period = int(x[1])
    adx_period = max(int(x[2]), 1)
    adx_threshold = float(x[3])
    roc_period = max(int(x[4]), 1)
    roc_threshold = float(x[5])
    atr_period = max(int(x[6]), 1)
    trailing_stop_pct = float(x[7])
    wait_buy = int(x[8])
    wait_sell = int(x[9])
    
    # Pre-compute indicators
    ema = ema_np(close, ema_period)
    sma = sma_np(close, sma_period)
    adx_data = adx_np(high, low, close, adx_period)
    adx = adx_data[0]
    roc = roc_np(close, roc_period)
    atr = atr_np(high, low, close, atr_period)
    obv = obv_np(close, volume)
    
    return _execute(close, 1_000_000.0, ema, sma, adx, roc, atr, obv,
                    adx_threshold, roc_threshold, atr_period,
                    trailing_stop_pct, wait_buy, wait_sell)

@njit(fastmath=True)
def _execute(close, start_cash, ema, sma, adx, roc, atr, obv,
             adx_threshold, roc_threshold, atr_period,
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
        c_adx = adx[i]
        c_roc = roc[i]
        c_atr = atr[i]
        c_obv = obv[i]
        
        # Skip NaN values (first period-1 values are NaN)
        if np.isnan(c_ema) or np.isnan(c_sma) or np.isnan(c_adx):
            continue
        if np.isnan(c_roc) or np.isnan(c_atr) or np.isnan(c_obv):
            continue
        
        # Update peak for trailing stop (track highest price since entry)
        if num_coins > 0 and price > peak_price:
            peak_price = price
        
        # TRAILING STOP: ATR-based exit to lock in gains
        # Protects from large drawdowns while letting winners run
        if num_coins > 0 and peak_price > 0:
            stop_price = peak_price * (1.0 - trailing_stop_pct)
            if price <= stop_price:
                cash, num_coins = sell_all(cash, num_coins, price)
                last_trade = i
                num_trades += 1
                peak_price = 0.0
                continue
        
        # BUY SIGNAL: Multiple confirmations
        if num_coins == 0:
            # 1. EMA > SMA (trend direction)
            # 2. ADX > threshold (strong trend exists, filters AAPL whipsaws)
            # 3. ROC > threshold (momentum positive, filters weak moves)
            # 4. OBV rising (volume confirms price action, filters fakeouts)
            # 5. Cooldown period respected
            if (c_ema > c_sma and 
                c_adx > adx_threshold and 
                c_roc > roc_threshold and
                (i == 0 or c_obv > obv[i-1]) and
                i > last_trade + wait_buy):
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