import numpy as np
from numba import njit
from strategy_helpers import *

def get_strategy() -> dict:
    return dict(
        name="ema_sma_adx_trend_v1",
        variables=["ema_period", "sma_period", "adx_period", "adx_threshold", 
                   "atr_mult", "wait_buy", "wait_sell"],
        bounds=([20, 40, 10, 15, 0.5, 10, 50],
                [60, 100, 30, 35, 3.0, 150, 200]),
        simulate=simulate,
    )

def simulate(close, high, low, volume, x):
    ema_period = max(int(x[0]), 1)
    sma_period = max(int(x[1]), 1)
    adx_period = max(int(x[2]), 1)
    atr_period = max(int(x[3]), 1)
    
    ema = ema_np(close, ema_period)
    sma = sma_np(close, sma_period)
    adx_result = adx_np(high, low, close, adx_period)
    adx, pdi, mdi = adx_result[0], adx_result[1], adx_result[2]
    atr = atr_np(high, low, close, atr_period)
    
    return _execute(close, 1_000_000.0, ema, sma, adx, pdi, mdi, atr,
                    float(x[3]), int(x[4]), int(x[5]), int(x[6]))

@njit(fastmath=True)
def _execute(close, start_cash, ema, sma, adx, pdi, mdi, atr,
             atr_mult, adx_threshold, wait_buy, wait_sell):
    cash = start_cash
    num_coins = 0
    last_trade = 0
    num_trades = 0
    peak = 0.0
    
    for i in range(len(close)):
        c_ema = ema[i]
        c_sma = sma[i]
        c_adx = adx[i]
        c_pdi = pdi[i]
        c_mdi = mdi[i]
        c_atr = atr[i]
        price = close[i]
        
        # Skip NaN values
        if np.isnan(c_ema) or np.isnan(c_sma) or np.isnan(c_adx):
            continue
        
        # Track peak for trailing stop
        if num_coins > 0 and price > peak:
            peak = price
        
        # Gentle trailing stop - only if position is up by at least 1.5x ATR
        if num_coins > 0 and peak > 0:
            min_gain = price + (1.5 * c_atr)
            stop_price = peak * (1.0 - atr_mult * c_atr / price)
            if price >= min_gain and price <= stop_price:
                cash, num_coins = sell_all(cash, num_coins, price)
                last_trade = i
                num_trades += 1
                peak = 0.0
        
        # BUY: EMA > SMA + ADX shows trend + PDI > MDI + cooldown
        # ADX > 20 ensures we're in a trending market (not choppy)
        if num_coins == 0:
            if (c_ema > c_sma and c_adx > adx_threshold and 
                c_pdi > c_mdi and i > last_trade + wait_buy):
                cash, num_coins = buy_all(cash, num_coins, price)
                last_trade = i
                num_trades += 1
                peak = price
        
        # SELL: EMA < SMA + cooldown
        elif num_coins > 0 and c_ema < c_sma and i > last_trade + wait_sell:
            cash, num_coins = sell_all(cash, num_coins, price)
            last_trade = i
            num_trades += 1
            peak = 0.0
    
    # Force-sell at end
    cash, num_coins = sell_all(cash, num_coins, close[-1])
    return cash / start_cash, num_trades