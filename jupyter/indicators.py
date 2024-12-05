import pandas as pd
from numba import jit
import numpy as np
import talib as ta
from typing import Union






jit(nopython=True)
def rma(arr, length):
    x = pd.Series(arr)
    x.iloc[:length] = x.rolling(length).mean().iloc[:length]
    x = x.ewm(alpha=(1.0/length),adjust=False).mean()
    return np.nan_to_num(x.values)


jit(nopython=True)
def barssince(cond):
#     condition = cond==False
#     cum_condition = np.invert(condition).cumsum()
    cum_condition = (cond==False).cumsum()
    result = ( cum_condition - np.maximum.accumulate((cond)*cum_condition ))
    return result

jit(nopython=True)
def barssincetime(st: Union[float, int], ed: np.array, tf: str):
    tfs = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240, '8h': 480, '12h': 720, '1d': 1440}
    return (ed-st)/60000/tfs[tf]


jit(nopython=True)
def shift(arr, length: int=1):
    return np.append(np.zeros(length), arr[:-length])


jit(nopython=True)
def crossover(x, y):
    return (np.where((x>y)&(np.append(0, x[:-1])<np.append(0, y[:-1])), 1, 0)^np.where((x<y)&(np.append(0, x[:-1])>np.append(0, y[:-1])), 1, 0))
    
    
jit(nopython=True)
def crossover_up(x, y):
    return np.where((x>y)&(np.append(0, x[:-1])<np.append(0, y[:-1])), 1, 0)
    
    
jit(nopython=True)
def crossover_down(x, y):
    return np.where((x<y)&(np.append(0, x[:-1])>np.append(0, y[:-1])), 1, 0)


jit(nopython=True)
def down(x):
    return np.less(x, np.append(0, x[:-1])).astype(float)


jit(nopython=True)
def up(x):
    return np.greater(x, np.append(0, x[:-1])).astype(float)


jit(nopython=True)
def updown(x):
    return np.where((np.append(0, x[:-2])<np.append(0, x[:-1]))&(np.append(0, x[:-1])>x), 1, 0).astype(float)


jit(nopython=True)
def downup(x):
    return np.where((np.append(0, x[:-2])>np.append(0, x[:-1]))&(np.append(0, x[:-1])<x), 1, 0).astype(float)


jit(nopython=True)
def rolling(a, window):
    a = np.concatenate((np.zeros(window-1), a))
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


jit(nopython=True)
def stochrsi(x, rsi_period: int=14, rsimin_period:int=14, rsimax_period:int=14, k_period: int=3, d_period: int=3):
    rsimin_period = rsi_period if rsimin_period is None else rsimin_period
    rsimax_period = rsi_period if rsimax_period is None else rsimax_period
    rsi = ta.RSI(x, timeperiod=rsi_period)
    rsi_high = ta.MAX(rsi, timeperiod=rsimax_period)
    rsi_low = ta.MIN(rsi, timeperiod=rsimin_period)
    stoch = (rsi - rsi_low) / (rsi_high - rsi_low)
    stoch = ta.SMA(stoch, k_period if k_period else 3)
    if d_period>1: 
        stoch = ta.SMA(stoch, d_period if d_period else 3)    
    return stoch


jit(nopython=True)
def cci(high, low, close, period: int=20, constant: float=.015):
    tp = (high + low + close)/3
    ma = ta.SMA(tp, period)
    cci = (tp-ma) / (constant * ta.SMA(np.absolute(tp-ma), period))
    return cci



jit(nopython=True)
def rsi_divergence(close, len_fast: int=5, len_slow: int=14, params: list=None, **kwargs):
    up_fast = rma(np.append(0, close[1:]-close[:-1]).clip(0), len_fast)
    down_fast = rma((-1*np.append(0, close[1:]-close[:-1])).clip(0), len_fast)
    rsi_fast = np.where(down_fast==0, 100, np.where(up_fast==0, 0, 100-(100/(1+up_fast/down_fast))))
    up_slow = rma(np.append(0, close[1:]-close[:-1]).clip(0), len_slow)
    down_slow = rma((-1*np.append(0, close[1:]-close[:-1])).clip(0), len_slow)
    rsi_slow = np.where(down_slow==0, 100, np.where(up_slow==0, 0, 100-(100/(1+up_slow/down_slow))))
    return rsi_fast - rsi_slow



jit(nopython=True)
def renko(c: np.array, box_size: float) -> np.array:
    last_price = c[0]
    closes = [last_price]
    for c_ in c[1:]:
        if c_ > last_price+last_price*box_size:
            last_price+=last_price*box_size
        elif c_ < last_price-last_price*box_size:
            last_price-=last_price*box_size
        closes.append(last_price)
    return np.array(closes)
