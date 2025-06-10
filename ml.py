import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from levels import get_all_levels
from fib_levels import get_fib_levels
import random
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import zscore
from tabulate import tabulate
import os
import argparse
import joblib

# === Parameter ===
SYMBOL = "SPXUSDT"
LIVEDATA_LEN = 1000
TRAIN_N = 2000
PUFFER = 0.02

MODEL_PATH = os.environ.get("MODEL_PATH", "elliott_model.joblib")
DATASET_PATH = os.environ.get("DATASET_PATH", "elliott_dataset.joblib")
CONFIDENCE_THRESHOLD = 0.3

FEATURES_BASE = [
    "returns",
    "range",
    "body",
    "ma_diff",
    "vol_ratio",
    "fibo_level",
    "wave_len_ratio",
    "rsi_z",
    "macd",
    "macd_signal",
    "stoch_k",
    "stoch_d",
    "obv",
    "atr",
    "kvo",
    "kvo_signal",
    "cmf",
    "high_z",
    "low_z",
    "vol_z",
    "ema_ratio",
    "bb_width",
    "roc_10",
    "corr_close_vol_10",
    "slope_5",
    "trend_len",
    "vol_atr_ratio",
    "rsi_4h",
    "close_4h",
    "vol_4h",
    "pattern_confidence",
    "pos_in_pattern",
    "prev_wave_code",
    "next_wave_code",
    "level_dist",
    "fib_dist_1d",
    "fib_near_1d",
    "fib_dist_1w",
    "fib_near_1w",
    "wave_fib_dist",
    "wave_fib_near",
]

def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def save_dataset(df, path):
    joblib.dump(df, path)


def load_dataset(path):
    return joblib.load(path)

PATTERN_PROJ_FACTORS = {
    "TRIANGLE": 1.0,
    "ZIGZAG": 0.9,
    "DOUBLE_ZIGZAG": 1.1,
    "FLAT": 0.8,
    "RUNNING_FLAT": 0.7,
    "EXPANDED_FLAT": 1.2,
    "TREND_REVERSAL": 1.0,
    "FALSE_BREAKOUT": 0.8,
    "GAP_EXTENSION": 1.2,
    "WXY": 1.0,
    "WXYXZ": 1.0,
    "WXYXZY": 1.0
}
# Mapping der erwarteten Folgewelle gemäß Elliott-Theorie. Manche Pattern
# können in mehrere Richtungen aufgelöst werden, weshalb hier Listen verwendet
# werden. "Abschluss" signalisiert das Ende einer Zyklusphase.
SPECIALPATTERN_NEXTWAVE = {
    "1": "2",
    "2": "3",
    "3": "4",
    "4": "5",
    "5": "A",
    "A": "B",
    "B": "C",
    "C": "1",
    "FLAT": ["3", "5", "C"],
    "EXPANDED_FLAT": ["3", "5", "C"],
    "ZIGZAG": ["B", "3"],
    "DOUBLE_ZIGZAG": ["C", "5"],
    "TRIANGLE": ["5", "C"],
    "WXY": ["Z", "Abschluss"],
    "WXYXZ": ["Abschluss"]
}
LABEL_MAP = {
    "1": "Impulswelle 1",
    "2": "Korrekturwelle 2",
    "3": "Impulswelle 3",
    "4": "Korrekturwelle 4",
    "5": "Impulswelle 5",
    "A": "Korrekturwelle A",
    "B": "Korrekturwelle B",
    "C": "Korrekturwelle C",
    "TRIANGLE": "Triangle (Seitwärts)",
    "ZIGZAG": "ZigZag (Korrektur)",
    "DOUBLE_ZIGZAG": "Double ZigZag",
    "FLAT": "Flat (Seitwärts)",
    "RUNNING_FLAT": "Running Flat",
    "EXPANDED_FLAT": "Expanded Flat",
    "TREND_REVERSAL": "Trend Reversal",
    "FALSE_BREAKOUT": "False Breakout",
    "GAP_EXTENSION": "Gap Extension",
    "W": "W",
    "X": "Zwischenwelle",
    "Y": "Y",
    "Z": "Z",
    "WXY": "Double Three",
    "WXYXZ": "Triple Three",
    "WXYXZY": "Complex Triple Three",
    "N": "Kein Muster",
    "INVALID_WAVE": "Ungültig"
}

def bold(x):
    """Return text without ANSI color codes."""
    return str(x)

def blue(x):
    return str(x)

def red(x):
    return str(x)

def green(x):
    return str(x)

def yellow(x):
    return str(x)

# === Indikator-Berechnungen ===
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta > 0, 0, -delta)
    gain_series = pd.Series(gain, index=series.index)
    loss_series = pd.Series(loss, index=series.index)
    avg_gain = gain_series.rolling(window=period, min_periods=period).mean()
    avg_loss = loss_series.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def calc_stoch_kd(df, k=14, d=3):
    low_min = df['low'].rolling(window=k).min()
    high_max = df['high'].rolling(window=k).max()
    stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-6)
    stoch_d = stoch_k.rolling(window=d).mean()
    return stoch_k, stoch_d

def calc_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        change = np.sign(df['close'].iloc[i] - df['close'].iloc[i-1])
        obv.append(obv[-1] + df['volume'].iloc[i] * change)
    return pd.Series(obv, index=df.index)

def calc_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr

def calc_klinger(df, fast=34, slow=55, signal=13):
    dm = df['high'] - df['low']
    cm = df['close'] - df['open']
    vf = np.abs((dm + cm) * df['volume'])
    ema_fast = vf.ewm(span=fast, adjust=False).mean()
    ema_slow = vf.ewm(span=slow, adjust=False).mean()
    kvo = ema_fast - ema_slow
    kvo_signal = kvo.ewm(span=signal, adjust=False).mean()
    return kvo, kvo_signal

def calc_cmf(df, period=20):
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) * df['volume']
    mfv_sum = mfv.rolling(window=period).sum()
    vol_sum = df['volume'].rolling(window=period).sum()
    return mfv_sum / (vol_sum + 1e-8)

def calc_slope(series, window=5):
    """Rolling slope of the given series using linear regression."""
    idx = np.arange(window)
    slopes = []
    for i in range(len(series)):
        if i < window - 1:
            slopes.append(np.nan)
            continue
        y = series.iloc[i-window+1:i+1]
        m, _ = np.polyfit(idx, y, 1)
        slopes.append(m)
    return pd.Series(slopes, index=series.index)

def calc_trend_length(series):
    """Length of consecutive positive/negative returns."""
    direction = np.sign(series)
    groups = (direction != direction.shift()).cumsum()
    return direction.groupby(groups).cumcount() + 1

def calc_pattern_confidence(series, window=10):
    """Estimate cleanliness of a price series (0-1)."""
    smoothed = series.rolling(window).mean()
    deviation = (series - smoothed).abs().rolling(window).mean()
    amplitude = series.rolling(window).max() - series.rolling(window).min()
    score = 1 - deviation / (amplitude + 1e-8)
    return score.clip(lower=0, upper=1).bfill()

# === Synthetische Muster & Generatoren ===
def validate_impulse_elliott(df):
    def wave_len(w):
        d = df[df['wave'] == w]
        return abs(d['close'].iloc[-1] - d['close'].iloc[0]) if len(d) > 1 else 0
    w1_len = wave_len('1')
    w3_len = wave_len('3')
    w5_len = wave_len('5')
    if w3_len < w1_len or w3_len < w5_len:
        df['wave'] = 'INVALID_WAVE'
        return df
    w1_max = df[df['wave']=='1']['close'].max()
    w4_min = df[df['wave']=='4']['close'].min()
    if w4_min < w1_max:
        df['wave'] = 'INVALID_WAVE'
        return df
    w1_start = df[df['wave']=='1']['close'].iloc[0]
    w2_min = df[df['wave']=='2']['close'].min()
    if w2_min < w1_start * 1.01:
        df['wave'] = 'INVALID_WAVE'
        return df
    w3_max = df[df['wave']=='3']['close'].max()
    w5_max = df[df['wave']=='5']['close'].max()
    if w5_max < w3_max:
        df['wave'] = 'TRUNCATED_5'
    return df

def subwaves(length, amp, noise, pattern):
    if pattern == 'impulse':
        segs = [length//5]*5
        vals = []
        price = amp
        for i in range(5):
            if i%2==0:
                segment = price + np.cumsum(np.abs(np.random.normal(amp/segs[0], noise, segs[i])))
            else:
                segment = price - np.cumsum(np.abs(np.random.normal(amp/segs[0], noise, segs[i])))
            vals.extend(segment)
            price = segment[-1]
        return np.array(vals)
    else:
        return amp + np.cumsum(np.random.normal(amp/length, noise, length))

def synthetic_elliott_wave_rulebased(lengths, amp, noise, puffer=PUFFER):
    pattern = ['1','2','3','4','5','A','B','C']
    prices, labels = [], []
    price = amp
    wave1_high = None
    for i, (l, w) in enumerate(zip(lengths, pattern)):
        if w in ['1','3','5'] and l >= 15:
            seg = subwaves(l, price, noise, 'impulse')
            segment = seg
            price = seg[-1]
        elif w == '2':
            tentative = price - np.cumsum(np.abs(np.random.normal(amp/l, noise, l)))
            max_level = wave1_high * (1 + puffer) if wave1_high else price
            segment = np.minimum(tentative, max_level)
            price = segment[-1]
        elif w == '4':
            segment = price - np.cumsum(np.abs(np.random.normal(amp/l, noise, l)))
            price = segment[-1]
        else:
            segment = price - np.cumsum(np.abs(np.random.normal(amp/l, noise, l)))
            price = segment[-1]
        prices.extend(segment)
        labels.extend([w]*l)
        if w == '1': wave1_high = segment[-1]
    n = min(len(prices), len(labels))
    df = pd.DataFrame({'close': prices[:n], 'wave': labels[:n]})
    df['open'] = df['close'].shift(1).fillna(df['close'][0])
    df['high'] = np.maximum(df['open'], df['close']) + np.random.uniform(0,1,len(df))
    df['low'] = np.minimum(df['open'], df['close']) - np.random.uniform(0,1,len(df))
    df['volume'] = np.random.uniform(100,1000,len(df))
    df = validate_impulse_elliott(df)
    return df.reset_index(drop=True)

def synthetic_triangle_pattern(length=40, amp=100, noise=3):
    base = amp
    step = amp*0.04
    a = base + np.cumsum(np.random.normal(step, noise, length//5))
    b = a[-1] - np.cumsum(np.abs(np.random.normal(step*0.8, noise, length//5)))
    c = b[-1] + np.cumsum(np.random.normal(step*0.7, noise, length//5))
    d = c[-1] - np.cumsum(np.abs(np.random.normal(step*0.5, noise, length//5)))
    e = d[-1] + np.cumsum(np.random.normal(step*0.4, noise, length - length//5*4))
    prices = np.concatenate([a,b,c,d,e])
    labels = ['TRIANGLE']*len(prices)
    n = min(len(prices), len(labels))
    df = pd.DataFrame({'close': prices[:n], 'wave': labels[:n]})
    df['open'] = df['close'].shift(1).fillna(df['close'][0])
    df['high'] = np.maximum(df['open'], df['close']) + np.random.uniform(0,1,len(df))
    df['low'] = np.minimum(df['open'], df['close']) - np.random.uniform(0,1,len(df))
    df['volume'] = np.random.uniform(100,1000,len(df))
    return df

def synthetic_zigzag_pattern(length=30, amp=100, noise=3):
    len1 = length//2
    len2 = length//4
    len3 = length - len1 - len2
    a = amp + np.cumsum(np.random.normal(amp/len1, noise, len1))
    b = a[-1] - np.cumsum(np.abs(np.random.normal(amp/len2, noise, len2)))
    c = b[-1] + np.cumsum(np.random.normal(amp/len3, noise, len3))
    prices = np.concatenate([a,b,c])
    labels = ['ZIGZAG']*len(prices)
    n = min(len(prices), len(labels))
    df = pd.DataFrame({'close': prices[:n], 'wave': labels[:n]})
    df['open'] = df['close'].shift(1).fillna(df['close'][0])
    df['high'] = np.maximum(df['open'], df['close']) + np.random.uniform(0,1,len(df))
    df['low'] = np.minimum(df['open'], df['close']) - np.random.uniform(0,1,len(df))
    df['volume'] = np.random.uniform(100,1000,len(df))
    return df

def synthetic_flat_pattern(length=30, amp=100, noise=3):
    len1 = length//3
    len2 = length//3
    len3 = length - len1 - len2
    a = amp + np.cumsum(np.random.normal(amp/len1, noise, len1))
    b = a[-1] + np.cumsum(np.random.normal(amp/len2, noise, len2))
    c = b[-1] - np.cumsum(np.abs(np.random.normal(amp/len3, noise, len3)))
    prices = np.concatenate([a,b,c])
    labels = ['FLAT']*len(prices)
    n = min(len(prices), len(labels))
    df = pd.DataFrame({'close': prices[:n], 'wave': labels[:n]})
    df['open'] = df['close'].shift(1).fillna(df['close'][0])
    df['high'] = np.maximum(df['open'], df['close']) + np.random.uniform(0,1,len(df))
    df['low'] = np.minimum(df['open'], df['close']) - np.random.uniform(0,1,len(df))
    df['volume'] = np.random.uniform(100,1000,len(df))
    return df

def synthetic_double_zigzag_pattern(length=50, amp=100, noise=3):
    zz1 = synthetic_zigzag_pattern(length//2, amp, noise)['close']
    zz2 = synthetic_zigzag_pattern(length//2, amp*0.9, noise)['close']
    prices = np.concatenate([zz1, zz2])
    labels = ['DOUBLE_ZIGZAG']*len(prices)
    df = pd.DataFrame({'close': prices, 'wave': labels})
    df['open'] = df['close'].shift(1).fillna(df['close'][0])
    df['high'] = np.maximum(df['open'], df['close']) + np.random.uniform(0,1,len(df))
    df['low'] = np.minimum(df['open'], df['close']) - np.random.uniform(0,1,len(df))
    df['volume'] = np.random.uniform(100,1000,len(df))
    return df

def synthetic_running_flat_pattern(length=30, amp=100, noise=3):
    a = amp + np.cumsum(np.random.normal(amp/length, noise, length//3))
    b = a[-1] + np.cumsum(np.random.normal(amp/length, noise, length//3))
    c = b[-1] + np.cumsum(np.random.normal(amp/length, noise, length - length//3*2))
    prices = np.concatenate([a,b,c])
    labels = ['RUNNING_FLAT']*len(prices)
    df = pd.DataFrame({'close': prices, 'wave': labels})
    df['open'] = df['close'].shift(1).fillna(df['close'][0])
    df['high'] = np.maximum(df['open'], df['close']) + np.random.uniform(0,1,len(df))
    df['low'] = np.minimum(df['open'], df['close']) - np.random.uniform(0,1,len(df))
    df['volume'] = np.random.uniform(100,1000,len(df))
    return df

def synthetic_expanded_flat_pattern(length=30, amp=100, noise=3):
    a = amp + np.cumsum(np.random.normal(amp/length, noise, length//3))
    b = a[-1] + np.cumsum(np.random.normal(amp/length*2, noise, length//3))
    c = b[-1] - np.cumsum(np.abs(np.random.normal(amp/length, noise, length - length//3*2)))
    prices = np.concatenate([a,b,c])
    labels = ['EXPANDED_FLAT']*len(prices)
    df = pd.DataFrame({'close': prices, 'wave': labels})
    df['open'] = df['close'].shift(1).fillna(df['close'][0])
    df['high'] = np.maximum(df['open'], df['close']) + np.random.uniform(0,1,len(df))
    df['low'] = np.minimum(df['open'], df['close']) - np.random.uniform(0,1,len(df))
    df['volume'] = np.random.uniform(100,1000,len(df))
    return df

def synthetic_trend_reversal_pattern(length=40, amp=100, noise=3, gap_chance=0.1):
    up_len = length // 2
    down_len = length - up_len
    up = amp + np.cumsum(np.abs(np.random.normal(amp/up_len, noise, up_len)))
    gap = np.random.uniform(amp*0.05, amp*0.15) if np.random.rand() < gap_chance else 0
    start_down = up[-1] + gap
    down = synthetic_zigzag_pattern(down_len, start_down, noise)['close']
    prices = np.concatenate([up, down])
    labels = ['TREND_REVERSAL'] * len(prices)
    df = pd.DataFrame({'close': prices, 'wave': labels})
    df['open'] = df['close'].shift(1).fillna(df['close'][0])
    df['high'] = np.maximum(df['open'], df['close']) + np.random.uniform(0,1,len(df))
    df['low'] = np.minimum(df['open'], df['close']) - np.random.uniform(0,1,len(df))
    df['volume'] = np.random.uniform(100,1000,len(df))
    return df

def synthetic_false_breakout_pattern(length=40, amp=100, noise=3, gap_chance=0.1):
    base_len = length // 3
    breakout_len = length // 4
    end_len = length - base_len - breakout_len
    base = amp + np.cumsum(np.random.normal(0, noise, base_len))
    direction = 1 if np.random.rand() < 0.5 else -1
    breakout = base[-1] + direction * np.cumsum(np.abs(np.random.normal(amp/breakout_len, noise, breakout_len)))
    if np.random.rand() < gap_chance:
        breakout[0] = base[-1] + direction * np.random.uniform(amp*0.05, amp*0.1)
    ret = breakout[-1] - direction * np.cumsum(np.abs(np.random.normal(amp/end_len, noise, end_len)))
    prices = np.concatenate([base, breakout, ret])
    labels = ['FALSE_BREAKOUT'] * len(prices)
    df = pd.DataFrame({'close': prices, 'wave': labels})
    df['open'] = df['close'].shift(1).fillna(df['close'][0])
    df['high'] = np.maximum(df['open'], df['close']) + np.random.uniform(0,1,len(df))
    df['low'] = np.minimum(df['open'], df['close']) - np.random.uniform(0,1,len(df))
    df['volume'] = np.random.uniform(100,1000,len(df))
    return df

def synthetic_gap_extension_pattern(length=40, amp=100, noise=3):
    l1 = length // 3
    l2 = length // 3
    l3 = length - l1 - l2
    first = amp + np.cumsum(np.random.normal(amp/l1, noise, l1))
    gap = np.random.uniform(-amp*0.15, amp*0.15)
    mid_start = first[-1] + gap
    mid_df = synthetic_triangle_pattern(l2, amp, noise)
    mid = mid_df['close'].to_numpy()
    mid = mid - mid[0] + mid_start
    ext = mid[-1] + np.cumsum(np.random.normal(amp/l3, noise, l3))
    prices = np.concatenate([first, mid, ext])
    labels = ['GAP_EXTENSION'] * len(prices)
    df = pd.DataFrame({'close': prices, 'wave': labels})
    df['open'] = df['close'].shift(1).fillna(df['close'][0])
    df['high'] = np.maximum(df['open'], df['close']) + np.random.uniform(0,1,len(df))
    df['low'] = np.minimum(df['open'], df['close']) - np.random.uniform(0,1,len(df))
    df['volume'] = np.random.uniform(100,1000,len(df))
    return df

def synthetic_wxy_pattern(length=60, amp=100, noise=3):
    lw = length // 3
    lx = length // 6
    ly = length - lw - lx
    w = amp + np.cumsum(np.random.normal(amp/lw, noise, lw))
    x = w[-1] - np.cumsum(np.abs(np.random.normal(amp/lx, noise, lx)))
    y = x[-1] + np.cumsum(np.random.normal(amp/ly, noise, ly))
    prices = np.concatenate([w, x, y])
    labels = ['W']*lw + ['X']*lx + ['Y']*ly
    df = pd.DataFrame({'close': prices, 'wave': labels})
    df['open'] = df['close'].shift(1).fillna(df['close'][0])
    df['high'] = np.maximum(df['open'], df['close']) + np.random.uniform(0,1,len(df))
    df['low'] = np.minimum(df['open'], df['close']) - np.random.uniform(0,1,len(df))
    df['volume'] = np.random.uniform(100,1000,len(df))
    return df

def synthetic_wxyxz_pattern(length=80, amp=100, noise=3):
    seg = length // 5
    w = amp + np.cumsum(np.random.normal(amp/seg*2, noise, seg*2))
    x1 = w[-1] - np.cumsum(np.abs(np.random.normal(amp/seg, noise, seg)))
    y = x1[-1] + np.cumsum(np.random.normal(amp/seg, noise, seg))
    x2 = y[-1] - np.cumsum(np.abs(np.random.normal(amp/(length-seg*4), noise, seg//2)))
    z = x2[-1] + np.cumsum(np.random.normal(amp/(length-seg*4), noise, length - seg*4 - seg//2))
    prices = np.concatenate([w, x1, y, x2, z])
    labels = ['W']*(seg*2) + ['X']*seg + ['Y']*seg + ['X']*(seg//2) + ['Z']*(length - seg*4 - seg//2)
    df = pd.DataFrame({'close': prices, 'wave': labels})
    df['open'] = df['close'].shift(1).fillna(df['close'][0])
    df['high'] = np.maximum(df['open'], df['close']) + np.random.uniform(0,1,len(df))
    df['low'] = np.minimum(df['open'], df['close']) - np.random.uniform(0,1,len(df))
    df['volume'] = np.random.uniform(100,1000,len(df))
    return df

def synthetic_wxyxzy_pattern(length=100, amp=100, noise=3):
    seg = length // 6
    w = amp + np.cumsum(np.random.normal(amp/seg, noise, seg))
    x1 = w[-1] - np.cumsum(np.abs(np.random.normal(amp/seg, noise, seg)))
    y1 = x1[-1] + np.cumsum(np.random.normal(amp/seg, noise, seg))
    x2 = y1[-1] - np.cumsum(np.abs(np.random.normal(amp/seg, noise, seg)))
    z = x2[-1] + np.cumsum(np.random.normal(amp/seg, noise, seg))
    y2 = z[-1] - np.cumsum(np.abs(np.random.normal(amp/seg, noise, length - seg*5)))
    prices = np.concatenate([w, x1, y1, x2, z, y2])
    labels = ['W']*seg + ['X']*seg + ['Y']*seg + ['X']*seg + ['Z']*seg + ['Y']*(length - seg*5)
    df = pd.DataFrame({'close': prices, 'wave': labels})
    df['open'] = df['close'].shift(1).fillna(df['close'][0])
    df['high'] = np.maximum(df['open'], df['close']) + np.random.uniform(0,1,len(df))
    df['low'] = np.minimum(df['open'], df['close']) - np.random.uniform(0,1,len(df))
    df['volume'] = np.random.uniform(100,1000,len(df))
    return df

def generate_negative_samples(length=100, amp=100, noise=20, outlier_chance=0.1, gap_chance=0.1):
    prices = amp + np.cumsum(np.random.normal(0, noise, length))
    for i in range(1, length):
        if np.random.rand() < gap_chance:
            gap = np.random.normal(0, noise*5)
            prices[i:] += gap
    mask = np.random.rand(length) < outlier_chance
    prices[mask] += np.random.normal(0, noise*10, mask.sum())
    labels = ['N']*len(prices)
    n = min(len(prices), len(labels))
    df = pd.DataFrame({'close': prices[:n], 'wave': labels[:n]})
    df['open'] = df['close'].shift(1).fillna(df['close'][0])
    df['high'] = np.maximum(df['open'], df['close']) + np.random.uniform(0,1,len(df))
    df['low'] = np.minimum(df['open'], df['close']) - np.random.uniform(0,1,len(df))
    df['volume'] = np.random.uniform(100,1000,len(df))
    return df

def _simple_wave_segment(label, start_price, length=8, noise=2):
    """Create a short price segment for a single wave label."""
    direction = 1 if np.random.rand() < 0.5 else -1
    step = max(start_price * 0.02, 1)
    prices = start_price + direction * np.cumsum(
        np.abs(np.random.normal(step, noise, length))
    )
    df = pd.DataFrame({"close": prices, "wave": [label] * length})
    df["open"] = df["close"].shift(1).fillna(df["close"][0])
    df["high"] = np.maximum(df["open"], df["close"]) + np.random.uniform(0, 1, len(df))
    df["low"] = np.minimum(df["open"], df["close"]) - np.random.uniform(0, 1, len(df))
    df["volume"] = np.random.uniform(100, 1000, len(df))
    return df


def generate_rulebased_synthetic_with_patterns(
    n=1000, negative_ratio=0.15, pattern_ratio=0.35
):
    num_pattern = int(n * pattern_ratio)
    num_neg = int(n * negative_ratio)
    num_pos = n - num_pattern - num_neg
    dfs = []
    for _ in range(num_pos):
        lengths = np.random.randint(12, 50, size=8)
        amp = np.random.uniform(60, 150)
        noise = np.random.uniform(1, 4)
        df = synthetic_elliott_wave_rulebased(lengths, amp, noise)
        dfs.append(df)
    pattern_funcs = [
        (synthetic_triangle_pattern, "TRIANGLE"),
        (synthetic_zigzag_pattern, "ZIGZAG"),
        (synthetic_flat_pattern, "FLAT"),
        (synthetic_double_zigzag_pattern, "DOUBLE_ZIGZAG"),
        (synthetic_running_flat_pattern, "RUNNING_FLAT"),
        (synthetic_expanded_flat_pattern, "EXPANDED_FLAT"),
        (synthetic_trend_reversal_pattern, "TREND_REVERSAL"),
        (synthetic_false_breakout_pattern, "FALSE_BREAKOUT"),
        (synthetic_gap_extension_pattern, "GAP_EXTENSION"),
        (synthetic_wxy_pattern, "WXY"),
        (synthetic_wxyxz_pattern, "WXYXZ"),
        (synthetic_wxyxzy_pattern, "WXYXZY"),
    ]
    for _ in range(num_pattern):
        segs = []
        for _ in range(np.random.randint(1, 3)):
            f, pname = random.choice(pattern_funcs)
            length = np.random.randint(32, 70)
            amp = np.random.uniform(60, 140)
            noise = np.random.uniform(1, 3.5)
            d = f(length=length, amp=amp, noise=noise)
            if np.random.rand() < 0.3:
                cut = np.random.randint(len(d) // 2, len(d))
                d = d.iloc[:cut]
            segs.append(d)
            if pname in SPECIALPATTERN_NEXTWAVE:
                nxt = SPECIALPATTERN_NEXTWAVE[pname]
                nxt = nxt if isinstance(nxt, list) else [nxt]
                start = d["close"].iloc[-1]
                for wave in nxt:
                    if wave == "Abschluss":
                        continue
                    follow_len = np.random.randint(5, 12)
                    follow = _simple_wave_segment(wave, start, length=follow_len)
                    segs.append(follow)
                    start = follow["close"].iloc[-1]
        df = pd.concat(segs, ignore_index=True)
        dfs.append(df)
    for _ in range(num_neg):
        length = np.random.randint(80, 250)
        amp = np.random.uniform(50, 120)
        noise = np.random.uniform(12, 35)
        df = generate_negative_samples(length=length, amp=amp, noise=noise,
                                       outlier_chance=0.2, gap_chance=0.2)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    return combined

def synthetic_subwaves(df, minlen=4, maxlen=9):
    df = df.copy()
    subwave_id = np.zeros(len(df), dtype=int)
    i = 0
    wave_counter = 1
    while i < len(df):
        sublen = np.random.randint(minlen, maxlen)
        if i + sublen > len(df):
            sublen = len(df) - i
        subwave_id[i:i+sublen] = np.arange(1, sublen+1)
        i += sublen
        wave_counter += 1
    df['subwave'] = subwave_id[:len(df)]
    return df

def compute_wave_fibs(df, label_col='wave_pred', buffer=PUFFER):
    """Add fibonacci levels per wave segment to ``df``.

    Parameters
    ----------
    df : DataFrame
        OHLCV data with wave labels/predictions.
    label_col : str, optional
        Column containing wave identifiers.
    buffer : float, optional
        Tolerance for the ``wave_fib_near`` flag.
    """
    if label_col not in df.columns:
        return df

    fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618]

    df = df.copy()
    df['wave_fib_dist'] = np.nan
    df['wave_fib_near'] = np.nan

    start = 0
    cur = df[label_col].iloc[0]
    for i in range(1, len(df) + 1):
        if i == len(df) or df[label_col].iloc[i] != cur:
            end = i - 1
            start_price = df['close'].iloc[start]
            end_price = df['close'].iloc[end]
            if end_price >= start_price:
                diff = end_price - start_price
                fibs = {
                    0.0: end_price,
                    0.236: end_price - diff * 0.236,
                    0.382: end_price - diff * 0.382,
                    0.5: end_price - diff * 0.5,
                    0.618: end_price - diff * 0.618,
                    0.786: end_price - diff * 0.786,
                    1.0: start_price,
                    1.618: start_price - diff * 0.618,
                    2.618: start_price - diff * 1.618,
                }
            else:
                diff = start_price - end_price
                fibs = {
                    0.0: end_price,
                    0.236: end_price + diff * 0.236,
                    0.382: end_price + diff * 0.382,
                    0.5: end_price + diff * 0.5,
                    0.618: end_price + diff * 0.618,
                    0.786: end_price + diff * 0.786,
                    1.0: start_price,
                    1.618: start_price + diff * 0.618,
                    2.618: start_price + diff * 1.618,
                }

            idx_slice = df.index[start:end + 1]
            closes = df['close'].loc[idx_slice]
            dists = pd.DataFrame({k: (closes - v).abs() for k, v in fibs.items()})
            min_dist = dists.min(axis=1)
            df.loc[idx_slice, 'wave_fib_dist'] = min_dist / closes
            df.loc[idx_slice, 'wave_fib_near'] = (min_dist / closes <= buffer).astype(int)

            start = i
            if i < len(df):
                cur = df[label_col].iloc[i]

    return df

# === Feature Engineering mit 4H-Integration ===
def make_features(df, df_4h=None, levels=None, fib_levels=None):
    df = df.copy()
    df['returns'] = df['close'].pct_change().fillna(0)
    df['range'] = (df['high'] - df['low']) / df['close']
    df['body'] = (df['close'] - df['open']).abs() / df['close']
    df['ma_fast'] = df['close'].rolling(5).mean().bfill()
    df['ma_slow'] = df['close'].rolling(34).mean().bfill()
    df['ma_diff'] = df['ma_fast'] - df['ma_slow']
    df['vol_ratio'] = df['volume'] / (df['volume'].rolling(5).mean().bfill() + 1e-6)
    df['fibo_level'] = (df['close'] - df['close'].rolling(21).min()) / (df['close'].rolling(21).max() - df['close'].rolling(21).min() + 1e-6)
    window = 20
    counts = df.groupby(df['wave']).cumcount() + 1 if 'wave' in df else 1
    df['wave_len_ratio'] = counts / (window)
    df['rsi'] = calc_rsi(df['close'], period=14).bfill()
    df['rsi_z'] = zscore(df['rsi'].fillna(50))
    df['macd'], df['macd_signal'] = calc_macd(df['close'])
    df['stoch_k'], df['stoch_d'] = calc_stoch_kd(df)
    df['obv'] = calc_obv(df)
    df['atr'] = calc_atr(df)
    df['vol_atr_ratio'] = df['volume'] / (df['atr'] + 1e-8)
    df['kvo'], df['kvo_signal'] = calc_klinger(df)
    df['cmf'] = calc_cmf(df)
    df['high_z'] = zscore(df['high'])
    df['low_z'] = zscore(df['low'])
    df['vol_z'] = zscore(df['volume'])
    df['ema_fast'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=26, adjust=False).mean()
    df['ema_ratio'] = df['ema_fast'] / (df['ema_slow'] + 1e-8)
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_width'] = (bb_std * 4) / (bb_mid + 1e-8)
    df['roc_10'] = df['close'].pct_change(10).fillna(0)
    df['corr_close_vol_10'] = df['close'].rolling(10).corr(df['volume']).fillna(0)
    df['slope_5'] = calc_slope(df['close'], window=5).bfill()
    df['trend_len'] = calc_trend_length(df['returns']).bfill()
    df['pattern_confidence'] = calc_pattern_confidence(df['close'])
    if 'subwave' not in df.columns:
        df = synthetic_subwaves(df)
    if 'wave' in df.columns:
        idx = df.groupby('wave').cumcount()
        total = df.groupby('wave')['wave'].transform('count').replace(0,1)
        df['pos_in_pattern'] = idx / total
        df['prev_wave'] = df['wave'].shift(1).fillna('START')
        df['next_wave'] = df['wave'].shift(-1).fillna('END')
        df['prev_wave_code'] = pd.Categorical(df['prev_wave']).codes
        df['next_wave_code'] = pd.Categorical(df['next_wave']).codes
    else:
        df['pos_in_pattern'] = 0.0
        df['prev_wave_code'] = 0
        df['next_wave_code'] = 0
    if levels is not None:
        level_prices = np.array([lvl['price'] for lvl in levels])
        if len(level_prices):
            df['level_dist'] = df['close'].apply(lambda x: np.min(np.abs(level_prices - x)))
        else:
            df['level_dist'] = 0.0
    else:
        df['level_dist'] = 0.0

    if fib_levels is not None:
        for tf, fibs in fib_levels.items():
            prices = np.array([f['price'] for f in fibs])
            if len(prices):
                dist = df['close'].apply(lambda x: np.min(np.abs(prices - x)) / x)
                df[f'fib_dist_{tf.lower()}'] = dist
                df[f'fib_near_{tf.lower()}'] = (dist <= 0.003).astype(int)
            else:
                df[f'fib_dist_{tf.lower()}'] = 1.0
                df[f'fib_near_{tf.lower()}'] = 0
    else:
        for tf in ['1d', '1w']:
            df[f'fib_dist_{tf}'] = 1.0
            df[f'fib_near_{tf}'] = 0
    if df_4h is not None:
        df_4h['rsi_4h'] = calc_rsi(df_4h['close'], period=14).bfill()
        df['rsi_4h'] = np.interp(df.index, np.linspace(0, len(df)-1, len(df_4h)), df_4h['rsi_4h'])
        df['close_4h'] = np.interp(df.index, np.linspace(0, len(df)-1, len(df_4h)), df_4h['close'])
        df['vol_4h'] = np.interp(df.index, np.linspace(0, len(df)-1, len(df_4h)), df_4h['volume'])

    # Local fib levels for actual or predicted waves
    if 'wave' in df.columns:
        df = compute_wave_fibs(df, 'wave', buffer=PUFFER)
    elif 'wave_pred' in df.columns:
        df = compute_wave_fibs(df, 'wave_pred', buffer=PUFFER)

    df = df.dropna().reset_index(drop=True)
    return df

# === OHLCV-Import Bitget API ===
def fetch_bitget_ohlcv_auto(symbol, interval="1H", target_len=1000, page_limit=1000, log=False):
    if isinstance(interval, (list, tuple, set)):
        result = {}
        for iv in interval:
            result[iv] = fetch_bitget_ohlcv_auto(symbol, iv, target_len=target_len,
                                                page_limit=page_limit, log=log)
        return result

    all_data = []
    end_time = None
    total = 0
    while total < target_len:
        url = f"https://api.bitget.com/api/v2/mix/market/candles?symbol={symbol}&granularity={interval}&productType=USDT-FUTURES&limit={page_limit}"
        if end_time:
            url += f"&endTime={end_time}"
        r = requests.get(url)
        if r.status_code != 200:
            raise Exception("Bitget API-Fehler: " + r.text)
        data = r.json()["data"]
        if not data or len(data) < 2:
            break
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "baseVolume", "quoteVolume"])
        all_data.append(df)
        min_timestamp = df["timestamp"].astype(int).min()
        end_time = min_timestamp - 1
        total = sum(len(x) for x in all_data)
    if not all_data:
        raise Exception("Keine Candles empfangen!")
    combined = pd.concat(all_data, ignore_index=True)
    combined[["open", "high", "low", "close", "baseVolume", "quoteVolume"]] = combined[["open", "high", "low", "close", "baseVolume", "quoteVolume"]].astype(float)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"].astype(int), unit="ms")
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined["volume"] = combined["baseVolume"]
    if len(combined) > target_len:
        combined = combined.iloc[-target_len:].reset_index(drop=True)
    if log:
        print(green(f"Loaded {len(combined)} bars for {symbol} {interval}"))
    return combined[["timestamp","open","high","low","close","volume"]]

# === ML Training ===
def train_ml(skip_grid_search=False, max_samples=None, model_type="rf", feature_selection=False):
    if os.path.exists(DATASET_PATH):
        print(yellow("Lade vorhandenes Dataset..."))
        df = load_dataset(DATASET_PATH)
    else:
        print(bold("Erzeuge und kombiniere alle Muster (Impuls, Korrektur, Triangle, usw.)..."))
        df = generate_rulebased_synthetic_with_patterns(
            n=TRAIN_N, negative_ratio=0.15, pattern_ratio=0.35)
        save_dataset(df, DATASET_PATH)

    print(f"{blue('Gesamtanzahl Datenpunkte:')} {len(df)}")
    df.index = pd.date_range("2020-01-01", periods=len(df), freq="1h")
    levels = get_all_levels(df, ["2H", "4H", "1D", "1W"])
    df = make_features(df, levels=levels)
    df_valid = df[~df['wave'].isin(['X','INVALID_WAVE'])].reset_index(drop=True)

    if max_samples is not None and len(df_valid) > max_samples:
        groups = df_valid.groupby('wave')
        per_class = max_samples // len(groups)
        sampled = [g.sample(min(len(g), per_class), random_state=42) for _, g in groups]
        df_valid = pd.concat(sampled).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"{blue('Nach Filterung gültige Datenpunkte:')} {len(df_valid)}")
    features = [f for f in FEATURES_BASE if f in df_valid.columns]
    X = df_valid[features]
    y = df_valid['wave'].astype(str)

    tscv = TimeSeriesSplit(n_splits=5)

    def create_model_params(mtype):
        if mtype == 'rf':
            base = RandomForestClassifier(random_state=42)
            grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'class_weight': [None, 'balanced'],
            }
            defaults = {'n_estimators': 200, 'max_depth': None, 'class_weight': None}
        elif mtype == 'xgb':
            base = XGBClassifier(random_state=42, verbosity=0, eval_metric='logloss')
            grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.05, 0.1],
            }
            defaults = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1}
        elif mtype == 'lgbm':
            base = LGBMClassifier(random_state=42)
            grid = {
                'n_estimators': [100, 200],
                'max_depth': [-1, 10],
                'learning_rate': [0.05, 0.1],
            }
            defaults = {'n_estimators': 200, 'max_depth': -1, 'learning_rate': 0.1}
        elif mtype == 'voting':
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            xgb = XGBClassifier(n_estimators=200, random_state=42, verbosity=0, eval_metric='logloss')
            lgb = LGBMClassifier(n_estimators=200, random_state=42)
            base = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgb)], voting='soft')
            grid = None
            defaults = None
        else:
            raise ValueError(f'Unbekannter Modeltyp: {mtype}')
        return base, grid, defaults

    base_model, param_grid, defaults = create_model_params(model_type)

    if not skip_grid_search and param_grid:
        grid = GridSearchCV(base_model, param_grid, cv=tscv, n_jobs=-1)
        print(yellow('Starte GridSearch zur Hyperparameteroptimierung...'))
        grid.fit(X, y)
        best_params = grid.best_params_
        print(green(f"Beste CV-Genauigkeit: {grid.best_score_:.3f} | Beste Parameter: {best_params}"))
    else:
        best_params = defaults or {}

    def instantiate(mtype, params):
        if mtype == 'rf':
            return RandomForestClassifier(**params, random_state=42)
        elif mtype == 'xgb':
            return XGBClassifier(**params, random_state=42, verbosity=0, eval_metric='logloss')
        elif mtype == 'lgbm':
            return LGBMClassifier(**params, random_state=42)
        elif mtype == 'voting':
            rf = RandomForestClassifier(n_estimators=params.get('n_estimators', 200), random_state=42)
            xgb = XGBClassifier(n_estimators=params.get('n_estimators', 200), random_state=42, verbosity=0, eval_metric='logloss')
            lgb = LGBMClassifier(n_estimators=params.get('n_estimators', 200), random_state=42)
            return VotingClassifier(estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgb)], voting='soft')

    model = instantiate(model_type, best_params)

    if feature_selection:
        selector = RFECV(model, step=1, cv=tscv, n_jobs=-1)
        print(yellow('Führe Feature Selection mittels RFECV durch...'))
        selector.fit(X, y)
        model = selector.estimator_
        features = [f for f, s in zip(features, selector.support_) if s]
        X = df_valid[features]

    print(yellow('Trainiere finales Modell...'))
    model.fit(X, y)

    cv_scores = cross_val_score(model, X, y, cv=tscv, n_jobs=-1)
    print(green(f"Durchschnittliche CV-Genauigkeit: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"))

    if hasattr(model, 'feature_importances_'):
        importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    else:
        importance = pd.Series(np.zeros(len(features)), index=features)

    save_model({'model': model, 'features': features}, MODEL_PATH)
    return model, features, importance

# === Fibo-Bereiche für Entry/TP/SL (automatisch, pro Welle) ===
def get_fibo_zones(df, wave, side="LONG"):
    idx = df[df['wave_pred'] == wave].index
    if len(idx) < 2:
        return None, None
    start, end = idx[0], idx[-1]
    price_start = df['close'].iloc[start]
    price_end = df['close'].iloc[end]
    diff = price_end - price_start

    # Determine actual direction by diff sign
    direction = "LONG" if diff >= 0 else "SHORT"
    diff = abs(diff)

    if direction == "LONG":
        e1 = price_start + diff * 0.236
        e2 = price_start + diff * 0.382
        t1 = price_start + diff * 0.618
        t2 = price_start + diff * 1.0
    else:
        e1 = price_start - diff * 0.236
        e2 = price_start - diff * 0.382
        t1 = price_start - diff * 0.618
        t2 = price_start - diff * 1.0

    # Expand zones using PUFFER
    entry_zone = [min(e1, e2) * (1 - PUFFER), max(e1, e2) * (1 + PUFFER)]
    tp_zone = [min(t1, t2) * (1 - PUFFER), max(t1, t2) * (1 + PUFFER)]
    return entry_zone, tp_zone

# === Zielprojektionen / Pattern-Targets ===
def pattern_target(df_features, current_pattern, last_complete_close):
    idx_pattern = df_features[df_features["wave_pred"] == current_pattern].index
    if not len(idx_pattern):
        return last_complete_close * 1.01
    close_last = df_features["close"].iloc[idx_pattern[-1]]
    if current_pattern in ["WXY", "WXYXZ", "WXYXZY"]:
        w_idx = _latest_segment_indices(df_features, "W")
        if len(w_idx) > 1:
            w_len = df_features["close"].iloc[w_idx[-1]] - df_features["close"].iloc[w_idx[0]]
            return close_last + 0.8 * w_len
    factor = PATTERN_PROJ_FACTORS.get(current_pattern, 1.0)
    high = df_features["high"].iloc[idx_pattern].max()
    low  = df_features["low"].iloc[idx_pattern].min()
    breakout_size = high - low
    return close_last + factor * breakout_size

def _latest_segment_indices(df, wave_label):
    """Return indices of the most recent contiguous segment for a wave."""
    idxs = df[df["wave_pred"] == wave_label].index.to_numpy()
    if len(idxs) == 0:
        return np.array([], dtype=int)
    splits = np.split(idxs, np.where(np.diff(idxs) != 1)[0] + 1)
    return splits[-1]

def get_next_wave(current_wave):
    """Return a list of expected follow-up waves for ``current_wave``."""
    wave = str(current_wave)
    if wave in SPECIALPATTERN_NEXTWAVE:
        nxt = SPECIALPATTERN_NEXTWAVE[wave]
        if isinstance(nxt, list):
            return [n for n in nxt if n and n != "Abschluss"]
        return [nxt] if nxt and nxt != "Abschluss" else []
    order = ['1', '2', '3', '4', '5', 'A', 'B', 'C']
    try:
        idx = order.index(wave)
        if idx + 1 < len(order):
            return [order[idx + 1]]
    except Exception:
        pass
    return []

def elliott_target(
    df_features,
    current_wave,
    last_complete_close,
    levels=None,
    level_tolerance=0.001,
):
    """Return projected target price for an Elliott wave segment.

    Parameters
    ----------
    df_features : DataFrame
        Feature dataframe containing OHLC data.
    current_wave : str
        Wave label to project the target for.
    last_complete_close : float
        Last closing price of the previously completed candle.
    levels : list[dict], optional
        Pre-computed price levels used to snap the target.
    level_tolerance : float, optional
        Tolerance as a fraction of ``wave_start_price`` used when snapping to
        nearby levels. If the nearest level differs from ``wave_start_price`` by
        less than this tolerance, the original target is kept.
    """
    idx = lambda wave: _latest_segment_indices(df_features, wave)
    start_idx = idx(current_wave)
    wave_start_price = (
        df_features["close"].iloc[start_idx[0]] if len(start_idx) > 0 else last_complete_close
    )
    last_wave_close = (
        df_features["close"].iloc[start_idx[-1]] if len(start_idx) > 0 else last_complete_close
    )

    target = None
    if str(current_wave) in PATTERN_PROJ_FACTORS:
        target = pattern_target(df_features, str(current_wave), last_complete_close)
    elif str(current_wave) == "1":
        target = wave_start_price * 1.02
    elif str(current_wave) == "2":
        idx1 = idx("1")
        if len(idx1) > 0:
            high1 = df_features["close"].iloc[idx1].max()
            low2 = wave_start_price
            retracement = 0.5
            target = high1 - retracement * (high1 - low2)
            if target >= low2:
                target = None
        else:
            target = last_complete_close * 0.98
    elif str(current_wave) == "3":
        idx1 = idx("1")
        if len(idx1) > 1 and len(start_idx) > 0:
            w1len = df_features["close"].iloc[idx1[-1]] - df_features["close"].iloc[idx1[0]]
            target = wave_start_price + 1.618 * w1len
        else:
            target = last_complete_close * 1.05
    elif str(current_wave) == "4":
        idx3 = idx("3")
        if len(idx3) > 1 and len(start_idx) > 0:
            w3len = df_features["close"].iloc[idx3[-1]] - df_features["close"].iloc[idx3[0]]
            target = wave_start_price - 0.382 * abs(w3len)
            if target >= wave_start_price:
                target = None
        else:
            target = last_complete_close * 0.98
    elif str(current_wave) == "5":
        idx1 = idx("1")
        idx3 = idx("3")
        if len(idx1) > 1 and len(start_idx) > 0:
            w1len = df_features["close"].iloc[idx1[-1]] - df_features["close"].iloc[idx1[0]]
            if len(idx3) > 1:
                w3len = df_features["close"].iloc[idx3[-1]] - df_features["close"].iloc[idx3[0]]
                proj_len = max(w1len, w3len) * 0.618
            else:
                proj_len = w1len
            target = wave_start_price + proj_len
        else:
            target = last_complete_close * 1.02
    elif str(current_wave) == "A":
        idx5 = idx("5")
        if len(idx5) > 1 and len(start_idx) > 0:
            w5len = df_features["close"].iloc[idx5[-1]] - df_features["close"].iloc[idx5[0]]
            target = wave_start_price - w5len
            if target >= wave_start_price:
                target = None
        else:
            target = last_complete_close * 0.98
    elif str(current_wave) == "B":
        idxa = idx("A")
        if len(idxa) > 1 and len(start_idx) > 0:
            wa = df_features["close"].iloc[idxa[-1]] - df_features["close"].iloc[idxa[0]]
            target = wave_start_price + 0.618 * abs(wa)
        else:
            target = last_complete_close * 1.01
    elif str(current_wave) == "C":
        idxa = idx("A")
        if len(idxa) > 1 and len(start_idx) > 0:
            wa = df_features["close"].iloc[idxa[-1]] - df_features["close"].iloc[idxa[0]]
            target = wave_start_price - 1.0 * abs(wa)
            if target >= wave_start_price:
                target = None
        else:
            target = last_complete_close * 0.98
    elif str(current_wave) in ["W", "X", "Y", "Z"]:
        idxw = idx("W")
        if len(idxw) > 1 and len(start_idx) > 0:
            wlen = df_features["close"].iloc[idxw[-1]] - df_features["close"].iloc[idxw[0]]
            if str(current_wave) in ["Y", "Z"]:
                target = wave_start_price + 0.618 * wlen
            else:
                target = wave_start_price - 0.382 * abs(wlen)
        else:
            target = last_complete_close
    else:
        target = last_complete_close * 1.01

    if target is not None and levels:
        prices = [lvl["price"] for lvl in levels]
        if prices:
            nearest = min(prices, key=lambda p: abs(p - target))
            # Only snap to the nearest level when it differs from the
            # wave's start by more than the tolerance. This avoids
            # returning a target equal to the wave's start unless the
            # level is meaningfully different.
            tolerance = abs(wave_start_price) * level_tolerance
            if abs(nearest - wave_start_price) >= tolerance:
                target = nearest

    return target, wave_start_price, last_wave_close

def suggest_trade(
    df,
    current_wave,
    target,
    last_close,
    entry_zone=None,
    tp_zone=None,
    risk=0.01,
    sl_puffer=0.005,
    levels=None,
    probability=None,
):
    """Suggest trade parameters for the given wave.

    When ``probability`` is provided, the risk (and thus resulting SL/TP and
    position size) is scaled by this value to reflect the confidence of the
    prediction.
    """

    entry = last_close
    if target is None:
        print(red("Kein gültiges Kursziel berechnet."))
        return None, None, None, None

    adj_risk = risk * probability if probability is not None else risk

    if target > entry:
        direction = "LONG"
        tp = target
        sl = entry * (1 - adj_risk - sl_puffer)
    else:
        direction = "SHORT"
        tp = target
        sl = entry * (1 + adj_risk + sl_puffer)

    local_cols = [c for c in df.columns if c.startswith(f"fib_{current_wave}_")]
    local_prices = []
    if local_cols:
        local_prices = df.iloc[-1][local_cols].dropna().tolist()

    if levels or local_prices:
        prices = []
        if levels:
            prices.extend([lvl["price"] for lvl in levels])
        prices.extend(local_prices)
        if prices:
            tp = min(prices, key=lambda p: abs(p - tp))
            entry = min(prices, key=lambda p: abs(p - entry))

    size = 1000 * adj_risk / abs(entry - sl) if abs(entry - sl) > 0 else 0
    print(
        bold(
            f"\n[TRADE-SETUP] {direction} | Entry: {entry:.4f} | SL: {sl:.4f} | TP: {tp:.4f} | PosSize: {size:.1f}x"
        )
    )
    # Entry- und TP-Zonen werden nicht ausgegeben
    return direction, sl, tp, entry

# === Auswertung der vorhergesagten Wellenstruktur ===
def evaluate_wave_structure(df, label_col="wave_pred"):
    """Prüfe Reihenfolge und Start-/Endpunkte der vorhergesagten Wellen."""
    if label_col not in df.columns:
        print(red("Keine Vorhersagen zur Auswertung gefunden."))
        return False

    segments = []
    current = df[label_col].iloc[0]
    start = 0
    for i in range(1, len(df)):
        if df[label_col].iloc[i] != current:
            segments.append((current, start, i - 1))
            current = df[label_col].iloc[i]
            start = i
    segments.append((current, start, len(df) - 1))

    table = [(lbl, s, e) for lbl, s, e in segments]
    print(bold("\nWave Segments (Prediction):"))
    print(tabulate(table, headers=["Wave", "Start", "End"]))

    order = ["1", "2", "3", "4", "5", "A", "B", "C"]
    last_idx = -1
    for lbl, s, e in segments:
        if lbl in order:
            idx = order.index(lbl)
            if idx < last_idx:
                print(red(f"Ungültige Reihenfolge bei Welle {lbl} (Start {s})"))
                return False
            last_idx = idx
    print(green("Wellenreihenfolge scheint konsistent."))
    return True

def run_pattern_analysis(df, model, features, levels=None):
    df_feat = make_features(df, levels=levels)
    preds = model.predict(df_feat[features])
    proba = model.predict_proba(df_feat[features])
    classes = [str(c) for c in model.classes_]
    results = []
    df_feat["wave_pred"] = preds
    df_feat = compute_wave_fibs(df_feat, 'wave_pred', buffer=PUFFER)
    for i, row in df_feat.iterrows():
        wave = str(row["wave_pred"])
        prob = proba[i, classes.index(wave)] if wave in classes else 0.0
        target, start_price, _ = elliott_target(df_feat.iloc[: i + 1], wave, row["close"], levels=levels)
        validity = "valid" if target is not None else "invalid"
        trade = None
        if validity == "valid":
            direction, sl, tp, entry = suggest_trade(
                df_feat.iloc[: i + 1],
                wave,
                target,
                row["close"],
                levels=levels,
                probability=prob,
            )
            trade = {"entry": entry, "sl": sl, "tp": tp, "probability": prob}
        results.append({
            "pattern_type": wave,
            "wave_id": wave,
            "target_projection": {"level": None, "price": target, "pattern": wave, "probability": prob},
            "tradesetup": trade,
            "validity": validity,
        })
    return results

# === Prediction Smoothing ===
def smooth_predictions(pred, window=5):
    """Return smoothed labels using majority voting."""
    half = window // 2
    smoothed = []
    for i in range(len(pred)):
        start = max(0, i - half)
        end = min(len(pred), i + half + 1)
        window_vals = pred[start:end]
        mode = pd.Series(window_vals).mode()
        smoothed.append(mode.iloc[0] if len(mode) else pred[i])
    return np.array(smoothed)

# === Hauptfunktion für Analyse & Grafik ===
def run_ml_on_bitget(model, features, importance, symbol=SYMBOL, interval="1H", livedata_len=LIVEDATA_LEN, extra_intervals=None):
    df_1h = fetch_bitget_ohlcv_auto(symbol, interval, target_len=livedata_len, page_limit=1000, log=True)
    df_4h = fetch_bitget_ohlcv_auto(symbol, "4H", target_len=800, page_limit=1000, log=True)
    df_2h = df_1d = df_1w = None
    if extra_intervals:
        if "2H" in extra_intervals:
            df_2h = fetch_bitget_ohlcv_auto(symbol, "2H", target_len=800, page_limit=1000, log=True)
        if "1D" in extra_intervals or "D" in extra_intervals:
            df_1d = fetch_bitget_ohlcv_auto(symbol, "1D", target_len=400, page_limit=1000, log=True)
        if "1W" in extra_intervals or "W" in extra_intervals:
            df_1w = fetch_bitget_ohlcv_auto(symbol, "1W", target_len=200, page_limit=1000, log=True)
    print(bold("\n==== BITGET DATA ===="))
    parts = [f"{len(df_1h)} (1H)"]
    if df_2h is not None:
        parts.append(f"{len(df_2h)} (2H)")
    parts.append(f"{len(df_4h)} (4H)")
    if df_1d is not None:
        parts.append(f"{len(df_1d)} (1D)")
    if df_1w is not None:
        parts.append(f"{len(df_1w)} (1W)")
    print(f"Symbol: {symbol} | Intervall: {interval} | Bars: " + " / ".join(parts))
    print(f"Letzter Timestamp: {df_1h['timestamp'].iloc[-1]}")
    last_complete_close = df_1h["close"].iloc[-2]

    levels_base = df_1h.copy()
    levels_base["timestamp"] = pd.to_datetime(levels_base["timestamp"])
    levels_base = levels_base.set_index("timestamp")
    levels = get_all_levels(levels_base, ["2H", "4H", "1D", "1W"])

    fib_levels = {}
    if df_1d is not None:
        tmp = df_1d.copy()
        tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])
        tmp = tmp.set_index("timestamp")
        fib_levels["1D"] = get_fib_levels(tmp, "1D")
    if df_1w is not None:
        tmp = df_1w.copy()
        tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])
        tmp = tmp.set_index("timestamp")
        fib_levels["1W"] = get_fib_levels(tmp, "1W")
    for fl in fib_levels.values():
        levels.extend(fl)

    df_features = make_features(df_1h, df_4h, levels=levels, fib_levels=fib_levels)

    # Sicherstellen, dass alle vom Modell erwarteten Features vorhanden sind
    missing = [f for f in features if f not in df_features.columns]
    for m in missing:
        df_features[m] = 0.0

    pred_raw = model.predict(df_features[features])
    pred = smooth_predictions(pred_raw)
    pred_proba = model.predict_proba(df_features[features])
    classes = [str(c) for c in model.classes_]
    df_features["wave_pred_raw"] = pred_raw
    df_features["wave_pred"] = pred
    df_features = compute_wave_fibs(df_features, 'wave_pred', buffer=PUFFER)
    proba_row = pred_proba[-1]
    current_wave = df_features["wave_pred"].iloc[-1]
    main_wave = str(pred[-1])
    main_wave_idx = classes.index(main_wave)
    main_wave_prob = proba_row[main_wave_idx]
    alt_wave, alt_prob = None, None
    if current_wave in ["N", "INVALID_WAVE", "X"]:
        valid_waves = [c for c in classes if c not in ["N", "X", "INVALID_WAVE"]]
        valid_indices = [i for i, c in enumerate(classes) if c in valid_waves]
        valid_probs = [proba_row[i] for i in valid_indices]
        if valid_probs:
            alt_idx = valid_indices[np.argmax(valid_probs)]
            alt_wave = classes[alt_idx]
            alt_prob = proba_row[alt_idx]
            print(yellow("\nAktuelle erkannte Welle: Noise/Invalid/X"))
            print(yellow(f"Alternativer Vorschlag (höchste ML-Prob.): {alt_wave} ({alt_prob*100:.1f}%) → {LABEL_MAP.get(alt_wave,alt_wave)}"))
            current_wave = alt_wave
            print(green(f"Aktuelle erkannte Welle (endgültig): {current_wave} ({alt_prob*100:.1f}%) - {LABEL_MAP.get(current_wave,current_wave)}"))
        else:
            print(red("Keine valide Welle mit hinreichender Wahrscheinlichkeit erkannt!"))
    else:
        print(green(f"Aktuelle erkannte Welle (endgültig): {current_wave} ({main_wave_prob*100:.1f}%) - {LABEL_MAP.get(current_wave,current_wave)}"))

    prob_sorted_idx = np.argsort(proba_row)[::-1]
    print(bold("\nTop-3 ML-Wahrscheinlichkeiten:"))
    for i in range(3):
        idx = prob_sorted_idx[i]
        label = classes[idx]
        print(f"  {LABEL_MAP.get(label,label)}: {proba_row[idx]*100:.1f}%")

    pattern_conf = df_features['pattern_confidence'].iloc[-1]
    if pattern_conf < CONFIDENCE_THRESHOLD:
        print(red(f"Muster-Konfidenz zu niedrig ({pattern_conf:.2f} < {CONFIDENCE_THRESHOLD}) - Trade-Setup übersprungen."))
        return

    # ==== Fibo-Zonen berechnen (nicht ausgeben) ====
    entry_zone, tp_zone = get_fibo_zones(
        df_features,
        current_wave,
        side="LONG" if current_wave in ['1', '3', '5', 'B', 'ZIGZAG', 'TRIANGLE'] else "SHORT",
    )

    # ==== Zielprojektionen ====
    target, wave_start_price, last_wave_close = elliott_target(
        df_features, current_wave, last_complete_close, levels=levels
    )
    # Bestmögliche Folgewelle anhand der ML-Wahrscheinlichkeit wählen
    next_wave_candidates = get_next_wave(current_wave)
    next_wave = None
    next_wave_start = None
    next_target = None
    next_wave_prob = -1
    for cand in next_wave_candidates:
        cand_prob = proba_row[classes.index(cand)] if cand in classes else 0.0
        if cand_prob > next_wave_prob:
            next_wave = cand
            next_wave_prob = cand_prob
    if next_wave:
        next_target, next_wave_start, _ = elliott_target(
            df_features,
            next_wave,
            target if target is not None else last_wave_close,
            levels=levels,
        )

    # Gleiches Vorgehen für die übernächste Welle
    next_next_wave_candidates = get_next_wave(next_wave) if next_wave else []
    next_next_wave = None
    next_next_wave_start = None
    next_next_target = None
    next_next_prob = -1
    for cand in next_next_wave_candidates:
        cand_prob = proba_row[classes.index(cand)] if cand in classes else 0.0
        if cand_prob > next_next_prob:
            next_next_wave = cand
            next_next_prob = cand_prob
    if next_next_wave:
        next_next_target, next_next_wave_start, _ = elliott_target(
            df_features,
            next_next_wave,
            next_target if next_target is not None else (
                target if target is not None else last_wave_close
            ),
            levels=levels,
        )

    print(bold("\n==== ZIELPROJEKTIONEN ===="))
    print(
        f"Aktuelle Welle: {current_wave} ({LABEL_MAP.get(current_wave,current_wave)}) | Start: {wave_start_price:.4f} | Ziel: {target:.4f}"
    )
    if next_wave and next_target:
        print(
            f"Nächste erwartete Welle: {next_wave} ({LABEL_MAP.get(next_wave,next_wave)}) | Start: {next_wave_start:.4f} | Ziel: {next_target:.4f}"
        )
    if next_next_wave and next_next_target:
        print(
            f"Darauffolgende erwartete Welle: {next_next_wave} ({LABEL_MAP.get(next_next_wave,next_next_wave)}) | Start: {next_next_wave_start:.4f} | Ziel: {next_next_target:.4f}"
        )

    # === Breakout Zone (letzte Patternrange) ===
    idx_pattern = df_features[df_features["wave_pred"] == current_wave].index
    breakout_zone = None
    if len(idx_pattern) > 1:
        high = df_features["high"].iloc[idx_pattern].max()
        low  = df_features["low"].iloc[idx_pattern].min()
        breakout_zone = (low, high)

    # === Trade-Setup Output ===
    trade_wave = str(next_wave) if next_target else str(current_wave)
    trade_target = next_target if next_target else target
    if trade_wave in classes:
        trade_wave_idx = classes.index(trade_wave)
        trade_prob = proba_row[trade_wave_idx]
    else:
        print(yellow(f"Unbekannte Trade-Welle {trade_wave} – Wahrscheinlichkeit auf 0 gesetzt."))
        trade_prob = 0.0

    fib_near = max(df_features.get('fib_near_1d', pd.Series([0])).iloc[-1],
                   df_features.get('fib_near_1w', pd.Series([0])).iloc[-1])
    naked_near = int(df_features['level_dist'].iloc[-1] / df_features['close'].iloc[-1] <= 0.003)
    prob_weight = (1 + 0.5 * fib_near) * (1 + 0.5 * naked_near)
    trade_prob *= prob_weight
    entry_exit_score = pattern_conf * prob_weight
    print(bold(f"Entry/Exit-Score: {entry_exit_score:.2f}"))
    direction, sl, tp, entry = suggest_trade(
        df_features,
        trade_wave,
        trade_target,
        last_complete_close,
        entry_zone,
        tp_zone,
        levels=levels,
        probability=trade_prob,
    )
    if direction is None:
        print(red("Trade-Setup konnte nicht erstellt werden."))

    evaluate_wave_structure(df_features)


    # Statistik- und Feature-Logs entfernt

    # === PRO-Level Grafik ===
    plt.figure(figsize=(17, 8))
    plt.plot(df_features["close"].values, label="Kurs", linewidth=1.3)
    colors = pd.Categorical(df_features["wave_pred"]).codes
    plt.scatter(df_features.index, df_features["close"].values, c=colors, cmap="rainbow", s=18, label="WaveClass", alpha=0.65)
    if target is not None:
        plt.axhline(target, color="black", linestyle="--", linewidth=1.5, label=f"Zielprojektion {current_wave} {target:.4f}")
    if next_wave and next_target:
        plt.axhline(next_target, color="grey", linestyle=":", linewidth=1.3, label=f"Nächste Welle {next_wave} Ziel {next_target:.4f}")
    # Breakout-Zone Highlight
    if breakout_zone:
        plt.axhspan(breakout_zone[0], breakout_zone[1], color="orange", alpha=0.19, label="Breakout-Zone")
    # Fibo Entry/TP-Zonen
    if entry_zone:
        plt.axhspan(entry_zone[0], entry_zone[1], color="green", alpha=0.15, label="Entry-Zone")
    if tp_zone:
        plt.axhspan(tp_zone[0], tp_zone[1], color="blue", alpha=0.14, label="TP-Zone")
    # Wellen/Pattern Text-Annotationen
    for wave in set(df_features["wave_pred"]):
        idxs = df_features[df_features["wave_pred"] == wave].index
        if len(idxs):
            mid = idxs[0] + (idxs[-1] - idxs[0]) // 2
            price = df_features["close"].iloc[mid]
            plt.text(mid, price, f"{LABEL_MAP.get(wave, wave)}", fontsize=8, ha='center', va='bottom', color="black", alpha=0.8)
    plt.title(f"{symbol} 1H Chart – {LABEL_MAP.get(current_wave, current_wave)}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Main ===
def main():
    global MODEL_PATH, DATASET_PATH
    parser = argparse.ArgumentParser(description="Elliott Wave ML")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Pfad zum Modell")
    parser.add_argument("--dataset-path", default=DATASET_PATH, help="Pfad zum Dataset")
    parser.add_argument("--skip-grid-search", action="store_true", help="GridSearch überspringen")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximale Anzahl Trainingssamples")
    parser.add_argument("--model", choices=["rf", "xgb", "lgbm", "voting"], default="rf", help="Modelltyp")
    parser.add_argument("--feature-selection", action="store_true", help="RFECV Feature Auswahl nutzen")
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    DATASET_PATH = args.dataset_path
    
    if os.path.exists(MODEL_PATH):
        print(yellow("Lade gespeichertes Modell..."))
        obj = load_model(MODEL_PATH)
        if isinstance(obj, dict) and 'model' in obj:
            model = obj['model']
            features = obj.get('features', FEATURES_BASE)
        else:
            model = obj
            if os.path.exists(DATASET_PATH):
                df_tmp = load_dataset(DATASET_PATH)
                df_tmp = make_features(df_tmp)
                df_tmp = df_tmp[~df_tmp['wave'].isin(['X','INVALID_WAVE'])].reset_index(drop=True)
                features = [f for f in FEATURES_BASE if f in df_tmp.columns]
            else:
                features = FEATURES_BASE
        if hasattr(model, 'feature_importances_'):
            importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        else:
            importance = pd.Series(index=features, dtype=float)
    else:
        model, features, importance = train_ml(skip_grid_search=args.skip_grid_search,
                                               max_samples=args.max_samples,
                                               model_type=args.model,
                                               feature_selection=args.feature_selection)
    run_ml_on_bitget(model, features, importance, symbol=SYMBOL, interval="1H", livedata_len=LIVEDATA_LEN,
                     extra_intervals=["2H", "4H", "1D", "1W"])

if __name__ == "__main__":
    main()
