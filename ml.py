import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from scipy.stats import zscore
from tabulate import tabulate
import os
import argparse
import joblib

# === Parameter ===
SYMBOL = "SPXUSDT"
LIVEDATA_LEN = 1000
TRAIN_N = 70000
PUFFER = 0.02

MODEL_PATH = os.environ.get("MODEL_PATH", "elliott_model.joblib")
DATASET_PATH = os.environ.get("DATASET_PATH", "elliott_dataset.joblib")

FEATURES_BASE = [
    "returns", "range", "body", "ma_diff", "vol_ratio", "fibo_level",
    "wave_len_ratio", "rsi_z", "macd", "macd_signal", "stoch_k", "stoch_d", "obv",
    "atr", "kvo", "kvo_signal", "cmf", "high_z", "low_z", "vol_z",
    "ema_ratio", "bb_width", "roc_10", "roll_corr_10", "slope_5",
    "trend_len", "vol_atr_ratio",
    "rsi_4h", "close_4h", "vol_4h",
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
    "EXPANDED_FLAT": 1.2
}
SPECIALPATTERN_NEXTWAVE = {
    "TRIANGLE": "5",
    "ZIGZAG": "1",
    "DOUBLE_ZIGZAG": "1",
    "FLAT": "1",
    "RUNNING_FLAT": "1",
    "EXPANDED_FLAT": "1"
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
    "N": "Kein Muster",
    "X": "Zwischenwelle",
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
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean()
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

def calc_rolling_corr(series1, series2, window=10):
    """Rolling correlation between two series."""
    return series1.rolling(window).corr(series2)

def calc_slope(series, window=5):
    """Linear regression slope over a rolling window."""
    idx = np.arange(window)
    def _slope(x):
        if len(x) < window:
            return np.nan
        sl, _ = np.polyfit(idx, x, 1)
        return sl
    return series.rolling(window).apply(_slope, raw=True)

def calc_trend_length(series):
    """Length of current trend measured by consecutive price moves."""
    diff = series.diff().fillna(0)
    direction = np.sign(diff)
    length = [1]
    for i in range(1, len(direction)):
        if direction.iloc[i] == 0:
            length.append(length[-1])
        elif direction.iloc[i] == direction.iloc[i-1]:
            length.append(length[-1] + 1)
        else:
            length.append(1)
    return pd.Series(length, index=series.index)

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

def generate_negative_samples(length=100, amp=100, noise=20):
    prices = amp + np.cumsum(np.random.normal(0, noise, length))
    labels = ['N']*len(prices)
    n = min(len(prices), len(labels))
    df = pd.DataFrame({'close': prices[:n], 'wave': labels[:n]})
    df['open'] = df['close'].shift(1).fillna(df['close'][0])
    df['high'] = np.maximum(df['open'], df['close']) + np.random.uniform(0,1,len(df))
    df['low'] = np.minimum(df['open'], df['close']) - np.random.uniform(0,1,len(df))
    df['volume'] = np.random.uniform(100,1000,len(df))
    return df

def generate_rulebased_synthetic_with_patterns(n=1000, negative_ratio=0.15, pattern_ratio=0.35):
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
        synthetic_triangle_pattern,
        synthetic_zigzag_pattern,
        synthetic_flat_pattern,
        synthetic_double_zigzag_pattern,
        synthetic_running_flat_pattern,
        synthetic_expanded_flat_pattern,
    ]
    for _ in range(num_pattern):
        f = np.random.choice(pattern_funcs)
        length = np.random.randint(32, 70)
        amp = np.random.uniform(60, 140)
        noise = np.random.uniform(1, 3.5)
        df = f(length=length, amp=amp, noise=noise)
        dfs.append(df)
    for _ in range(num_neg):
        length = np.random.randint(80, 250)
        amp = np.random.uniform(50, 120)
        noise = np.random.uniform(12, 35)
        df = generate_negative_samples(length=length, amp=amp, noise=noise)
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

# === Feature Engineering mit 4H-Integration ===
def make_features(df, df_4h=None):
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
    df['roll_corr_10'] = calc_rolling_corr(df['close'], df['volume']).bfill()
    df['slope_5'] = calc_slope(df['close']).bfill()
    df['trend_len'] = calc_trend_length(df['close'])
    if 'subwave' not in df.columns:
        df = synthetic_subwaves(df)
    if df_4h is not None:
        df_4h['rsi_4h'] = calc_rsi(df_4h['close'], period=14).bfill()
        df['rsi_4h'] = np.interp(df.index, np.linspace(0, len(df)-1, len(df_4h)), df_4h['rsi_4h'])
        df['close_4h'] = np.interp(df.index, np.linspace(0, len(df)-1, len(df_4h)), df_4h['close'])
        df['vol_4h'] = np.interp(df.index, np.linspace(0, len(df)-1, len(df_4h)), df_4h['volume'])
    df = df.dropna().reset_index(drop=True)
    return df

# === OHLCV-Import Bitget API ===
def fetch_bitget_ohlcv_auto(symbol, interval="1H", target_len=1000, page_limit=1000):
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
    return combined[["timestamp","open","high","low","close","volume"]]

# === ML Training ===
def train_ml(skip_grid_search=False, max_samples=None):
    if os.path.exists(DATASET_PATH):
        print(yellow("Lade vorhandenes Dataset..."))
        df = load_dataset(DATASET_PATH)
    else:
        print(bold("Erzeuge und kombiniere alle Muster (Impuls, Korrektur, Triangle, usw.)..."))
        df = generate_rulebased_synthetic_with_patterns(
            n=TRAIN_N, negative_ratio=0.15, pattern_ratio=0.35)
        save_dataset(df, DATASET_PATH)
    print(f"{blue('Gesamtanzahl Datenpunkte:')} {len(df)}")
    df = make_features(df)
    df_valid = df[~df['wave'].isin(['X','INVALID_WAVE'])].reset_index(drop=True)
    if max_samples is not None and len(df_valid) > max_samples:
        groups = df_valid.groupby('wave')
        per_class = max_samples // len(groups)
        sampled = [g.sample(min(len(g), per_class), random_state=42) for _, g in groups]
        df_valid = pd.concat(sampled).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"{blue('Nach Filterung gültige Datenpunkte:')} {len(df_valid)}")
    features = [f for f in FEATURES_BASE if f in df_valid.columns]
    X = df_valid[features]
    y = df_valid["wave"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=43)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "class_weight": [None, "balanced"],
    }

    base_model = RandomForestClassifier(random_state=42)
    if skip_grid_search:
        best_params = {"n_estimators": 200, "max_depth": None, "class_weight": None}
    else:
        tscv = TimeSeriesSplit(n_splits=3)
        grid = GridSearchCV(base_model, param_grid, cv=tscv, n_jobs=-1)
        print(yellow("Starte GridSearch zur Hyperparameteroptimierung..."))
        grid.fit(X_train, y_train)
        print(green(f"Beste CV-Genauigkeit: {grid.best_score_:.3f} | Beste Parameter: {grid.best_params_}"))
        best_params = grid.best_params_

    rf = RandomForestClassifier(**best_params, random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    model = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
    print(yellow("Trainiere finales Modell..."))
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(green(f"Training fertig. Genauigkeit (Train): {train_score:.3f} | (Test): {test_score:.3f}"))
    print(f"{blue('Wichtigste ML-Features:')}")
    importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    print(importance.head(8).round(3))
    save_model(model, MODEL_PATH)
    return model, features, importance

# === Fibo-Bereiche für Entry/TP/SL (automatisch, pro Welle) ===
def get_fibo_zones(df, wave, side="LONG"):
    idx = df[df['wave_pred'] == wave].index
    if len(idx) < 2:
        return None, None
    start, end = idx[0], idx[-1]
    price_start = df['close'].iloc[start]
    price_end   = df['close'].iloc[end]
    diff = price_end - price_start
    if side == "LONG":
        entry_zone = [price_start + diff * 0.236, price_start + diff * 0.382]
        tp_zone    = [price_start + diff * 0.618, price_start + diff * 1.0]
    else:
        entry_zone = [price_start - diff * 0.236, price_start - diff * 0.382]
        tp_zone    = [price_start - diff * 0.618, price_start - diff * 1.0]
    return entry_zone, tp_zone

# === Zielprojektionen / Pattern-Targets ===
def pattern_target(df_features, current_pattern, last_complete_close):
    idx_pattern = df_features[df_features["wave_pred"] == current_pattern].index
    if not len(idx_pattern):
        return last_complete_close * 1.01
    close_last = df_features["close"].iloc[idx_pattern[-1]]
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
    if current_wave in SPECIALPATTERN_NEXTWAVE:
        return SPECIALPATTERN_NEXTWAVE[current_wave]
    order = ['1','2','3','4','5','A','B','C']
    try:
        idx = order.index(str(current_wave))
        if idx+1 < len(order):
            return order[idx+1]
        else:
            return None
    except Exception:
        return None

def elliott_target(df_features, current_wave, last_complete_close):
    idx = lambda wave: _latest_segment_indices(df_features, wave)
    start_idx = idx(current_wave)
    wave_start_price = (
        df_features["close"].iloc[start_idx[0]] if len(start_idx) > 0 else last_complete_close
    )
    last_wave_close = (
        df_features["close"].iloc[start_idx[-1]] if len(start_idx) > 0 else last_complete_close
    )

    if str(current_wave) in PATTERN_PROJ_FACTORS:
        target = pattern_target(df_features, str(current_wave), last_complete_close)
        return target, wave_start_price, last_wave_close

    if str(current_wave) == "1":
        return wave_start_price * 1.02, wave_start_price, last_wave_close
    elif str(current_wave) == "2":
        idx1 = idx("1")
        if len(idx1) > 0:
            high1 = df_features["close"].iloc[idx1].max()
            low2 = wave_start_price
            retracement = 0.5
            target = high1 - retracement * (high1 - low2)
            if target >= low2:
                return None
            return target, wave_start_price, last_wave_close
        else:
            return last_complete_close * 0.98, wave_start_price, last_wave_close
    elif str(current_wave) == "3":
        idx1 = idx("1")
        if len(idx1) > 1 and len(start_idx) > 0:
            w1len = df_features["close"].iloc[idx1[-1]] - df_features["close"].iloc[idx1[0]]
            return wave_start_price + 1.618 * w1len, wave_start_price, last_wave_close
        else:
            return last_complete_close * 1.05, wave_start_price, last_wave_close
    elif str(current_wave) == "4":
        idx3 = idx("3")
        if len(idx3) > 1 and len(start_idx) > 0:
            w3len = df_features["close"].iloc[idx3[-1]] - df_features["close"].iloc[idx3[0]]
            target = wave_start_price - 0.382 * abs(w3len)
            if target >= wave_start_price:
                return None
            return target, wave_start_price, last_wave_close
        else:
            return last_complete_close * 0.98, wave_start_price, last_wave_close
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
            return wave_start_price + proj_len, wave_start_price, last_wave_close
        else:
            return last_complete_close * 1.02, wave_start_price, last_wave_close
    elif str(current_wave) == "A":
        idx5 = idx("5")
        if len(idx5) > 1 and len(start_idx) > 0:
            w5len = df_features["close"].iloc[idx5[-1]] - df_features["close"].iloc[idx5[0]]
            target = wave_start_price - w5len
            if target >= wave_start_price:
                return None
            return target, wave_start_price, last_wave_close
        else:
            return last_complete_close * 0.98, wave_start_price, last_wave_close
    elif str(current_wave) == "B":
        idxa = idx("A")
        if len(idxa) > 1 and len(start_idx) > 0:
            wa = df_features["close"].iloc[idxa[-1]] - df_features["close"].iloc[idxa[0]]
            return wave_start_price + 0.618 * abs(wa), wave_start_price, last_wave_close
        else:
            return last_complete_close * 1.01, wave_start_price, last_wave_close
    elif str(current_wave) == "C":
        idxa = idx("A")
        if len(idxa) > 1 and len(start_idx) > 0:
            wa = df_features["close"].iloc[idxa[-1]] - df_features["close"].iloc[idxa[0]]
            target = wave_start_price - 1.0 * abs(wa)
            if target >= wave_start_price:
                return None
            return target, wave_start_price, last_wave_close
        else:
            return last_complete_close * 0.98, wave_start_price, last_wave_close
    else:
        return last_complete_close * 1.01, wave_start_price, last_wave_close

def suggest_trade(df, current_wave, target, last_close, entry_zone=None, tp_zone=None, risk=0.01, sl_puffer=0.005):
    entry = last_close
    if target is None:
        print(red("Kein gültiges Kursziel berechnet."))
        return None, None, None, None

    if target > entry:
        direction = "LONG"
        tp = target
        sl = entry * (1 - risk - sl_puffer)
    else:
        direction = "SHORT"
        tp = target
        sl = entry * (1 + risk + sl_puffer)

    size = 1000 * risk / abs(entry - sl) if abs(entry - sl) > 0 else 0
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
def run_ml_on_bitget(model, features, importance, symbol=SYMBOL, interval="1H", livedata_len=LIVEDATA_LEN):
    df_1h = fetch_bitget_ohlcv_auto(symbol, interval, target_len=livedata_len, page_limit=1000)
    df_4h = fetch_bitget_ohlcv_auto(symbol, "4H", target_len=800, page_limit=1000)
    print(bold("\n==== BITGET DATA ===="))
    print(f"Symbol: {symbol} | Intervall: {interval} | Bars: {len(df_1h)} (1H) / {len(df_4h)} (4H)")
    print(f"Letzter Timestamp: {df_1h['timestamp'].iloc[-1]}")
    last_complete_close = df_1h["close"].iloc[-2]
    df_features = make_features(df_1h, df_4h)
    pred_raw = model.predict(df_features[features])
    pred = smooth_predictions(pred_raw)
    pred_proba = model.predict_proba(df_features[features])
    classes = model.classes_
    df_features["wave_pred_raw"] = pred_raw
    df_features["wave_pred"] = pred
    df_features["confidence"] = pred_proba.max(axis=1)
    proba_row = pred_proba[-1]
    current_wave = df_features["wave_pred"].iloc[-1]
    main_wave = pred[-1]
    main_wave_idx = list(classes).index(main_wave)
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
            confidence = alt_prob - np.partition(proba_row, -2)[-2]
            print(green(f"Aktuelle erkannte Welle (endgültig): {current_wave} ({alt_prob*100:.1f}%) - {LABEL_MAP.get(current_wave,current_wave)} | Conf: {confidence*100:.1f}%"))
        else:
            print(red("Keine valide Welle mit hinreichender Wahrscheinlichkeit erkannt!"))
    else:
        confidence = main_wave_prob - np.partition(proba_row, -2)[-2]
        print(green(f"Aktuelle erkannte Welle (endgültig): {current_wave} ({main_wave_prob*100:.1f}%) - {LABEL_MAP.get(current_wave,current_wave)} | Conf: {confidence*100:.1f}%"))

    prob_sorted_idx = np.argsort(proba_row)[::-1]
    print(bold("\nTop-3 ML-Wahrscheinlichkeiten:"))
    for i in range(3):
        idx = prob_sorted_idx[i]
        label = classes[idx]
        print(f"  {LABEL_MAP.get(label,label)}: {proba_row[idx]*100:.1f}%")

    # ==== Fibo-Zonen berechnen (nicht ausgeben) ====
    entry_zone, tp_zone = get_fibo_zones(
        df_features,
        current_wave,
        side="LONG" if current_wave in ['1', '3', '5', 'B', 'ZIGZAG', 'TRIANGLE'] else "SHORT",
    )

    # ==== Zielprojektionen ====
    target, wave_start_price, last_wave_close = elliott_target(
        df_features, current_wave, last_complete_close
    )
    next_wave = get_next_wave(current_wave)
    if next_wave:
        next_target, next_wave_start, _ = elliott_target(
            df_features,
            next_wave,
            target if target is not None else last_wave_close,
        )
    else:
        next_target = None
        next_wave_start = None
    next_next_wave = get_next_wave(next_wave) if next_wave else None
    if next_next_wave:
        next_next_target, next_next_wave_start, _ = elliott_target(
            df_features,
            next_next_wave,
            next_target if next_target is not None else (
                target if target is not None else last_wave_close
            ),
        )
    else:
        next_next_target = None
        next_next_wave_start = None

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
    trade_wave = next_wave if next_target else current_wave
    trade_target = next_target if next_target else target
    direction, sl, tp, entry = suggest_trade(
        df_features, trade_wave, trade_target, last_complete_close, entry_zone, tp_zone
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
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    DATASET_PATH = args.dataset_path
    
    if os.path.exists(MODEL_PATH):
        print(yellow("Lade gespeichertes Modell..."))
        model = load_model(MODEL_PATH)
        if os.path.exists(DATASET_PATH):
            df_tmp = load_dataset(DATASET_PATH)
            df_tmp = make_features(df_tmp)
            df_tmp = df_tmp[~df_tmp['wave'].isin(['X','INVALID_WAVE'])].reset_index(drop=True)
            features = [f for f in FEATURES_BASE if f in df_tmp.columns]
        else:
            features = FEATURES_BASE
        importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    else:
        model, features, importance = train_ml(skip_grid_search=args.skip_grid_search, max_samples=args.max_samples)
    run_ml_on_bitget(model, features, importance, symbol=SYMBOL, interval="1H", livedata_len=LIVEDATA_LEN)

if __name__ == "__main__":
    main()
