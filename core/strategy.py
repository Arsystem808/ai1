from __future__ import annotations
import pandas as pd, numpy as np
from typing import Literal, TypedDict, Optional, Dict

Horizon = Literal["short","swing","position"]
Action  = Literal["BUY","SHORT","WAIT"]

DEFAULTS = {
    "ema_fast": 20, "ema_mid": 50, "ema_slow": 200,
    "rsi_hi": 68, "rsi_lo": 32,
    "macd_bars_short": 2, "macd_bars_swing": 3, "macd_bars_pos": 3,
    "weak_tail_ratio": 0.6,
    "weak_body_ratio": 0.35,
    "short_tp1":0.6, "short_tp2":1.2, "short_sl":0.9,
    "swing_tp1":0.8, "swing_tp2":1.6, "swing_sl":1.0,
    "pos_tp1":1.0,   "pos_tp2":2.2,   "pos_sl":1.4,
}

class Signal(TypedDict):
    symbol: str
    horizon: Horizon
    action: Action
    confidence: float
    entry: float; tp1: float; tp2: float; sl: float
    key_mark: float; upper_zone: float; lower_zone: float
    pivot_P: float; R1: float; R2: float; R3: float; S1: float; S2: float; S3: float

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    r_up  = up.ewm(alpha=1/period, adjust=False).mean()
    r_dn  = dn.ewm(alpha=1/period, adjust=False).mean()
    rs = r_up / r_dn.replace(0, np.nan)
    rsi = 100 - 100/(1+rs)
    return rsi.fillna(50)

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h,l,c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = np.maximum(h-l, np.maximum((h-pc).abs(), (l-pc).abs()))
    w = min(period, max(5, len(df)//2))
    return tr.rolling(w).mean().fillna(tr.ewm(span=w, adjust=False).mean())

def _macd_hist(close: pd.Series, fast=12, slow=26, sig=9) -> pd.Series:
    ema_fast = _ema(close, fast); ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=sig, adjust=False).mean()
    return macd - signal

def _consecutive_sign(series: pd.Series) -> int:
    s = series.dropna()
    if s.empty: return 0
    last = s.iloc[-1]
    if abs(last) == 0: return 0
    sign = 1 if last > 0 else -1
    cnt = 0
    for x in reversed(s.values):
        if (x > 0 and sign > 0) or (x < 0 and sign < 0):
            cnt += 1
        else:
            break
    return sign * cnt

def _landmarks(df: pd.DataFrame) -> pd.DataFrame:
    h1, l1, c1 = df["High"].shift(1), df["Low"].shift(1), df["Close"].shift(1)
    key_mark   = (h1 + l1 + c1)/3.0
    upper_zone = 2*key_mark - l1
    lower_zone = 2*key_mark - h1
    return pd.DataFrame({"key_mark":key_mark, "upper_zone":upper_zone, "lower_zone":lower_zone})

def _floor_pivots(H: float, L: float, C: float):
    P  = (H + L + C) / 3.0
    R1 = 2*P - L
    S1 = 2*P - H
    R2 = P + (H - L)
    S2 = P - (H - L)
    R3 = H + 2*(P - L)
    S3 = L - 2*(H - P)
    return float(P), float(R1), float(R2), float(R3), float(S1), float(S2), float(S3)

def _pivots_by_scope(df: pd.DataFrame, scope: str) -> dict[str, float]:
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.set_index("Date").sort_index()
    if scope == "weekly":
        agg = d[["High","Low","Close"]].resample("W-FRI").agg({"High":"max","Low":"min","Close":"last"}).dropna()
    elif scope == "monthly":
        agg = d[["High","Low","Close"]].resample("M").agg({"High":"max","Low":"min","Close":"last"}).dropna()
    else:  # daily
        if len(d) < 2:
            px = float(d["Close"].iloc[-1])
            return {"P":px,"R1":px,"R2":px,"R3":px,"S1":px,"S2":px,"S3":px}
        prev = d.iloc[-2]
        P,R1,R2,R3,S1,S2,S3 = _floor_pivots(prev["High"], prev["Low"], prev["Close"])
        return {"P":P,"R1":R1,"R2":R2,"R3":R3,"S1":S1,"S2":S2,"S3":S3}
    if len(agg) < 2:
        px = float(d["Close"].iloc[-1])
        return {"P":px,"R1":px,"R2":px,"R3":px,"S1":px,"S2":px,"S3":px}
    prev = agg.iloc[-2]
    P,R1,R2,R3,S1,S2,S3 = _floor_pivots(prev.High, prev.Low, prev.Close)
    return {"P":P,"R1":R1,"R2":R2,"R3":R3,"S1":S1,"S2":S2,"S3":S3}

def compute_signal(df: pd.DataFrame, symbol: str, horizon: Horizon, params: Optional[Dict]=None) -> Signal:
    P = DEFAULTS.copy()
    if params:
        P.update({k:v for k,v in params.items() if k in DEFAULTS})
    if df is None or df.empty:
        raise ValueError("empty dataframe")

    d = df.copy()
    d["EMA_fast"] = _ema(d["Close"], P["ema_fast"])
    d["EMA_mid"]  = _ema(d["Close"], P["ema_mid"])
    d["EMA_slow"] = _ema(d["Close"], P["ema_slow"])
    d["RSI14"]    = _rsi(d["Close"], 14)
    d["ATR14"]    = _atr(d, 14)
    d["MACD_H"]   = _macd_hist(d["Close"])
    d = pd.concat([d, _landmarks(d)], axis=1)

    last = d.iloc[-1]
    rng = (last["High"] - last["Low"]) or 1e-9
    upper_tail = last["High"] - max(last["Close"], last["Open"])
    body = abs(last["Close"] - last["Open"])
    weak_candle = (upper_tail / rng >= P["weak_tail_ratio"]) and (body / rng <= P["weak_body_ratio"])

    if horizon == "short":
        pivot_scope = "daily"; need_hist = P["macd_bars_short"]
        k_tp1, k_tp2, k_sl = P["short_tp1"], P["short_tp2"], P["short_sl"]; trend_weight = 0.6
    elif horizon == "position":
        pivot_scope = "monthly"; need_hist = P["macd_bars_pos"]
        k_tp1, k_tp2, k_sl = P["pos_tp1"], P["pos_tp2"], P["pos_sl"]; trend_weight = 0.7
    else:
        pivot_scope = "weekly"; need_hist = P["macd_bars_swing"]
        k_tp1, k_tp2, k_sl = P["swing_tp1"], P["swing_tp2"], P["swing_sl"]; trend_weight = 0.65

    x  = d.iloc[-1]
    px = float(x["Close"])
    atr = float(x["ATR14"]) if pd.notna(x["ATR14"]) else 0.0
    km = float(x["key_mark"]) if pd.notna(x["key_mark"]) else px
    uz = float(x["upper_zone"]) if pd.notna(x["upper_zone"]) else px
    lz = float(x["lower_zone"]) if pd.notna(x["lower_zone"]) else px

    piv = _pivots_by_scope(d, pivot_scope)
    Pv, R1, R2, R3, S1, S2, S3 = piv["P"], piv["R1"], piv["R2"], piv["R3"], piv["S1"], piv["S2"], piv["S3"]

    trend_pos = (x["EMA_fast"] > x["EMA_mid"] > x["EMA_slow"]) and (d["EMA_mid"].iloc[-1] > d["EMA_mid"].iloc[-5])
    trend_neg = (x["EMA_fast"] < x["EMA_mid"] < x["EMA_slow"]) and (d["EMA_mid"].iloc[-1] < d["EMA_mid"].iloc[-5])
    hist_seq  = _consecutive_sign(d["MACD_H"])
    macd_ok_buy, macd_ok_short = (hist_seq >= need_hist), (hist_seq <= -need_hist)

    rsi = float(x["RSI14"]) if pd.notna(x["RSI14"]) else 50.0
    rsi_bias_up, rsi_bias_down = (rsi <= P["rsi_hi"]), (rsi >= P["rsi_lo"])

    pos_vs_key = px - km
    buy_bias   = trend_pos and macd_ok_buy   and (pos_vs_key >= 0) and rsi_bias_up
    short_bias = trend_neg and macd_ok_short and (pos_vs_key <= 0) and rsi_bias_down

    if weak_candle and trend_neg:
        short_bias = True
    if weak_candle and trend_pos:
        buy_bias = False

    def _next_up_targets(price: float):
        levels = [lvl for lvl in [R1, R2, R3] if lvl > price]
        return levels[:2]
    def _next_down_targets(price: float):
        levels = [lvl for lvl in [S1, S2, S3] if lvl < price]
        levels.sort(reverse=True)
        return levels[:2]

    def _conf(sign: int) -> float:
        if atr <= 0: base = 0.5
        else:        base = 0.5 + sign * (pos_vs_key / (2.8*atr))
        base = float(np.clip(base, 0.0, 1.0))
        trend_part = trend_weight if (trend_pos and sign>0) or (trend_neg and sign<0) else 0.0
        adj = -0.07 if (sign>0 and weak_candle) else (0.07 if (sign<0 and weak_candle) else 0.0)
        return float(np.clip(0.5*base + 0.5*trend_part + adj, 0.0, 1.0))

    if buy_bias and atr > 0:
        action = "BUY"; confidence = _conf(+1); entry = px
        targets = _next_up_targets(px)
        if targets:
            tp1 = max(targets[0], px + 0.2*atr)
            tp2 = max(targets[1] if len(targets)>1 else targets[0] + 0.6*atr, tp1 + 0.2*atr)
        else:
            tp1, tp2 = px + k_tp1*atr, px + k_tp2*atr
        sl_cands = [lvl for lvl in [S1, S2, S3] if lvl < px]
        sl = min(sl_cands) if sl_cands else px - k_sl*atr
        tp1, tp2 = max(tp1, px), max(tp2, tp1)
        sl       = min(sl, px - 0.01)
    elif short_bias and atr > 0:
        action = "SHORT"; confidence = _conf(-1); entry = px
        targets = _next_down_targets(px)
        if targets:
            tp1 = min(targets[0], px - 0.2*atr)
            tp2 = min(targets[1] if len(targets)>1 else targets[0] - 0.6*atr, tp1 - 0.2*atr)
        else:
            tp1, tp2 = px - k_tp1*atr, px - k_tp2*atr
        sl_cands = [lvl for lvl in [R1, R2, R3] if lvl > px]
        sl = max(sl_cands) if sl_cands else px + k_sl*atr
        tp1, tp2 = min(tp1, px), min(tp2, tp1)
        sl       = max(sl, px + 0.01)
    else:
        action = "WAIT"
        confidence = float(np.clip(0.5 + (pos_vs_key/(4.0*atr) if atr>0 else 0.0), 0.0, 1.0))
        entry = px
        tp1, tp2, sl = px + 0.6*atr, px + 1.2*atr, px - 0.9*atr

    return {
        "symbol": symbol, "horizon": horizon, "action": action, "confidence": round(confidence,2),
        "entry": round(float(entry),2), "tp1": round(float(tp1),2), "tp2": round(float(tp2),2), "sl": round(float(sl),2),
        "key_mark": round(float(km),2), "upper_zone": round(float(uz),2), "lower_zone": round(float(lz),2),
        "pivot_P": round(float(Pv),2), "R1": round(float(R1),2), "R2": round(float(R2),2), "R3": round(float(R3),2),
        "S1": round(float(S1),2), "S2": round(float(S2),2), "S3": round(float(S3),2),
    }
