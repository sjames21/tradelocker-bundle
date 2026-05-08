# crypto_live_exec.py
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import time
import math
from dataclasses import dataclass
from zoneinfo import ZoneInfo
from tradelocker import TLAPI
import os
import sys
import logging
from contextlib import contextmanager

import pandas as pd
from tqdm import tqdm

load_dotenv()

# ============================================================
# LOGGING / OUTPUT CONTROL
# ============================================================

def suppress_noisy_logs():
    """Silence noisy TradeLocker/debug loggers."""
    noisy_loggers = [
        "tradelocker",
        "tradelocker.tradelocker_api",
        "tradelocker_api",
        "utils",
        "utils.wrapper",
        "utils.timestamps_from_lookback",
    ]

    logging.basicConfig(level=logging.CRITICAL)

    for name in noisy_loggers:
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False

        for handler in logger.handlers:
            handler.setLevel(logging.CRITICAL)


@contextmanager
def quiet_tradelocker_call():
    """Temporarily suppress stdout/stderr during noisy TradeLocker SDK calls."""
    suppress_noisy_logs()
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        suppress_noisy_logs()


def quiet_print(message: str):
    print(message, flush=True)


suppress_noisy_logs()

# ============================================================
# SETTINGS
# ============================================================

SYMBOLS = ["BTCUSD","XAUUSD"]
TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h"]

TF_WEIGHTS = {
    "5m": 2,
    "15m": 2,
    "30m": 2,
    "1h": 1,
    "4h": 1,
}

DUPLICATE_ENTRY_TOLERANCE = {
    "BTCUSD": 5.0,
    "XAUUSD": 0.50,
}

DUPLICATE_SL_TOLERANCE = {
    "BTCUSD": 5.0,
    "XAUUSD": 0.50,
}

DYNAMIC_STOP_ENABLED = True

STOP_ATR_MULTIPLIER = {
    "BTCUSD": 1.75,
    "XAUUSD": 1.50,
}

MIN_STOP_DISTANCE = {
    "BTCUSD": 300.0,   # $300 minimum BTC stop distance
    "XAUUSD": 4.00,    # $4 minimum gold stop distance
}

MAX_STOP_DISTANCE = {
    "BTCUSD": 2500.0,
    "XAUUSD": 25.00,
}

MIN_BUY_SCORE = 5
MIN_SELL_SCORE = -5
LOOKBACK_BARS = 300

EASTERN = ZoneInfo("America/New_York")

EMA_FAST = 50
EMA_SLOW = 200
RSI_PERIOD = 14
ATR_PERIOD = 14

MAX_DISTANCE_FROM_VWAP_ATR = 1.25
MAX_RSI_BUY = 72
MIN_RSI_SELL = 28

ENTRY_BUFFER_ATR = 0.15
SL_BUFFER_ATR = 0.20

MIN_ATR_PCT = 0.0015  # 0.15% minimum volatility
MAX_SIGNALS_PER_DAY = 100

SCAN_SECONDS = 60

CONFIRM_BODY_ATR = 0.25
CONFIRM_CLOSE_NEAR_HIGH = 0.65
CONFIRM_CLOSE_NEAR_LOW = 0.35

FVG_LOOKBACK = 20
MIN_FVG_ATR = 0.15

USE_BREAKOUT_ENTRY = False
BREAKOUT_BUFFER_ATR = 0.05
MAX_BREAKOUT_DISTANCE_ATR = 0.75
MIN_CLOSE_ABOVE_VWAP_ATR = 0.10

REQUIRE_CONFIRMATION_CANDLE = True
REQUIRE_FVG_IMBALANCE = True

LIVE_TRADING_ENABLED = True

MAX_RISK_DOLLARS = {
        "BTCUSD": 7,
        "XAUUSD": 4,
    }
LOT_STEP = {
    "BTCUSD": 0.001,
    "XAUUSD": 0.01,
}

MIN_ORDER_QUANTITY = {
    "BTCUSD": 0.001,
    "XAUUSD": 0.01,
}

MAX_ORDER_QUANTITY = {
    "BTCUSD": 0.04,
    "XAUUSD": 0.02,
}

CANCEL_REPLACE_ONLY_ON_VALID_SIGNAL = True

# ============================================================
# TRADELOCKER DATA PLACEHOLDER
# Replace this function with your actual TradeLocker candle fetch.
# It must return a DataFrame with:
# time, open, high, low, close, volume
# ============================================================

TL_ENVIRONMENT_DEMO = os.getenv("TL_ENVIRONMENT_DEMO")
TL_USERNAME = os.getenv("TL_USERNAME")
TL_PASSWORD = os.getenv("TL_PASSWORD")
TL_SERVER = os.getenv("TL_SERVER")
TL_ACCOUNT_ID_DEMO = os.getenv("TL_ACCOUNT_ID_DEMO")
TL_ACC_NUM_DEMO = os.getenv("TL_ACC_NUM_DEMO")

tl = TLAPI(
    environment=TL_ENVIRONMENT_DEMO,
    username=TL_USERNAME,
    password=TL_PASSWORD,
    server=TL_SERVER,
    account_id=int(TL_ACCOUNT_ID_DEMO),
    acc_num=int(TL_ACC_NUM_DEMO),
)

suppress_noisy_logs()
quiet_print("[CONNECTED] TradeLocker connection successful.")

try:
    for _symbol in SYMBOLS:
        with quiet_tradelocker_call():
            _instrument_id = tl.get_instrument_id_from_symbol_name(_symbol)
        quiet_print(f"[CONNECTED] {_symbol} instrument ID: {_instrument_id}")
except Exception as e:
    quiet_print(f"[CONNECTION ERROR] TradeLocker connection check failed: {e}")


def analyze_timeframe(symbol: str, timeframe: str) -> dict:
    df = fetch_tradelocker_candles(symbol, timeframe, LOOKBACK_BARS)
    df = convert_time_to_est(df)
    df = add_indicators(df)

    candle = df.iloc[-1]
    prev_high, prev_low = get_previous_day_levels(df)

    if prev_high is None or prev_low is None:
        return {
            "timeframe": timeframe,
            "score": 0,
            "bias": "NEUTRAL",
            "reason": "Missing previous day levels",
            "df": df,
        }

    ema_bull = candle["ema50"] > candle["ema200"]
    ema_bear = candle["ema50"] < candle["ema200"]

    above_vwap = candle["close"] > candle["vwap"]
    below_vwap = candle["close"] < candle["vwap"]

    rsi_bull = candle["rsi"] > 50
    rsi_bear = candle["rsi"] < 50

    recent = df.tail(12)

    bull_sweep = (
        (recent["low"] < prev_low) &
        (recent["close"] > prev_low)
    ).any()

    bear_sweep = (
        (recent["high"] > prev_high) &
        (recent["close"] < prev_high)
    ).any()

    score = 0
    reasons = []

    if ema_bull:
        score += 1
        reasons.append("EMA bullish")
    elif ema_bear:
        score -= 1
        reasons.append("EMA bearish")

    if above_vwap:
        score += 1
        reasons.append("above VWAP")
    elif below_vwap:
        score -= 1
        reasons.append("below VWAP")

    if rsi_bull:
        score += 1
        reasons.append("RSI bullish")
    elif rsi_bear:
        score -= 1
        reasons.append("RSI bearish")

    if bull_sweep:
        score += 2
        reasons.append("bullish liquidity sweep")
    elif bear_sweep:
        score -= 2
        reasons.append("bearish liquidity sweep")

    # Cap each timeframe so no single timeframe dominates
    weighted_score = score * TF_WEIGHTS.get(timeframe, 1)
    weighted_score = max(min(weighted_score, 4), -4)

    if weighted_score > 0:
        bias = "BUY"
    elif weighted_score < 0:
        bias = "SELL"
    else:
        bias = "NEUTRAL"

    return {
        "timeframe": timeframe,
        "score": weighted_score,
        "raw_score": score,
        "bias": bias,
        "reason": ", ".join(reasons),
        "df": df,
        "prev_high": prev_high,
        "prev_low": prev_low,
    }

def fetch_tradelocker_candles(symbol: str, interval: str, lookback: int) -> pd.DataFrame:
    with quiet_tradelocker_call():
        instrument_id = tl.get_instrument_id_from_symbol_name(symbol)

    resolution_map = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1H",
        "4h": "4H",
        "1d": "1D",
    }

    resolution = resolution_map.get(interval.lower())
    if resolution is None:
        raise ValueError(f"Unsupported interval: {interval}")

    # Real timestamps in milliseconds
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=3)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    # quiet_print(f"[FETCH] {symbol} {interval} candles...")

    with quiet_tradelocker_call():
        price_history = tl.get_price_history(
            instrument_id,
            resolution=resolution,
            start_timestamp=start_ms,
            end_timestamp=end_ms,
        )

    df = pd.DataFrame(price_history)

    rename_map = {
        "t": "time",
        "time": "time",
        "o": "open",
        "open": "open",
        "h": "high",
        "high": "high",
        "l": "low",
        "low": "low",
        "c": "close",
        "close": "close",
        "v": "volume",
        "volume": "volume",
    }

    df = df.rename(columns=rename_map)

    required = ["time", "open", "high", "low", "close"]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing candle columns: {missing}. Returned columns: {df.columns.tolist()}"
        )

    if "volume" not in df.columns:
        df["volume"] = 1

    df = df[["time", "open", "high", "low", "close", "volume"]].copy()

    # Handles millisecond timestamps
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True, errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().tail(lookback).reset_index(drop=True)

    if df.empty:
        raise ValueError("TradeLocker returned candles, but dataframe is empty after cleanup.")

    return df


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema50"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    typical = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical * df["volume"]).cumsum() / df["volume"].cumsum()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()

    return df


# ============================================================
# SESSION FILTER
# ============================================================

def convert_time_to_est(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["time_est"] = df["time"].dt.tz_convert(EASTERN)
    return df


# ============================================================
# KEY LEVELS
# ============================================================

def get_previous_day_levels(df: pd.DataFrame):
    df = df.copy()
    df["date_est"] = df["time_est"].dt.date

    today = datetime.now(EASTERN).date()
    previous_days = df[df["date_est"] < today]

    if previous_days.empty:
        return None, None

    last_date = previous_days["date_est"].max()
    prev_day = previous_days[previous_days["date_est"] == last_date]

    prev_high = prev_day["high"].max()
    prev_low = prev_day["low"].min()

    return prev_high, prev_low


# ============================================================
# SIGNAL LOGIC
# ============================================================

@dataclass
class Signal:
    symbol: str
    direction: str
    entry_low: float
    entry_high: float
    preferred_entry_low: float
    preferred_entry_high: float
    stop_loss: float
    tp1: float
    tp2: float
    confidence: str
    reason: str

def candle_body_strength(candle, atr):
    body = abs(candle["close"] - candle["open"])
    return body / atr if atr > 0 else 0


def close_position_in_range(candle):
    candle_range = candle["high"] - candle["low"]
    if candle_range <= 0:
        return 0.5
    return (candle["close"] - candle["low"]) / candle_range


def bullish_confirmation_candle(candle, atr):
    body_ok = candle_body_strength(candle, atr) >= CONFIRM_BODY_ATR
    close_ok = close_position_in_range(candle) >= CONFIRM_CLOSE_NEAR_HIGH
    green_ok = candle["close"] > candle["open"]

    return body_ok and close_ok and green_ok


def bearish_confirmation_candle(candle, atr):
    body_ok = candle_body_strength(candle, atr) >= CONFIRM_BODY_ATR
    close_ok = close_position_in_range(candle) <= CONFIRM_CLOSE_NEAR_LOW
    red_ok = candle["close"] < candle["open"]

    return body_ok and close_ok and red_ok


def detect_bullish_fvg(df, atr):
    recent = df.tail(FVG_LOOKBACK)

    for i in range(2, len(recent)):
        c1 = recent.iloc[i - 2]
        c3 = recent.iloc[i]

        # Bullish fair value gap:
        # candle 1 high is below candle 3 low
        gap = c3["low"] - c1["high"]

        if gap > atr * MIN_FVG_ATR:
            return True, c1["high"], c3["low"]

    return False, None, None


def detect_bearish_fvg(df, atr):
    recent = df.tail(FVG_LOOKBACK)

    for i in range(2, len(recent)):
        c1 = recent.iloc[i - 2]
        c3 = recent.iloc[i]

        # Bearish fair value gap:
        # candle 1 low is above candle 3 high
        gap = c1["low"] - c3["high"]

        if gap > atr * MIN_FVG_ATR:
            return True, c3["high"], c1["low"]

    return False, None, None

def get_breakout_trigger(df, direction, atr):
    """
    Uses the latest candle as confirmation candle.
    BUY: price must break above confirmation high.
    SELL: price must break below confirmation low.
    """
    confirm = df.iloc[-2]   # confirmation candle
    current = df.iloc[-1]   # current breakout candle

    buffer = atr * BREAKOUT_BUFFER_ATR

    if direction == "BUY":
        trigger = confirm["high"] + buffer
        breakout_valid = current["high"] >= trigger
        breakout_distance = abs(current["close"] - trigger) / atr

        return breakout_valid, trigger, confirm, breakout_distance

    if direction == "SELL":
        trigger = confirm["low"] - buffer
        breakout_valid = current["low"] <= trigger
        breakout_distance = abs(current["close"] - trigger) / atr

        return breakout_valid, trigger, confirm, breakout_distance

    return False, None, None, None

def build_multi_timeframe_signal(symbol: str) -> Signal | None:

    results = []

    with tqdm(
            total=len(TIMEFRAMES),
            desc=f"Scanning {symbol}",
            ncols=90,
            leave=True,
            dynamic_ncols=True
    ) as pbar:
        for tf in TIMEFRAMES:
            try:
                pbar.set_postfix_str(f"timeframe={tf}")
                result = analyze_timeframe(symbol, tf)
                results.append(result)
                pbar.update(1)
            except Exception as e: results.append({
                "timeframe": tf,
                "score": 0,
                "bias": "ERROR",
                "reason": str(e),
                "df": None,
            })

    if not results:
        return None

    total_score = sum(r["score"] for r in results)

    lower_tf_score = sum(
        r["score"] for r in results
        if r["timeframe"] in ["5m", "15m", "30m"]
    )

    higher_tf_score = sum(
        r["score"] for r in results
        if r["timeframe"] in ["1h", "4h"]
    )

    print(f"\nMULTI-TIMEFRAME 🪙 {symbol}🪙  ANALYSIS")
    print("-" * 70)

    for r in results:
        print(
            f"{r['timeframe']:>4} | "
            f"Bias: {r['bias']:<7} | "
            f"Score: {r['score']:<4} | "
            f"{r['reason']}"
        )

    print(f"TOTAL SCORE: {total_score}")
    print("-" * 70)

    five_min = next(
        (r for r in results if r["timeframe"] == "5m" and r.get("df") is not None),
        None
    )

    if five_min is None:
        print("[WARN] 5m data unavailable, so no entry range can be built.")
        return None

    df = five_min["df"]

    if df is None or df.empty:
        print("[WARN] 5m dataframe empty, so no entry range can be built.")
        return None

    candle = df.iloc[-1]

    atr = candle["atr"]
    if math.isnan(atr) or atr <= 0:
        return None

    bull_confirm_candle = bullish_confirmation_candle(candle, atr)
    bear_confirm_candle = bearish_confirmation_candle(candle, atr)

    bull_fvg, bull_fvg_low, bull_fvg_high = detect_bullish_fvg(df, atr)
    bear_fvg, bear_fvg_low, bear_fvg_high = detect_bearish_fvg(df, atr)

    distance_from_vwap_atr = abs(candle["close"] - candle["vwap"]) / atr

    overextended_buy = (
            candle["close"] > candle["vwap"]
            and distance_from_vwap_atr > MAX_DISTANCE_FROM_VWAP_ATR
            and candle["rsi"] > MAX_RSI_BUY
    )

    overextended_sell = (
            candle["close"] < candle["vwap"]
            and distance_from_vwap_atr > MAX_DISTANCE_FROM_VWAP_ATR
            and candle["rsi"] < MIN_RSI_SELL
    )

    prev_high = five_min["prev_high"]
    prev_low = five_min["prev_low"]

    entry_buffer = atr * ENTRY_BUFFER_ATR
    sl_buffer = atr * SL_BUFFER_ATR

    recent = df.tail(12)

    abs_score = abs(total_score)

    # Grading Logic
    if abs_score >= 17:
        grade = "A+"
    elif abs_score >= 12:
        grade = "A"
    elif abs_score >= 7:
        grade = "B"
    else:
        grade = "C"

    if total_score >= MIN_BUY_SCORE and lower_tf_score > 0 and higher_tf_score > 0:
        if REQUIRE_CONFIRMATION_CANDLE and not bull_confirm_candle:
            print("[FILTERED] BUY blocked: no bullish confirmation candle.")
            return None

        if REQUIRE_FVG_IMBALANCE and not bull_fvg:
            print("[FILTERED] BUY blocked: no bullish imbalance/FVG detected.")
            return None

        if overextended_buy:
            print(
                f"[FILTERED] BUY overextended | "
                f"Close too far above VWAP: {distance_from_vwap_atr:.2f} ATR | "
                f"RSI={candle['rsi']:.2f}"
            )
            return None

        if USE_BREAKOUT_ENTRY:
            breakout_valid, trigger, confirm_candle, breakout_distance = get_breakout_trigger(df, "BUY", atr)

            if not breakout_valid:
                print("[FILTERED] BUY blocked: price has not broken confirmation candle high.")
                return None

            if breakout_distance > MAX_BREAKOUT_DISTANCE_ATR:
                print(
                    f"[FILTERED] BUY blocked: breakout too extended "
                    f"({breakout_distance:.2f} ATR from trigger)."
                )
                return None
            if (candle["close"] - candle["vwap"]) / atr < MIN_CLOSE_ABOVE_VWAP_ATR:
                return None

        sweep_low = recent["low"].min()

        if USE_BREAKOUT_ENTRY:
            entry_low = trigger
            entry_high = trigger + (atr * 0.20)
        elif bull_fvg:
            entry_low = bull_fvg_low
            entry_high = bull_fvg_high
        else:
            entry_low = min(candle["vwap"], candle["close"]) - entry_buffer
            entry_high = max(candle["vwap"], candle["close"]) + entry_buffer



        # Preferred = lower half of the range
        mid = (entry_low + entry_high) / 2

        preferred_entry_low = entry_low
        preferred_entry_high = mid

        stop_loss = min(sweep_low, prev_low) - sl_buffer

        risk = entry_high - stop_loss
        tp1 = entry_high + risk
        tp2 = entry_high + risk * 2

        return Signal(
            symbol=symbol,
            direction="BUY",
            entry_low=round(entry_low, 2),
            entry_high=round(entry_high, 2),
            preferred_entry_low=round(preferred_entry_low, 2),
            preferred_entry_high=round(preferred_entry_high, 2),
            stop_loss=round(stop_loss, 2),
            tp1=round(tp1, 2),
            tp2=round(tp2, 2),
            confidence=grade,
            reason=f"Multi-timeframe bullish alignment. Total score: {total_score}."
        )

    if total_score <= MIN_SELL_SCORE and lower_tf_score < 0 and higher_tf_score < 0:
        if REQUIRE_CONFIRMATION_CANDLE and not bear_confirm_candle:
            print("[FILTERED] SELL blocked: no bearish confirmation candle.")
            return None

        if REQUIRE_FVG_IMBALANCE and not bear_fvg:
            print("[FILTERED] SELL blocked: no bearish imbalance/FVG detected.")
            return None

        if overextended_sell:
            print(
                f"[FILTERED] SELL overextended | "
                f"Close too far below VWAP: {distance_from_vwap_atr:.2f} ATR | "
                f"RSI={candle['rsi']:.2f}"
            )
            return None

        if USE_BREAKOUT_ENTRY:
            breakout_valid, trigger, confirm_candle, breakout_distance = get_breakout_trigger(df, "SELL", atr)

            if not breakout_valid:
                print("[FILTERED] SELL blocked: price has not broken confirmation candle low.")
                return None

            if breakout_distance > MAX_BREAKOUT_DISTANCE_ATR:
                print(
                    f"[FILTERED] SELL blocked: breakout too extended "
                    f"({breakout_distance:.2f} ATR from trigger)."
                )
                return None

            if (candle["vwap"] - candle["close"]) / atr < MIN_CLOSE_ABOVE_VWAP_ATR:
                return None

        sweep_high = recent["high"].max()

        if USE_BREAKOUT_ENTRY:
            entry_high = trigger
            entry_low = trigger - (atr * 0.20)
        elif bear_fvg:
            entry_low = bear_fvg_low
            entry_high = bear_fvg_high
        else:
            entry_low = min(candle["vwap"], candle["close"]) - entry_buffer
            entry_high = max(candle["vwap"], candle["close"]) + entry_buffer

        # Preferred = lower half of the range
        mid = (entry_low + entry_high) / 2

        preferred_entry_low = mid
        preferred_entry_high = entry_high

        stop_loss = max(sweep_high, prev_high) + sl_buffer

        risk = stop_loss - entry_low
        tp1 = entry_low - risk
        tp2 = entry_low - risk * 2

        return Signal(
            symbol=symbol,
            direction="SELL",
            entry_low=round(entry_low, 2),
            entry_high=round(entry_high, 2),
            preferred_entry_low=round(preferred_entry_low, 2),
            preferred_entry_high=round(preferred_entry_high, 2),
            stop_loss=round(stop_loss, 2),
            tp1=round(tp1, 2),
            tp2=round(tp2, 2),
            confidence=grade,
            reason=f"Multi-timeframe bearish alignment. Total score: {total_score}."
        )

    return None


# ============================================================
# OUTPUT
# ============================================================

def print_signal(signal: Signal):
    now = datetime.now(EASTERN).strftime("%m-%d-%Y %I:%M %p EST")

    print("\n" + "=" * 70)
    print(f"{signal.symbol} SIGNAL | {now}")
    print("=" * 70)
    print(f"Symbol:      {signal.symbol}")
    print(f"Direction:   {signal.direction}")
    print(f"Confidence:  {signal.confidence}")
    print(f"Entry Range: {signal.entry_low} - {signal.entry_high}")
    print(f"Preferred Entry Range: {signal.preferred_entry_low} - {signal.preferred_entry_high}")
    print(f"Stop Loss:   {signal.stop_loss}")
    print(f"TP1:         {signal.tp1}")
    print(f"TP2:         {signal.tp2}")
    print(f"Reason:      {signal.reason}")
    print("=" * 70 + "\n")


# ============================================================
# MAIN LOOP
# ============================================================

def get_instrument_id(symbol: str):
    with quiet_tradelocker_call():
        return tl.get_instrument_id_from_symbol_name(symbol)


def get_signal_order_price(signal: Signal) -> float:
    """
    BUY uses preferred lower entry.
    SELL uses preferred upper entry.
    """
    if signal.direction == "BUY":
        return float(signal.preferred_entry_low)

    if signal.direction == "SELL":
        return float(signal.preferred_entry_high)

    raise ValueError(f"Unknown signal direction: {signal.direction}")


def get_existing_pending_orders_for_symbol(symbol: str):
    try:
        instrument_id = int(get_instrument_id(symbol))

        orders = tl.get_all_orders()

        if orders is None:
            return []

        # IMPORTANT: TradeLocker returns a DataFrame here
        if isinstance(orders, pd.DataFrame):
            rows = orders.to_dict("records")
        elif isinstance(orders, dict):
            rows = orders.get("orders") or orders.get("data") or orders.get("d") or []
        else:
            rows = orders

        pending_orders = []

        for order in rows:
            try:
                order_instrument_id = int(order.get("tradableInstrumentId", -1))
                order_status = str(order.get("status", "")).lower()
                order_type = str(order.get("type", "")).lower()
                is_open = bool(order.get("isOpen", False))

                matches_symbol = order_instrument_id == instrument_id

                is_pending = (
                    is_open
                    and order_status not in ["filled", "cancelled", "canceled", "rejected", "closed"]
                    and order_type in ["limit", "stop", "stoplimit", "stop_limit"]
                )

                if matches_symbol and is_pending:
                    pending_orders.append(order)

            except Exception as e:
                print(f"[ORDER PARSE ERROR] {symbol}: {e}")

        return pending_orders

    except Exception as e:
        print(f"[ORDER FETCH ERROR] {symbol}: {e}")
        return []


def extract_order_id(order):
    for key in ["id", "orderId", "order_id", "positionId", "tradableOrderId"]:
        if key in order and order[key]:
            return order[key]

    raise ValueError(f"Could not extract order id from order: {order}")


def cancel_pending_order(order):
    order_id = extract_order_id(order)

    with quiet_tradelocker_call():
        if hasattr(tl, "delete_order"):
            return tl.delete_order(order_id)

        if hasattr(tl, "cancel_order"):
            return tl.cancel_order(order_id)

        if hasattr(tl, "remove_order"):
            return tl.remove_order(order_id)

    raise AttributeError(
        "Could not find cancel method on TLAPI. "
        "Check your wrapper for delete_order(), cancel_order(), or remove_order()."
    )


def cancel_existing_pending_orders(symbol: str):
    existing_orders = get_existing_pending_orders_for_symbol(symbol)

    print(f"[ORDER CHECK] {symbol}: {len(existing_orders)} pending order(s) found.")

    if not existing_orders:
        return

    for order in existing_orders:
        try:
            order_id = extract_order_id(order)
            cancel_pending_order(order)
            print(f"[ORDER] Cancelled existing {symbol} pending order: {order_id}")
        except Exception as e:
            print(f"[ORDER ERROR] Failed to cancel pending order: {e}")

def round_down_to_step(value: float, step: float) -> float:
    return math.floor(value / step) * step


def calculate_quantity_from_risk(signal: Signal, order_price: float, stop_loss: float) -> float:
    risk_dollars = MAX_RISK_DOLLARS[signal.symbol]
    stop_distance = abs(order_price - stop_loss)

    if stop_distance <= 0:
        raise ValueError("Invalid stop distance. Entry and stop loss are too close.")

    raw_quantity = risk_dollars / stop_distance

    min_qty = MIN_ORDER_QUANTITY[signal.symbol]
    max_qty = MAX_ORDER_QUANTITY[signal.symbol]
    step = LOT_STEP[signal.symbol]

    quantity = min(raw_quantity, max_qty)
    quantity = round_down_to_step(quantity, step)

    if quantity < min_qty:
        print(
            f"[RISK FILTER] {signal.symbol} skipped. "
            f"Calculated qty {quantity} is below min qty {min_qty}."
        )
        return 0

    return round(quantity, 4)

def find_number_in_order(order, keys):
    if not isinstance(order, dict):
        return None

    for key in keys:
        if key in order and order[key] is not None:
            try:
                return float(order[key])
            except Exception:
                pass

    return None


def has_duplicate_pending_order(signal: Signal, order_price: float, stop_loss: float) -> bool:
    pending_orders = get_existing_pending_orders_for_symbol(signal.symbol)

    entry_tol = DUPLICATE_ENTRY_TOLERANCE[signal.symbol]
    sl_tol = DUPLICATE_SL_TOLERANCE[signal.symbol]

    for order in pending_orders:
        order_str = str(order).lower()

        existing_price = find_number_in_order(order, [
            "price",
            "limitPrice",
            "limit_price",
            "stopPrice",
            "stop_price",
            "entryPrice",
            "entry_price",
            "orderPrice",
            "order_price",
            "avgPrice",
            "avg_price",
            "openPrice",
            "open_price",
        ])

        existing_sl = find_number_in_order(order, [
            "stopLoss",
            "stop_loss",
            "sl",
            "stopLossPrice",
            "stop_loss_price",
            "stopPrice",
            "stop_price",
        ])

        same_direction = signal.direction.lower() in order_str

        same_entry = (
            existing_price is not None
            and abs(existing_price - order_price) <= entry_tol
        )

        same_sl = (
            existing_sl is not None
            and abs(existing_sl - stop_loss) <= sl_tol
        )

        # Main duplicate rule
        if same_direction and same_entry and same_sl:
            print(
                f"[DUPLICATE FILTER] Existing {signal.symbol} {signal.direction} order found. "
                f"Entry={existing_price}, SL={existing_sl}. New Entry={order_price}, New SL={stop_loss}. Skipping."
            )
            return True

        # Backup rule: if entry matches exactly/near exactly, still block it
        if same_direction and same_entry:
            print(
                f"[DUPLICATE FILTER] Existing {signal.symbol} {signal.direction} order near same entry. "
                f"Existing Entry={existing_price}, New Entry={order_price}. Skipping."
            )
            return True

    return False

def get_dynamic_stop_loss(signal: Signal, order_price: float) -> float:
    """
    Makes the stop wider/more realistic using ATR-style distance logic.
    Keeps original stop if it is already wider.
    """

    original_stop = float(signal.stop_loss)
    original_distance = abs(order_price - original_stop)

    min_distance = MIN_STOP_DISTANCE[signal.symbol]
    max_distance = MAX_STOP_DISTANCE[signal.symbol]

    # Use the wider of original stop distance or minimum required distance
    stop_distance = max(original_distance, min_distance)

    # Prevent absurdly wide stops
    stop_distance = min(stop_distance, max_distance)

    if signal.direction == "BUY":
        return round(order_price - stop_distance, 2)

    if signal.direction == "SELL":
        return round(order_price + stop_distance, 2)

    raise ValueError(f"Unknown direction: {signal.direction}")

def place_pending_order_from_signal(signal: Signal):
    instrument_id = get_instrument_id(signal.symbol)
    order_price = get_signal_order_price(signal)
    stop_loss = (
        get_dynamic_stop_loss(signal, order_price)
        if DYNAMIC_STOP_ENABLED
        else float(signal.stop_loss)
    )

    if has_duplicate_pending_order(signal, order_price, stop_loss):
        return None

    qty = calculate_quantity_from_risk(signal, order_price, stop_loss)
    if qty <= 0:
        return None

    risk_per_unit = abs(order_price - float(stop_loss))
    actual_risk = risk_per_unit * qty
    max_risk = MAX_RISK_DOLLARS[signal.symbol]

    if actual_risk > max_risk + 0.25:
        print(
            f"[RISK FILTER] {signal.symbol} skipped. "
            f"Risk = ${actual_risk:.2f} > ${max_risk:.2f}"
        )
        return None

    side = signal.direction.lower()

    print(
        f"[ORDER] Placing {signal.symbol} {signal.direction} pending order | "
        f"Entry={order_price} | SL={stop_loss} | TP={signal.tp1} | "
        f"Qty={qty} | Risk=${actual_risk:.2f}"
    )

    with quiet_tradelocker_call():
        order_id = tl.create_order(
            instrument_id=instrument_id,
            quantity=qty,
            side=side,
            type_="limit",
            price=order_price,
            stop_loss=stop_loss,
            stop_loss_type="absolute",
            take_profit=float(signal.tp1),
            take_profit_type="absolute",
            validity="GTC",
        )

    print(f"[ORDER] New {signal.symbol} pending order placed: {order_id}")
    return order_id

def get_open_positions_for_symbol(symbol: str):
    with quiet_tradelocker_call():
        if hasattr(tl, "get_all_positions"):
            positions = tl.get_all_positions()
        elif hasattr(tl, "get_positions"):
            positions = tl.get_positions()
        elif hasattr(tl, "get_open_positions"):
            positions = tl.get_open_positions()
        else:
            raise AttributeError(
                "Could not find position-list method on TLAPI."
            )

    if isinstance(positions, pd.DataFrame):
        rows = positions.to_dict("records")
    elif isinstance(positions, dict):
        rows = positions.get("positions") or positions.get("data") or positions.get("d") or []
    else:
        rows = positions or []

    instrument_id = str(get_instrument_id(symbol))
    symbol_positions = []

    for pos in rows:
        pos_str = str(pos).lower()

        is_symbol = (
            symbol.lower() in pos_str
            or instrument_id in pos_str
        )

        is_open = not any(
            word in pos_str
            for word in ["closed", "cancelled", "canceled"]
        )

        if is_symbol and is_open:
            symbol_positions.append(pos)

    return symbol_positions


def has_open_position_for_symbol(symbol: str) -> bool:
    return len(get_open_positions_for_symbol(symbol)) > 0

def cancel_replace_pending_order(signal: Signal):
    if not LIVE_TRADING_ENABLED:
        print("[ORDER] Live trading disabled. Signal printed only.")
        return

    if has_open_position_for_symbol(signal.symbol):
        print(
            f"[POSITION FILTER] Open {signal.symbol} position exists. "
            f"No new order placed."
        )
        return

    order_price = get_signal_order_price(signal)

    stop_loss = (
        get_dynamic_stop_loss(signal, order_price)
        if DYNAMIC_STOP_ENABLED
        else float(signal.stop_loss)
    )

    # If exact/near duplicate already exists, do nothing
    if has_duplicate_pending_order(signal, order_price, stop_loss):
        return

    # Otherwise cancel old pending orders for this symbol and replace
    cancel_existing_pending_orders(signal.symbol)
    place_pending_order_from_signal(signal)

def main():
    signals_today = 0
    current_day = datetime.now(EASTERN).date()

    print(f"TradeLocker signal bot started.")
    print("Demo execution enabled. Pending orders only.")
    print("Active window: 24/7 (all sessions).\n")

    while True:
        now = datetime.now(EASTERN)

        if now.date() != current_day:
            current_day = now.date()
            signals_today = 0

        if signals_today >= MAX_SIGNALS_PER_DAY:
            print("Max signals reached for today.")
            time.sleep(SCAN_SECONDS)
            continue

        print("\n" + "=" * 80)
        print(f"[CYCLE] Starting scan cycle at {now.strftime('%m-%d-%Y %I:%M %p EST')}")
        print("=" * 80)

        try:
            for symbol in SYMBOLS:
                signal = build_multi_timeframe_signal(symbol)

                if signal:
                    print_signal(signal)
                    cancel_replace_pending_order(signal)
                else:
                    print(f"No valid {symbol} setup.")
                # Do nothing. Existing pending orders remain untouched.

        except Exception as e:
            print(f"[{now.strftime('%I:%M %p')}] Multi-timeframe scan completed with errors.")
            print(f"Error: {e}")

        hour = now.hour
        if 0 <= hour < 3:
            print("[INFO] Low liquidity hours — signals may be weaker.")

        time.sleep(SCAN_SECONDS)

if __name__ == "__main__":
    main()
