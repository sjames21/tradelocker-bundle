# btc_weekend_tradelocker_signal_bot.py
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import time
import math
from dataclasses import dataclass
from zoneinfo import ZoneInfo
from tradelocker import TLAPI
import os
import requests
import logging
import sys
from contextlib import contextmanager

from tqdm import tqdm
import pandas as pd

load_dotenv()

# ============================================================
# LOG CLEANUP
# ============================================================

logging.basicConfig(level=logging.INFO)


# ============================================================
# SETTINGS
# ============================================================

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
SEND_SLACK_ALERTS = True

SYMBOLS = ["BTCUSD","XAUUSD"]
TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h"]

SYMBOL_CONFIG = {
    "BTCUSD": {
        "min_buy_score": 7,
        "min_sell_score": -7,
        "min_atr_pct": 0.0015,
    },
    "XAUUSD": {
        "min_buy_score": 7,
        "min_sell_score": -7,
        "min_atr_pct": 0.0008,
    },
}

TF_WEIGHTS = {
    "5m": 2,
    "15m": 2,
    "30m": 2,
    "1h": 1,
    "4h": 1,
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

SCAN_SECONDS = 300

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
TEST_SLACK_ON_NO_SIGNAL = False  # set to False after testing

# ============================================================
# TRADELOCKER DATA PLACEHOLDER
# Replace this function with your actual TradeLocker candle fetch.
# It must return a DataFrame with:
# time, open, high, low, close, volume
# ============================================================

TL_ENVIRONMENT = os.getenv("TL_ENVIRONMENT")
TL_USERNAME = os.getenv("TL_USERNAME")
TL_PASSWORD = os.getenv("TL_PASSWORD")
TL_SERVER = os.getenv("TL_SERVER")
TL_ACCOUNT_ID = os.getenv("TL_ACCOUNT_ID")
TL_ACC_NUM = os.getenv("TL_ACC_NUM")

tl = TLAPI(
    environment=TL_ENVIRONMENT,
    username=TL_USERNAME,
    password=TL_PASSWORD,
    server=TL_SERVER,
    account_id=int(TL_ACCOUNT_ID),
    acc_num=int(TL_ACC_NUM),
)


def suppress_noisy_logs():
    """Suppress noisy TradeLocker/debug logs without hiding our own bot output."""
    noisy_loggers = [
        "tradelocker",
        "tradelocker.tradelocker_api",
        "tradelocker_api",
        "utils",
        "utils.wrapper",
        "utils.timestamps_from_lookback",
    ]

    for name in noisy_loggers:
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False

        for handler in logger.handlers:
            handler.setLevel(logging.CRITICAL)


@contextmanager
def quiet_tradelocker_call():
    """Hide stdout/stderr spam produced inside the TradeLocker wrapper call."""
    suppress_noisy_logs()
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    class _DevNull:
        def write(self, _):
            return None

        def flush(self):
            return None

    try:
        sys.stdout = _DevNull()
        sys.stderr = _DevNull()
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        suppress_noisy_logs()


suppress_noisy_logs()

try:
    print("[CONNECTED] TradeLocker connection successful.")
    for _symbol in SYMBOLS:
        with quiet_tradelocker_call():
            _instrument_id = tl.get_instrument_id_from_symbol_name(_symbol)
        print(f"[CONNECTED] {_symbol} instrument ID: {_instrument_id}")
except Exception as e:
    print(f"[CONNECTION ERROR] TradeLocker connection failed: {e}")


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
            except Exception as e:
                results.append({
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

    print(f"\nMULTI-TIMEFRAME 🪙 {symbol}🪙 ANALYSIS")
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
    atr_pct = atr / candle["close"]

    if atr_pct < SYMBOL_CONFIG[symbol]["min_atr_pct"]:
        print(f"[FILTERED] {symbol} ATR too low: {atr_pct:.5f}")
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

    if total_score >= SYMBOL_CONFIG[symbol]["min_buy_score"] and lower_tf_score > 0 and higher_tf_score > 0:
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

    if total_score <= SYMBOL_CONFIG[symbol]["min_sell_score"] and lower_tf_score < 0 and higher_tf_score < 0:
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
    print(f"Entry Price:  {(signal.preferred_entry_low + signal.preferred_entry_high)/2}")
    print(f"Stop Loss:   {signal.stop_loss}")
    print(f"TP1:         {signal.tp1}")
    print(f"TP2:         {signal.tp2}")
    print(f"Reason:      {signal.reason}")
    print("=" * 70 + "\n")


# ============================================================
# MAIN LOOP
# ============================================================

def send_slack_signal(signal: Signal):
    if not SEND_SLACK_ALERTS:
        return

    if not SLACK_WEBHOOK_URL:
        print("[SLACK] Missing SLACK_WEBHOOK_URL in .env")
        return

    emoji = "🟢" if signal.direction == "BUY" else "🔴"

    message = {
        "text": f"{emoji} {signal.symbol} {signal.direction} Signal",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {signal.symbol} {signal.direction} Signal",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Symbol:*\n{signal.symbol}"},
                    {"type": "mrkdwn", "text": f"*Direction:*\n{signal.direction}"},
                    {"type": "mrkdwn", "text": f"*Confidence:*\n{signal.confidence}"},
                    {"type": "mrkdwn", "text": f"*Strategy:*\nMTF FVG Pullback"},
                ],
            },
            {"type": "divider"},
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Preferred Entry:*\n{signal.preferred_entry_low} - {signal.preferred_entry_high}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Full Entry:*\n{signal.entry_low} - {signal.entry_high}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Entry Price:*\n{(signal.preferred_entry_low + signal.preferred_entry_high)/2}",
                    },
                    {"type": "mrkdwn", "text": f"*Stop Loss:*\n{signal.stop_loss}"},
                    {"type": "mrkdwn", "text": f"*Take Profit 1:*\n{signal.tp1}"},
                    {"type": "mrkdwn", "text": f"*Take Profit 2:*\n{signal.tp2}"},
                ],
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Reason:*\n{signal.reason}",
                },
            },
        ],
    }

    try:
        response = requests.post(
            SLACK_WEBHOOK_URL,
            json=message,
            timeout=10,
        )

        if response.status_code != 200:
            print(f"[SLACK ERROR] {response.status_code}: {response.text}")
        else:
            print("[SLACK] Trading alert sent.")

    except Exception as e:
        print(f"[SLACK ERROR] {e}")

def main():
    signals_today = 0
    current_day = datetime.now(EASTERN).date()

    print("TradeLocker signal bot started.")
    print("No execution. Signals only.")
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

        for symbol in SYMBOLS:
            try:
                signal = build_multi_timeframe_signal(symbol)

                if signal:
                    print_signal(signal)
                    send_slack_signal(signal)

                    # Only place orders in execution script
                else:
                    print(f"[{now.strftime('%I:%M %p')}] No valid {symbol} setup.")

            except Exception as e:
                print(f"[{symbol} ERROR] {e}")

        hour = now.hour
        if 0 <= hour < 3:
            print("[INFO] Low liquidity hours — signals may be weaker.")

        time.sleep(SCAN_SECONDS)

if __name__ == "__main__":
    main()
