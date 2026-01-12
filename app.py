import datetime as dt
import itertools
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="Weekly Market Regime Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def load_weekly_data(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        return data
    data = normalize_ohlcv_columns(data, ticker)
    if data.empty:
        return data
    expected = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in expected if col not in data.columns]
    if missing:
        return pd.DataFrame()
    data = data[expected]
    weekly = data.resample("W-FRI").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    return weekly.dropna()


def normalize_ohlcv_columns(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if not isinstance(data.columns, pd.MultiIndex):
        return data

    primary = ticker.replace(",", " ").split()
    primary = primary[0].upper() if primary else ""

    level0 = data.columns.get_level_values(0)
    level1 = data.columns.get_level_values(1)
    if "Open" in level0:
        tickers = list(dict.fromkeys(level1))
        pick = primary if primary in tickers else tickers[0]
        return data.xs(pick, axis=1, level=1)

    tickers = list(dict.fromkeys(level0))
    pick = primary if primary in tickers else tickers[0]
    return data[pick]


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(int(length)).mean()


def rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(int(length)).mean()
    avg_loss = loss.rolling(int(length)).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(int(length)).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    up_move = high.diff()
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_val = tr.rolling(int(length)).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=high.index).rolling(int(length)).mean() / atr_val)
    minus_di = 100 * (pd.Series(minus_dm, index=high.index).rolling(int(length)).mean() / atr_val)

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / denom) * 100
    return dx.rolling(int(length)).mean()


def compute_regime_indicators(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    out = df.copy()
    out["MA_FAST"] = sma(out["Close"], settings["ma_fast_len"])
    out["MA_SLOW"] = sma(out["Close"], settings["ma_slow_len"])
    out["RSI"] = rsi(out["Close"], settings["rsi_len"])
    out["RSI_FAST"] = rsi(out["Close"], settings["rsi_fast_len"])
    out["ADX"] = adx(out["High"], out["Low"], out["Close"], settings["adx_len"])
    out["ADX_FAST"] = adx(out["High"], out["Low"], out["Close"], settings["adx_fast_len"])
    out["ATR"] = atr(out["High"], out["Low"], out["Close"], settings["atr_len"])
    out["ATR_FAST"] = atr(out["High"], out["Low"], out["Close"], settings["atr_fast_len"])
    out["ATR_PCT"] = out["ATR"] / out["MA_FAST"] * 100.0
    out["ATR_PCT_FAST"] = out["ATR_FAST"] / out["MA_FAST"] * 100.0
    weekly_ret = out["Close"].pct_change()
    out["VOL"] = weekly_ret.rolling(settings["vol_len"]).std() * np.sqrt(52)
    out["VOL_FAST"] = weekly_ret.rolling(settings["vol_fast_len"]).std() * np.sqrt(52)
    out["Volume_MA"] = sma(out["Volume"], settings["volume_ma_len"])
    out["Volume_MA_FAST"] = sma(out["Volume"], settings["volume_ma_fast_len"])
    out["Ret_n"] = out["Close"].pct_change(settings["return_window"])
    return out


def classify_downtrend(df: pd.DataFrame, settings: dict) -> tuple[pd.Series, dict[str, pd.Series], pd.Series]:
    label = pd.Series("none", index=df.index)

    atr_fast_short = df["ATR_PCT_FAST"].rolling(3).max()
    atr_fast_long = df["ATR_PCT_FAST"].shift(4).rolling(78).max()
    atr_fast_roll2 = df["ATR_PCT_FAST"].rolling(2).mean()
    atr_fast_shift2_roll2 = df["ATR_PCT_FAST"].shift(2).rolling(2).mean()
    atr_fast_shift1_roll5 = df["ATR_PCT_FAST"].shift(1).rolling(5).mean()
    vol_fast_shift6_roll15 = df["Volume_MA_FAST"].shift(6).rolling(15).max()
    vol_fast_shift1_roll5 = df["Volume_MA_FAST"].shift(1).rolling(5).max()
    adx_roll2 = df["ADX"].rolling(2).mean()
    atr_slow_short = df["ATR_PCT"].rolling(5).mean()
    atr_slow_long = df["ATR_PCT"].shift(6).rolling(5).mean()
    vol_slow_shift6 = df["Volume_MA"].shift(6)
    atr_pct_shift4 = df["ATR_PCT"].shift(4)
    atr_pct_shift12 = df["ATR_PCT"].shift(12)
    atr_pct_shift24 = df["ATR_PCT"].shift(24)
    atr_pct_shift30 = df["ATR_PCT"].shift(30)

    fast_down_trend_raw = (
        ((df["MA_FAST"] < df["MA_FAST"].shift(1))
        #& (df["ADX"].rolling(2).mean() > settings["trend_adx_threshold"])
        & (atr_fast_short > atr_fast_long)
        & (df["Volume_MA_FAST"] > vol_fast_shift6_roll15)
        & (df["MA_FAST"] < df["MA_SLOW"]))
        | ((df["Close"] < df["MA_SLOW"]) & (df["Close"].shift(1) < df["MA_SLOW"])
        & (df["ATR_PCT_FAST"] > atr_fast_shift1_roll5)
        & (df["Volume_MA_FAST"] > 1.1 * vol_fast_shift1_roll5)
        & (adx_roll2 > settings["trend_adx_threshold"])
        & (df["RSI_FAST"] > 30)
        )
    )

    slow_down_trend_raw = (
        (df["MA_SLOW"] < df["MA_SLOW"].shift(1))
        & (df["ADX"] > settings["trend_adx_threshold"])
        & (atr_slow_short > atr_slow_long)
        & (df["Volume_MA"] > vol_slow_shift6)
        & (df["ATR_PCT"] > atr_pct_shift4)
        & (atr_pct_shift4 > atr_pct_shift12)
        & (atr_pct_shift12 > atr_pct_shift24)
        & (atr_pct_shift24 > atr_pct_shift30)
    )

    freeze_uptrend = pd.Series(False, index=df.index)
    freeze_active = False
    ma_slow_up = df["MA_SLOW"] > df["MA_SLOW"].shift(1)
    for idx in df.index:
        if slow_down_trend_raw.loc[idx]:
            freeze_active = True
        if freeze_active and ma_slow_up.loc[idx]:
            freeze_active = False
        freeze_uptrend.loc[idx] = freeze_active

    fast_up_trend = (
        (df["MA_FAST"] > df["MA_FAST"].shift(1))
        #& (df["ADX_FAST"] > settings["trend_adx_threshold"])
        & (atr_fast_roll2 < atr_fast_shift2_roll2)
        & ~freeze_uptrend
    )

    slow_up_trend = (
        (df["MA_SLOW"] > df["MA_SLOW"].shift(1))
        & (df["ADX"] < settings["trend_adx_threshold"])
        & (atr_slow_short > atr_slow_long)
        & ~freeze_uptrend
    )

    panic = (
        (df["RSI"] < settings["panic_rsi_max"])
        & (df["Ret_n"] < settings["panic_return_max"])
        & (df["Volume"] > settings["panic_volume_mult"] * df["Volume_MA"])
    )

    required_series = [
        df["MA_FAST"],
        df["MA_FAST"].shift(1),
        df["MA_SLOW"],
        df["MA_SLOW"].shift(1),
        atr_fast_short,
        atr_fast_long,
        atr_fast_roll2,
        atr_fast_shift2_roll2,
        atr_fast_shift1_roll5,
        vol_fast_shift6_roll15,
        vol_fast_shift1_roll5,
        adx_roll2,
        atr_slow_short,
        atr_slow_long,
        vol_slow_shift6,
        atr_pct_shift4,
        atr_pct_shift12,
        atr_pct_shift24,
        atr_pct_shift30,
        df["ADX"],
        df["RSI"],
        df["RSI_FAST"],
        df["Ret_n"],
        df["Volume"],
        df["Volume_MA"],
    ]
    valid_mask = pd.concat(required_series, axis=1).notna().all(axis=1)

    freeze_downtrend = pd.Series(False, index=df.index)
    if settings.get("freeze_down_on_panic", False):
        freeze_active = False
        for idx in df.index:
            if panic.loc[idx]:
                freeze_active = True
            if freeze_active and (fast_up_trend.loc[idx] or slow_up_trend.loc[idx]):
                freeze_active = False
            freeze_downtrend.loc[idx] = freeze_active

    fast_down_trend = fast_down_trend_raw & ~freeze_downtrend
    slow_down_trend = slow_down_trend_raw & ~freeze_downtrend

    rules = {
        "fast_down_trend": fast_down_trend,
        "slow_down_trend": slow_down_trend,
        "fast_up_trend": fast_up_trend,
        "slow_up_trend": slow_up_trend,
        "panic": panic,
    }
    priority_order = settings.get("priority_order") or list(rules.keys())
    for name in priority_order:
        condition = rules.get(name)
        if condition is not None:
            label.loc[condition] = name

    return label, rules, valid_mask


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 52, risk_free: float = 0.0) -> float:
    clean = returns.dropna()
    if len(clean) < 2:
        return np.nan
    excess = clean - (risk_free / periods_per_year)
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return np.nan
    return np.sqrt(periods_per_year) * excess.mean() / vol


def compute_equity_stats(returns: pd.Series) -> dict:
    clean = returns.dropna()
    if len(clean) < 2:
        return {}

    equity = (1 + clean).cumprod()
    num_weeks = len(equity)
    total_return = equity.iloc[-1] - 1
    cagr = (equity.iloc[-1] ** (52 / num_weeks)) - 1

    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    max_dd = drawdown.min()
    sharpe = sharpe_ratio(clean)
    volatility = clean.std() * np.sqrt(52)

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Max Drawdown": max_dd,
        "Sharpe": sharpe,
        "Volatility": volatility,
        "Returns": clean,
        "Equity": equity,
        "Drawdown": drawdown,
    }


def compute_trade_returns(
    returns: pd.Series, buy_signal: pd.Series, sell_signal: pd.Series, fee: float
) -> tuple[pd.Series, pd.Series]:
    buy_signal = buy_signal.reindex(returns.index, fill_value=False)
    sell_signal = sell_signal.reindex(returns.index, fill_value=False)
    position_raw = pd.Series(np.nan, index=returns.index, dtype=float)
    position_raw.loc[buy_signal] = 1.0
    position_raw.loc[sell_signal] = 0.0
    position_raw = position_raw.ffill().fillna(0.0)
    position = position_raw.shift(1).fillna(0.0)
    trade_cost = position.diff().abs()
    trade_cost.iloc[0] = position.iloc[0]
    strategy_returns = returns * position - trade_cost * fee
    return strategy_returns, position


def compute_strategy_cagr(
    raw: pd.DataFrame, settings: dict, start_date: dt.date, fee: float = 0.003
) -> float:
    data = compute_regime_indicators(raw, settings)
    _, rules, valid_mask = classify_downtrend(data, settings)
    valid_index = valid_mask[valid_mask].index
    if valid_index.empty:
        return np.nan
    effective_start = max(pd.Timestamp(start_date), valid_index[0])
    data = data.loc[data.index >= effective_start].copy()
    if data.empty:
        return np.nan
    weekly_ret = data["Close"].pct_change()
    active_regimes = set(settings["priority_order"])
    buy_signal = pd.Series(False, index=data.index)
    sell_signal = pd.Series(False, index=data.index)
    if "fast_up_trend" in active_regimes:
        buy_signal |= rules["fast_up_trend"].reindex(data.index, fill_value=False)
    if "slow_up_trend" in active_regimes:
        buy_signal |= rules["slow_up_trend"].reindex(data.index, fill_value=False)
    if "panic" in active_regimes:
        buy_signal |= rules["panic"].reindex(data.index, fill_value=False)
    if "fast_down_trend" in active_regimes:
        sell_signal |= rules["fast_down_trend"].reindex(data.index, fill_value=False)
    if "slow_down_trend" in active_regimes:
        sell_signal |= rules["slow_down_trend"].reindex(data.index, fill_value=False)
    strategy_returns, _ = compute_trade_returns(weekly_ret, buy_signal, sell_signal, fee=fee)
    stats = compute_equity_stats(strategy_returns)
    return stats.get("CAGR", np.nan)


def compute_regime_summary(returns: pd.Series, labels: pd.Series, order: list[str]) -> pd.DataFrame:
    rows = []
    total_weeks = len(labels)
    for name in order:
        mask = labels == name
        sample = returns[mask]
        stats = compute_equity_stats(sample)
        rows.append(
            {
                "Regime": name,
                "Weeks": int(mask.sum()),
                "Share": (mask.sum() / total_weeks) if total_weeks else np.nan,
                "Total Return": stats.get("Total Return", np.nan),
                "CAGR": stats.get("CAGR", np.nan),
                "Max Drawdown": stats.get("Max Drawdown", np.nan),
                "Sharpe": stats.get("Sharpe", np.nan),
                "Volatility": stats.get("Volatility", np.nan),
            }
        )
    return pd.DataFrame(rows)


def format_pct(value: float) -> str:
    return "N/A" if np.isnan(value) else f"{value:.2%}"


def format_num(value: float) -> str:
    return "N/A" if np.isnan(value) else f"{value:.2f}"


def parse_priority_order(raw: str, options: list[str]) -> list[str]:
    tokens = [token.strip().lower() for token in re.split(r"[,\s]+", raw) if token.strip()]
    seen = set()
    ordered = []
    for token in tokens:
        if token in options and token not in seen:
            ordered.append(token)
            seen.add(token)
    return ordered


def get_state_or_default(key: str, default):
    return st.session_state.get(key, default)


WARMUP_KEYS = [
    "ma_fast_len",
    "ma_slow_len",
    "atr_len",
    "atr_fast_len",
    "adx_len",
    "adx_fast_len",
    "rsi_len",
    "rsi_fast_len",
    "vol_len",
    "vol_fast_len",
    "volume_ma_len",
    "volume_ma_fast_len",
    "return_window",
]

OPT_PARAM_SPECS = [
    {"key": "ma_fast_len", "label": "MA fast", "type": "int", "min": 2, "max": 300, "step": 1},
    {"key": "ma_slow_len", "label": "MA slow", "type": "int", "min": 10, "max": 400, "step": 1},
    {"key": "rsi_len", "label": "RSI slow", "type": "int", "min": 2, "max": 100, "step": 1},
    {"key": "rsi_fast_len", "label": "RSI fast", "type": "int", "min": 2, "max": 100, "step": 1},
    {"key": "adx_len", "label": "ADX slow", "type": "int", "min": 2, "max": 100, "step": 1},
    {"key": "adx_fast_len", "label": "ADX fast", "type": "int", "min": 2, "max": 100, "step": 1},
    {"key": "atr_len", "label": "ATR slow", "type": "int", "min": 2, "max": 100, "step": 1},
    {"key": "atr_fast_len", "label": "ATR fast", "type": "int", "min": 2, "max": 100, "step": 1},
    {"key": "vol_len", "label": "Vol slow", "type": "int", "min": 2, "max": 104, "step": 1},
    {"key": "vol_fast_len", "label": "Vol fast", "type": "int", "min": 2, "max": 104, "step": 1},
    {"key": "return_window", "label": "Panic return window", "type": "int", "min": 2, "max": 52, "step": 1},
    {"key": "volume_ma_len", "label": "Volume MA slow", "type": "int", "min": 2, "max": 60, "step": 1},
    {"key": "volume_ma_fast_len", "label": "Volume MA fast", "type": "int", "min": 2, "max": 60, "step": 1},
    {"key": "trend_adx_threshold", "label": "Trend ADX threshold", "type": "float", "min": 1.0, "max": 100.0, "step": 1.0},
    {"key": "panic_rsi_max", "label": "Panic RSI max", "type": "float", "min": 0.0, "max": 50.0, "step": 1.0},
    {"key": "panic_return_max", "label": "Panic return max", "type": "float", "min": -1.0, "max": 0.0, "step": 0.01},
    {"key": "panic_volume_mult", "label": "Panic volume mult", "type": "float", "min": 1.0, "max": 10.0, "step": 0.1},
]
OPT_SPEC_MAP = {spec["key"]: spec for spec in OPT_PARAM_SPECS}


def calc_warmup_weeks(values: dict) -> int:
    base = max(values.get(key, 0) for key in WARMUP_KEYS)
    return int(base + 120)


def build_param_values(min_val: float, max_val: float, step: float, kind: str) -> list:
    if step <= 0 or max_val < min_val:
        return []
    if kind == "int":
        step_int = max(1, int(step))
        return list(range(int(min_val), int(max_val) + 1, step_int))
    count = int(round((max_val - min_val) / step))
    values = [min_val + i * step for i in range(count + 1)]
    return [float(f"{val:.6f}") for val in values if min_val - 1e-12 <= val <= max_val + 1e-12]


def load_presets(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {name: values for name, values in data.items() if isinstance(values, dict)}


def save_presets(path: Path, presets: dict) -> None:
    path.write_text(json.dumps(presets, indent=2, sort_keys=True))


st.title("Weekly Market Regime Dashboard")
st.caption("Label weekly downtrend regimes and compare performance by regime.")
st.caption("Lookbacks are in weeks and can be edited in the sidebar.")

today = dt.date.today()
default_start = today - dt.timedelta(days=365 * 10)
presets_path = Path("regime_presets.json")
presets = load_presets(presets_path)
preset_keys = [
    "ma_fast_len",
    "ma_slow_len",
    "rsi_len",
    "rsi_fast_len",
    "adx_len",
    "adx_fast_len",
    "atr_len",
    "atr_fast_len",
    "vol_len",
    "vol_fast_len",
    "return_window",
    "volume_ma_len",
    "volume_ma_fast_len",
    "trend_adx_threshold",
    "panic_rsi_max",
    "panic_return_max",
    "panic_volume_mult",
    "freeze_down_on_panic",
    "priority_raw",
]
opt_param_keys = list(OPT_SPEC_MAP.keys())
if "pending_preset" in st.session_state:
    pending_name = st.session_state.pop("pending_preset")
    pending = presets.get(pending_name)
    if pending:
        for key in preset_keys:
            if key in pending:
                st.session_state[key] = pending[key]
        st.session_state["preset_notice"] = f"Loaded preset: {pending_name}"
    else:
        st.session_state["preset_warning"] = "Preset not found."
if "pending_opt_params" in st.session_state:
    pending = st.session_state.pop("pending_opt_params")
    for key in opt_param_keys:
        if key in pending:
            spec = OPT_SPEC_MAP.get(key)
            value = pending[key]
            if spec and spec["type"] == "int":
                value = int(value)
            elif spec:
                value = float(value)
            st.session_state[key] = value
    st.session_state["opt_notice"] = "Applied optimized parameters."
regime_options = [
    "fast_up_trend",
    "slow_up_trend",
    "slow_down_trend",
    "fast_down_trend",
    "panic",
]

with st.sidebar:
    st.header("Parameters")
    ticker = st.text_input("Benchmark ticker", value=get_state_or_default("ticker", "SPY"), key="ticker")
    start_date = st.date_input(
        "Start date", value=get_state_or_default("start_date", default_start), key="start_date"
    )
    end_date = st.date_input(
        "End date",
        value=get_state_or_default("end_date", today),
        min_value=dt.date(2000, 1, 1),
        max_value=dt.date(2040, 12, 31),
        key="end_date",
    )
    show_markers = st.checkbox("Show regime markers", value=get_state_or_default("show_markers", True), key="show_markers")
    st.subheader("Regime settings")
    ma_cols = st.columns(2)
    ma_fast_len = ma_cols[0].number_input(
        "MA fast length (weeks)",
        min_value=2,
        max_value=300,
        value=int(get_state_or_default("ma_fast_len", 50)),
        step=1,
        key="ma_fast_len",
    )
    ma_slow_len = ma_cols[1].number_input(
        "MA slow length (weeks)",
        min_value=10,
        max_value=400,
        value=int(get_state_or_default("ma_slow_len", 200)),
        step=1,
        key="ma_slow_len",
    )
    rsi_cols = st.columns(2)
    rsi_len = rsi_cols[0].number_input(
        "RSI slow length (weeks)",
        min_value=2,
        max_value=100,
        value=int(get_state_or_default("rsi_len", 14)),
        step=1,
        key="rsi_len",
    )
    rsi_fast_len = rsi_cols[1].number_input(
        "RSI fast length (weeks)",
        min_value=2,
        max_value=100,
        value=int(get_state_or_default("rsi_fast_len", 7)),
        step=1,
        key="rsi_fast_len",
    )
    adx_cols = st.columns(2)
    adx_len = adx_cols[0].number_input(
        "ADX slow length (weeks)",
        min_value=2,
        max_value=100,
        value=int(get_state_or_default("adx_len", 14)),
        step=1,
        key="adx_len",
    )
    adx_fast_len = adx_cols[1].number_input(
        "ADX fast length (weeks)",
        min_value=2,
        max_value=100,
        value=int(get_state_or_default("adx_fast_len", 7)),
        step=1,
        key="adx_fast_len",
    )
    atr_cols = st.columns(2)
    atr_len = atr_cols[0].number_input(
        "ATR slow length (weeks)",
        min_value=2,
        max_value=100,
        value=int(get_state_or_default("atr_len", 14)),
        step=1,
        key="atr_len",
    )
    atr_fast_len = atr_cols[1].number_input(
        "ATR fast length (weeks)",
        min_value=2,
        max_value=100,
        value=int(get_state_or_default("atr_fast_len", 7)),
        step=1,
        key="atr_fast_len",
    )
    vol_cols = st.columns(2)
    vol_len = vol_cols[0].number_input(
        "Volatility slow length (weeks)",
        min_value=2,
        max_value=104,
        value=int(get_state_or_default("vol_len", 20)),
        step=1,
        key="vol_len",
    )
    vol_fast_len = vol_cols[1].number_input(
        "Volatility fast length (weeks)",
        min_value=2,
        max_value=104,
        value=int(get_state_or_default("vol_fast_len", 10)),
        step=1,
        key="vol_fast_len",
    )
    return_window = st.number_input(
        "Panic return window (weeks)",
        min_value=2,
        max_value=52,
        value=int(get_state_or_default("return_window", 5)),
        step=1,
        key="return_window",
    )
    volume_cols = st.columns(2)
    volume_ma_len = volume_cols[0].number_input(
        "Volume MA slow length (weeks)",
        min_value=2,
        max_value=60,
        value=int(get_state_or_default("volume_ma_len", 20)),
        step=1,
        key="volume_ma_len",
    )
    volume_ma_fast_len = volume_cols[1].number_input(
        "Volume MA fast length (weeks)",
        min_value=2,
        max_value=60,
        value=int(get_state_or_default("volume_ma_fast_len", 10)),
        step=1,
        key="volume_ma_fast_len",
    )
    trend_adx_threshold = st.number_input(
        "Trend ADX threshold",
        min_value=1.0,
        max_value=100.0,
        value=float(get_state_or_default("trend_adx_threshold", 40.0)),
        step=1.0,
        key="trend_adx_threshold",
    )
    panic_rsi_max = st.number_input(
        "Panic RSI max",
        min_value=0.0,
        max_value=50.0,
        value=float(get_state_or_default("panic_rsi_max", 25.0)),
        step=1.0,
        key="panic_rsi_max",
    )
    panic_return_max = st.number_input(
        "Panic return max",
        min_value=-1.0,
        max_value=0.0,
        value=float(get_state_or_default("panic_return_max", -0.10)),
        step=0.01,
        format="%.2f",
        key="panic_return_max",
    )
    panic_volume_mult = st.number_input(
        "Panic volume multiple",
        min_value=1.0,
        max_value=10.0,
        value=float(get_state_or_default("panic_volume_mult", 2.0)),
        step=0.1,
        format="%.1f",
        key="panic_volume_mult",
    )
    freeze_down_on_panic = st.checkbox(
        "Freeze down-trend after panic until up-trend",
        value=bool(get_state_or_default("freeze_down_on_panic", True)),
        key="freeze_down_on_panic",
    )
    priority_raw = st.text_input(
        "Regime priority order (comma or space separated)",
        value=get_state_or_default("priority_raw", ", ".join(regime_options)),
        help="Only listed regimes are active; later rules override earlier ones; 'none' is implied.",
        key="priority_raw",
    )
    st.caption(f"Available regimes: {', '.join(regime_options)}")
    st.subheader("Presets")
    preset_name = st.text_input("Preset name", value="", key="preset_name")
    save_preset = st.button("Save preset", key="save_preset")
    preset_options = ["(select)"] + sorted(presets.keys())
    preset_to_load = st.selectbox("Load preset", options=preset_options, key="preset_to_load")
    load_preset = st.button("Load preset", key="load_preset")
    optimize_button = st.button("Optimize parameters", key="optimize_button")

priority_order = parse_priority_order(priority_raw, regime_options)
if save_preset:
    name = preset_name.strip()
    if not name:
        st.sidebar.warning("Preset name is required.")
    else:
        presets[name] = {key: st.session_state.get(key) for key in preset_keys}
        save_presets(presets_path, presets)
        st.session_state["preset_notice"] = f"Saved preset: {name}"
        st.rerun()

if load_preset:
    if preset_to_load == "(select)":
        st.sidebar.warning("Choose a preset to load.")
    else:
        st.session_state["pending_preset"] = preset_to_load
        st.rerun()

if "preset_warning" in st.session_state:
    st.sidebar.warning(st.session_state.pop("preset_warning"))
if "preset_notice" in st.session_state:
    st.sidebar.success(st.session_state.pop("preset_notice"))
if "opt_notice" in st.session_state:
    st.sidebar.success(st.session_state.pop("opt_notice"))

settings = {
    "ma_fast_len": int(ma_fast_len),
    "ma_slow_len": int(ma_slow_len),
    "rsi_len": int(rsi_len),
    "rsi_fast_len": int(rsi_fast_len),
    "adx_len": int(adx_len),
    "adx_fast_len": int(adx_fast_len),
    "atr_len": int(atr_len),
    "atr_fast_len": int(atr_fast_len),
    "vol_len": int(vol_len),
    "vol_fast_len": int(vol_fast_len),
    "return_window": int(return_window),
    "volume_ma_len": int(volume_ma_len),
    "volume_ma_fast_len": int(volume_ma_fast_len),
    "trend_adx_threshold": float(trend_adx_threshold),
    "panic_rsi_max": float(panic_rsi_max),
    "panic_return_max": float(panic_return_max),
    "panic_volume_mult": float(panic_volume_mult),
    "freeze_down_on_panic": bool(freeze_down_on_panic),
    "priority_order": priority_order,
}


if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

warmup_weeks = calc_warmup_weeks(settings)
warmup_start = start_date - dt.timedelta(weeks=int(warmup_weeks))
raw = load_weekly_data(ticker, warmup_start, end_date)
if raw.empty:
    st.error("No data returned. Check the ticker or date range.")
    st.stop()

if settings["ma_fast_len"] >= settings["ma_slow_len"]:
    st.warning("MA fast length should be smaller than MA slow length.")


@st.dialog("Parameter Optimization")
def optimization_dialog() -> None:
    st.write("Optimize in-sample CAGR using the current date range.")
    st.caption(f"Date range: {start_date} to {end_date}")
    search_mode = st.selectbox("Search method", ["Grid", "Random"], key="opt_search_mode")
    max_trials = st.number_input("Max trials", min_value=1, max_value=5000, value=200, step=1, key="opt_max_trials")
    seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42, step=1, key="opt_seed")
    range_pct = st.number_input(
        "Range around current (%)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0,
        key="opt_range_pct",
    )

    prev_range = st.session_state.get("opt_prev_range_pct")
    if st.session_state.pop("opt_reset_ranges", False) or prev_range is None or not math.isclose(prev_range, range_pct):
        pct = range_pct / 100.0
        for spec in OPT_PARAM_SPECS:
            key = spec["key"]
            current = st.session_state.get(key, settings.get(key))
            if spec["type"] == "int":
                current_val = int(current)
                min_val = max(spec["min"], math.floor(current_val * (1 - pct)))
                max_val = min(spec["max"], math.ceil(current_val * (1 + pct)))
                if min_val > max_val:
                    min_val = max_val = current_val
                st.session_state[f"opt_min_{key}"] = int(min_val)
                st.session_state[f"opt_max_{key}"] = int(max_val)
                st.session_state[f"opt_step_{key}"] = int(spec["step"])
            else:
                current_val = float(current)
                delta = abs(current_val) * pct
                min_val = max(spec["min"], current_val - delta)
                max_val = min(spec["max"], current_val + delta)
                if min_val > max_val:
                    min_val = max_val = current_val
                st.session_state[f"opt_min_{key}"] = float(min_val)
                st.session_state[f"opt_max_{key}"] = float(max_val)
                st.session_state[f"opt_step_{key}"] = float(spec["step"])
        st.session_state["opt_prev_range_pct"] = range_pct

    st.markdown("**Tune parameters**")
    header = st.columns([2.4, 1, 1, 1])
    header[0].markdown("Parameter")
    header[1].markdown("Min")
    header[2].markdown("Max")
    header[3].markdown("Step")

    param_inputs = {}
    spec_map = OPT_SPEC_MAP
    for spec in OPT_PARAM_SPECS:
        key = spec["key"]
        current = st.session_state.get(key, settings.get(key))
        if spec["type"] == "int":
            current_val = int(current)
        else:
            current_val = float(current)
        current_val = min(max(current_val, spec["min"]), spec["max"])
        row = st.columns([2.4, 1, 1, 1])
        enabled = row[0].checkbox(spec["label"], value=False, key=f"opt_enable_{key}")
        min_val = row[1].number_input(
            "min",
            min_value=spec["min"],
            max_value=spec["max"],
            value=current_val,
            step=spec["step"],
            key=f"opt_min_{key}",
            label_visibility="collapsed",
        )
        max_val = row[2].number_input(
            "max",
            min_value=spec["min"],
            max_value=spec["max"],
            value=current_val,
            step=spec["step"],
            key=f"opt_max_{key}",
            label_visibility="collapsed",
        )
        step_max = max(spec["step"], spec["max"] - spec["min"])
        step_val = row[3].number_input(
            "step",
            min_value=spec["step"],
            max_value=step_max,
            value=spec["step"],
            step=spec["step"],
            key=f"opt_step_{key}",
            label_visibility="collapsed",
        )
        param_inputs[key] = {"enabled": enabled, "min": min_val, "max": max_val, "step": step_val}

    run_opt = st.button("Run optimization", key="run_optimization")
    if not run_opt:
        return

    param_values = {}
    for key, values in param_inputs.items():
        if not values["enabled"]:
            continue
        spec = spec_map[key]
        if values["max"] < values["min"]:
            st.warning(f"{spec['label']}: max must be >= min.")
            return
        grid = build_param_values(values["min"], values["max"], values["step"], spec["type"])
        if not grid:
            st.warning(f"{spec['label']}: invalid range or step.")
            return
        param_values[key] = grid

    if not param_values:
        st.warning("Enable at least one parameter to tune.")
        return

    combinations = 1
    for grid in param_values.values():
        combinations *= len(grid)

    warmup_values = {key: settings[key] for key in WARMUP_KEYS}
    for key, grid in param_values.items():
        if key in warmup_values:
            warmup_values[key] = max(grid)
    warmup_weeks = calc_warmup_weeks(warmup_values)
    warmup_start = start_date - dt.timedelta(weeks=int(warmup_weeks))
    raw_opt = raw
    if raw_opt.empty or raw_opt.index.min() > pd.Timestamp(warmup_start):
        raw_opt = load_weekly_data(ticker, warmup_start, end_date)
    if raw_opt.empty:
        st.error("No data returned for the optimization range.")
        return

    candidates = []
    if search_mode == "Grid" and combinations <= max_trials:
        keys = list(param_values.keys())
        for combo in itertools.product(*(param_values[key] for key in keys)):
            candidates.append(dict(zip(keys, combo)))
    else:
        if search_mode == "Grid":
            st.info("Grid size exceeds max trials; using random sampling.")
        rng = random.Random(seed)
        keys = list(param_values.keys())
        for _ in range(int(max_trials)):
            candidates.append({key: rng.choice(param_values[key]) for key in keys})

    total = len(candidates)
    progress = st.progress(0.0) if total > 1 else None
    results = []
    for idx, candidate in enumerate(candidates, 1):
        test_settings = settings.copy()
        for key, value in candidate.items():
            spec = spec_map[key]
            test_settings[key] = int(value) if spec["type"] == "int" else float(value)
        if test_settings["ma_fast_len"] >= test_settings["ma_slow_len"]:
            continue
        cagr = compute_strategy_cagr(raw_opt, test_settings, start_date)
        if not np.isnan(cagr):
            results.append({"CAGR": cagr, **candidate})
        if progress and idx % 5 == 0:
            progress.progress(idx / total)
    if progress:
        progress.empty()

    if not results:
        st.warning("No valid trials produced a CAGR.")
        return

    result_df = pd.DataFrame(results).sort_values("CAGR", ascending=False)
    display = result_df.copy()
    display["CAGR"] = display["CAGR"].apply(format_pct)
    st.dataframe(display.head(20))

    best_row = result_df.iloc[0]
    best_params = {}
    for key in param_values.keys():
        spec = spec_map[key]
        value = best_row[key]
        best_params[key] = int(value) if spec["type"] == "int" else float(value)
    st.write(f"Best CAGR: {format_pct(best_row['CAGR'])}")
    if st.button("Apply best parameters", key="apply_best_params"):
        st.session_state["pending_opt_params"] = best_params
        st.rerun()


if optimize_button:
    st.session_state["opt_reset_ranges"] = True
    optimization_dialog()

data = compute_regime_indicators(raw, settings)
regime_labels, regime_rules, valid_mask = classify_downtrend(data, settings)
data["Regime"] = regime_labels
valid_index = valid_mask[valid_mask].index
if valid_index.empty:
    st.error("Not enough data to compute indicators with the current settings.")
    st.stop()
valid_start = valid_index[0]
effective_start = max(pd.Timestamp(start_date), valid_start)
if effective_start > pd.Timestamp(start_date):
    st.info(f"Start date adjusted to {effective_start.date()} to allow indicator warm-up.")
data = data.loc[data.index >= effective_start].copy()

weekly_ret = data["Close"].pct_change()
active_regimes = set(settings["priority_order"])
buy_signal = pd.Series(False, index=data.index)
sell_signal = pd.Series(False, index=data.index)
if "fast_up_trend" in active_regimes:
    buy_signal |= regime_rules["fast_up_trend"].reindex(data.index, fill_value=False)
if "slow_up_trend" in active_regimes:
    buy_signal |= regime_rules["slow_up_trend"].reindex(data.index, fill_value=False)
if "panic" in active_regimes:
    buy_signal |= regime_rules["panic"].reindex(data.index, fill_value=False)
if "fast_down_trend" in active_regimes:
    sell_signal |= regime_rules["fast_down_trend"].reindex(data.index, fill_value=False)
if "slow_down_trend" in active_regimes:
    sell_signal |= regime_rules["slow_down_trend"].reindex(data.index, fill_value=False)
strategy_returns, _ = compute_trade_returns(weekly_ret, buy_signal, sell_signal, fee=0.003)
overall = compute_equity_stats(weekly_ret)
strategy_stats = compute_equity_stats(strategy_returns)
strategy_equity = None
if strategy_stats:
    strategy_equity = strategy_stats["Equity"].reindex(data.index) * 10000.0

label_order = settings["priority_order"] + ["none"]
summary = compute_regime_summary(weekly_ret, data["Regime"], label_order)
priority_text = " -> ".join(settings["priority_order"]) or "none"

st.subheader("Performance metrics")
st.caption("Signal backtest uses fast/slow up-trend to enter and fast/slow down-trend to exit. Fees: 0.30% per side.")
if overall or strategy_stats:
    metrics = [
        "Total Return",
        "CAGR",
        "Max Drawdown",
        "Sharpe (ann.)",
        "Volatility (ann.)",
    ]
    combined = pd.DataFrame({"Metric": metrics})
    if overall:
        combined["Buy & Hold"] = [
            format_pct(overall["Total Return"]),
            format_pct(overall["CAGR"]),
            format_pct(overall["Max Drawdown"]),
            format_num(overall["Sharpe"]),
            format_pct(overall["Volatility"]),
        ]
    if strategy_stats:
        combined["Signal Backtest"] = [
            format_pct(strategy_stats["Total Return"]),
            format_pct(strategy_stats["CAGR"]),
            format_pct(strategy_stats["Max Drawdown"]),
            format_num(strategy_stats["Sharpe"]),
            format_pct(strategy_stats["Volatility"]),
        ]
    st.table(combined)
else:
    st.info("Not enough data to compute performance metrics.")

st.subheader("Price and regimes")
fig = make_subplots(
    rows=6,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.42, 0.14, 0.14, 0.1, 0.1, 0.1],
    subplot_titles=("Price with MAs", "RSI", "ADX", "ATR%", "Volatility", "Volume"),
    specs=[
        [{"secondary_y": True}],
        [{"secondary_y": False}],
        [{"secondary_y": False}],
        [{"secondary_y": False}],
        [{"secondary_y": False}],
        [{"secondary_y": False}],
    ],
)

fig.add_trace(
    go.Scatter(x=data.index, y=data["Close"], name="Close", line=dict(width=1.4)),
    row=1,
    col=1,
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["MA_FAST"],
        name=f"MA{settings['ma_fast_len']}",
        line=dict(width=1),
    ),
    row=1,
    col=1,
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["MA_SLOW"],
        name=f"MA{settings['ma_slow_len']}",
        line=dict(width=1),
    ),
    row=1,
    col=1,
    secondary_y=False,
)
if strategy_equity is not None:
    fig.add_trace(
        go.Scatter(
            x=strategy_equity.index,
            y=strategy_equity,
            name="Equity (10k)",
            line=dict(width=1.2, color="#2b8cbe"),
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

regime_colors = {
    "fast_down_trend": "#8c564b",
    "slow_down_trend": "#c49c94",
    "fast_up_trend": "#17becf",
    "slow_up_trend": "#9edae5",
    "panic": "#d62728",
    "none": "#7f7f7f",
}

if show_markers:
    for regime in label_order:
        if regime == "none":
            continue
        mask = data["Regime"] == regime
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=data.index[mask],
                    y=data["Close"][mask],
                    mode="markers",
                    marker=dict(size=6, color=regime_colors.get(regime, "#7f7f7f")),
                    name=regime,
                ),
                row=1,
                col=1,
            )

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["RSI"],
        name=f"RSI{settings['rsi_len']}",
        line=dict(width=1),
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["RSI_FAST"],
        name=f"RSI{settings['rsi_fast_len']}",
        line=dict(width=1),
    ),
    row=2,
    col=1,
)
fig.add_hline(y=70, line=dict(width=1, dash="dot", color="gray"), row=2, col=1)
fig.add_hline(y=30, line=dict(width=1, dash="dot", color="gray"), row=2, col=1)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["ADX"],
        name=f"ADX{settings['adx_len']}",
        line=dict(width=1),
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["ADX_FAST"],
        name=f"ADX{settings['adx_fast_len']}",
        line=dict(width=1),
    ),
    row=3,
    col=1,
)
fig.add_hline(y=20, line=dict(width=1, dash="dot", color="gray"), row=3, col=1)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["ATR_PCT"],
        name=f"ATR%{settings['atr_len']}",
        line=dict(width=1),
    ),
    row=4,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["ATR_PCT_FAST"],
        name=f"ATR%{settings['atr_fast_len']}",
        line=dict(width=1),
    ),
    row=4,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["VOL"],
        name=f"Vol{settings['vol_len']}",
        line=dict(width=1),
    ),
    row=5,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["VOL_FAST"],
        name=f"Vol{settings['vol_fast_len']}",
        line=dict(width=1),
    ),
    row=5,
    col=1,
)

fig.add_trace(
    go.Bar(
        x=data.index,
        y=data["Volume"],
        name="Volume",
        marker_color="rgba(120, 120, 120, 0.5)",
    ),
    row=6,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["Volume_MA"],
        name=f"Volume_MA{settings['volume_ma_len']}",
        line=dict(width=1.2, color="#1f77b4"),
    ),
    row=6,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["Volume_MA_FAST"],
        name=f"Volume_MA{settings['volume_ma_fast_len']}",
        line=dict(width=1.2, color="#ff7f0e"),
    ),
    row=6,
    col=1,
)

fig.update_layout(height=1050, legend=dict(orientation="h"), yaxis_type="log", yaxis2_type="log")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Regime definitions")
st.markdown(
    "\n".join(
        [
            "Rules are evaluated on weekly bars.",
            (
                f"Priority order (later rules override earlier ones): `{priority_text}`."
            ),
            "After `slow_down_trend`, up-trend regimes are suppressed until MA_slow > prior MA_slow.",
            "Optional: if enabled, `panic` freezes down-trend regimes until an up-trend fires.",
            "",
            (
                f"- `fast_down_trend`: MA{settings['ma_fast_len']} < prior MA{settings['ma_fast_len']} AND "
                f"ADX{settings['adx_fast_len']} < {settings['trend_adx_threshold']:.0f} AND "
                "ATR% fast 3w MA > ATR% fast shifted-4w 3w MA AND "
                "Volume fast MA > Volume fast shifted-4w."
            ),
            (
                f"- `slow_down_trend`: MA{settings['ma_slow_len']} < prior MA{settings['ma_slow_len']} AND "
                f"ADX{settings['adx_len']} < {settings['trend_adx_threshold']:.0f} AND "
                "ATR% slow 5w MA > ATR% slow shifted-6w 5w MA AND "
                "Volume slow MA > Volume slow shifted-6w."
            ),
            (
                f"- `fast_up_trend`: MA{settings['ma_fast_len']} > prior MA{settings['ma_fast_len']} AND "
                f"ADX{settings['adx_fast_len']} > {settings['trend_adx_threshold']:.0f} AND "
                "ATR% fast 2w MA < ATR% fast shifted-2w 2w MA."
            ),
            (
                f"- `slow_up_trend`: MA{settings['ma_slow_len']} > prior MA{settings['ma_slow_len']} AND "
                f"ADX{settings['adx_len']} < {settings['trend_adx_threshold']:.0f} AND "
                "ATR% slow 5w MA > ATR% slow shifted-6w 5w MA."
            ),
            (
                f"- `panic`: RSI{settings['rsi_len']} < {settings['panic_rsi_max']:.0f} AND "
                f"{settings['return_window']}w return < {settings['panic_return_max']:.2f} AND "
                f"Volume > {settings['panic_volume_mult']:.1f}x Volume_MA{settings['volume_ma_len']}."
            ),
            "- `none`: no rule matched.",
        ]
    )
)

st.subheader("Regime metrics")
if summary.empty:
    st.info("Not enough data to compute regime metrics.")
else:
    display = summary.copy()
    display["Share"] = display["Share"].apply(format_pct)
    display["Total Return"] = display["Total Return"].apply(format_pct)
    display["CAGR"] = display["CAGR"].apply(format_pct)
    display["Max Drawdown"] = display["Max Drawdown"].apply(format_pct)
    display["Sharpe"] = display["Sharpe"].apply(format_num)
    display["Volatility"] = display["Volatility"].apply(format_pct)
    st.table(display)

st.subheader("Regime label sample")
st.dataframe(data[["Close", "Regime"]].tail(12))
