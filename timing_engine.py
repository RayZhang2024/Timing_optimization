import datetime as dt
import re

import numpy as np
import pandas as pd


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


def prepare_ohlcv_input(
    data: pd.DataFrame, *, ticker: str | None = None, resample_rule: str | None = "W-FRI"
) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    df = data.copy()
    df = normalize_ohlcv_columns(df, ticker or "")

    rename_map = {}
    for col in df.columns:
        key = str(col).strip()
        key_clean = re.sub(r"[_\s]+", " ", key.lower())
        if key_clean in {"open", "high", "low", "close"}:
            rename_map[col] = key_clean.title()
        elif key_clean in {"adj close", "adjclose"}:
            rename_map[col] = "Adj Close"
        elif key_clean in {"volume", "vol"}:
            rename_map[col] = "Volume"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df[required]
    if "Date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("Date")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=required)
    if resample_rule:
        df = df.resample(resample_rule).agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
    return df.dropna()


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
        (
            (df["MA_FAST"] < df["MA_FAST"].shift(1))
            & (atr_fast_short > atr_fast_long)
            & (df["Volume_MA_FAST"] > vol_fast_shift6_roll15)
            & (df["MA_FAST"] < df["MA_SLOW"])
        )
        | (
            (df["Close"] < df["MA_SLOW"])
            & (df["Close"].shift(1) < df["MA_SLOW"])
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


def build_trade_signals(
    rules: dict[str, pd.Series], index: pd.Index, active_regimes: list[str] | None
) -> tuple[pd.Series, pd.Series]:
    active = set(active_regimes or rules.keys())
    buy_signal = pd.Series(False, index=index)
    sell_signal = pd.Series(False, index=index)
    if "fast_up_trend" in active:
        buy_signal |= rules["fast_up_trend"].reindex(index, fill_value=False)
    if "slow_up_trend" in active:
        buy_signal |= rules["slow_up_trend"].reindex(index, fill_value=False)
    if "panic" in active:
        buy_signal |= rules["panic"].reindex(index, fill_value=False)
    if "fast_down_trend" in active:
        sell_signal |= rules["fast_down_trend"].reindex(index, fill_value=False)
    if "slow_down_trend" in active:
        sell_signal |= rules["slow_down_trend"].reindex(index, fill_value=False)
    return buy_signal, sell_signal


def compute_signals_window(
    raw: pd.DataFrame,
    settings: dict,
    window_start: dt.date,
    window_end: dt.date,
    *,
    end_exclusive: bool = False,
) -> dict:
    if raw.empty:
        return {
            "data": pd.DataFrame(),
            "labels": pd.Series(dtype=object),
            "rules": {},
            "valid_mask": pd.Series(dtype=bool),
            "effective_start": None,
            "weekly_returns": pd.Series(dtype=float),
            "buy_signal": pd.Series(dtype=bool),
            "sell_signal": pd.Series(dtype=bool),
        }
    data = compute_regime_indicators(raw, settings)
    labels, rules, valid_mask = classify_downtrend(data, settings)
    valid_index = valid_mask[valid_mask].index
    if valid_index.empty:
        return {
            "data": pd.DataFrame(),
            "labels": labels.iloc[0:0],
            "rules": rules,
            "valid_mask": valid_mask,
            "effective_start": None,
            "weekly_returns": pd.Series(dtype=float),
            "buy_signal": pd.Series(dtype=bool),
            "sell_signal": pd.Series(dtype=bool),
        }
    effective_start = max(pd.Timestamp(window_start), valid_index[0])
    window_end_ts = pd.Timestamp(window_end)
    if end_exclusive:
        data = data.loc[(data.index >= effective_start) & (data.index < window_end_ts)].copy()
    else:
        data = data.loc[(data.index >= effective_start) & (data.index <= window_end_ts)].copy()
    if data.empty:
        return {
            "data": data,
            "labels": labels.reindex(data.index),
            "rules": rules,
            "valid_mask": valid_mask,
            "effective_start": effective_start,
            "weekly_returns": pd.Series(dtype=float),
            "buy_signal": pd.Series(dtype=bool),
            "sell_signal": pd.Series(dtype=bool),
        }
    weekly_returns = data["Close"].pct_change()
    buy_signal, sell_signal = build_trade_signals(rules, data.index, settings.get("priority_order"))
    return {
        "data": data,
        "labels": labels.reindex(data.index, fill_value="none"),
        "rules": rules,
        "valid_mask": valid_mask,
        "effective_start": effective_start,
        "weekly_returns": weekly_returns,
        "buy_signal": buy_signal,
        "sell_signal": sell_signal,
    }


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


def compute_partial_oos_stats(returns: pd.Series) -> dict:
    clean = returns.dropna()
    if len(clean) < 1:
        return {"Total Return": np.nan, "CAGR": np.nan}
    equity = (1 + clean).cumprod()
    total_return = equity.iloc[-1] - 1
    num_weeks = len(equity)
    cagr = (equity.iloc[-1] ** (52 / num_weeks)) - 1 if num_weeks else np.nan
    return {"Total Return": total_return, "CAGR": cagr}


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


def compute_strategy_returns_window(
    raw: pd.DataFrame,
    settings: dict,
    window_start: dt.date,
    window_end: dt.date,
    fee: float = 0.003,
    end_exclusive: bool = False,
) -> pd.Series:
    signals = compute_signals_window(
        raw, settings, window_start, window_end, end_exclusive=end_exclusive
    )
    weekly_ret = signals["weekly_returns"]
    if weekly_ret.empty:
        return pd.Series(dtype=float)
    strategy_returns, _ = compute_trade_returns(
        weekly_ret, signals["buy_signal"], signals["sell_signal"], fee=fee
    )
    return strategy_returns


def compute_strategy_cagr(
    raw: pd.DataFrame, settings: dict, start_date: dt.date, fee: float = 0.003
) -> float:
    if raw.empty:
        return np.nan
    window_end = raw.index.max() + pd.Timedelta(days=1)
    strategy_returns = compute_strategy_returns_window(
        raw, settings, start_date, window_end, fee=fee, end_exclusive=True
    )
    stats = compute_equity_stats(strategy_returns)
    return stats.get("CAGR", np.nan)


def compute_strategy_cagr_window(
    raw: pd.DataFrame,
    settings: dict,
    window_start: dt.date,
    window_end: dt.date,
    fee: float = 0.003,
    end_exclusive: bool = True,
) -> float:
    strategy_returns = compute_strategy_returns_window(
        raw, settings, window_start, window_end, fee=fee, end_exclusive=end_exclusive
    )
    stats = compute_equity_stats(strategy_returns)
    return stats.get("CAGR", np.nan)


def compute_strategy_objective_from_returns(returns: pd.Series) -> tuple[float, float, float]:
    stats = compute_equity_stats(returns)
    cagr = stats.get("CAGR", np.nan)
    max_dd = stats.get("Max Drawdown", np.nan)
    if np.isnan(cagr) or np.isnan(max_dd):
        return np.nan, cagr, max_dd
    objective = cagr + 0.1 * min(max_dd, -0.3)
    return objective, cagr, max_dd


def compute_strategy_objective(
    raw: pd.DataFrame, settings: dict, start_date: dt.date, fee: float = 0.003
) -> tuple[float, float, float]:
    if raw.empty:
        return np.nan, np.nan, np.nan
    window_end = raw.index.max() + pd.Timedelta(days=1)
    strategy_returns = compute_strategy_returns_window(
        raw, settings, start_date, window_end, fee=fee, end_exclusive=True
    )
    return compute_strategy_objective_from_returns(strategy_returns)


def compute_strategy_objective_window(
    raw: pd.DataFrame,
    settings: dict,
    window_start: dt.date,
    window_end: dt.date,
    fee: float = 0.003,
    end_exclusive: bool = True,
) -> tuple[float, float, float]:
    strategy_returns = compute_strategy_returns_window(
        raw, settings, window_start, window_end, fee=fee, end_exclusive=end_exclusive
    )
    return compute_strategy_objective_from_returns(strategy_returns)


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


def parse_priority_order(raw: str, options: list[str]) -> list[str]:
    tokens = [token.strip().lower() for token in re.split(r"[,\s]+", raw) if token.strip()]
    seen = set()
    ordered = []
    for token in tokens:
        if token in options and token not in seen:
            ordered.append(token)
            seen.add(token)
    return ordered


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
