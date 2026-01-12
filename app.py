import datetime as dt
import re

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


def classify_downtrend(df: pd.DataFrame, settings: dict) -> tuple[pd.Series, dict[str, pd.Series]]:
    label = pd.Series("none", index=df.index)

    atr_fast_short = df["ATR_PCT_FAST"].rolling(3).max()
    atr_fast_long = df["ATR_PCT_FAST"].shift(4).rolling(78).max()
    atr_slow_short = df["ATR_PCT"].rolling(5).mean()
    atr_slow_long = df["ATR_PCT"].shift(6).rolling(5).mean()

    fast_down_trend = (
        ((df["MA_FAST"] < df["MA_FAST"].shift(1))
        #& (df["ADX"].rolling(2).mean() > settings["trend_adx_threshold"])
        & (atr_fast_short > atr_fast_long)
        & (df["Volume_MA_FAST"] > df["Volume_MA_FAST"].shift(6).rolling(15).max())
        & (df["MA_FAST"] < df["MA_SLOW"]))
        | ((df["Close"] < df["MA_SLOW"]) & (df["Close"].shift(1) < df["MA_SLOW"])
        & (df["ATR_PCT_FAST"] > df["ATR_PCT_FAST"].shift(1).rolling(5).mean())
        & (df["Volume_MA_FAST"] > 1.1* df["Volume_MA_FAST"].shift(1).rolling(5).max())
        & (df["ADX"].rolling(2).mean() > settings["trend_adx_threshold"])
        & (df["RSI_FAST"] > 30)
        )
    )

    slow_down_trend = (
        (df["MA_SLOW"] < df["MA_SLOW"].shift(1))
        & (df["ADX"] > settings["trend_adx_threshold"])
        & (atr_slow_short > atr_slow_long)
        & (df["Volume_MA"] > df["Volume_MA"].shift(6))
        & (df["ATR_PCT"] > df["ATR_PCT"].shift(4))
        & (df["ATR_PCT"].shift(4) > df["ATR_PCT"].shift(12))
        & (df["ATR_PCT"].shift(12) > df["ATR_PCT"].shift(24))
        & (df["ATR_PCT"].shift(24) > df["ATR_PCT"].shift(30))
    )

    freeze_uptrend = pd.Series(False, index=df.index)
    freeze_active = False
    ma_slow_up = df["MA_SLOW"] > df["MA_SLOW"].shift(1)
    for idx in df.index:
        if slow_down_trend.loc[idx]:
            freeze_active = True
        if freeze_active and ma_slow_up.loc[idx]:
            freeze_active = False
        freeze_uptrend.loc[idx] = freeze_active

    fast_up_trend = (
        (df["MA_FAST"] > df["MA_FAST"].shift(1))
        #& (df["ADX_FAST"] > settings["trend_adx_threshold"])
        & (df["ATR_PCT_FAST"].rolling(2).mean() < df["ATR_PCT_FAST"].shift(2).rolling(2).mean())
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

    return label, rules


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


st.title("Weekly Market Regime Dashboard")
st.caption("Label weekly downtrend regimes and compare performance by regime.")
st.caption("Lookbacks are in weeks and can be edited in the sidebar.")

today = dt.date.today()
default_start = today - dt.timedelta(days=365 * 10)
regime_options = [
    "fast_up_trend",
    "slow_up_trend",
    "slow_down_trend",
    "fast_down_trend",
    "panic",
]

with st.sidebar:
    st.header("Parameters")
    ticker = st.text_input("Benchmark ticker", value="SPY")
    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input(
        "End date",
        value=today,
        min_value=dt.date(2000, 1, 1),
        max_value=dt.date(2040, 12, 31),
    )
    show_markers = st.checkbox("Show regime markers", value=True)
    st.subheader("Regime settings")
    ma_fast_len = st.number_input("MA fast length (weeks)", min_value=2, max_value=300, value=50, step=1)
    ma_slow_len = st.number_input("MA slow length (weeks)", min_value=10, max_value=400, value=200, step=1)
    rsi_len = st.number_input("RSI slow length (weeks)", min_value=2, max_value=100, value=14, step=1)
    rsi_fast_len = st.number_input("RSI fast length (weeks)", min_value=2, max_value=100, value=7, step=1)
    adx_len = st.number_input("ADX slow length (weeks)", min_value=2, max_value=100, value=14, step=1)
    adx_fast_len = st.number_input("ADX fast length (weeks)", min_value=2, max_value=100, value=7, step=1)
    atr_len = st.number_input("ATR slow length (weeks)", min_value=2, max_value=100, value=14, step=1)
    atr_fast_len = st.number_input("ATR fast length (weeks)", min_value=2, max_value=100, value=7, step=1)
    vol_len = st.number_input("Volatility slow length (weeks)", min_value=2, max_value=104, value=20, step=1)
    vol_fast_len = st.number_input("Volatility fast length (weeks)", min_value=2, max_value=104, value=10, step=1)
    return_window = st.number_input("Panic return window (weeks)", min_value=2, max_value=52, value=5, step=1)
    volume_ma_len = st.number_input("Volume MA slow length (weeks)", min_value=2, max_value=60, value=20, step=1)
    volume_ma_fast_len = st.number_input(
        "Volume MA fast length (weeks)",
        min_value=2,
        max_value=60,
        value=10,
        step=1,
    )
    trend_adx_threshold = st.number_input(
        "Trend ADX threshold", min_value=1.0, max_value=100.0, value=40.0, step=1.0
    )
    panic_rsi_max = st.number_input("Panic RSI max", min_value=0.0, max_value=50.0, value=25.0, step=1.0)
    panic_return_max = st.number_input(
        "Panic return max", min_value=-1.0, max_value=0.0, value=-0.10, step=0.01, format="%.2f"
    )
    panic_volume_mult = st.number_input(
        "Panic volume multiple", min_value=1.0, max_value=10.0, value=2.0, step=0.1, format="%.1f"
    )
    priority_raw = st.text_input(
        "Regime priority order (comma or space separated)",
        value=", ".join(regime_options),
        help="Only listed regimes are active; later rules override earlier ones; 'none' is implied.",
    )
    st.caption(f"Available regimes: {', '.join(regime_options)}")

priority_order = parse_priority_order(priority_raw, regime_options)

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
    "priority_order": priority_order,
}


if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

raw = load_weekly_data(ticker, start_date, end_date)
if raw.empty:
    st.error("No data returned. Check the ticker or date range.")
    st.stop()

if settings["ma_fast_len"] >= settings["ma_slow_len"]:
    st.warning("MA fast length should be smaller than MA slow length.")

data = compute_regime_indicators(raw, settings)
regime_labels, regime_rules = classify_downtrend(data, settings)
data["Regime"] = regime_labels

weekly_ret = data["Close"].pct_change()
active_regimes = set(settings["priority_order"])
buy_signal = pd.Series(False, index=data.index)
sell_signal = pd.Series(False, index=data.index)
if "fast_up_trend" in active_regimes:
    buy_signal |= regime_rules["fast_up_trend"]
if "slow_up_trend" in active_regimes:
    buy_signal |= regime_rules["slow_up_trend"]
if "fast_down_trend" in active_regimes:
    sell_signal |= regime_rules["fast_down_trend"]
if "slow_down_trend" in active_regimes:
    sell_signal |= regime_rules["slow_down_trend"]
strategy_returns, _ = compute_trade_returns(weekly_ret, buy_signal, sell_signal, fee=0.003)
overall = compute_equity_stats(weekly_ret)
strategy_stats = compute_equity_stats(strategy_returns)
strategy_equity = None
if strategy_stats:
    strategy_equity = strategy_stats["Equity"].reindex(data.index) * 10000.0

label_order = settings["priority_order"] + ["none"]
summary = compute_regime_summary(weekly_ret, data["Regime"], label_order)
priority_text = " -> ".join(settings["priority_order"]) or "none"

latest = data.tail(1).copy()
st.subheader("Latest snapshot")
snapshot = latest[["Close", "MA_FAST", "MA_SLOW", "RSI", "ADX", "Regime"]].copy()
snapshot = snapshot.rename(
    columns={
        "MA_FAST": f"MA{settings['ma_fast_len']}",
        "MA_SLOW": f"MA{settings['ma_slow_len']}",
        "RSI": f"RSI{settings['rsi_len']}",
        "ADX": f"ADX{settings['adx_len']}",
    }
)
st.dataframe(snapshot)

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

st.subheader("Regime label sample")
st.dataframe(data[["Close", "Regime"]].tail(12))
