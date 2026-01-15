import datetime as dt
import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

from timing_engine import (
    WARMUP_KEYS,
    build_param_values,
    calc_warmup_weeks,
    compute_equity_stats,
    compute_partial_oos_stats,
    compute_regime_summary,
    compute_signals_window,
    compute_strategy_objective,
    compute_strategy_objective_window,
    compute_strategy_returns_window,
    compute_trade_returns,
    prepare_ohlcv_input,
    parse_priority_order,
)

st.set_page_config(page_title="Weekly Market Regime Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def load_weekly_data(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1wk",
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    if data.empty:
        return data
    weekly = prepare_ohlcv_input(data, ticker=ticker, resample_rule=None)
    return weekly


def format_pct(value: float) -> str:
    return "N/A" if np.isnan(value) else f"{value:.2%}"


def format_num(value: float) -> str:
    return "N/A" if np.isnan(value) else f"{value:.2f}"

def get_state_or_default(key: str, default):
    return st.session_state.get(key, default)


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


def _build_opt_dialog_state_keys() -> list[str]:
    keys = [
        "opt_search_mode",
        "opt_max_trials",
        "opt_seed",
        "opt_range_pct",
        "opt_prev_range_pct",
        "wfo_ins_len",
        "wfo_oos_len",
        "wfo_mode",
        "wfo_shifted",
        "wfo_shift_step",
    ]
    for spec in OPT_PARAM_SPECS:
        key = spec["key"]
        keys.extend(
            [
                f"opt_enable_{key}",
                f"opt_min_{key}",
                f"opt_max_{key}",
                f"opt_step_{key}",
            ]
        )
    return keys


OPT_DIALOG_STATE_KEYS = _build_opt_dialog_state_keys()


def save_opt_dialog_state() -> None:
    snapshot = dict(st.session_state.get("opt_dialog_state", {}))
    updated = False
    for key in OPT_DIALOG_STATE_KEYS:
        if key in st.session_state:
            snapshot[key] = st.session_state[key]
            updated = True
    if updated:
        st.session_state["opt_dialog_state"] = snapshot


def restore_opt_dialog_state() -> None:
    snapshot = st.session_state.get("opt_dialog_state")
    if not snapshot:
        return
    for key, value in snapshot.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_param_values_from_state(settings: dict) -> tuple[dict, bool]:
    restore_opt_dialog_state()
    snapshot = st.session_state.get("opt_dialog_state", {})
    param_values = {}
    invalid = False
    range_pct = float(st.session_state.get("opt_range_pct", snapshot.get("opt_range_pct", 20.0))) / 100.0
    for spec in OPT_PARAM_SPECS:
        key = spec["key"]
        default_enabled = key not in {"vol_len", "vol_fast_len"}
        enabled = st.session_state.get(f"opt_enable_{key}", snapshot.get(f"opt_enable_{key}", default_enabled))
        if not enabled:
            continue
        min_key = f"opt_min_{key}"
        max_key = f"opt_max_{key}"
        step_key = f"opt_step_{key}"
        min_val = st.session_state.get(min_key, snapshot.get(min_key))
        max_val = st.session_state.get(max_key, snapshot.get(max_key))
        step_val = st.session_state.get(step_key, snapshot.get(step_key))
        if min_val is None or max_val is None or step_val is None:
            current = st.session_state.get(key, settings.get(key))
            if spec["type"] == "int":
                current_val = int(current)
                min_val = max(spec["min"], math.floor(current_val * (1 - range_pct)))
                max_val = min(spec["max"], math.ceil(current_val * (1 + range_pct)))
                if min_val > max_val:
                    min_val = max_val = current_val
                step_val = int(spec["step"])
            else:
                current_val = float(current)
                delta = abs(current_val) * range_pct
                min_val = max(spec["min"], current_val - delta)
                max_val = min(spec["max"], current_val + delta)
                if min_val > max_val:
                    min_val = max_val = current_val
                step_val = float(spec["step"])
            st.session_state[f"opt_min_{key}"] = min_val
            st.session_state[f"opt_max_{key}"] = max_val
            st.session_state[f"opt_step_{key}"] = step_val
        if spec["type"] == "int":
            min_val = int(min_val)
            max_val = int(max_val)
            step_val = int(step_val)
        else:
            min_val = float(min_val)
            max_val = float(max_val)
            step_val = float(step_val)
        if max_val < min_val:
            st.warning(f"{spec['label']}: max must be >= min.")
            invalid = True
            continue
        grid = build_param_values(min_val, max_val, step_val, spec["type"])
        if not grid:
            st.warning(f"{spec['label']}: invalid range or step.")
            invalid = True
            continue
        param_values[key] = grid

    if not param_values:
        st.warning("Enable at least one parameter to tune.")
        invalid = True

    return param_values, invalid


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
    current_settings = {
        "ma_fast_len": ma_fast_len,
        "ma_slow_len": ma_slow_len,
        "rsi_len": rsi_len,
        "rsi_fast_len": rsi_fast_len,
        "adx_len": adx_len,
        "adx_fast_len": adx_fast_len,
        "atr_len": atr_len,
        "atr_fast_len": atr_fast_len,
        "vol_len": vol_len,
        "vol_fast_len": vol_fast_len,
        "return_window": return_window,
        "volume_ma_len": volume_ma_len,
        "volume_ma_fast_len": volume_ma_fast_len,
        "trend_adx_threshold": trend_adx_threshold,
        "panic_rsi_max": panic_rsi_max,
        "panic_return_max": panic_return_max,
        "panic_volume_mult": panic_volume_mult,
    }
    with st.expander("Optimization settings", expanded=False):
        restore_opt_dialog_state()
        st.write("Optimize in-sample objective: CAGR + 0.1 * min(Max Drawdown, -0.3).")
        search_mode = st.selectbox("Search method", ["Grid", "Random"], key="opt_search_mode")
        if "opt_max_trials" in st.session_state:
            max_trials = st.number_input(
                "Max trials", min_value=1, max_value=5000, step=1, key="opt_max_trials"
            )
        else:
            max_trials = st.number_input(
                "Max trials", min_value=1, max_value=5000, value=200, step=1, key="opt_max_trials"
            )
        if "opt_seed" in st.session_state:
            seed = st.number_input(
                "Random seed", min_value=0, max_value=100000, step=1, key="opt_seed"
            )
        else:
            seed = st.number_input(
                "Random seed", min_value=0, max_value=100000, value=42, step=1, key="opt_seed"
            )
        if "opt_range_pct" in st.session_state:
            range_pct = st.number_input(
                "Range around current (%)",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                key="opt_range_pct",
            )
        else:
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
                current = st.session_state.get(key, current_settings.get(key))
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

        for spec in OPT_PARAM_SPECS:
            key = spec["key"]
            current = st.session_state.get(key, current_settings.get(key))
            current_val = int(current) if spec["type"] == "int" else float(current)
            current_val = min(max(current_val, spec["min"]), spec["max"])
            row = st.columns([2.4, 1, 1, 1])
            default_enabled = key not in {"vol_len", "vol_fast_len"}
            if f"opt_enable_{key}" in st.session_state:
                row[0].checkbox(spec["label"], key=f"opt_enable_{key}")
            else:
                row[0].checkbox(spec["label"], value=default_enabled, key=f"opt_enable_{key}")
            if f"opt_min_{key}" in st.session_state:
                row[1].number_input(
                    "min",
                    min_value=spec["min"],
                    max_value=spec["max"],
                    step=spec["step"],
                    key=f"opt_min_{key}",
                    label_visibility="collapsed",
                )
            else:
                row[1].number_input(
                    "min",
                    min_value=spec["min"],
                    max_value=spec["max"],
                    value=current_val,
                    step=spec["step"],
                    key=f"opt_min_{key}",
                    label_visibility="collapsed",
                )
            if f"opt_max_{key}" in st.session_state:
                row[2].number_input(
                    "max",
                    min_value=spec["min"],
                    max_value=spec["max"],
                    step=spec["step"],
                    key=f"opt_max_{key}",
                    label_visibility="collapsed",
                )
            else:
                row[2].number_input(
                    "max",
                    min_value=spec["min"],
                    max_value=spec["max"],
                    value=current_val,
                    step=spec["step"],
                    key=f"opt_max_{key}",
                    label_visibility="collapsed",
                )
            step_max = max(spec["step"], spec["max"] - spec["min"])
            if f"opt_step_{key}" in st.session_state:
                row[3].number_input(
                    "step",
                    min_value=spec["step"],
                    max_value=step_max,
                    step=spec["step"],
                    key=f"opt_step_{key}",
                    label_visibility="collapsed",
                )
            else:
                row[3].number_input(
                    "step",
                    min_value=spec["step"],
                    max_value=step_max,
                    value=spec["step"],
                    step=spec["step"],
                    key=f"opt_step_{key}",
                    label_visibility="collapsed",
                )

        st.markdown("**Walk-forward optimization**")
        if "wfo_ins_len" in st.session_state:
            st.number_input(
                "In-sample length (weeks)", min_value=26, max_value=520, step=1, key="wfo_ins_len"
            )
        else:
            st.number_input(
                "In-sample length (weeks)", min_value=26, max_value=520, value=156, step=1, key="wfo_ins_len"
            )
        if "wfo_oos_len" in st.session_state:
            wfo_oos_len = st.number_input(
                "Out-of-sample length (weeks)", min_value=13, max_value=260, step=1, key="wfo_oos_len"
            )
        else:
            wfo_oos_len = st.number_input(
                "Out-of-sample length (weeks)", min_value=13, max_value=260, value=52, step=1, key="wfo_oos_len"
            )
        st.caption("Step size equals the out-of-sample length.")
        st.selectbox("In-sample window", ["Rolling", "Expanding"], key="wfo_mode")
        if "wfo_shifted" in st.session_state:
            shifted_wfo = st.checkbox("Shifted walk-forward", key="wfo_shifted")
        else:
            shifted_wfo = st.checkbox("Shifted walk-forward", value=False, key="wfo_shifted")
        shift_default = max(1, int(wfo_oos_len) // 4)
        if "wfo_shift_step" in st.session_state:
            st.number_input(
                "Shift step (weeks)",
                min_value=1,
                max_value=int(wfo_oos_len),
                step=1,
                key="wfo_shift_step",
                disabled=not shifted_wfo,
            )
        else:
            st.number_input(
                "Shift step (weeks)",
                min_value=1,
                max_value=int(wfo_oos_len),
                value=shift_default,
                step=1,
                key="wfo_shift_step",
                disabled=not shifted_wfo,
            )
        if shifted_wfo:
            st.caption("Shifted walk-forward runs multiple offsets within the OOS window length.")
        save_opt_dialog_state()

    run_optimization = st.button("Run optimization", key="run_optimization_sidebar")
    run_wfo = st.button("Run walk-forward", key="run_wfo_sidebar")
    if run_optimization:
        st.session_state["run_optimization"] = True
    if run_wfo:
        st.session_state["run_wfo"] = True

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


st.subheader("Optimization results")
opt_progress_slot = st.empty()
run_optimization = st.session_state.pop("run_optimization", False)
if run_optimization:
    param_values, invalid = build_param_values_from_state(settings)
    if not invalid:
        search_mode = st.session_state.get("opt_search_mode", "Grid")
        max_trials = int(st.session_state.get("opt_max_trials", 200))
        seed = int(st.session_state.get("opt_seed", 42))
        save_opt_dialog_state()
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
        else:
            spec_map = OPT_SPEC_MAP
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
            progress = opt_progress_slot.progress(0.0) if total > 1 else None
            results = []
            for idx, candidate in enumerate(candidates, 1):
                test_settings = settings.copy()
                for key, value in candidate.items():
                    spec = spec_map[key]
                    test_settings[key] = int(value) if spec["type"] == "int" else float(value)
                if test_settings["ma_fast_len"] >= test_settings["ma_slow_len"]:
                    continue
                objective, cagr, max_dd = compute_strategy_objective(raw_opt, test_settings, start_date)
                if not np.isnan(objective):
                    results.append({"Objective": objective, "CAGR": cagr, "Max Drawdown": max_dd, **candidate})
                if progress and idx % 5 == 0:
                    progress.progress(idx / total)
            if progress:
                progress.empty()

            if results:
                result_df = pd.DataFrame(results).sort_values("Objective", ascending=False)
                best_row = result_df.iloc[0]
                best_params = {}
                for key in param_values.keys():
                    spec = spec_map[key]
                    value = best_row[key]
                    best_params[key] = int(value) if spec["type"] == "int" else float(value)
                st.session_state["opt_results"] = results
                st.session_state["opt_best_params"] = best_params
                st.session_state["opt_best_objective"] = float(best_row["Objective"])
                st.session_state["opt_best_cagr"] = float(best_row["CAGR"])
                st.session_state["opt_best_max_dd"] = float(best_row["Max Drawdown"])
            else:
                st.warning("No valid trials produced an objective score.")

results = st.session_state.get("opt_results")
if results:
    result_df = pd.DataFrame(results)
    if "Objective" not in result_df.columns:
        st.warning("Optimization results are from an older run. Please rerun to compute the new objective.")
        st.session_state.pop("opt_results", None)
        st.session_state.pop("opt_best_params", None)
        st.session_state.pop("opt_best_objective", None)
        st.session_state.pop("opt_best_cagr", None)
        st.session_state.pop("opt_best_max_dd", None)
        st.rerun()
    result_df = result_df.sort_values("Objective", ascending=False)
    display = result_df.copy()
    display["Objective"] = display["Objective"].apply(format_pct)
    display["CAGR"] = display["CAGR"].apply(format_pct)
    display["Max Drawdown"] = display["Max Drawdown"].apply(format_pct)
    st.dataframe(display.head(20))

    best_params = st.session_state.get("opt_best_params")
    best_objective = st.session_state.get("opt_best_objective")
    best_cagr = st.session_state.get("opt_best_cagr")
    best_max_dd = st.session_state.get("opt_best_max_dd")
    if best_params is None:
        best_row = result_df.iloc[0]
        best_params = {}
        for key in result_df.columns:
            if key in {"Objective", "CAGR", "Max Drawdown"}:
                continue
            spec = OPT_SPEC_MAP.get(key)
            if not spec:
                continue
            value = best_row[key]
            best_params[key] = int(value) if spec["type"] == "int" else float(value)
    if best_objective is None:
        best_objective = float(result_df.iloc[0]["Objective"])
    if best_cagr is None:
        best_cagr = float(result_df.iloc[0]["CAGR"])
    if best_max_dd is None:
        best_max_dd = float(result_df.iloc[0]["Max Drawdown"])
    st.write(
        "Best objective: "
        f"{format_pct(best_objective)} "
        f"(CAGR: {format_pct(best_cagr)}, Max DD: {format_pct(best_max_dd)})"
    )
    if st.button("Apply best parameters", key="apply_best_params"):
        st.session_state["pending_opt_params"] = best_params
        st.rerun()
    if st.button("Clear optimization results", key="clear_opt_results"):
        st.session_state.pop("opt_results", None)
        st.session_state.pop("opt_best_params", None)
        st.session_state.pop("opt_best_objective", None)
        st.session_state.pop("opt_best_cagr", None)
        st.session_state.pop("opt_best_max_dd", None)
        st.rerun()
else:
    st.info("Run optimization to see results.")

st.subheader("Walk-forward results")
wfo_progress_slot = st.empty()
run_wfo = st.session_state.pop("run_wfo", False)
if run_wfo:
    param_values, invalid = build_param_values_from_state(settings)
    if not invalid:
        search_mode = st.session_state.get("opt_search_mode", "Grid")
        max_trials = int(st.session_state.get("opt_max_trials", 200))
        seed = int(st.session_state.get("opt_seed", 42))
        wfo_ins_len = int(st.session_state.get("wfo_ins_len", 156))
        wfo_oos_len = int(st.session_state.get("wfo_oos_len", 52))
        wfo_mode = st.session_state.get("wfo_mode", "Rolling")
        shifted_wfo = bool(st.session_state.get("wfo_shifted", False))
        shift_step_weeks = int(
            st.session_state.get("wfo_shift_step", max(1, int(wfo_oos_len) // 4))
        )
        save_opt_dialog_state()

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
        else:
            spec_map = OPT_SPEC_MAP
            data_end_ts = min(pd.Timestamp(end_date), raw_opt.index.max())
            end_limit = data_end_ts + pd.Timedelta(days=1)
            oos_len_weeks = int(wfo_oos_len)
            step_weeks = oos_len_weeks
            ins_len_weeks = int(wfo_ins_len)
            shift_step = max(1, min(int(shift_step_weeks), oos_len_weeks))
            if shifted_wfo:
                shift_offsets = list(range(0, oos_len_weeks, shift_step))
            else:
                shift_offsets = [0]

            total_windows = 0
            for offset in shift_offsets:
                oos_cursor = pd.Timestamp(start_date) + pd.Timedelta(weeks=offset)
                while True:
                    oos_start = oos_cursor
                    if oos_start > data_end_ts:
                        break
                    oos_end = min(oos_start + pd.Timedelta(weeks=oos_len_weeks), end_limit)
                    if oos_end <= oos_start:
                        break
                    total_windows += 1
                    oos_cursor = oos_cursor + pd.Timedelta(weeks=step_weeks)

            progress = wfo_progress_slot.progress(0.0) if total_windows > 0 else None
            wfo_results = []
            oos_returns_by_shift: dict[int, list[pd.Series]] = {}
            window_idx = 0
            for offset in shift_offsets:
                oos_cursor = pd.Timestamp(start_date) + pd.Timedelta(weeks=offset)
                initial_ins_start = oos_cursor - pd.Timedelta(weeks=ins_len_weeks)
                while True:
                    oos_start = oos_cursor
                    if oos_start > data_end_ts:
                        break
                    if wfo_mode == "Rolling":
                        ins_start = oos_start - pd.Timedelta(weeks=ins_len_weeks)
                    else:
                        ins_start = initial_ins_start
                    ins_end = oos_start
                    oos_end = min(oos_start + pd.Timedelta(weeks=oos_len_weeks), end_limit)
                    if oos_end <= oos_start:
                        break
                    combinations = 1
                    for grid in param_values.values():
                        combinations *= len(grid)
                    candidates = []
                    if search_mode == "Grid" and combinations <= max_trials:
                        keys = list(param_values.keys())
                        for combo in itertools.product(*(param_values[key] for key in keys)):
                            candidates.append(dict(zip(keys, combo)))
                    else:
                        rng = random.Random(seed + window_idx)
                        keys = list(param_values.keys())
                        for _ in range(int(max_trials)):
                            candidates.append({key: rng.choice(param_values[key]) for key in keys})

                    best_objective = np.nan
                    best_cagr = np.nan
                    best_max_dd = np.nan
                    best_params = None
                    for candidate in candidates:
                        test_settings = settings.copy()
                        for key, value in candidate.items():
                            spec = spec_map[key]
                            test_settings[key] = int(value) if spec["type"] == "int" else float(value)
                        if test_settings["ma_fast_len"] >= test_settings["ma_slow_len"]:
                            continue
                        objective, cagr, max_dd = compute_strategy_objective_window(
                            raw_opt, test_settings, ins_start, ins_end
                        )
                        if np.isnan(objective):
                            continue
                        if np.isnan(best_objective) or objective > best_objective:
                            best_objective = objective
                            best_cagr = cagr
                            best_max_dd = max_dd
                            best_params = test_settings.copy()

                    if not best_params:
                        break

                    oos_returns = compute_strategy_returns_window(
                        raw_opt, best_params, oos_start, oos_end, fee=0.003, end_exclusive=True
                    )
                    oos_stats = compute_partial_oos_stats(oos_returns)
                    if not oos_returns.empty:
                        oos_returns_by_shift.setdefault(int(offset), []).append(oos_returns)
                    params_display = ", ".join(
                        f"{key}={best_params[key]}" for key in param_values.keys() if key in best_params
                    )
                    wfo_results.append(
                        {
                            "Shift (weeks)": int(offset),
                            "In-sample start": ins_start.date(),
                            "In-sample end": ins_end.date(),
                            "OOS start": oos_start.date(),
                            "OOS end": (oos_end - pd.Timedelta(days=1)).date(),
                            "IS Objective": best_objective,
                            "IS CAGR": best_cagr,
                            "IS Max Drawdown": best_max_dd,
                            "OOS CAGR": oos_stats.get("CAGR", np.nan),
                            "OOS Total Return": oos_stats.get("Total Return", np.nan),
                            "Best params": params_display,
                        }
                    )
                    oos_cursor = oos_cursor + pd.Timedelta(weeks=step_weeks)
                    window_idx += 1
                    if progress:
                        progress.progress(min(window_idx / total_windows, 1.0))

            st.session_state["wfo_results"] = wfo_results
            oos_shift_stats = []
            wfo_shift_plot_data: dict[int, dict[str, pd.Series]] = {}
            for shift_key, series_list in sorted(oos_returns_by_shift.items()):
                if not series_list:
                    continue
                oos_all = pd.concat(series_list).sort_index()
                if oos_all.empty:
                    continue
                price_full = raw_opt["Close"].loc[oos_all.index.min() : oos_all.index.max()]
                if price_full.empty:
                    continue
                plot_returns = oos_all.reindex(price_full.index).fillna(0.0)
                equity = 10000.0 * (1 + plot_returns).cumprod()
                price = price_full
                stats = compute_equity_stats(oos_all)
                if not stats:
                    partial = compute_partial_oos_stats(oos_all)
                    stats = {
                        "Total Return": partial.get("Total Return", np.nan),
                        "CAGR": partial.get("CAGR", np.nan),
                        "Max Drawdown": np.nan,
                        "Sharpe": np.nan,
                        "Volatility": np.nan,
                    }
                oos_shift_stats.append(
                    {
                        "Shift (weeks)": shift_key,
                        "Total Return": stats.get("Total Return", np.nan),
                        "CAGR": stats.get("CAGR", np.nan),
                        "Max Drawdown": stats.get("Max Drawdown", np.nan),
                        "Sharpe (ann.)": stats.get("Sharpe", np.nan),
                        "Volatility (ann.)": stats.get("Volatility", np.nan),
                    }
                )
                if not price.empty and not equity.empty:
                    wfo_shift_plot_data[int(shift_key)] = {"price": price, "equity": equity}
            st.session_state["wfo_oos_shift_stats"] = oos_shift_stats
            st.session_state["wfo_shift_plot_data"] = wfo_shift_plot_data
            if progress:
                progress.empty()

wfo_results = st.session_state.get("wfo_results")
if wfo_results:
    wfo_df = pd.DataFrame(wfo_results)
    wfo_display = wfo_df.copy()
    if "IS Objective" in wfo_display.columns:
        wfo_display["IS Objective"] = wfo_display["IS Objective"].apply(format_pct)
    wfo_display["IS CAGR"] = wfo_display["IS CAGR"].apply(format_pct)
    if "IS Max Drawdown" in wfo_display.columns:
        wfo_display["IS Max Drawdown"] = wfo_display["IS Max Drawdown"].apply(format_pct)
    wfo_display["OOS CAGR"] = wfo_display["OOS CAGR"].apply(format_pct)
    wfo_display["OOS Total Return"] = wfo_display["OOS Total Return"].apply(format_pct)
    st.dataframe(wfo_display)
    oos_shift_stats = st.session_state.get("wfo_oos_shift_stats", [])
    if oos_shift_stats:
        st.write("Walk-forward OOS summary by shift")
        shift_df = pd.DataFrame(oos_shift_stats)
        shift_display = shift_df.copy()
        shift_display["Total Return"] = shift_display["Total Return"].apply(format_pct)
        shift_display["CAGR"] = shift_display["CAGR"].apply(format_pct)
        shift_display["Max Drawdown"] = shift_display["Max Drawdown"].apply(format_pct)
        shift_display["Sharpe (ann.)"] = shift_display["Sharpe (ann.)"].apply(format_num)
        shift_display["Volatility (ann.)"] = shift_display["Volatility (ann.)"].apply(format_pct)
        st.table(shift_display)
    wfo_shift_plot_data = st.session_state.get("wfo_shift_plot_data", {})
    if wfo_shift_plot_data:
        st.write("Walk-forward price and equity by shift")
        for shift_key in sorted(wfo_shift_plot_data.keys()):
            series = wfo_shift_plot_data[shift_key]
            price = series.get("price", pd.Series(dtype=float))
            equity = series.get("equity", pd.Series(dtype=float))
            if price.empty or equity.empty:
                continue
            expanded = len(wfo_shift_plot_data) == 1
            with st.expander(f"Shift {shift_key} weeks", expanded=expanded):
                fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Scatter(x=price.index, y=price, name="Price", line=dict(width=1.2, color="#7a7a7a")),
                    row=1,
                    col=1,
                    secondary_y=False,
                )
                fig.add_trace(
                    go.Scatter(x=equity.index, y=equity, name="Equity (10k)", line=dict(width=1.2, color="#2b8cbe")),
                    row=1,
                    col=1,
                    secondary_y=True,
                )
                fig.update_layout(height=320, legend=dict(orientation="h"), yaxis_type="log", yaxis2_type="log")
                st.plotly_chart(fig, use_container_width=True)
    if st.button("Clear walk-forward results", key="clear_wfo_results"):
        st.session_state.pop("wfo_results", None)
        st.session_state.pop("wfo_oos_shift_stats", None)
        st.session_state.pop("wfo_shift_plot_data", None)
        st.rerun()
else:
    st.info("Run walk-forward to see results.")

window_end = raw.index.max() + pd.Timedelta(days=1)
signals = compute_signals_window(raw, settings, start_date, window_end, end_exclusive=True)
data = signals["data"]
regime_labels = signals["labels"]
regime_rules = signals["rules"]
valid_mask = signals["valid_mask"]
effective_start = signals["effective_start"]
if data.empty or effective_start is None:
    st.error("Not enough data to compute indicators with the current settings.")
    st.stop()
if effective_start > pd.Timestamp(start_date):
    st.info(f"Start date adjusted to {effective_start.date()} to allow indicator warm-up.")
data["Regime"] = regime_labels
weekly_ret = signals["weekly_returns"]
buy_signal = signals["buy_signal"]
sell_signal = signals["sell_signal"]
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
