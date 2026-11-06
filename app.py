import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------
# 0. 頁面總體配置
# ---------------------------------------------------------
st.set_page_config(
    page_title="雙核心 ETF Canary 戰術輪動與動態優化系統",
    layout="wide",
)

# ---------------------------------------------------------
# 側邊欄：策略選擇與動態參數調整
# ---------------------------------------------------------
st.sidebar.title("🎯 策略與動態參數尋優")
strategy_choice = st.sidebar.radio(
    "選擇要運行的戰術策略：",
    [
        "1. ACWI 全球戰術輪動策略 (US-Core & Non-US Canary)",
        "2. VTI 美股加強輪動策略 (SMH Alpha & IWM Canary)",
    ],
)

st.sidebar.write("---")
st.sidebar.subheader("⚙️ 量化參數微調 (即時回測)")

ma_window = st.sidebar.slider(
    "金絲雀 Canary 均線天數 (MA)",
    min_value=20,
    max_value=250,
    value=130,
    step=10,
    help="建議改用 130MA (半年線) 以提高敏銳度，減少 lag 損失",
)

smh_alpha_weight = st.sidebar.slider(
    "VTI 策略 Risk-On 時 SMH 權重 (%)",
    min_value=15,
    max_value=35,
    value=25,
    step=5,
    help="提高科技/半導體龍頭的超額權重以拉高 Alpha",
) / 100.0

risk_off_mode = st.sidebar.selectbox(
    "Risk-Off (雙紅燈) 時防禦機制：",
    ["1. IVV 100% 抱緊美股大盤防禦", "2. 50% IVV + 50% BIL 現金避險"],
    help="美股具高 ROE 與長牛特性，全撥回 IVV 通常能有效避免在低點拋售避險",
)

st.sidebar.write("---")
start_date = st.sidebar.date_input(
    "起始日期", pd.to_datetime("2013-01-01")
).strftime("%Y-%m-%d")
end_date = st.sidebar.date_input(
    "結束日期", pd.to_datetime("2026-11-01")
).strftime("%Y-%m-%d")


# ---------------------------------------------------------
# 通用數據下載與安全解析 (穩定快取)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_etf_data(symbols, start, end):
    raw_df = yf.download(symbols, start=start, end=end, auto_adjust=True)
    if isinstance(raw_df.columns, pd.MultiIndex):
        if "Close" in raw_df.columns.levels[0]:
            df = raw_df["Close"]
        else:
            df = raw_df.xs(raw_df.columns.levels[0][0], axis=1, level=0)
    else:
        df = raw_df

    df = df.reindex(columns=symbols).ffill().bfill()
    return df


def get_drawdown_series(cum_returns):
    peak = cum_returns.cummax()
    return (cum_returns - peak) / peak * 100


# =========================================================
# 策略一：ACWI 全球戰術輪動策略 (動態尋優版)
# =========================================================
if "ACWI" in strategy_choice:
    st.title("🌐 ACWI 全球戰術輪動優化系統 (無前視偏誤對齊)")

    tickers_acwi = ["IVV", "VGK", "EWJ", "EPP", "IEMG", "BIL"]
    benchmark_weights_acwi = pd.Series(
        [0.63, 0.13, 0.05, 0.09, 0.11, 0.00], index=tickers_acwi
    )

    with st.spinner("正在讀取數據並執行無前視偏誤回測..."):
        try:
            data = load_etf_data(tickers_acwi, start_date, end_date)
        except Exception as e:
            st.error(f"❌ 數據下載失敗：{e}")
            st.stop()

    if data.empty or len(data) < ma_window:
        st.error(f"❌ 歷史數據不足 {ma_window} 天，無法執行回測。")
        st.stop()

    ma_target = data.rolling(window=ma_window).mean()
    returns = data.pct_change().fillna(0)

    canary_iemg_series = data["IEMG"] > ma_target["IEMG"]
    canary_epp_series = data["EPP"] > ma_target["EPP"]

    df_signals = pd.DataFrame(index=data.index)
    df_signals["Canary_IEMG"] = canary_iemg_series
    df_signals["Canary_EPP"] = canary_epp_series

    target_weights = []
    modes = []
    for d in data.index:
        c_iemg = df_signals.loc[d, "Canary_IEMG"]
        c_epp = df_signals.loc[d, "Canary_EPP"]

        if c_iemg and c_epp:
            mode = "Risk-On (🟢🟢)"
            w = [0.70, 0.10, 0.05, 0.05, 0.10, 0.00]
        elif not c_iemg and not c_epp:
            mode = "Risk-Off (🔴🔴)"
            if "100%" in risk_off_mode:
                w = [1.00, 0.00, 0.00, 0.00, 0.00, 0.00]  # IVV 全額大盤防衛
            else:
                w = [0.50, 0.00, 0.00, 0.00, 0.00, 0.50]  # 現金避險
        else:
            mode = "Risk-Neutral (🟡)"
            w = [0.80, 0.10, 0.10, 0.00, 0.00, 0.00]

        modes.append(mode)
        target_weights.append(w)

    df_w_target = pd.DataFrame(
        target_weights, index=data.index, columns=tickers_acwi
    )
    df_signals["Mode"] = modes

    # 無前視偏誤核心：Lag-1 Shift
    df_w_executed = df_w_target.shift(1).dropna()
    exec_dates = df_w_executed.index[
        df_w_executed.index >= data.index[ma_window]
    ]

    daily_records = []
    for d in exec_dates:
        daily_ret = returns.loc[d]
        strat_w = df_w_executed.loc[d]
        mode_applied = df_signals.loc[d, "Mode"]

        bm_ret = (daily_ret * benchmark_weights_acwi).sum()
        strat_ret = (daily_ret * strat_w).sum()

        rec = {
            "Date": d,
            "Mode": mode_applied,
            "Canary_IEMG": (
                "🟢 均線上"
                if df_signals.loc[d, "Canary_IEMG"]
                else "🔴 均線下"
            ),
            "Canary_EPP": (
                "🟢 均線上"
                if df_signals.loc[d, "Canary_EPP"]
                else "🔴 均線下"
            ),
            "Strat_Ret": strat_ret,
            "BM_Ret": bm_ret,
            "W_IVV": strat_w["IVV"],
            "W_VGK": strat_w["VGK"],
            "W_EWJ": strat_w["EWJ"],
            "W_EPP": strat_w["EPP"],
            "W_IEMG": strat_w["IEMG"],
            "W_BIL": strat_w["BIL"],
        }
        for t in tickers_acwi:
            rec[f"R_{t}"] = daily_ret[t]

        daily_records.append(rec)

    df_daily = pd.DataFrame(daily_records).set_index("Date")

    strat_cum = (1 + df_daily["Strat_Ret"]).cumprod()
    bm_cum = (1 + df_daily["BM_Ret"]).cumprod()

    total_days = (exec_dates[-1] - exec_dates[0]).days
    years = total_days / 365.25

    strat_total_return = (strat_cum.iloc[-1] - 1) * 100
    bm_total_return = (bm_cum.iloc[-1] - 1) * 100

    strat_cagr = ((strat_cum.iloc[-1]) ** (1 / years) - 1) * 100
    bm_cagr = ((bm_cum.iloc[-1]) ** (1 / years) - 1) * 100

    strat_dd_series = get_drawdown_series(strat_cum)
    bm_dd_series = get_drawdown_series(bm_cum)

    strat_mdd = strat_dd_series.min()
    bm_mdd = bm_dd_series.min()

    agg_dict = {
        "Strat_Ret": lambda x: (1 + x).prod() - 1,
        "BM_Ret": lambda x: (1 + x).prod() - 1,
        "Mode": "first",
        "Canary_IEMG": "last",
        "Canary_EPP": "last",
        "W_IVV": "first",
        "W_VGK": "first",
        "W_EWJ": "first",
        "W_EPP": "first",
        "W_IEMG": "first",
        "W_BIL": "first",
    }
    for t in tickers_acwi:
        agg_dict[f"R_{t}"] = lambda x, symbol=t: (1 + x).prod() - 1

    monthly_df = df_daily.resample("ME").agg(agg_dict)
    win_rate = (monthly_df["Strat_Ret"] > monthly_df["BM_Ret"]).mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "策略累積總報酬",
        f"{strat_total_return:.2f}%",
        f"年化 CAGR: {strat_cagr:.2f}%",
    )
    col2.metric(
        "ACWI 基準總報酬",
        f"{bm_total_return:.2f}%",
        f"年化 CAGR: {bm_cagr:.2f}%",
    )
    col3.metric("策略月度超越勝率", f"{win_rate:.2f}%")

    st.write("")
    col4, col5 = st.columns(2)
    col4.metric("策略最大回撤 (Max Drawdown)", f"{strat_mdd:.2f}%")
    col5.metric("ACWI 基準最大回撤 (Max Drawdown)", f"{bm_mdd:.2f}%")

    st.write("### 📈 累積淨值走勢比較 (基期 = 1.0)")
    st.line_chart(
        pd.DataFrame(
            {"Canary 戰術輪動策略": strat_cum, "ACWI 基準對標": bm_cum}
        )
    )

    st.write("### 📉 風險水庫：水面下回撤走勢圖 (Drawdown %)")
    st.line_chart(
        pd.DataFrame(
            {
                "Canary 策略 Drawdown (%)": strat_dd_series,
                "ACWI 基準 Drawdown (%)": bm_dd_series,
            }
        )
    )

    st.write("---")
    st.subheader(
        "📋 每月詳細策略報酬與對標歷程 (月度第一天執行對齊)"
    )

    monthly_display = monthly_df.copy()
    monthly_display.index = monthly_display.index.strftime("%Y-%m-%d")

    monthly_display["策略月報酬"] = (
        monthly_display["Strat_Ret"] * 100
    ).map("{:.2f}%".format)
    monthly_display["ACWI 月報酬"] = (
        monthly_display["BM_Ret"] * 100
    ).map("{:.2f}%".format)
    monthly_display["月度勝負"] = np.where(
        monthly_display["Strat_Ret"] > monthly_display["BM_Ret"],
        "🏆 擊敗 ACWI",
        "🔻 落後 ACWI",
    )

    for t in tickers_acwi:
        monthly_display[f"{t} 月報酬"] = (
            monthly_display[f"R_{t}"] * 100
        ).map("{:.2f}%".format)
        monthly_display[f"W_{t}"] = (monthly_display[f"W_{t}"] * 100).map(
            "{:.0f}%".format
        )

    table_cols = [
        "Mode",
        "策略月報酬",
        "ACWI 月報酬",
        "月度勝負",
        "IVV 月報酬",
        "VGK 月報酬",
        "EWJ 月報酬",
        "EPP 月報酬",
        "IEMG 月報酬",
        "BIL 月報酬",
        "Canary_IEMG",
        "Canary_EPP",
        "W_IVV",
        "W_VGK",
        "W_EWJ",
        "W_EPP",
        "W_IEMG",
        "W_BIL",
    ]
    st.dataframe(
        monthly_display[table_cols].sort_index(ascending=False),
        use_container_width=True,
    )


# =========================================================
# 策略二：VTI 美股加強輪動策略 (動態尋優版)
# =========================================================
else:
    st.title("🚀 VTI 美股核心加強型優化系統 (無前視偏誤對齊)")

    tickers_vti = ["VTI", "IVV", "SMH", "IWM", "BIL"]

    with st.spinner("正在讀取數據並執行無前視偏誤回測..."):
        try:
            data = load_etf_data(tickers_vti, start_date, end_date)
        except Exception as e:
            st.error(f"❌ 數據下載失敗：{e}")
            st.stop()

    if data.empty or len(data) < ma_window:
        st.error(f"❌ 歷史數據不足 {ma_window} 天，無法執行回測。")
        st.stop()

    ma_target = data.rolling(window=ma_window).mean()
    returns = data.pct_change().fillna(0)

    canary_smh_series = data["SMH"] > ma_target["SMH"]
    canary_iwm_series = data["IWM"] > ma_target["IWM"]

    df_signals = pd.DataFrame(index=data.index)
    df_signals["Canary_SMH"] = canary_smh_series
    df_signals["Canary_IWM"] = canary_iwm_series

    target_weights = []
    modes = []
    for d in data.index:
        c_smh = df_signals.loc[d, "Canary_SMH"]
        c_iwm = df_signals.loc[d, "Canary_IWM"]

        if c_smh and c_iwm:
            mode = "Risk-On (🟢🟢)"
            ivw_w = 1.0 - smh_alpha_weight - 0.15
            w = [0.00, round(ivw_w, 2), smh_alpha_weight, 0.15, 0.00]
        elif not c_smh and not c_iwm:
            mode = "Risk-Off (🔴🔴)"
            if "100%" in risk_off_mode:
                w = [0.00, 1.00, 0.00, 0.00, 0.00]  # IVV 100% 大盤防衛
            else:
                w = [0.00, 0.50, 0.00, 0.00, 0.50]  # 現金避險
        else:
            mode = "Risk-Neutral (🟡)"
            if c_smh:
                w = [
                    0.00,
                    round(1.0 - smh_alpha_weight, 2),
                    smh_alpha_weight,
                    0.00,
                    0.00,
                ]
            else:
                w = [0.00, 0.85, 0.00, 0.15, 0.00]

        modes.append(mode)
        target_weights.append(w)

    df_w_target = pd.DataFrame(
        target_weights, index=data.index, columns=tickers_vti
    )
    df_signals["Mode"] = modes

    # 無前視偏誤核心：Lag-1 Shift
    df_w_executed = df_w_target.shift(1).dropna()
    exec_dates = df_w_executed.index[
        df_w_executed.index >= data.index[ma_window]
    ]

    daily_records = []
    for d in exec_dates:
        daily_ret = returns.loc[d]
        strat_w = df_w_executed.loc[d]
        mode_applied = df_signals.loc[d, "Mode"]

        vti_ret = daily_ret["VTI"]
        strat_ret = (daily_ret * strat_w).sum()

        rec = {
            "Date": d,
            "Mode": mode_applied,
            "Canary_SMH": (
                "🟢 均線上"
                if df_signals.loc[d, "Canary_SMH"]
                else "🔴 均線下"
            ),
            "Canary_IWM": (
                "🟢 均線上"
                if df_signals.loc[d, "Canary_IWM"]
                else "🔴 均線下"
            ),
            "Strat_Ret": strat_ret,
            "VTI_Ret": vti_ret,
            "W_IVV": strat_w["IVV"],
            "W_SMH": strat_w["SMH"],
            "W_IWM": strat_w["IWM"],
            "W_BIL": strat_w["BIL"],
        }
        for t in tickers_vti:
            rec[f"R_{t}"] = daily_ret[t]

        daily_records.append(rec)

    df_daily = pd.DataFrame(daily_records).set_index("Date")

    strat_cum = (1 + df_daily["Strat_Ret"]).cumprod()
    vti_cum = (1 + df_daily["VTI_Ret"]).cumprod()

    total_days = (exec_dates[-1] - exec_dates[0]).days
    years = total_days / 365.25

    strat_total_return = (strat_cum.iloc[-1] - 1) * 100
    vti_total_return = (vti_cum.iloc[-1] - 1) * 100

    strat_cagr = ((strat_cum.iloc[-1]) ** (1 / years) - 1) * 100
    vti_cagr = ((vti_cum.iloc[-1]) ** (1 / years) - 1) * 100

    strat_dd_series = get_drawdown_series(strat_cum)
    vti_dd_series = get_drawdown_series(vti_cum)

    strat_mdd = strat_dd_series.min()
    vti_mdd = vti_dd_series.min()

    agg_dict = {
        "Strat_Ret": lambda x: (1 + x).prod() - 1,
        "VTI_Ret": lambda x: (1 + x).prod() - 1,
        "Mode": "first",
        "Canary_SMH": "last",
        "Canary_IWM": "last",
        "W_IVV": "first",
        "W_SMH": "first",
        "W_IWM": "first",
        "W_BIL": "first",
    }
    for t in tickers_vti:
        agg_dict[f"R_{t}"] = lambda x, symbol=t: (1 + x).prod() - 1

    monthly_df = df_daily.resample("ME").agg(agg_dict)
    win_rate = (monthly_df["Strat_Ret"] > monthly_df["VTI_Ret"]).mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "戰術策略累積總報酬",
        f"{strat_total_return:.2f}%",
        f"年化 CAGR: {strat_cagr:.2f}%",
    )
    col2.metric(
        "VTI 買入持有總報酬",
        f"{vti_total_return:.2f}%",
        f"年化 CAGR: {vti_cagr:.2f}%",
    )
    col3.metric("月度超越 VTI 勝率", f"{win_rate:.2f}%")

    st.write("")
    col4, col5 = st.columns(2)
    col4.metric("策略最大回撤 (Max Drawdown)", f"{strat_mdd:.2f}%")
    col5.metric("VTI 基準最大回撤 (Max Drawdown)", f"{vti_mdd:.2f}%")

    st.write("### 📈 累積淨值走勢比較 (對標 VTI, 基期 = 1.0)")
    st.line_chart(
        pd.DataFrame(
            {
                "SMH/IWM 戰術輪動策略": strat_cum,
                "VTI Buy & Hold 基準": vti_cum,
            }
        )
    )

    st.write("### 📉 風險水庫：水面下回撤走勢圖 (Drawdown %)")
    st.line_chart(
        pd.DataFrame(
            {
                "策略 Drawdown (%)": strat_dd_series,
                "VTI 基準 Drawdown (%)": vti_dd_series,
            }
        )
    )

    st.write("---")
    st.subheader(
        "📋 每月詳細策略報酬與對標歷程 (月度第一天執行對齊)"
    )

    monthly_display = monthly_df.copy()
    monthly_display.index = monthly_display.index.strftime("%Y-%m-%d")

    monthly_display["策略月報酬"] = (
        monthly_display["Strat_Ret"] * 100
    ).map("{:.2f}%".format)
    monthly_display["VTI 月報酬"] = (
        monthly_display["VTI_Ret"] * 100
    ).map("{:.2f}%".format)
    monthly_display["月度勝負"] = np.where(
        monthly_display["Strat_Ret"] > monthly_display["VTI_Ret"],
        "🏆 擊敗 VTI",
        "🔻 落後 VTI",
    )

    for t in tickers_vti:
        monthly_display[f"{t} 月報酬"] = (
            monthly_display[f"R_{t}"] * 100
        ).map("{:.2f}%".format)
        if t != "VTI":
            monthly_display[f"W_{t}"] = (
                monthly_display[f"W_{t}"] * 100
            ).map("{:.0f}%".format)

    table_cols = [
        "Mode",
        "策略月報酬",
        "VTI 月報酬",
        "月度勝負",
        "IVV 月報酬",
        "SMH 月報酬",
        "IWM 月報酬",
        "BIL 月報酬",
        "Canary_SMH",
        "Canary_IWM",
        "W_IVV",
        "W_SMH",
        "W_IWM",
        "W_BIL",
    ]
    st.dataframe(
        monthly_display[table_cols].sort_index(ascending=False),
        use_container_width=True,
    )
