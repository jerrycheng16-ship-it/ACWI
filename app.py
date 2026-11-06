import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------
# 0. 頁面總體配置
# ---------------------------------------------------------
st.set_page_config(
    page_title="雙核心 ETF Canary 戰術輪動與對標回測系統",
    layout="wide",
)

# ---------------------------------------------------------
# 側邊欄：策略選擇與時間設定
# ---------------------------------------------------------
st.sidebar.title("🎯 策略與參數控制")
strategy_choice = st.sidebar.radio(
    "選擇要運行的戰術策略：",
    [
        "1. ACWI 全球戰術輪動策略 (US-Core & Non-US Canary)",
        "2. VTI 美股加強輪動策略 (SMH Alpha & IWM Canary)",
    ],
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
# 策略一：ACWI 全球戰術輪動策略 (嚴格無前視偏誤版)
# =========================================================
if "ACWI" in strategy_choice:
    st.title("🌐 ACWI 全球戰術輪動與基準對標系統")

    tickers_acwi = ["IVV", "VGK", "EWJ", "EPP", "IEMG", "BIL"]
    benchmark_weights_acwi = pd.Series(
        [0.63, 0.13, 0.05, 0.09, 0.11, 0.00], index=tickers_acwi
    )

    with st.expander("🏷️ 點此查看：ACWI 策略邏輯與金絲雀機制說明"):
        st.markdown(
            """
        ### 📌 策略核心邏輯 (US-Core & Non-US Canary TAA)
        以 **IVV** 作為核心，並利用 **IEMG (新興)** 與 **EPP (亞太不含日本)** 作為全球流動性與避險的金絲雀 (Canary) 指標。
        * **🟢 Risk-On (全球牛市 / 雙綠燈)**：IEMG 與 EPP 收盤價皆高於 200MA。
          * 配置：`IVV 70%` | `VGK 10%` | `EWJ 5%` | `EPP 5%` | `IEMG 10%` | `BIL 0%`
        * **🟡 Risk-Neutral (防守切換 / 單綠燈)**：IEMG 與 EPP 僅有 1 支在 200MA 之上。
          * 配置：`IVV 80%` | `VGK 10%` | `EWJ 10%` | `EPP 0%` | `IEMG 0%` | `BIL 0%`
        * **🔴 Risk-Off (避險防衛 / 雙紅燈)**：IEMG 與 EPP 皆跌破 200MA。
          * 配置：`IVV 50%` | `BIL 50%` | 其餘 0%
        """
        )

    with st.spinner("正在讀取 ACWI 策略歷史數據並計算無前視偏誤回測..."):
        try:
            data = load_etf_data(tickers_acwi, start_date, end_date)
        except Exception as e:
            st.error(f"❌ 數據下載失敗：{e}")
            st.stop()

    if data.empty or len(data) < 200:
        st.error("❌ 歷史數據不足 200 天，無法執行回測。")
        st.stop()

    ma200 = data.rolling(window=200).mean()
    returns = data.pct_change().fillna(0)

    # 1. 每日計算金絲雀訊號與理論目標權重 (Signal generated at Day T)
    canary_iemg_series = data["IEMG"] > ma200["IEMG"]
    canary_epp_series = data["EPP"] > ma200["EPP"]

    df_signals = pd.DataFrame(index=data.index)
    df_signals["Canary_IEMG"] = canary_iemg_series
    df_signals["Canary_EPP"] = canary_epp_series

    # 根據 T 日訊號決定 T 日產生的目標權重
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
            w = [0.50, 0.00, 0.00, 0.00, 0.00, 0.50]
        else:
            mode = "Risk-Neutral (🟡)"
            w = [0.80, 0.10, 0.10, 0.00, 0.00, 0.00]

        modes.append(mode)
        target_weights.append(w)

    df_w_target = pd.DataFrame(
        target_weights, index=data.index, columns=tickers_acwi
    )
    df_signals["Mode"] = modes

    # 2. 核心修正：實施 Lag-1 滯後處理 (Day T 產生的訊號與權重，在 Day T+1 執行套用)
    df_w_executed = df_w_target.shift(1).dropna()
    exec_dates = df_w_executed.index[
        df_w_executed.index >= data.index[200]
    ]  # 確保過 200MA

    # 3. 逐日計算執行報酬與 ACWI 基準報酬
    daily_records = []
    for d in exec_dates:
        daily_ret = returns.loc[d]
        strat_w = df_w_executed.loc[d]
        mode_applied = df_signals.loc[d, "Mode"]  # 當天實際套用的市場模式

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

    # 4. 月度彙整 (月度顯示的權重與模式，對齊上個月末產生的決定)
    agg_dict = {
        "Strat_Ret": lambda x: (1 + x).prod() - 1,
        "BM_Ret": lambda x: (1 + x).prod() - 1,
        "Mode": "first",  # 當月第一天套用的模式 (即上月底決策)
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

    # 5. UI 呈現
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
    st.subheader("🔍 金絲雀數據對照：EPP & IEMG 股價 vs 200MA")
    tab1, tab2 = st.tabs(["EPP (亞太不含日本)", "IEMG (新興市場)"])
    with tab1:
        st.line_chart(
            pd.DataFrame(
                {
                    "EPP 收盤價": data["EPP"].loc[exec_dates],
                    "EPP 200MA": ma200["EPP"].loc[exec_dates],
                }
            )
        )
    with tab2:
        st.line_chart(
            pd.DataFrame(
                {
                    "IEMG 收盤價": data["IEMG"].loc[exec_dates],
                    "IEMG 200MA": ma200["IEMG"].loc[exec_dates],
                }
            )
        )

    # 6. 印出當前最新訊號（收盤即時決策）
    st.subheader("🔔 當前最新金絲雀訊號與推薦配置 (即時最新交易日)")
    latest_date = data.index[-1]
    latest_mode = df_signals.loc[latest_date, "Mode"]
    latest_iemg = (
        "🟢 均線上"
        if df_signals.loc[latest_date, "Canary_IEMG"]
        else "🔴 均線下"
    )
    latest_epp = (
        "🟢 均線上"
        if df_signals.loc[latest_date, "Canary_EPP"]
        else "🔴 均線下"
    )
    latest_w = df_w_target.loc[latest_date]

    c_rec1, c_rec2, c_rec3 = st.columns(3)
    c_rec1.info(f"**最新市場狀態**：{latest_mode}")
    c_rec2.write(f"**IEMG Canary**: {latest_iemg} | **EPP Canary**: {latest_epp}")
    c_rec3.write(
        f"**建議持股比例**：IVV {latest_w['IVV']*100:.0f}% | VGK {latest_w['VGK']*100:.0f}% | EWJ {latest_w['EWJ']*100:.0f}% | EPP {latest_w['EPP']*100:.0f}% | IEMG {latest_w['IEMG']*100:.0f}% | BIL {latest_w['BIL']*100:.0f}%"
    )

    st.write("---")
    st.subheader(
        "📋 每月詳細策略報酬、對標與各 ETF 獨立月報酬歷程 (無未來偏誤對齊)"
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

    monthly_display = monthly_display.rename(
        columns={
            "Mode": "當月執行模式",
            "Canary_IEMG": "IEMG 訊號",
            "Canary_EPP": "EPP 訊號",
            "W_IVV": "IVV 權重",
            "W_VGK": "VGK 權重",
            "W_EWJ": "EWJ 權重",
            "W_EPP": "EPP 權重",
            "W_IEMG": "IEMG 權重",
            "W_BIL": "BIL 權重",
        }
    )

    table_cols = [
        "當月執行模式",
        "策略月報酬",
        "ACWI 月報酬",
        "月度勝負",
        "IVV 月報酬",
        "VGK 月報酬",
        "EWJ 月報酬",
        "EPP 月報酬",
        "IEMG 月報酬",
        "BIL 月報酬",
        "IEMG 訊號",
        "EPP 訊號",
        "IVV 權重",
        "VGK 權重",
        "EWJ 權重",
        "EPP 權重",
        "IEMG 權重",
        "BIL 權重",
    ]
    st.dataframe(
        monthly_display[table_cols].sort_index(ascending=False),
        use_container_width=True,
    )


# =========================================================
# 策略二：VTI 美股加強輪動策略 (嚴格無前視偏誤版)
# =========================================================
else:
    st.title("🚀 VTI 美股核心加強型戰術輪動系統")

    tickers_vti = ["VTI", "IVV", "SMH", "IWM", "BIL"]

    with st.expander("🏷️ 點此查看：VTI 策略邏輯與金絲雀機制說明"):
        st.markdown(
            """
        ### 📌 策略核心邏輯 (US-Large Core + SMH Alpha & IWM Canary)
        本策略直接對標 **VTI**，以 **IVV** 為穩健底倉，利用 **SMH (半導體)** 為 Alpha 攻擊引擎，**IWM (小型股)** 作為流動性金絲雀預警標的。
        * **🟢 Risk-On (強勢牛市 / 雙綠燈)**：SMH 與 IWM 均在 200MA 之上。
          * 配置：`IVV 65%` | `SMH 20% (Alpha加碼)` | `IWM 15%` | `BIL 0%`
        * **🟡 Risk-Neutral (防守分化 / 單綠燈)**：SMH 與 IWM 僅 1 支在 200MA 之上。
          * 配置：`IVV 80% (大盤防守)` | 僅持有強勢者 20% | `BIL 0%`
        * **🔴 Risk-Off (系統熊市 / 雙紅燈)**：SMH 與 IWM 皆跌破 200MA。
          * 配置：`IVV 50%` | `BIL 50%` | 其餘 0%
        """
        )

    with st.spinner("正在讀取 VTI 策略歷史數據並計算無前視偏誤回測..."):
        try:
            data = load_etf_data(tickers_vti, start_date, end_date)
        except Exception as e:
            st.error(f"❌ 數據下載失敗：{e}")
            st.stop()

    if data.empty or len(data) < 200:
        st.error("❌ 歷史數據不足 200 天，無法執行回測。")
        st.stop()

    ma200 = data.rolling(window=200).mean()
    returns = data.pct_change().fillna(0)

    # 1. 每日計算金絲雀訊號與目標權重
    canary_smh_series = data["SMH"] > ma200["SMH"]
    canary_iwm_series = data["IWM"] > ma200["IWM"]

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
            w = [0.00, 0.65, 0.20, 0.15, 0.00]
        elif not c_smh and not c_iwm:
            mode = "Risk-Off (🔴🔴)"
            w = [0.00, 0.50, 0.00, 0.00, 0.50]
        else:
            mode = "Risk-Neutral (🟡)"
            if c_smh:
                w = [0.00, 0.80, 0.20, 0.00, 0.00]
            else:
                w = [0.00, 0.80, 0.00, 0.20, 0.00]

        modes.append(mode)
        target_weights.append(w)

    df_w_target = pd.DataFrame(
        target_weights, index=data.index, columns=tickers_vti
    )
    df_signals["Mode"] = modes

    # 2. 核心修正：實施 Lag-1 滯後處理
    df_w_executed = df_w_target.shift(1).dropna()
    exec_dates = df_w_executed.index[df_w_executed.index >= data.index[200]]

    # 3. 逐日計算
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

    # 4. 月度彙整
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
    st.subheader("🔍 金絲雀數據驗證：SMH & IWM 股價 vs 200MA")
    tab1, tab2 = st.tabs(["SMH (半導體 Alpha)", "IWM (小型股金絲雀)"])
    with tab1:
        st.line_chart(
            pd.DataFrame(
                {
                    "SMH 收盤價": data["SMH"].loc[exec_dates],
                    "SMH 200MA": ma200["SMH"].loc[exec_dates],
                }
            )
        )
    with tab2:
        st.line_chart(
            pd.DataFrame(
                {
                    "IWM 收盤價": data["IWM"].loc[exec_dates],
                    "IWM 200MA": ma200["IWM"].loc[exec_dates],
                }
            )
        )

    # 6. 即時最新訊號
    st.subheader("🔔 當前最新金絲雀訊號與推薦配置 (即時最新交易日)")
    latest_date = data.index[-1]
    latest_mode = df_signals.loc[latest_date, "Mode"]
    latest_smh = (
        "🟢 均線上"
        if df_signals.loc[latest_date, "Canary_SMH"]
        else "🔴 均線下"
    )
    latest_iwm = (
        "🟢 均線上"
        if df_signals.loc[latest_date, "Canary_IWM"]
        else "🔴 均線下"
    )
    latest_w = df_w_target.loc[latest_date]

    c_rec1, c_rec2, c_rec3 = st.columns(3)
    c_rec1.info(f"**最新市場狀態**：{latest_mode}")
    c_rec2.write(f"**SMH Canary**: {latest_smh} | **IWM Canary**: {latest_iwm}")
    c_rec3.write(
        f"**建議持股比例**：IVV {latest_w['IVV']*100:.0f}% | SMH {latest_w['SMH']*100:.0f}% | IWM {latest_w['IWM']*100:.0f}% | BIL {latest_w['BIL']*100:.0f}%"
    )

    st.write("---")
    st.subheader(
        "📋 每月詳細策略報酬、對標與各 ETF 獨立月報酬歷程 (無未來偏誤對齊)"
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

    monthly_display = monthly_display.rename(
        columns={
            "Mode": "當月執行模式",
            "Canary_SMH": "SMH 訊號",
            "Canary_IWM": "IWM 訊號",
            "W_IVV": "IVV 權重",
            "W_SMH": "SMH 權重",
            "W_IWM": "IWM 權重",
            "W_BIL": "BIL 權重",
        }
    )

    table_cols = [
        "當月執行模式",
        "策略月報酬",
        "VTI 月報酬",
        "月度勝負",
        "IVV 月報酬",
        "SMH 月報酬",
        "IWM 月報酬",
        "BIL 月報酬",
        "SMH 訊號",
        "IWM 訊號",
        "IVV 權重",
        "SMH 權重",
        "IWM 權重",
        "BIL 權重",
    ]
    st.dataframe(
        monthly_display[table_cols].sort_index(ascending=False),
        use_container_width=True,
    )
