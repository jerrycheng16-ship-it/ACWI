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
# 側邊欄：策略選擇與動態參數設定
# ---------------------------------------------------------
st.sidebar.title("🎯 策略與動態參數尋優")
strategy_choice = st.sidebar.radio(
    "選擇要運行的戰術策略：",
    [
        "1. ACWI 全球戰術輪動策略 (US-Core & Non-US Canary)",
        "2. VTI 美股相對 RSI 均值回歸策略 (SMH / IWM Ratio RSI)",
    ],
)

st.sidebar.write("---")
st.sidebar.subheader("⚙️ 量化參數微調 (即時回測)")

ma_window = st.sidebar.slider(
    "金絲雀 Canary / 均線過濾天數 (MA)",
    min_value=20,
    max_value=250,
    value=130,
    step=10,
    help="建議使用 130MA (半年線) 以提高靈敏度，減少 T+1 滯後造成的平滑損失",
)

rel_rsi_oversold = st.sidebar.slider(
    "VTI 策略：相對 RSI 超跌抄底門檻",
    min_value=20,
    max_value=45,
    value=35,
    step=5,
    help="當 SMH/VTI 或 IWM/VTI 的 RSI 低於此門檻且站在均線上時，觸發逆勢加碼",
)

smh_add_weight = st.sidebar.slider(
    "VTI 策略：觸發抄底時 SMH 加碼權重 (%)",
    min_value=20,
    max_value=40,
    value=35,
    step=5,
    help="觸發加碼時 SMH 的持股比例 (預設常態為 15%)",
) / 100.0

risk_off_mode_acwi = st.sidebar.selectbox(
    "ACWI 策略：Risk-Off (雙紅燈) 防禦機制：",
    ["1. IVV 100% 抱緊美股大盤防衛", "2. 50% IVV + 50% BIL 現金避險"],
    help="美股具高 ROE 與長牛特性，全撥回 IVV 通常能有效避開低點拋售避險造成的傷害",
)

st.sidebar.write("---")
start_date = st.sidebar.date_input(
    "起始日期", pd.to_datetime("2013-01-01")
).strftime("%Y-%m-%d")
end_date = st.sidebar.date_input(
    "結束日期", pd.to_datetime("2026-11-01")
).strftime("%Y-%m-%d")


# ---------------------------------------------------------
# 通用數據下載與計算函式 (穩定快取)
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


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def get_drawdown_series(cum_returns):
    peak = cum_returns.cummax()
    return (cum_returns - peak) / peak * 100


# =========================================================
# 策略一：ACWI 全球戰術輪動策略 (以 EPP 替代 IAPD，無前視偏誤)
# =========================================================
if "ACWI" in strategy_choice:
    st.title("🌐 ACWI 全球戰術輪動對標系統 (嚴格無前視偏誤對齊)")

    tickers_acwi = ["IVV", "VGK", "EWJ", "EPP", "IEMG", "BIL"]
    benchmark_weights_acwi = pd.Series(
        [0.63, 0.13, 0.05, 0.09, 0.11, 0.00], index=tickers_acwi
    )

    with st.expander("🏷️ 點此查看：ACWI 策略邏輯與金絲雀機制說明"):
        st.markdown(
            """
        ### 📌 策略核心邏輯 (US-Core & Non-US Canary TAA)
        以 **IVV** 作為核心，並利用 **IEMG (新興)** 與 **EPP (亞太不含日本)** 作為全球流動性與避險的金絲雀 (Canary) 指標。
        * **🟢 Risk-On (全球牛市 / 雙綠燈)**：IEMG 與 EPP 收盤價皆高於 均線 (MA)。
          * 配置：`IVV 70%` | `VGK 10%` | `EWJ 5%` | `EPP 5%` | `IEMG 10%` | `BIL 0%`
        * **🟡 Risk-Neutral (防守切換 / 單綠燈)**：IEMG 與 EPP 僅有 1 支在 均線 之上。
          * 配置：`IVV 80%` | `VGK 10%` | `EWJ 10%` | `EPP 0%` | `IEMG 0%` | `BIL 0%`
        * **🔴 Risk-Off (避險防衛 / 雙紅燈)**：IEMG 與 EPP 皆跌破 均線。
          * 配置 (根據側邊欄)：`IVV 100% (大盤防衛)` 或 `IVV 50% + BIL 50% (現金避險)`
        """
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

    # 1. T 日計算金絲雀訊號與目標權重
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
            if "100%" in risk_off_mode_acwi:
                w = [1.00, 0.00, 0.00, 0.00, 0.00, 0.00]
            else:
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

    # 2. 嚴格實施 Lag-1 滯後處理 (Shift-1) 消除前視偏誤
    df_w_executed = df_w_target.shift(1).dropna()
    exec_dates = df_w_executed.index[
        df_w_executed.index >= data.index[ma_window]
    ]

    daily_records = []
    for d in exec_dates:
        daily_ret = returns.loc[d]
        strat_w = df_w_executed.loc[d]
        mode_applied = df_signals.shift(1).loc[d, "Mode"]

        bm_ret = (daily_ret * benchmark_weights_acwi).sum()
        strat_ret = (daily_ret * strat_w).sum()

        rec = {
            "Date": d,
            "Mode": mode_applied,
            "Canary_IEMG": (
                "🟢 均線上"
                if df_signals.shift(1).loc[d, "Canary_IEMG"]
                else "🔴 均線下"
            ),
            "Canary_EPP": (
                "🟢 均線上"
                if df_signals.shift(1).loc[d, "Canary_EPP"]
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

    # 月度彙整 (精準對齊當月第一天實際執行的模式與權重)
    agg_dict = {
        "Strat_Ret": lambda x: (1 + x).prod() - 1,
        "BM_Ret": lambda x: (1 + x).prod() - 1,
        "Mode": "first",
        "Canary_IEMG": "first",
        "Canary_EPP": "first",
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

    # 畫面呈現
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
        f"🔍 金絲雀數據對照：EPP & IEMG 股價 vs {ma_window}MA"
    )
    tab1, tab2 = st.tabs(["EPP (亞太不含日本)", "IEMG (新興市場)"])
    with tab1:
        st.line_chart(
            pd.DataFrame(
                {
                    "EPP 收盤價": data["EPP"].loc[exec_dates],
                    f"EPP {ma_window}MA": ma_target["EPP"].loc[exec_dates],
                }
            )
        )
    with tab2:
        st.line_chart(
            pd.DataFrame(
                {
                    "IEMG 收盤價": data["IEMG"].loc[exec_dates],
                    f"IEMG {ma_window}MA": ma_target["IEMG"].loc[exec_dates],
                }
            )
        )

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
        "📋 每月詳細策略報酬與各 ETF 獨立月報酬歷程 (T+1 執行對齊)"
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
# 策略二：VTI 美股相對 RSI 均值回歸策略 (SMH / IWM Ratio RSI)
# =========================================================
else:
    st.title("⚖️ 相對強度 (SMH/VTI & IWM/VTI) RSI 均值回歸戰術系統")

    tickers_vti = ["VTI", "IVV", "SMH", "IWM", "BIL"]

    with st.expander("🏷️ 點此查看：相對 RSI (Ratio RSI) 均值回歸邏輯說明"):
        st.markdown(
            f"""
        ### 📌 相對強度 RSI 策略設計原理 (Relative Strength RSI Reversion)
        本策略監控 **「相對大盤 (VTI) 的比值 RSI」**：
        * **`SMH / VTI Ratio`**：評估半導體相較於整體美股的超買/超賣強度。
        * **`IWM / VTI Ratio`**：評估小型股相較於整體美股的強弱輪動。
        * **常態持股 (無抄底訊號)**：持有 `85% IVV` (高勝率核心) + `15% SMH` (常態動能)。
        * **抄底機制**：當 `SMH/VTI` 的 RSI < `{rel_rsi_oversold}`，且 SMH 股價在 `{ma_window}MA` 之上時，**將 SMH 加碼至 {int(smh_add_weight*100)}% (IVV 自動降至 {int(round((1-smh_add_weight)*100, 0))}%)**。
        * **無前視偏誤 (No Look-Ahead Bias)**：嚴格實施 **T+1 日滯後執行 (Shift-1)**。
        """
        )

    with st.spinner("正在讀取歷史數據並計算 SMH/VTI 與 IWM/VTI 相對 RSI..."):
        try:
            data = load_etf_data(tickers_vti, start_date, end_date)
        except Exception as e:
            st.error(f"❌ 數據下載失敗：{e}")
            st.stop()

    if data.empty or len(data) < ma_window:
        st.error(f"❌ 歷史數據不足 {ma_window} 天，無法執行回測。")
        st.stop()

    # 1. 計算相對比值 (Ratio)
    ratio_smh_vti = data["SMH"] / data["VTI"]
    ratio_iwm_vti = data["IWM"] / data["VTI"]

    # 2. 計算相對比值的 14 日 RSI
    rel_rsi_smh = calculate_rsi(ratio_smh_vti, period=14)
    rel_rsi_iwm = calculate_rsi(ratio_iwm_vti, period=14)

    # 均線過濾網 (絕對股價 vs 均線)
    ma_filter = data.rolling(window=ma_window).mean()
    returns = data.pct_change().fillna(0)

    # 3. 逐日產生目標權重 (T 日產生的訊號)
    target_weights = []
    modes = []

    for d in data.index:
        smh_above_ma = data.loc[d, "SMH"] > ma_filter.loc[d, "SMH"]
        iwm_above_ma = data.loc[d, "IWM"] > ma_filter.loc[d, "IWM"]

        c_smh_rel_rsi = rel_rsi_smh.loc[d]
        c_iwm_rel_rsi = rel_rsi_iwm.loc[d]

        smh_buy_signal = (c_smh_rel_rsi < rel_rsi_oversold) and smh_above_ma
        iwm_buy_signal = (c_iwm_rel_rsi < rel_rsi_oversold) and iwm_above_ma

        if smh_buy_signal and not iwm_buy_signal:
            mode = f"加碼 SMH (抄底至 {int(smh_add_weight*100)}%)"
            w = [
                0.00,
                round(1.0 - smh_add_weight, 2),
                smh_add_weight,
                0.00,
                0.00,
            ]
        elif iwm_buy_signal and not smh_buy_signal:
            mode = "加碼 IWM (逆勢抄底至 30%)"
            w = [0.00, 0.70, 0.00, 0.30, 0.00]
        elif smh_buy_signal and iwm_buy_signal:
            mode = "雙資產相對超跌 ➔ 雙重加碼"
            w = [0.00, 0.60, 0.20, 0.20, 0.00]
        else:
            mode = "常態持股 (IVV 85% + SMH 15%)"
            w = [0.00, 0.85, 0.15, 0.00, 0.00]

        modes.append(mode)
        target_weights.append(w)

    df_w_target = pd.DataFrame(
        target_weights, index=data.index, columns=tickers_vti
    )
    df_signals = pd.DataFrame(
        {
            "Mode": modes,
            "Rel_RSI_SMH": rel_rsi_smh,
            "Rel_RSI_IWM": rel_rsi_iwm,
        },
        index=data.index,
    )

    # 4. Lag-1 Shift 消除前視偏誤 (T 日產生的權重與模式，於 T+1 實施套用)
    df_w_executed = df_w_target.shift(1).dropna()
    df_signals_executed = df_signals.shift(1).dropna()

    exec_dates = df_w_executed.index[
        df_w_executed.index >= data.index[ma_window]
    ]

    daily_records = []
    for d in exec_dates:
        daily_ret = returns.loc[d]
        strat_w = df_w_executed.loc[d]
        mode_applied = df_signals_executed.loc[d, "Mode"]

        vti_ret = daily_ret["VTI"]
        strat_ret = (daily_ret * strat_w).sum()

        rec = {
            "Date": d,
            "Mode": mode_applied,
            "Rel_RSI_SMH": f"{df_signals_executed.loc[d, 'Rel_RSI_SMH']:.1f}",
            "Rel_RSI_IWM": f"{df_signals_executed.loc[d, 'Rel_RSI_IWM']:.1f}",
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

    # 月度統計 (統一使用 first 對齊當月第一天實際執行的權重與模式)
    agg_dict = {
        "Strat_Ret": lambda x: (1 + x).prod() - 1,
        "VTI_Ret": lambda x: (1 + x).prod() - 1,
        "Mode": "first",
        "W_IVV": "first",
        "W_SMH": "first",
        "W_IWM": "first",
        "W_BIL": "first",
    }
    for t in tickers_vti:
        agg_dict[f"R_{t}"] = lambda x, symbol=t: (1 + x).prod() - 1

    monthly_df = df_daily.resample("ME").agg(agg_dict)
    win_rate = (monthly_df["Strat_Ret"] > monthly_df["VTI_Ret"]).mean() * 100

    # 5. 畫面呈現
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "相對 RSI 策略總報酬",
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

    st.write("### 📈 累積淨值走勢比較 (基期 = 1.0)")
    st.line_chart(
        pd.DataFrame(
            {
                "相對 RSI 均值回歸策略": strat_cum,
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

    # 6. 相對 RSI 圖表
    st.write("---")
    st.subheader(
        "📊 相對強度 RSI (Ratio RSI) 歷史對照圖 (SMH/VTI & IWM/VTI)"
    )
    tab_rel1, tab_rel2 = st.tabs(["SMH / VTI 相對 RSI", "IWM / VTI 相對 RSI"])

    with tab_rel1:
        st.line_chart(
            pd.DataFrame(
                {
                    "SMH/VTI 相對 14日 RSI": rel_rsi_smh.loc[exec_dates],
                    "超跌買入門檻": rel_rsi_oversold,
                }
            )
        )

    with tab_rel2:
        st.line_chart(
            pd.DataFrame(
                {
                    "IWM/VTI 相對 14日 RSI": rel_rsi_iwm.loc[exec_dates],
                    "超跌買入門檻": rel_rsi_oversold,
                }
            )
        )

    st.write("---")
    st.subheader(
        "📋 每月詳細策略報酬與各 ETF 獨立月報酬歷程 (T+1 執行對齊)"
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
        "IVV 權重",
        "SMH 權重",
        "IWM 權重",
        "BIL 權重",
    ]
    st.dataframe(
        monthly_display[table_cols].sort_index(ascending=False),
        use_container_width=True,
    )
