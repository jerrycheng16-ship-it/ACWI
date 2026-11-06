import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------
# 0. 頁面總體配置
# ---------------------------------------------------------
st.set_page_config(
    page_title="雙核心 ETF Canary & RSI 戰術輪動系統",
    layout="wide",
)

# ---------------------------------------------------------
# 側邊欄：策略選擇與動態參數設定
# ---------------------------------------------------------
st.sidebar.title("🎯 策略與動態參數尋優")
strategy_choice = st.sidebar.radio(
    "選擇要運行的戰術策略：",
    [
        "1. ACWI 全球戰術輪動策略 (IEMG/ACWI & EPP/ACWI 比值 RSI)",
        "2. VTI 美股相對 RSI 均值回歸策略 (SMH / IWM Ratio RSI)",
    ],
)

st.sidebar.write("---")
st.sidebar.subheader("⚙️ 量化參數微調 (即時回測)")

ma_window = st.sidebar.slider(
    "均線過濾天數 (MA Filter)",
    min_value=20,
    max_value=250,
    value=130,
    step=10,
    help="建議使用 130MA (半年線) 以提高靈敏度，減少 T+1 滯後造成的平滑損失",
)

rel_rsi_oversold = st.sidebar.slider(
    "相對 RSI 超跌抄底門檻 (進場)",
    min_value=20,
    max_value=45,
    value=35,
    step=5,
    help="當相對 RSI 低於此門檻且站在均線上時，觸發逆勢加碼",
)

rel_rsi_overbought = st.sidebar.slider(
    "相對 RSI 獲利了結門檻 (出場)",
    min_value=50,
    max_value=85,
    value=70,
    step=5,
    help="當相對 RSI 高於此門檻時，獲利了結切回常態/大盤防守持股",
)

smh_add_weight = (
    st.sidebar.slider(
        "VTI 策略：觸發抄底時 SMH 加碼權重 (%)",
        min_value=20,
        max_value=40,
        value=35,
        step=5,
        help="觸發加碼時 SMH 的持股比例 (預設常態為 15%)",
    )
    / 100.0
)

risk_off_mode_acwi = st.sidebar.selectbox(
    "ACWI 策略：Risk-Off 防禦機制：",
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
# 策略一：ACWI 全球戰術輪動策略 (IEMG/ACWI & EPP/ACWI)
# =========================================================
if "ACWI" in strategy_choice:
    st.title("🌐 ACWI 全球戰術輪動系統 (IEMG/ACWI & EPP/ACWI 比值 RSI)")

    tickers_acwi = ["ACWI", "IVV", "VGK", "EWJ", "EPP", "IEMG", "BIL"]
    benchmark_weights_acwi = pd.Series(
        [0.00, 0.63, 0.13, 0.05, 0.09, 0.11, 0.00], index=tickers_acwi
    )

    with st.expander("🏷️ 點此查看：ACWI 相對 RSI (對標全球大盤) 雙門檻邏輯說明"):
        st.markdown(
            f"""
        ### 📌 ACWI 相對 RSI 策略原理 (Relative to Global Market)
        本策略監控 **`IEMG / ACWI` (新興市場比全球大盤)** 與 **`EPP / ACWI` (亞太比全球大盤)** 的 14 日相對 RSI：
        * **進場加碼 (Entry)**：當相對 ACWI 的 RSI < `{rel_rsi_oversold}` 且股價 > `{ma_window}MA` 時，觸發非美市場進場加碼。
        * **離場獲利了結 (Exit)**：當相對 RSI > `{rel_rsi_overbought}` (相對超買高點) 或 股價跌破 `{ma_window}MA` 時，全數獲利了結。
        * **無訊號常態**：撥回 `100% IVV (美股大盤防禦)` 或 `50% IVV + 50% BIL (現金避險)` (視側邊欄選項而定)。
        * **無前視偏誤 (No Look-Ahead Bias)**：嚴格實施 **T+1 日滯後執行 (Shift-1)**。
        """
        )

    with st.spinner("正在讀取歷史數據並計算 IEMG/ACWI 與 EPP/ACWI 相對 RSI..."):
        try:
            data = load_etf_data(tickers_acwi, start_date, end_date)
        except Exception as e:
            st.error(f"❌ 數據下載失敗：{e}")
            st.stop()

    if data.empty or len(data) < ma_window:
        st.error(f"❌ 歷史數據不足 {ma_window} 天，無法執行回測。")
        st.stop()

    # 1. 計算相對比值 (Ratio vs ACWI) 與 14 日 Relative RSI
    ratio_iemg_acwi = data["IEMG"] / data["ACWI"]
    ratio_epp_acwi = data["EPP"] / data["ACWI"]

    rel_rsi_iemg = calculate_rsi(ratio_iemg_acwi, period=14)
    rel_rsi_epp = calculate_rsi(ratio_epp_acwi, period=14)

    ma_filter = data.rolling(window=ma_window).mean()
    returns = data.pct_change().fillna(0)

    # 2. 狀態機控制進出場
    target_weights = []
    modes = []

    holding_iemg_add = False
    holding_epp_add = False

    for d in data.index:
        iemg_above_ma = data.loc[d, "IEMG"] > ma_filter.loc[d, "IEMG"]
        epp_above_ma = data.loc[d, "EPP"] > ma_filter.loc[d, "EPP"]

        c_iemg_rsi = rel_rsi_iemg.loc[d]
        c_epp_rsi = rel_rsi_epp.loc[d]

        # 離場判斷
        if c_iemg_rsi > rel_rsi_overbought or not iemg_above_ma:
            holding_iemg_add = False
        if c_epp_rsi > rel_rsi_overbought or not epp_above_ma:
            holding_epp_add = False

        # 進場判斷
        if (c_iemg_rsi < rel_rsi_oversold) and iemg_above_ma:
            holding_iemg_add = True
        if (c_epp_rsi < rel_rsi_oversold) and epp_above_ma:
            holding_epp_add = True

        # 權重分配 [ACWI, IVV, VGK, EWJ, EPP, IEMG, BIL]
        if holding_iemg_add and holding_epp_add:
            mode = "全球牛市 (🟢🟢 雙重加碼)"
            w = [0.00, 0.60, 0.10, 0.05, 0.10, 0.15, 0.00]
        elif holding_iemg_add and not holding_epp_add:
            mode = "加碼 IEMG (新興市場超跌)"
            w = [0.00, 0.70, 0.10, 0.05, 0.00, 0.15, 0.00]
        elif holding_epp_add and not holding_iemg_add:
            mode = "加碼 EPP (亞太市場超跌)"
            w = [0.00, 0.70, 0.10, 0.05, 0.15, 0.00, 0.00]
        else:
            mode = "大盤常態防禦 (🔴🔴)"
            if "100%" in risk_off_mode_acwi:
                w = [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00]
            else:
                w = [0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.50]

        modes.append(mode)
        target_weights.append(w)

    df_w_target = pd.DataFrame(
        target_weights, index=data.index, columns=tickers_acwi
    )
    df_signals = pd.DataFrame(
        {
            "Mode": modes,
            "Rel_RSI_IEMG": rel_rsi_iemg,
            "Rel_RSI_EPP": rel_rsi_epp,
        },
        index=data.index,
    )

    # 3. Lag-1 Shift 消除前視偏誤
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

        bm_ret = (daily_ret * benchmark_weights_acwi).sum()
        strat_ret = (daily_ret * strat_w).sum()

        rec = {
            "Date": d,
            "Mode": mode_applied,
            "Rel_RSI_IEMG": f"{df_signals_executed.loc[d, 'Rel_RSI_IEMG']:.1f}",
            "Rel_RSI_EPP": f"{df_signals_executed.loc[d, 'Rel_RSI_EPP']:.1f}",
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
            if t != "ACWI":
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

    # 月度統計 (統一用 first 對齊當月首個交易日)
    agg_dict = {
        "Strat_Ret": lambda x: (1 + x).prod() - 1,
        "BM_Ret": lambda x: (1 + x).prod() - 1,
        "Mode": "first",
        "W_IVV": "first",
        "W_VGK": "first",
        "W_EWJ": "first",
        "W_EPP": "first",
        "W_IEMG": "first",
        "W_BIL": "first",
    }
    for t in tickers_acwi:
        if t != "ACWI":
            agg_dict[f"R_{t}"] = lambda x, symbol=t: (1 + x).prod() - 1

    monthly_df = df_daily.resample("ME").agg(agg_dict)
    win_rate = (monthly_df["Strat_Ret"] > monthly_df["BM_Ret"]).mean() * 100

    # 畫面呈現
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "ACWI RSI 策略總報酬",
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
            {"ACWI Ratio RSI 策略": strat_cum, "ACWI 基準對標": bm_cum}
        )
    )

    st.write("### 📉 風險水庫：水面下回撤走勢圖 (Drawdown %)")
    st.line_chart(
        pd.DataFrame(
            {
                "RSI 策略 Drawdown (%)": strat_dd_series,
                "ACWI 基準 Drawdown (%)": bm_dd_series,
            }
        )
    )

    # ACWI 最新訊號卡片
    st.write("---")
    st.subheader("🔔 當前 11 月最新交易訊號與建倉推薦配置 (即時盤後)")
    latest_date_acwi = data.index[-1]
    latest_mode_acwi = df_signals.loc[latest_date_acwi, "Mode"]
    latest_rsi_iemg_val = rel_rsi_iemg.loc[latest_date_acwi]
    latest_rsi_epp_val = rel_rsi_epp.loc[latest_date_acwi]
    latest_w_acwi = df_w_target.loc[latest_date_acwi]

    a_rec1, a_rec2, a_rec3 = st.columns(3)
    a_rec1.info(
        f"**最新交易日**: {latest_date_acwi.strftime('%Y-%m-%d')}\n\n**當前市場模式**: {latest_mode_acwi}"
    )
    a_rec2.warning(
        f"**IEMG/ACWI 相對 RSI**: {latest_rsi_iemg_val:.1f}\n\n**EPP/ACWI 相對 RSI**: {latest_rsi_epp_val:.1f}"
    )
    a_rec3.success(
        f"**建議 11 月配置比例**:\n\nIVV: {latest_w_acwi['IVV']*100:.0f}% | VGK: {latest_w_acwi['VGK']*100:.0f}% | EWJ: {latest_w_acwi['EWJ']*100:.0f}%\n\nEPP: {latest_w_acwi['EPP']*100:.0f}% | IEMG: {latest_w_acwi['IEMG']*100:.0f}% | BIL: {latest_w_acwi['BIL']*100:.0f}%"
    )

    # 相對 RSI 圖表
    st.write("---")
    st.subheader(
        "📊 相對強度 RSI (Ratio vs ACWI) 歷史對照圖與進/出場門檻"
    )
    tab_acwi1, tab_acwi2 = st.tabs(
        ["IEMG / ACWI 相對 RSI", "EPP / ACWI 相對 RSI"]
    )

    with tab_acwi1:
        st.line_chart(
            pd.DataFrame(
                {
                    "IEMG/ACWI 相對 14日 RSI": rel_rsi_iemg.loc[exec_dates],
                    "超跌抄底進場門檻": rel_rsi_oversold,
                    "超買獲利了結門檻": rel_rsi_overbought,
                }
            )
        )

    with tab_acwi2:
        st.line_chart(
            pd.DataFrame(
                {
                    "EPP/ACWI 相對 14日 RSI": rel_rsi_epp.loc[exec_dates],
                    "超跌抄底進場門檻": rel_rsi_oversold,
                    "超買獲利了結門檻": rel_rsi_overbought,
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
    monthly_display["ACWI 月報酬"] = (
        monthly_display["BM_Ret"] * 100
    ).map("{:.2f}%".format)
    monthly_display["月度勝負"] = np.where(
        monthly_display["Strat_Ret"] > monthly_display["BM_Ret"],
        "🏆 擊敗 ACWI",
        "🔻 落後 ACWI",
    )

    for t in tickers_acwi:
        if t != "ACWI":
            monthly_display[f"{t} 月報酬"] = (
                monthly_display[f"R_{t}"] * 100
            ).map("{:.2f}%".format)
            monthly_display[f"W_{t}"] = (monthly_display[f"W_{t}"] * 100).map(
                "{:.0f}%".format
            )

    monthly_display = monthly_display.rename(
        columns={
            "Mode": "當月執行模式",
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
# 策略二：VTI 美股相對 RSI 均值回歸策略 (進場 + 出場門檻控制)
# =========================================================
else:
    st.title("⚖️ 相對強度 (SMH/VTI & IWM/VTI) RSI 均值回歸戰術系統")

    tickers_vti = ["VTI", "IVV", "SMH", "IWM", "BIL"]

    with st.expander("🏷️ 點此查看：進場/出場雙重門檻控制邏輯說明"):
        st.markdown(
            f"""
        ### 📌 雙重門檻控制邏輯 (Entry & Exit Rules)
        * **常態持股 (無加碼訊號)**：持有 `85% IVV` (高勝率核心) + `15% SMH` (常態動能)。
        * **進場條件 (超跌抄底 Buy-in)**：
          * 當 `SMH/VTI` RSI < `{rel_rsi_oversold}` 且 SMH > `{ma_window}MA` ➔ **加碼 SMH 至 {int(smh_add_weight*100)}%**。
          * 當 `IWM/VTI` RSI < `{rel_rsi_oversold}` 且 IWM > `{ma_window}MA` ➔ **加碼 IWM 至 30%**。
        * **出場/獲利了結條件 (Profit Take Exit)**：
          * 當相對 RSI > `{rel_rsi_overbought}` (超買高點) 或 股價跌破 `{ma_window}MA` ➔ **全數獲利了結，權重回歸常態 15%**。
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

    # 3. 狀態機與離場條件邏輯
    target_weights = []
    modes = []

    holding_smh_add = False
    holding_iwm_add = False

    for d in data.index:
        smh_above_ma = data.loc[d, "SMH"] > ma_filter.loc[d, "SMH"]
        iwm_above_ma = data.loc[d, "IWM"] > ma_filter.loc[d, "IWM"]

        c_smh_rsi = rel_rsi_smh.loc[d]
        c_iwm_rsi = rel_rsi_iwm.loc[d]

        # 離場觸發 (高於出場門檻 或 跌破均線)
        if c_smh_rsi > rel_rsi_overbought or not smh_above_ma:
            holding_smh_add = False
        if c_iwm_rsi > rel_rsi_overbought or not iwm_above_ma:
            holding_iwm_add = False

        # 進場觸發 (低於進場門檻 且 站在均線上)
        if (c_smh_rsi < rel_rsi_oversold) and smh_above_ma:
            holding_smh_add = True
        if (c_iwm_rsi < rel_rsi_oversold) and iwm_above_ma:
            holding_iwm_add = True

        # 決定當日目標權重
        if holding_smh_add and not holding_iwm_add:
            mode = f"加碼 SMH (抄底至 {int(smh_add_weight*100)}%)"
            w = [
                0.00,
                round(1.0 - smh_add_weight, 2),
                smh_add_weight,
                0.00,
                0.00,
            ]
        elif holding_iwm_add and not holding_smh_add:
            mode = "加碼 IWM (逆勢抄底至 30%)"
            w = [0.00, 0.70, 0.00, 0.30, 0.00]
        elif holding_smh_add and holding_iwm_add:
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

    # 4. Lag-1 Shift 消除前視偏誤
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

    # 月度統計
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
    col3.metric("策略月度超越 VTI 勝率", f"{win_rate:.2f}%")

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

    # VTI 最新訊號卡片
    st.write("---")
    st.subheader("🔔 當前 11 月最新交易訊號與建倉推薦配置 (即時盤後)")
    latest_date_vti = data.index[-1]
    latest_mode_vti = df_signals.loc[latest_date_vti, "Mode"]
    latest_rsi_smh_val = rel_rsi_smh.loc[latest_date_vti]
    latest_rsi_iwm_val = rel_rsi_iwm.loc[latest_date_vti]
    latest_w_vti = df_w_target.loc[latest_date_vti]

    v_rec1, v_rec2, v_rec3 = st.columns(3)
    v_rec1.info(
        f"**最新交易日**: {latest_date_vti.strftime('%Y-%m-%d')}\n\n**當前市場模式**: {latest_mode_vti}"
    )
    v_rec2.warning(
        f"**SMH/VTI 相對 RSI**: {latest_rsi_smh_val:.1f}\n\n**IWM/VTI 相對 RSI**: {latest_rsi_iwm_val:.1f}"
    )
    v_rec3.success(
        f"**建議 11 月配置比例**:\n\nIVV: {latest_w_vti['IVV']*100:.0f}% | SMH: {latest_w_vti['SMH']*100:.0f}%\n\nIWM: {latest_w_vti['IWM']*100:.0f}% | BIL: {latest_w_vti['BIL']*100:.0f}%"
    )

    # 6. 相對 RSI 圖表
    st.write("---")
    st.subheader(
        "📊 相對強度 RSI (Ratio RSI) 歷史對照圖與進/出場門檻"
    )
    tab_rel1, tab_rel2 = st.tabs(["SMH / VTI 相對 RSI", "IWM / VTI 相對 RSI"])

    with tab_rel1:
        st.line_chart(
            pd.DataFrame(
                {
                    "SMH/VTI 相對 14日 RSI": rel_rsi_smh.loc[exec_dates],
                    "超跌抄底進場門檻": rel_rsi_oversold,
                    "超買獲利了結門檻": rel_rsi_overbought,
                }
            )
        )

    with tab_rel2:
        st.line_chart(
            pd.DataFrame(
                {
                    "IWM/VTI 相對 14日 RSI": rel_rsi_iwm.loc[exec_dates],
                    "超跌抄底進場門檻": rel_rsi_oversold,
                    "超買獲利了結門檻": rel_rsi_overbought,
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
