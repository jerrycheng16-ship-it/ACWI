import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------
# 0. 頁面總體配置
# ---------------------------------------------------------
st.set_page_config(
    page_title="相對 RSI (Ratio RSI) 均值回歸戰術系統",
    layout="wide",
)

# ---------------------------------------------------------
# 側邊欄：相對 RSI 參數設定
# ---------------------------------------------------------
st.sidebar.title("🎯 相對 RSI (Relative RSI) 參數")

st.sidebar.subheader("⚙️ 均線過濾與相對 RSI 超買超賣門檻")
filter_ma_window = st.sidebar.slider(
    "均線過濾網天數 (MA Filter)",
    min_value=20,
    max_value=250,
    value=130,
    step=10,
    help="確保標的本身高於均線，避免價值陷阱",
)

rel_rsi_oversold = st.sidebar.slider(
    "相對 RSI 超跌抄底門檻",
    min_value=20,
    max_value=45,
    value=35,
    step=5,
    help="當 SMH/VTI 或 IWM/VTI 的 RSI 低於此門檻時觸發抄底",
)

rel_rsi_overbought = st.sidebar.slider(
    "相對 RSI 超買獲利了結門檻",
    min_value=60,
    max_value=85,
    value=70,
    step=5,
    help="當相對 RSI 高於此門檻時獲利了結轉回 IVV/VTI",
)

st.sidebar.write("---")
start_date = st.sidebar.date_input(
    "起始日期", pd.to_datetime("2013-01-01")
).strftime("%Y-%m-%d")
end_date = st.sidebar.date_input(
    "結束日期", pd.to_datetime("2026-11-01")
).strftime("%Y-%m-%d")


# ---------------------------------------------------------
# 通用數據下載與計算函式
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
# 主程式：對標 VTI 的相對 RSI (Ratio RSI) 戰術系統
# =========================================================
st.title("⚖️ 相對強度 (SMH/VTI & IWM/VTI) RSI 均值回歸戰術系統")

with st.expander("🏷️ 點此查看：相對 RSI (Ratio RSI) 邏輯說明"):
    st.markdown(
        f"""
    ### 📌 相對強度 RSI 策略設計原理 (Relative Strength RSI Reversion)
    本策略不再單純監控單一 ETF 的 RSI，而是監控 **「相對大盤 (VTI) 的比值 RSI」**[cite: 9]：
    * **`SMH / VTI Ratio`**：評估半導體相較於整體美股的超買/超賣強度[cite: 9]。
    * **`IWM / VTI Ratio`**：評估小型股相較於整體美股的強弱輪動[cite: 9]。
    * **抄底機制**：當 `SMH/VTI` 的 RSI < `{rel_rsi_oversold}`，且 SMH 股價在 `{filter_ma_window}MA` 之上時，代表**半導體相對大盤極度超跌但長期趨勢未壞**，觸發加碼。
    * **無前視偏誤 (No Look-Ahead Bias)**：嚴格實施 **T+1 日滯後執行 (Shift-1)**。
    """
    )

tickers_vti = ["VTI", "IVV", "SMH", "IWM", "BIL"]

with st.spinner("正在讀取歷史數據並計算 SMH/VTI 與 IWM/VTI 相對 RSI..."):
    try:
        data = load_etf_data(tickers_vti, start_date, end_date)
    except Exception as e:
        st.error(f"❌ 數據下載失敗：{e}")
        st.stop()

if data.empty or len(data) < filter_ma_window:
    st.error(f"❌ 歷史數據不足 {filter_ma_window} 天，無法執行回測。")
    st.stop()

# 1. 計算相對比值 Series (Ratio)
ratio_smh_vti = data["SMH"] / data["VTI"]
ratio_iwm_vti = data["IWM"] / data["VTI"]

# 2. 計算相對比值的 14 日 RSI
rel_rsi_smh = calculate_rsi(ratio_smh_vti, period=14)
rel_rsi_iwm = calculate_rsi(ratio_iwm_vti, period=14)

# 均線過濾網 (絕對股價 vs 均線)
ma_filter = data.rolling(window=filter_ma_window).mean()
returns = data.pct_change().fillna(0)

# 3. 逐日產生目標權重
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
        mode = "SMH/VTI 相對超跌 ➔ 加碼 SMH"
        w = [0.00, 0.70, 0.30, 0.00, 0.00]
    elif iwm_buy_signal and not smh_buy_signal:
        mode = "IWM/VTI 相對超跌 ➔ 加碼 IWM"
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

# 4. Lag-1 Shift 消除前視偏誤
df_w_executed = df_w_target.shift(1).dropna()
exec_dates = df_w_executed.index[
    df_w_executed.index >= data.index[filter_ma_window]
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
        "Rel_RSI_SMH": f"{df_signals.loc[d, 'Rel_RSI_SMH']:.1f}",
        "Rel_RSI_IWM": f"{df_signals.loc[d, 'Rel_RSI_IWM']:.1f}",
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

# ---------------------------------------------------------
# 5. 畫面呈現：KPI 指標與核心圖表
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 6. 相對 RSI (SMH/VTI & IWM/VTI) 專屬視圖
# ---------------------------------------------------------
st.write("---")
st.subheader(
    "📊 相對強度 RSI (Ratio RSI) 歷史對照圖 (SMH/VTI & IWM/VTI)"
)
tab_rel1, tab_rel2 = st.tabs(["SMH / VTI 相對 RSI", "IWM / VTI 相對 RSI"])

with tab_rel1:
    df_rel_smh_plot = pd.DataFrame(
        {
            "SMH/VTI 相對 14日 RSI": rel_rsi_smh.loc[exec_dates],
            "相對超跌買入門檻": rel_rsi_oversold,
            "相對超買賣出門檻": rel_rsi_overbought,
        }
    )
    st.line_chart(df_rel_smh_plot)

with tab_rel2:
    df_rel_iwm_plot = pd.DataFrame(
        {
            "IWM/VTI 相對 14日 RSI": rel_rsi_iwm.loc[exec_dates],
            "相對超跌買入門檻": rel_rsi_oversold,
            "相對超買賣出門檻": rel_rsi_overbought,
        }
    )
    st.line_chart(df_rel_iwm_plot)

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

table_cols = [
    "Mode",
    "策略月報酬",
    "VTI 月報酬",
    "月度勝負",
    "IVV 月報酬",
    "SMH 月報酬",
    "IWM 月報酬",
    "BIL 月報酬",
    "W_IVV",
    "W_SMH",
    "W_IWM",
    "W_BIL",
]
st.dataframe(
    monthly_display[table_cols].sort_index(ascending=False),
    use_container_width=True,
)
