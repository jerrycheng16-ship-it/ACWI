import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------
# 0. 頁面總體配置
# ---------------------------------------------------------
st.set_page_config(
    page_title="均值回歸與均線過濾戰術輪動系統",
    layout="wide",
)

# ---------------------------------------------------------
# 側邊欄：策略與動態均值回歸參數
# ---------------------------------------------------------
st.sidebar.title("🎯 均值回歸戰術參數設定")

st.sidebar.subheader("⚙️ 均線過濾網與 RSI 超買超賣觸發")
filter_ma_window = st.sidebar.slider(
    "均線過濾網天數 (MA Filter)",
    min_value=20,
    max_value=250,
    value=130,
    step=10,
    help="確保低勝率資產已止跌打底，站在均線上才允許逢低加碼，避免價值陷阱",
)

rsi_oversold = st.sidebar.slider(
    "超跌買入門檻 (RSI 門檻)",
    min_value=20,
    max_value=45,
    value=35,
    step=5,
    help="當低勝率資產 RSI 低於此數值時，觸發均值回歸逆勢加碼",
)

rsi_overbought = st.sidebar.slider(
    "高檔獲利了結門檻 (RSI 門檻)",
    min_value=60,
    max_value=85,
    value=70,
    step=5,
    help="當低勝率資產 RSI 高於此數值時，獲利了結轉回 IVV 核心",
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
# 主程式：VTI 均值回歸加強戰術策略
# =========================================================
st.title("🔄 反向均值回歸與均線過濾戰術系統 (對標 VTI)")

with st.expander("🏷️ 點此查看：均值回歸與價值陷阱防禦邏輯說明"):
    st.markdown(
        f"""
    ### 📌 策略設計原理 (Anti-Trend & Filtered Mean Reversion)
    針對 **IVV (美股大盤)** 勝率高、而 **SMH (半導體)** 與 **IWM (小型股)** 滾動勝率低於五成的特性設計：
    * **核心常態底倉**：預設持有 `85% IVV` 作為高勝率核心引擎。
    * **逢低抄底機制 (Mean Reversion Buy)**：當 IWM 或 SMH 的 RSI 低於 `{rsi_oversold}`，且股價**高於 {filter_ma_window}MA (無崩盤危機，已止跌)**，撥出 15%~30% 資金逆勢加碼。
    * **高檔獲利了結 (Profit Take)**：當 RSI > `{rsi_overbought}` 時全數獲利了結，資金 100% 轉回 IVV 避開後續拉回。
    * **無前視偏誤 (No Look-Ahead Bias)**：完全採納 **T+1 日滯後執行 (Shift-1)**。
    """
    )

tickers_vti = ["VTI", "IVV", "SMH", "IWM", "BIL"]

with st.spinner("正在讀取歷史數據並計算 RSI 與均線過濾指標..."):
    try:
        data = load_etf_data(tickers_vti, start_date, end_date)
    except Exception as e:
        st.error(f"❌ 數據下載失敗：{e}")
        st.stop()

if data.empty or len(data) < filter_ma_window:
    st.error(f"❌ 歷史數據不足 {filter_ma_window} 天，無法執行回測。")
    st.stop()

# 計算技術指標
ma_filter = data.rolling(window=filter_ma_window).mean()
rsi_smh = calculate_rsi(data["SMH"], period=14)
rsi_iwm = calculate_rsi(data["IWM"], period=14)
returns = data.pct_change().fillna(0)

# 逐日產生訊號與目標權重
target_weights = []
modes = []

for d in data.index:
    # 判斷是否高於均線 (止跌過濾)
    smh_above_ma = data.loc[d, "SMH"] > ma_filter.loc[d, "SMH"]
    iwm_above_ma = data.loc[d, "IWM"] > ma_filter.loc[d, "IWM"]

    c_smh_rsi = rsi_smh.loc[d]
    c_iwm_rsi = rsi_iwm.loc[d]

    # 1. 超跌抄底條件：RSI 小於超賣門檻 且 站在均線之上
    smh_buy_signal = (c_smh_rsi < rsi_oversold) and smh_above_ma
    iwm_buy_signal = (c_iwm_rsi < rsi_oversold) and iwm_above_ma

    # 2. 決定動態權重
    if smh_buy_signal and not iwm_buy_signal:
        mode = "加碼 SMH (逆勢抄底)"
        w = [0.00, 0.70, 0.30, 0.00, 0.00]
    elif iwm_buy_signal and not smh_buy_signal:
        mode = "加碼 IWM (逆勢抄底)"
        w = [0.00, 0.70, 0.00, 0.30, 0.00]
    elif smh_buy_signal and iwm_buy_signal:
        mode = "雙資產抄底"
        w = [0.00, 0.60, 0.20, 0.20, 0.00]
    else:
        # 預設回到高勝率美股核心 IVV
        mode = "IVV 核心常態持股"
        w = [0.00, 0.85, 0.15, 0.00, 0.00]

    modes.append(mode)
    target_weights.append(w)

df_w_target = pd.DataFrame(
    target_weights, index=data.index, columns=tickers_vti
)
df_signals = pd.DataFrame(
    {"Mode": modes, "RSI_SMH": rsi_smh, "RSI_IWM": rsi_iwm}, index=data.index
)

# 實施 Lag-1 滯後處理 (消除前視偏誤)
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
        "RSI_SMH": f"{df_signals.loc[d, 'RSI_SMH']:.1f}",
        "RSI_IWM": f"{df_signals.loc[d, 'RSI_IWM']:.1f}",
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
# 畫面呈現：KPI 指標與核心圖表
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric(
    "均值回歸策略總報酬",
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
            "均值回歸過濾策略": strat_cum,
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
# 7. RSI 水準對照圖 (SMH & IWM RSI 走勢)
# ---------------------------------------------------------
st.write("---")
st.subheader("📊 RSI 指標水準歷史走勢與超買/超賣門檻對照 (14日 RSI)")
tab_rsi1, tab_rsi2 = st.tabs(["SMH (半導體) RSI", "IWM (小型股) RSI"])

with tab_rsi1:
    df_rsi_smh_plot = pd.DataFrame({
        "SMH 14日 RSI": rsi_smh.loc[exec_dates],
        "超賣買入門檻": rsi_oversold,
        "超買獲利了結門檻": rsi_overbought,
    })
    st.line_chart(df_rsi_smh_plot)

with tab_rsi2:
    df_rsi_iwm_plot = pd.DataFrame({
        "IWM 14日 RSI": rsi_iwm.loc[exec_dates],
        "超賣買入門檻": rsi_oversold,
        "超買獲利了結門檻": rsi_overbought,
    })
    st.line_chart(df_rsi_iwm_plot)

st.write("---")
st.subheader("📋 每月詳細策略報酬與各 ETF 獨立月報酬歷程 (T+1 執行對齊)")

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
