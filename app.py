import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="ACWI 戰術輪動策略回測", layout="wide"
)

# ---------------------------------------------------------
# 1. 標的與基準參數設定
# ---------------------------------------------------------
tickers = ["IVV", "VGK", "EWJ", "IAPD", "IEMG", "BIL"]
start_date = "2013-01-01"
end_date = "2026-11-01"

# 靜態 ACWI 基準對標配比 (63% IVV, 13% VGK, 5% EWJ, 9% IAPD, 11% IEMG, 0% BIL)
benchmark_weights = np.array([0.63, 0.13, 0.05, 0.09, 0.11, 0.00])

# ---------------------------------------------------------
# 2. 標題與策略邏輯 Tag 說明
# ---------------------------------------------------------
st.title("📊 ETF Canary 戰術輪動與 ACWI 基準回測系統")

with st.expander("🏷️ 點此查看：策略邏輯與金絲雀（Canary）機制說明"):
    st.markdown(
        """
    ### 📌 策略核心邏輯 (US-Core & Non-US Canary TAA)
    本策略利用美股大盤 **IVV** 高勝率與強勁 ROE 的特性作為核心，並利用波動較大的非美股市 (**IEMG**、**IAPD**) 作為預警的金絲雀 (Canary) 指標。
    
    * **🟢 Risk-On (全球牛市 / 雙綠燈)**：
      * **條件**：IEMG 與 IAPD 收盤價皆高於其 200 日移動平均線 (200MA)。
      * **配置**：`IVV 70%` | `VGK 10%` | `EWJ 5%` | `IAPD 5%` | `IEMG 10%` | `BIL 0%`
    * **🟡 Risk-Neutral (防守切換 / 單綠燈)**：
      * **條件**：IEMG 與 IAPD 僅有其中 1 支在 200MA 之上。
      * **配置**：`IVV 80%` | `VGK 10%` | `EWJ 10%` | `IAPD 0%` | `IEMG 0%` | `BIL 0%`
    * **🔴 Risk-Off (避險防衛 / 雙紅燈)**：
      * **條件**：IEMG 與 IAPD 皆跌破 200MA。
      * **配置**：`IVV 50%` | `BIL (超短期美債/現金) 50%` | 其餘 0%
    """
    )


# ---------------------------------------------------------
# 3. 數據下載與安全解析
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


with st.spinner("正在讀取 ETF 歷史股價數據並計算關鍵指標..."):
    try:
        data = load_etf_data(tickers, start_date, end_date)
    except Exception as e:
        st.error(f"❌ 數據下載失敗：{e}")
        st.stop()

if data.empty or len(data) < 200:
    st.error("❌ 歷史數據不足，無法完成回測。")
    st.stop()

# ---------------------------------------------------------
# 4. 逐日回測與動態權重計算
# ---------------------------------------------------------
ma200 = data.rolling(window=200).mean()
returns = data.pct_change().fillna(0)

returns_matrix = returns.values
data_matrix = data.values
ma200_matrix = ma200.values

dates = data.index[200:]
idx_iapd = 3
idx_iemg = 4

daily_records = []

for i in range(200, len(data)):
    d = data.index[i]
    canary_iemg = data_matrix[i, idx_iemg] > ma200_matrix[i, idx_iemg]
    canary_iapd = data_matrix[i, idx_iapd] > ma200_matrix[i, idx_iapd]
    daily_ret = returns_matrix[i]

    # ACWI 基準報酬
    bm_ret = np.dot(daily_ret, benchmark_weights)

    # 決定動態策略權重
    if canary_iemg and canary_iapd:
        mode = "Risk-On (🟢🟢)"
        strat_w = np.array([0.70, 0.10, 0.05, 0.05, 0.10, 0.00])
    elif not canary_iemg and not canary_iapd:
        mode = "Risk-Off (🔴🔴)"
        strat_w = np.array([0.50, 0.00, 0.00, 0.00, 0.00, 0.50])
    else:
        mode = "Risk-Neutral (🟡)"
        strat_w = np.array([0.80, 0.10, 0.10, 0.00, 0.00, 0.00])

    strat_ret = np.dot(daily_ret, strat_w)

    daily_records.append(
        {
            "Date": d,
            "Mode": mode,
            "Canary_IEMG": "🟢 均線上" if canary_iemg else "🔴 均線下",
            "Canary_IAPD": "🟢 均線上" if canary_iapd else "🔴 均線下",
            "Strat_Ret": strat_ret,
            "BM_Ret": bm_ret,
            "W_IVV": strat_w[0],
            "W_VGK": strat_w[1],
            "W_EWJ": strat_w[2],
            "W_IAPD": strat_w[3],
            "W_IEMG": strat_w[4],
            "W_BIL": strat_w[5],
        }
    )

df_daily = pd.DataFrame(daily_records).set_index("Date")

# 累積總淨值
strat_cum = (1 + df_daily["Strat_Ret"]).cumprod()
bm_cum = (1 + df_daily["BM_Ret"]).cumprod()

# ---------------------------------------------------------
# 5. 年化報酬率 (CAGR) 與 Drawdown 走勢計算
# ---------------------------------------------------------
total_days = (dates[-1] - dates[0]).days
years = total_days / 365.25

strat_total_return = (strat_cum.iloc[-1] - 1) * 100
bm_total_return = (bm_cum.iloc[-1] - 1) * 100

strat_cagr = ((strat_cum.iloc[-1]) ** (1 / years) - 1) * 100
bm_cagr = ((bm_cum.iloc[-1]) ** (1 / years) - 1) * 100

# 計算逐日 Drawdown 序列
def get_drawdown_series(cum_returns):
    peak = cum_returns.cummax()
    return (cum_returns - peak) / peak * 100

strat_dd_series = get_drawdown_series(strat_cum)
bm_dd_series = get_drawdown_series(bm_cum)

strat_mdd = strat_dd_series.min()
bm_mdd = bm_dd_series.min()

# 月度勝率
monthly_df = df_daily.resample("ME").agg(
    {
        "Strat_Ret": lambda x: (1 + x).prod() - 1,
        "BM_Ret": lambda x: (1 + x).prod() - 1,
        "Mode": "last",
        "Canary_IEMG": "last",
        "Canary_IAPD": "last",
        "W_IVV": "last",
        "W_VGK": "last",
        "W_EWJ": "last",
        "W_IAPD": "last",
        "W_IEMG": "last",
        "W_BIL": "last",
    }
)
win_rate = (monthly_df["Strat_Ret"] > monthly_df["BM_Ret"]).mean() * 100

# ---------------------------------------------------------
# 6. 畫面呈現：KPI 指標與兩張核心圖表
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("策略累積總報酬", f"{strat_total_return:.2f}%", f"年化 CAGR: {strat_cagr:.2f}%")
col2.metric("ACWI 基準總報酬", f"{bm_total_return:.2f}%", f"年化 CAGR: {bm_cagr:.2f}%")
col3.metric("策略月度超越勝率", f"{win_rate:.2f}%")

st.write("")
col4, col5 = st.columns(2)
col4.metric("策略最大回撤 (Max Drawdown)", f"{strat_mdd:.2f}%")
col5.metric("ACWI 基準最大回撤 (Max Drawdown)", f"{bm_mdd:.2f}%")

st.write("### 📈 累積淨值走勢比較 (基期 = 1.0)")
df_cum_plot = pd.DataFrame(
    {"Canary 戰術輪動策略": strat_cum, "ACWI 基準對標": bm_cum}
)
st.line_chart(df_cum_plot)

st.write("### 📉 風險水庫：水面下回撤走勢圖 (Drawdown %)")
df_dd_plot = pd.DataFrame(
    {"Canary 策略 Drawdown (%)": strat_dd_series, "ACWI 基準 Drawdown (%)": bm_dd_series}
)
st.line_chart(df_dd_plot)

# ---------------------------------------------------------
# 7. 最新訊號與詳細表格
# ---------------------------------------------------------
st.write("---")
st.subheader("🔔 當前最新金絲雀訊號與推薦配置")
latest_rec = df_daily.iloc[-1]

c_rec1, c_rec2, c_rec3 = st.columns(
