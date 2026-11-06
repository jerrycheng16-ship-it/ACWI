import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="ACWI 戰術輪動策略回測", layout="wide")

# ---------------------------------------------------------
# 1. 標的與基準參數設定
# ---------------------------------------------------------
tickers = ["IVV", "VGK", "EWJ", "IAPD", "IEMG", "BIL"]
start_date = "2013-01-01"
end_date = "2026-11-01"

# 靜態 ACWI 基準配比
benchmark_weights = pd.Series([0.63, 0.13, 0.05, 0.09, 0.11, 0.00], index=tickers)

# ---------------------------------------------------------
# 2. 標題與策略邏輯 Tag 說明
# ---------------------------------------------------------
st.title("📊 ETF Canary 戰術輪動與 ACWI 基準回測系統")

with st.expander("🏷️ 點此查看：策略邏輯與金絲雀（Canary）機制說明"):
    st.markdown("""
    ### 📌 策略核心邏輯 (US-Core & Non-US Canary TAA)
    本策略利用美股大盤 **IVV** 高勝率與強勁 ROE 的特性作為核心，並利用波動較大的非美股市 (**IEMG**、**IAPD**) 作為預警的金絲雀 (Canary) 指標。
    
    * **🟢 Risk-On (全球牛市 / 雙綠燈)**：IEMG 與 IAPD 收盤價皆高於 200MA。
    * **🟡 Risk-Neutral (防守切換 / 單綠燈)**：IEMG 與 IAPD 僅有 1 支在 200MA 之上。
    * **🔴 Risk-Off (避險防衛 / 雙紅燈)**：IEMG 與 IAPD 皆跌破 200MA。
    """)

# ---------------------------------------------------------
# 3. 精準數據下載與修復 (避免填補無效歷史價格)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_etf_data(symbols, start, end):
    raw_df = yf.download(symbols, start=start, end=end, auto_adjust=True)
    if isinstance(raw_df.columns, pd.MultiIndex):
        if 'Close' in raw_df.columns.levels[0]:
            df = raw_df['Close']
        else:
            df = raw_df.xs(raw_df.columns.levels[0][0], axis=1, level=0)
    else:
        df = raw_df
    
    # 僅做個股內部的向前填補 (ffill)，剔除休市日，不盲目 bfill 毀壞 200MA
    df = df[symbols].ffill().dropna()
    return df

with st.spinner("正在精準下載 ETF 數據並驗證 200MA 均線..."):
    try:
        data = load_etf_data(tickers, start_date, end_date)
    except Exception as e:
        st.error(f"❌ 數據下載失敗：{e}")
        st.stop()

if data.empty or len(data) < 200:
    st.error("❌ 歷史數據不足，無法完成回測。")
    st.stop()

# ---------------------------------------------------------
# 4. 逐日與月度戰術回測計算
# ---------------------------------------------------------
ma200 = data.rolling(window=200).mean()
returns = data.pct_change().fillna(0)

# 金絲雀訊號布林邏輯 (直接指定 Ticker 欄位)
canary_iemg_series = data["IEMG"] > ma200["IEMG"]
canary_iapd_series = data["IAPD"] > ma200["IAPD"]

dates = data.index[200:]
daily_records = []

for d in dates:
    canary_iemg = canary_iemg_series.loc[d]
    canary_iapd = canary_iapd_series.loc[d]
    daily_ret = returns.loc[d]

    # ACWI 基準報酬
    bm_ret = (daily_ret * benchmark_weights).sum()

    # 決定動態權重
    if canary_iemg and canary_iapd:
        mode = "Risk-On (🟢🟢)"
        strat_w = pd.Series([0.70, 0.10, 0.05, 0.05, 0.10, 0.00], index=tickers)
    elif not canary_iemg and not canary_iapd:
        mode = "Risk-Off (🔴🔴)"
        strat_w = pd.Series([0.50, 0.00, 0.00, 0.00, 0.00, 0.50], index=tickers)
    else:
        mode = "Risk-Neutral (🟡)"
        strat_w = pd.Series([0.80, 0.10, 0.10, 0.00, 0.00, 0.00], index=tickers)

    strat_ret = (daily_ret * strat_w).sum()

    daily_records.append({
        "Date": d,
        "Mode": mode,
        "Canary_IEMG": "🟢 均線上" if canary_iemg else "🔴 均線下",
        "Canary_IAPD": "🟢 均線上" if canary_iapd else "🔴 均線下",
        "Strat_Ret": strat_ret,
        "BM_Ret": bm_ret,
        "W_IVV": strat_w["IVV"],
        "W_VGK": strat_w["VGK"],
        "W_EWJ": strat_w["EWJ"],
        "W_IAPD": strat_w["IAPD"],
        "W_IEMG": strat_w["IEMG"],
        "W_BIL": strat_w["BIL"]
    })

df_daily = pd.DataFrame(daily_records).set_index("Date")

# 累積總淨值
strat_cum = (1 + df_daily["Strat_Ret"]).cumprod()
bm_cum = (1 + df_daily["BM_Ret"]).cumprod()

# ---------------------------------------------------------
# 5. 年化報酬率 (CAGR) 與 Drawdown 計算
# ---------------------------------------------------------
total_days = (dates[-1] - dates[0]).days
years = total_days / 365.25

strat_total_return = (strat_cum.iloc[-1] - 1) * 100
bm_total_return = (bm_cum.iloc[-1] - 1) * 100

strat_cagr = ((strat_cum.iloc[-1]) ** (1 / years) - 1) * 100
bm_cagr = ((bm_cum.iloc[-1]) ** (1 / years) - 1) * 100

def get_drawdown_series(cum_returns):
    peak = cum_returns.cummax()
    return (cum_returns - peak) / peak * 100

strat_dd_series = get_drawdown_series(strat_cum)
bm_dd_series = get_drawdown_series(bm_cum)

strat_mdd = strat_dd_series.min()
bm_mdd = bm_dd_series.min()

# 月度勝率
monthly_df = df_daily.resample("ME").agg({
    "Strat_Ret": lambda x: (1 + x).prod() - 1,
    "BM_Ret": lambda x: (1 + x).prod() - 1,
    "Mode": "last",
    "Canary_IEMG": "last",
    "Canary_IAPD": "last",
    "W_IVV": "last", "W_VGK": "last", "W_EWJ": "last", 
    "W_IAPD": "last", "W_IEMG": "last", "W_BIL": "last"
})
win_rate = (monthly_df["Strat_Ret"] > monthly_df["BM_Ret"]).mean() * 100

# ---------------------------------------------------------
# 6. 畫面呈現：KPI 指標與核心圖表
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
st.line_chart(pd.DataFrame({"Canary 戰術輪動策略": strat_cum, "ACWI 基準對標": bm_cum}))

st.write("### 📉 風險水庫：水面下回撤走勢圖 (Drawdown %)")
st.line_chart(pd.DataFrame({"Canary 策略 Drawdown (%)": strat_dd_series, "ACWI 基準 Drawdown (%)": bm_dd_series}))

# ---------------------------------------------------------
# 7. 驗證金絲雀：IAPD / IEMG 股價 vs 200MA 對照圖
# ---------------------------------------------------------
st.write("---")
st.subheader("🔍 金絲雀數據驗證：IAPD 股價 vs 200MA 走勢對照")
df_iapd_verify = pd.DataFrame({
    "IAPD 收盤價 (USD)": data["IAPD"].loc[dates],
    "IAPD 200日均線 (200MA)": ma200["IAPD"].loc[dates]
})
st.line_chart(df_iapd_verify)

# ---------------------------------------------------------
# 8. 當前最新訊號與表格歷史
# ---------------------------------------------------------
st.subheader("🔔 當前最新金絲雀訊號與推薦配置")
latest_rec = df_daily.iloc[-1]

c_rec1, c_rec2, c_rec3 = st.columns(3)
c_rec1.info(f"**最新市場狀態**：{latest_rec['Mode']}")
c_rec2.write(f"**IEMG Canary**: {latest_rec['Canary_IEMG']} | **IAPD Canary**: {latest_rec['Canary_IAPD']}")
c_rec3.write(f"**建議持股比例**：IVV {latest_rec['W_IVV']*100:.0f}% | VGK {latest_rec['W_VGK']*100:.0f}% | EWJ {latest_rec['W_EWJ']*100:.0f}% | IAPD {latest_rec['W_IAPD']*100:.0f}% | IEMG {latest_rec['W_IEMG']*100:.0f}% | BIL {latest_rec['W_BIL']*100:.0f}%")

st.write("---")
st.subheader("📋 每月詳細策略報酬與配置歷程 (Monthly History Breakdown)")

monthly_display = monthly_df.copy()
monthly_display["策略月報酬"] = (monthly_display["Strat_Ret"] * 100).map("{:.2f}%".format)
monthly_display["ACWI 月報酬"] = (monthly_display["BM_Ret"] * 100).map("{:.2f}%".format)
monthly_display["月度勝負"] = np.where(monthly_display["Strat_Ret"] > monthly_display["BM_Ret"], "🏆 擊敗 ACWI", "🔻 落後 ACWI")

for col in ["W_IVV", "W_VGK", "W_EWJ", "W_IAPD", "W_IEMG", "W_BIL"]:
    monthly_display[col] = (monthly_display[col] * 100).map("{:.0f}%".format)

monthly_display = monthly_display.rename(columns={
    "Mode": "市場狀態", "Canary_IEMG": "IEMG 訊號", "Canary_IAPD": "IAPD 訊號",
    "W_IVV": "IVV 權重", "W_VGK": "VGK 權重", "W_EWJ": "EWJ 權重",
    "W_IAPD": "IAPD 權重", "W_IEMG": "IEMG 權重", "W_BIL": "BIL 權重"
})

table_columns = ["市場狀態", "策略月報酬", "ACWI 月報酬", "月度勝負", "IEMG 訊號", "IAPD 訊號", "IVV 權重", "VGK 權重", "EWJ 權重", "IAPD 權重", "IEMG 權重", "BIL 權重"]
st.dataframe(monthly_display[table_columns].sort_index(ascending=False), use_container_width=True)
