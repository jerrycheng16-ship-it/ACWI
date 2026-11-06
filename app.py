import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="ACWI 戰術輪動策略回測", layout="wide"
)
st.title("📊 ETF Canary 戰術輪動與 ACWI 基準回測系統")

# 1. 邊界與參數設定
tickers = ["IVV", "VGK", "EWJ", "IAPD", "IEMG", "BIL"]
start_date = "2013-01-01"
end_date = "2026-11-01"

benchmark_weights = pd.Series(
    [0.63, 0.13, 0.05, 0.09, 0.11, 0.00], index=tickers
)

# 2. 抓取歷史數據與防呆處理
with st.spinner("正在下載並處理 ETF 歷史數據..."):
    df_raw = yf.download(tickers, start=start_date, end=end_date)[
        "Adj Close"
    ]

    # 確保欄位依據 tickers 順序對齊，並進行前向填補 (ffill) 防止掉檔
    data = df_raw.reindex(columns=tickers).ffill().bfill()

if data.empty or len(data) < 200:
    st.error("❌ 無法取得足夠的 ETF 數據，請檢查網路連線或日期區間。")
    st.stop()

# 3. 計算 200 日移動平均線 (200MA)
ma200 = data.rolling(window=200).mean()

# 4. 計算每日報酬率
returns = data.pct_change().fillna(0)

# 5. 進行逐日戰術模擬
strategy_returns = []
benchmark_returns = []
dates = data.index[200:]

for d in dates:
    # 判斷 Canary 訊號 (IEMG 與 IAPD 是否位於 200MA 之上)
    canary_iemg = data.loc[d, "IEMG"] > ma200.loc[d, "IEMG"]
    canary_iapd = data.loc[d, "IAPD"] > ma200.loc[d, "IAPD"]

    # 當日各資產報酬率
    daily_ret = returns.loc[d, tickers]

    # 計算 ACWI 基準報酬 (63% IVV / 13% VGK / 5% EWJ / 9% IAPD / 11% IEMG)
    bm_ret = (daily_ret * benchmark_weights).sum()
    benchmark_returns.append(bm_ret)

    # 動態權重配置
    if canary_iemg and canary_iapd:
        # Risk-On: 美股加碼至 70%
        w = pd.Series([0.70, 0.10, 0.05, 0.05, 0.10, 0.00], index=tickers)
    elif not canary_iemg and not canary_iapd:
        # Risk-Off: 避險模式 (50% IVV + 50% BIL 現金)
        w = pd.Series([0.50, 0.00, 0.00, 0.00, 0.00, 0.50], index=tickers)
    else:
        # Risk-Neutral: 核心防守 (80% IVV + 10% VGK + 10% EWJ)
        w = pd.Series([0.80, 0.10, 0.10, 0.00, 0.00, 0.00], index=tickers)

    strat_ret = (daily_ret * w).sum()
    strategy_returns.append(strat_ret)

# 6. 計算累積總報酬率
strat_cum = (1 + pd.Series(strategy_returns, index=dates)).cumprod()
bm_cum = (1 + pd.Series(benchmark_returns, index=dates)).cumprod()

# 防呆驗證：確保數值不為空
if len(strat_cum) == 0 or len(bm_cum) == 0:
    st.error("❌ 計算結果為空，無法顯示績效。")
    st.stop()

# 7. 數據繪圖與指標統計展示
col1, col2, col3 = st.columns(3)

strat_total_return = (strat_cum.iloc[-1] - 1) * 100
bm_total_return = (bm_cum.iloc[-1] - 1) * 100

# 計算月度勝率
monthly_strat = (
    pd.Series(strategy_returns, index=dates)
    .resample("ME")
    .apply(lambda x: (1 + x).prod() - 1)
)
monthly_bm = (
    pd.Series(benchmark_returns, index=dates)
    .resample("ME")
    .apply(lambda x: (1 + x).prod() - 1)
)
win_rate = (monthly_strat > monthly_bm).mean() * 100

col1.metric("策略累積總報酬", f"{strat_total_return:.2f}%")
col2.metric("ACWI 基準總報酬", f"{bm_total_return:.2f}%")
col3.metric("策略月度超越勝率", f"{win_rate:.2f}%")

# 8. 走勢圖表
df_plot = pd.DataFrame(
    {"Canary 輪動策略": strat_cum, "ACWI 基準配比": bm_cum}
)
st.line_chart(df_plot)
