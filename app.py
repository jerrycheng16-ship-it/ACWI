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

# 靜態 ACWI 基準配比 (IVV 63%, VGK 13%, EWJ 5%, IAPD 9%, IEMG 11%, BIL 0%)
benchmark_weights = np.array([0.63, 0.13, 0.05, 0.09, 0.11, 0.00])


# 2. 歷史數據抓取與快取 (防止每次重整網頁都重新下載)
@st.cache_data(ttl=3600)
def load_etf_data(symbols, start, end):
    # 下載數據，強制解開 MultiIndex 欄位
    df = yf.download(symbols, start=start, end=end)["Adj Close"]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # 確保所有 tickers 都在欄位中且順序正確
    df = df[symbols]
    # 前向與後向補值，避免休市日全變 NaN
    df = df.ffill().bfill()
    return df


with st.spinner("正在下載並驗證 ETF 歷史數據..."):
    try:
        data = load_etf_data(tickers, start_date, end_date)
    except Exception as e:
        st.error(f"❌ 數據下載失敗：{e}")
        st.stop()

if data.empty or len(data) < 200:
    st.error("❌ 無法取得足夠的 ETF 歷史數據，請確認連線或時間範圍。")
    st.stop()

# 3. 計算 200 日移動平均線 (200MA) 與每日報酬率
ma200 = data.rolling(window=200).mean()
returns = data.pct_change().fillna(0)

# 4. 逐日回測邏輯 (使用 NumPy 矩陣乘法，徹底避免 Pandas 索引錯位)
returns_matrix = returns.values
data_matrix = data.values
ma200_matrix = ma200.values

strategy_returns = []
benchmark_returns = []
dates = data.index[200:]

# IEMG 與 IAPD 在 tickers 中的索引位置 (tickers = ['IVV', 'VGK', 'EWJ', 'IAPD', 'IEMG', 'BIL'])
idx_iapd = 3
idx_iemg = 4

for i in range(200, len(data)):
    # 判斷 Canary 訊號 (當天收盤價是否低於 200MA)
    canary_iemg = data_matrix[i, idx_iemg] > ma200_matrix[i, idx_iemg]
    canary_iapd = data_matrix[i, idx_iapd] > ma200_matrix[i, idx_iapd]

    # 當日各資產報酬率向量
    daily_ret = returns_matrix[i]

    # 1. 計算 ACWI 基準當日報酬率
    bm_ret = np.dot(daily_ret, benchmark_weights)
    benchmark_returns.append(bm_ret)

    # 2. 根據 Canary 訊號決定當日策略動態權重
    if canary_iemg and canary_iapd:
        # Risk-On (全球牛市): IVV 超配至 70%
        strat_w = np.array([0.70, 0.10, 0.05, 0.05, 0.10, 0.00])
    elif not canary_iemg and not canary_iapd:
        # Risk-Off (全球熊市): 50% IVV + 50% BIL (短債/現金)
        strat_w = np.array([0.50, 0.00, 0.00, 0.00, 0.00, 0.50])
    else:
        # Risk-Neutral (單綠燈/防守): 80% IVV + 10% VGK + 10% EWJ
        strat_w = np.array([0.80, 0.10, 0.10, 0.00, 0.00, 0.00])

    # 計算策略當日報酬率
    strat_ret = np.dot(daily_ret, strat_w)
    strategy_returns.append(strat_ret)

# 5. 計算累積淨值與報酬率
strat_ret_series = pd.Series(strategy_returns, index=dates)
bm_ret_series = pd.Series(benchmark_returns, index=dates)

strat_cum = (1 + strat_ret_series).cumprod()
bm_cum = (1 + bm_ret_series).cumprod()

# 6. 計算核心統計指標
strat_total_return = (strat_cum.iloc[-1] - 1) * 100
bm_total_return = (bm_cum.iloc[-1] - 1) * 100

# 計算月度超越勝率
monthly_strat = strat_ret_series.resample("ME").apply(
    lambda x: (1 + x).prod() - 1
)
monthly_bm = bm_ret_series.resample("ME").apply(lambda x: (1 + x).prod() - 1)
win_rate = (monthly_strat > monthly_bm).mean() * 100

# 7. Streamlit 畫面呈現
col1, col2, col3 = st.columns(3)
col1.metric("策略累積總報酬", f"{strat_total_return:.2f}%")
col2.metric("ACWI 基準總報酬", f"{bm_total_return:.2f}%")
col3.metric("策略月度超越勝率", f"{win_rate:.2f}%")

st.write("### 📈 累積淨值走勢比較 (基期 = 1.0)")
df_plot = pd.DataFrame(
    {"Canary 戰術輪動策略": strat_cum, "ACWI 基準對標": bm_cum}
)
st.line_chart(df_plot)
