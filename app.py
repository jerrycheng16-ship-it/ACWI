import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# 1. 設定回測標的與時間區間
tickers = ["IVV", "VGK", "EWJ", "IAPD", "IEMG", "BIL"]
start_date = "2013-01-01"  # 確保 IEMG 數據完整
end_date = "2026-11-01"

# 2. 抓取每日調整後收盤價 (Adjusted Close)
print("正在下載 ETF 歷史數據...")
data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
data = data.dropna()

# 3. 計算 Canary 訊號 (以 200日均線為例)
ma200 = data.rolling(window=200).mean()

# 4. 初始化策略報酬率容器
returns = data.pct_change()
strategy_returns = []
benchmark_returns = []  # 靜態 ACWI 模擬配比 (63/13/5/9/11)

benchmark_weights = np.array([0.63, 0.13, 0.05, 0.09, 0.11, 0.00])  # IVV~BIL

# 5. 執行逐日策略回測
dates = data.index[200:]  # 從第 200 天開始（需計算 200MA）

for d in dates:
    # 判斷 Canary 訊號 (IEMG 與 IAPD 是否位於 200MA 之上)
    canary_iemg = data.loc[d, "IEMG"] > ma200.loc[d, "IEMG"]
    canary_iapd = data.loc[d, "IAPD"] > ma200.loc[d, "IAPD"]

    # 當日各資產報酬率
    daily_ret = returns.loc[d, tickers].values

    # 計算靜態基準 (ACWI 模擬) 每日報酬
    bm_ret = np.sum(daily_ret * benchmark_weights)
    benchmark_returns.append(bm_ret)

    # 決定動態權重
    if canary_iemg and canary_iapd:
        # Risk-On (全球牛市): IVV 超配至 70%
        strat_weights = np.array([0.70, 0.10, 0.05, 0.05, 0.10, 0.00])
    elif not canary_iemg and not canary_iapd:
        # Risk-Off (避險模式): 50% IVV + 50% BIL(現金)
        strat_weights = np.array([0.50, 0.00, 0.00, 0.00, 0.00, 0.50])
    else:
        # Risk-Neutral: IVV 80% 核心防守
        strat_weights = np.array([0.80, 0.10, 0.10, 0.00, 0.00, 0.00])

    strat_ret = np.sum(daily_ret * strat_weights)
    strategy_returns.append(strat_ret)

# 6. 計算累積報酬率
strat_cum = (1 + pd.Series(strategy_returns, index=dates)).cumprod()
bm_cum = (1 + pd.Series(benchmark_returns, index=dates)).cumprod()

# 7. 印出關鍵績效與勝率
print("\n=== 回測結果摘要 ===")
print(f"策略累積總報酬率: {(strat_cum.iloc[-1]-1)*100:.2f}%")
print(f"基準 (ACWI) 累積總報酬率: {(bm_cum.iloc[-1]-1)*100:.2f}%")

# 勝率計算 (月度勝率)
monthly_strat = (
    pd.Series(strategy_returns, index=dates).resample("M").apply(lambda x: (1+x).prod()-1)
)
monthly_bm = (
    pd.Series(benchmark_returns, index=dates).resample("M").apply(lambda x: (1+x).prod()-1)
)
win_rate = (monthly_strat > monthly_bm).mean()
print(f"策略對 ACWI 月度勝率: {win_rate*100:.2f}%")
