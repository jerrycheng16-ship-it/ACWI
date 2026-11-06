# 讀取側邊欄設定的加碼比例 (例如 30%)
add_on_smh_w = smh_alpha_weight  # 使用 slider 選擇的數值 (預設 0.25~0.30)

if smh_buy_signal and not iwm_buy_signal:
    mode = "加碼 SMH (逆勢抄底)"
    # 加碼 SMH 至 30% (或 slider 設定值)，其餘給 IVV
    w = [0.00, round(1.0 - add_on_smh_w, 2), add_on_smh_w, 0.00, 0.00]

elif iwm_buy_signal and not smh_buy_signal:
    mode = "加碼 IWM (逆勢抄底)"
    # 加碼 IWM 至 30%，其餘給 IVV
    w = [0.00, 0.70, 0.00, 0.30, 0.00]

elif smh_buy_signal and iwm_buy_signal:
    mode = "雙資產相對超跌 ➔ 雙重加碼"
    w = [0.00, 0.60, 0.20, 0.20, 0.00]

else:
    mode = "常態持股 (IVV 85% + SMH 15%)"
    # 常態基準：IVV 85% + SMH 15%
    w = [0.00, 0.85, 0.15, 0.00, 0.00]
