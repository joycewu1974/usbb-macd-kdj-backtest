"""
MACD + KDJ 短線策略回測
執行: pip install -r requirements.txt  然後  streamlit run macd_kdj_backtest.py

四個版本同期比較 (日線、只做多、訊號在收盤確認、隔天開盤成交):
  0. 買進持有
  A. 純 MACD+KDJ : DIF>DEA 且 KDJ 金叉 進場 ; KDJ 死叉 出場
  B. A + 趨勢過濾 : 另外要求 收盤>MA50>MA200 且 DIF>0 才進場
  C. B + ATR 停損 : 出場改為 ATR 移動停損 或 MACD 死叉 (讓獲利奔跑)
  D. 二次確認    : KDJ 金叉後 N 天內 MACD 帶量金叉 買進 ;
                   KDJ 死叉後 N 天內 MACD 死叉 賣出 (可選擇是否也要帶量)
"""
import numpy as np
import pandas as pd

# ---------------- 指標 ----------------
def add_indicators(df, kdj_n=9, atr_n=14, confirm_n=5, vol_mult=1.2, sell_need_vol=False):
    c, h, l = df["Close"], df["High"], df["Low"]
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["HIST"] = 2 * (df["DIF"] - df["DEA"])

    llv, hhv = l.rolling(kdj_n).min(), h.rolling(kdj_n).max()
    rsv = ((c - llv) / (hhv - llv).replace(0, np.nan) * 100).fillna(50)
    df["K"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    df["D"] = df["K"].ewm(alpha=1 / 3, adjust=False).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / atr_n, adjust=False).mean()
    df["MA50"] = c.rolling(50).mean()
    df["MA200"] = c.rolling(200).mean()

    df["KDJ_GOLD"] = (df["K"] > df["D"]) & (df["K"].shift() <= df["D"].shift())
    df["KDJ_DEAD"] = (df["K"] < df["D"]) & (df["K"].shift() >= df["D"].shift())
    df["MACD_DEAD"] = (df["DIF"] < df["DEA"]) & (df["DIF"].shift() >= df["DEA"].shift())
    df["MACD_GOLD"] = (df["DIF"] > df["DEA"]) & (df["DIF"].shift() <= df["DEA"].shift())
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()
    df["VOL_OK"] = df["Volume"] > vol_mult * df["VOL_MA20"]
    kg_recent = df["KDJ_GOLD"].astype(int).rolling(confirm_n, min_periods=1).max().astype(bool)
    kd_recent = df["KDJ_DEAD"].astype(int).rolling(confirm_n, min_periods=1).max().astype(bool)
    df["D_BUY"] = kg_recent & df["MACD_GOLD"] & df["VOL_OK"]
    df["D_SELL"] = kd_recent & df["MACD_DEAD"] & (df["VOL_OK"] if sell_need_vol else True)
    df["TREND_OK"] = (c > df["MA50"]) & (df["MA50"] > df["MA200"]) & (df["DIF"] > 0)
    return df


# ---------------- 回測引擎 ----------------
def backtest(df, strategy, cost=0.001, atr_init=2.0, atr_trail=3.0, capital=10000.0):
    """strategy: 'BH','A','B','C'. cost = 單邊成本(手續費+滑價)."""
    o, c = df["Open"].values, df["Close"].values
    cash, shares, pending = float(capital), 0.0, None
    cost_basis = 0.0
    entry_px = entry_date = stop = None
    equity, trades, in_pos = [], [], []

    for i in range(len(df)):
        # 1) 執行前一天收盤產生的訂單 (今天開盤價)
        if pending == "buy" and shares == 0:
            cost_basis = cash
            shares = cash * (1 - cost) / o[i]
            cash, entry_px, entry_date = 0.0, o[i], df.index[i]
            if strategy == "C":
                stop = o[i] - atr_init * df["ATR"].iat[i - 1]
        elif pending == "sell" and shares > 0:
            cash = shares * o[i] * (1 - cost)
            trades.append({"進場日": entry_date, "出場日": df.index[i],
                           "進場價": entry_px, "出場價": o[i], "股數": shares,
                           "損益$": cash - cost_basis,
                           "報酬%": (cash / cost_basis - 1) * 100})
            shares, stop = 0.0, None
        pending = None

        equity.append(cash + shares * c[i])
        in_pos.append(shares > 0)

        # 2) 收盤後產生訊號
        r = df.iloc[i]
        if strategy == "BH":
            if i == 0:
                pending = "buy"
            continue
        if strategy == "D":
            if shares == 0 and r["D_BUY"]:
                pending = "buy"
            elif shares > 0 and r["D_SELL"]:
                pending = "sell"
            continue
        if shares == 0:
            entry = r["KDJ_GOLD"] and r["DIF"] > r["DEA"]
            if strategy in ("B", "C"):
                entry = entry and r["TREND_OK"]
            if entry:
                pending = "buy"
        else:
            if strategy in ("A", "B"):
                if r["KDJ_DEAD"]:
                    pending = "sell"
            else:  # C
                stop = max(stop, c[i] - atr_trail * r["ATR"])
                if c[i] < stop or r["MACD_DEAD"]:
                    pending = "sell"

    eq = pd.Series(equity, index=df.index)
    return eq, pd.DataFrame(trades), float(np.mean(in_pos))


def metrics(eq, trades, exposure):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    out = {"期末資金$": eq.iloc[-1], "損益$": eq.iloc[-1] - eq.iloc[0], "總報酬%": total * 100, "年化%": cagr * 100, "最大回撤%": mdd * 100,
           "交易次數": len(trades), "持倉時間%": exposure * 100,
           "勝率%": np.nan, "平均賺%": np.nan, "平均賠%": np.nan, "獲利因子": np.nan}
    if len(trades):
        r = trades["報酬%"]
        w, lo = r[r > 0], r[r <= 0]
        out["勝率%"] = len(w) / len(r) * 100
        out["平均賺%"] = w.mean() if len(w) else 0
        out["平均賠%"] = lo.mean() if len(lo) else 0
        out["獲利因子"] = w.sum() / abs(lo.sum()) if lo.sum() != 0 else np.inf
    return out


STRATS = {"BH": "買進持有", "A": "A 純MACD+KDJ", "B": "B 加趨勢過濾", "C": "C 加ATR停損", "D": "D 二次確認帶量"}


def run_all(raw, start, cost, atr_init, atr_trail, capital=10000.0,
            confirm_n=5, vol_mult=1.2, sell_need_vol=False):
    df = add_indicators(raw.copy(), confirm_n=confirm_n, vol_mult=vol_mult, sell_need_vol=sell_need_vol)
    df = df[df.index >= pd.Timestamp(start)].dropna(subset=["MA200"])
    res = {}
    for k in STRATS:
        eq, tr, ex = backtest(df, k, cost, atr_init, atr_trail, capital)
        res[k] = {"eq": eq, "trades": tr, "m": metrics(eq, tr, ex)}
    return df, res


# ---------------- 介面 ----------------
def main():
    import streamlit as st
    import yfinance as yf
    import plotly.graph_objects as go

    st.set_page_config(page_title="MACD+KDJ 短線回測", layout="wide")
    st.title("MACD + KDJ 短線策略回測")
    st.caption("日線、只做多、收盤確認訊號、隔日開盤成交。僅供研究與教學，不構成投資建議。")

    with st.sidebar:
        tickers = st.text_input("股票代號 (逗號分隔)", "NVDA, TSLA, AAPL, AMD, SPY")
        start = st.date_input("回測起始日", pd.Timestamp("2018-01-01"))
        capital = st.number_input("每檔起始資金 (美元)", 1000, 10_000_000, 10_000, 1000)
        cost = st.number_input("單邊成本 % (手續費+滑價)", 0.0, 1.0, 0.10, 0.05) / 100
        atr_init = st.number_input("C: 初始停損 ATR 倍數", 0.5, 5.0, 2.0, 0.5)
        atr_trail = st.number_input("C: 移動停損 ATR 倍數", 0.5, 6.0, 3.0, 0.5)
        st.markdown("**D 二次確認設定**")
        confirm_n = st.number_input("KDJ 交叉後幾天內等 MACD 確認", 1, 20, 5, 1)
        vol_mult = st.number_input("帶量定義: 成交量 > 20日均量的幾倍", 1.0, 3.0, 1.2, 0.1)
        sell_need_vol = st.checkbox("賣出也要帶量", False)
        go_btn = st.button("開始回測", type="primary")

    if not go_btn:
        st.info("在左側輸入股票代號後按「開始回測」。")
        return

    fetch_from = pd.Timestamp(start) - pd.Timedelta(days=450)  # 預留 MA200 暖機
    rows, all_res = [], {}
    for t in [x.strip().upper() for x in tickers.split(",") if x.strip()]:
        raw = yf.Ticker(t).history(start=fetch_from, auto_adjust=True)
        if raw.empty or len(raw) < 300:
            st.warning(f"{t}: 資料不足，已略過")
            continue
        raw.index = raw.index.tz_localize(None)
        df, res = run_all(raw[["Open", "High", "Low", "Close", "Volume"]], start, cost, atr_init, atr_trail,
                          capital, confirm_n, vol_mult, sell_need_vol)
        all_res[t] = (df, res)
        for k, v in res.items():
            rows.append({"代號": t, "策略": STRATS[k], **v["m"]})

    if not rows:
        return
    summary = pd.DataFrame(rows)
    st.subheader("總覽")
    st.dataframe(summary.style.format(precision=2), use_container_width=True, hide_index=True)

    st.subheader("各策略平均 (跨所有股票)")
    avg = summary.groupby("策略", sort=False).mean(numeric_only=True)
    st.dataframe(avg.style.format(precision=2), use_container_width=True)

    st.subheader("所有股票合計損益 (美元)")
    tot = summary.groupby("策略", sort=False)[["損益$"]].sum()
    st.dataframe(tot.style.format("{:,.0f}"), use_container_width=True)

    st.subheader("單檔細看")
    pick = st.selectbox("選擇股票", list(all_res))
    df, res = all_res[pick]

    fig = go.Figure()
    for k, v in res.items():
        fig.add_trace(go.Scatter(x=v["eq"].index, y=v["eq"], name=STRATS[k]))
    fig.update_layout(title=f"{pick} 資金曲線 (美元)", height=420, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    sk = st.radio("顯示買賣點", ["A", "B", "C", "D"], format_func=lambda k: STRATS[k], horizontal=True)
    tr = res[sk]["trades"]
    k = go.Figure(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                 increasing_line_color="#16a34a", decreasing_line_color="#dc2626", name=pick))
    k.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50", line=dict(width=1)))
    k.add_trace(go.Scatter(x=df.index, y=df["MA200"], name="MA200", line=dict(width=1)))
    if len(tr):
        k.add_trace(go.Scatter(x=tr["進場日"], y=tr["進場價"], mode="markers", name="買",
                               marker=dict(symbol="triangle-up", size=10, color="#2563eb")))
        k.add_trace(go.Scatter(x=tr["出場日"], y=tr["出場價"], mode="markers", name="賣",
                               marker=dict(symbol="triangle-down", size=10, color="#f59e0b")))
    k.update_layout(height=520, xaxis_rangeslider_visible=False)
    st.plotly_chart(k, use_container_width=True)

    with st.expander("交易明細"):
        st.dataframe(tr.style.format(precision=2), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
