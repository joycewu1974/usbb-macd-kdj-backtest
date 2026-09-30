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
    df["BB_MID"] = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["BB_UP"] = df["BB_MID"] + 2 * bb_std
    df["BB_LOW"] = df["BB_MID"] - 2 * bb_std
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

    # --- VCP + ATR (簡化版) ---
    df["MA150"] = c.rolling(150).mean()
    df["MA200_UP"] = df["MA200"] > df["MA200"].shift(22)
    df["HI52"] = h.rolling(252, min_periods=200).max()
    df["LO52"] = l.rolling(252, min_periods=200).min()
    df["ATR10"] = tr.ewm(alpha=1 / 10, adjust=False).mean()
    df["ATR50"] = tr.ewm(alpha=1 / 50, adjust=False).mean()
    df["HH20"] = h.rolling(20).max().shift()
    df["RANGE15"] = (h.rolling(15).max() - l.rolling(15).min()) / c
    df["TT7"] = ((c > df["MA150"]) & (c > df["MA200"]) & (df["MA150"] > df["MA200"]) & df["MA200_UP"]
                 & (df["MA50"] > df["MA150"]) & (c > df["MA50"])
                 & (c >= 1.3 * df["LO52"]) & (c >= 0.75 * df["HI52"]))
    # 波動收縮: 突破前 10 天內 ATR10/ATR50 曾低於 0.85，且前一天 15 日振幅 < 15%
    squeeze = (df["ATR10"] / df["ATR50"]).shift().rolling(10).min() < 0.85
    tight = squeeze & (df["RANGE15"].shift() < 0.15)
    df["V_BUY"] = (df["TT7"].shift(fill_value=False) & tight
                   & (c > df["HH20"]) & (df["Volume"] > 1.4 * df["VOL_MA20"]))

    # --- ADX(14) 趨勢強度 ---
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / atr14
    mdi = 100 * mdm.ewm(alpha=1 / 14, adjust=False).mean() / atr14
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    df["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
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
            if strategy in ("C", "V"):
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
            if strategy == "V":
                entry = bool(r["V_BUY"])
            if entry:
                pending = "buy"
        else:
            if strategy in ("A", "B"):
                if r["KDJ_DEAD"]:
                    pending = "sell"
            else:  # C
                stop = max(stop, c[i] - atr_trail * r["ATR"])
                if c[i] < stop or (strategy == "C" and r["MACD_DEAD"]):
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


STRATS = {"BH": "買進持有", "A": "A 純MACD+KDJ", "B": "B 加趨勢過濾", "C": "C 加ATR停損", "D": "D 二次確認帶量",
          "V": "V VCP+ATR(簡化)"}


def run_all(raw, start, cost, atr_init, atr_trail, capital=10000.0,
            confirm_n=5, vol_mult=1.2, sell_need_vol=False):
    df = add_indicators(raw.copy(), confirm_n=confirm_n, vol_mult=vol_mult, sell_need_vol=sell_need_vol)
    df = df[df.index >= pd.Timestamp(start)].dropna(subset=["MA200"])
    res = {}
    for k in STRATS:
        eq, tr, ex = backtest(df, k, cost, atr_init, atr_trail, capital)
        res[k] = {"eq": eq, "trades": tr, "m": metrics(eq, tr, ex)}
    return df, res


# ---------------- K線 + 布林 + 成交量 + MACD + KDJ ----------------
UP, DOWN = "#16a34a", "#dc2626"  # 美股慣例: 綠漲紅跌


def kline_chart(df, tr, name, show_bb=True, show_ma=True, days="近6個月"):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.5, 0.14, 0.18, 0.18],
                        subplot_titles=(f"{name} 日K", "成交量", "MACD (12,26,9)", "KDJ (9,3,3)"))
    # 主圖
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                 increasing_line_color=UP, decreasing_line_color=DOWN,
                                 increasing_fillcolor=UP, decreasing_fillcolor=DOWN, name="K線"), 1, 1)
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_UP"], name="布林上軌",
                                 line=dict(width=1, color="#8b5cf6")), 1, 1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOW"], name="布林下軌", fill="tonexty",
                                 fillcolor="rgba(139,92,246,0.08)", line=dict(width=1, color="#8b5cf6")), 1, 1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_MID"], name="布林中軌(MA20)",
                                 line=dict(width=1, color="#8b5cf6", dash="dot")), 1, 1)
    if show_ma:
        fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50", line=dict(width=1.2, color="#0ea5e9")), 1, 1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], name="MA200", line=dict(width=1.2, color="#f97316")), 1, 1)
    if len(tr):
        fig.add_trace(go.Scatter(x=tr["進場日"], y=tr["進場價"], mode="markers", name="買",
                                 marker=dict(symbol="triangle-up", size=12, color="#2563eb",
                                             line=dict(width=1, color="white"))), 1, 1)
        fig.add_trace(go.Scatter(x=tr["出場日"], y=tr["出場價"], mode="markers", name="賣",
                                 marker=dict(symbol="triangle-down", size=12, color="#f59e0b",
                                             line=dict(width=1, color="white"))), 1, 1)
    # 成交量
    vcol = np.where(df["Close"] >= df["Open"], UP, DOWN)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vcol, name="成交量", showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["VOL_MA20"], name="20日均量",
                             line=dict(width=1, color="#6b7280")), 2, 1)
    # MACD
    hcol = np.where(df["HIST"] >= 0, UP, DOWN)
    fig.add_trace(go.Bar(x=df.index, y=df["HIST"], marker_color=hcol, name="MACD柱", showlegend=False), 3, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["DIF"], name="DIF", line=dict(width=1.2, color="#2563eb")), 3, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["DEA"], name="DEA", line=dict(width=1.2, color="#f59e0b")), 3, 1)
    # KDJ
    fig.add_trace(go.Scatter(x=df.index, y=df["K"], name="K", line=dict(width=1.2, color="#2563eb")), 4, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["D"], name="D", line=dict(width=1.2, color="#f59e0b")), 4, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["J"], name="J", line=dict(width=1, color="#a855f7")), 4, 1)
    for lvl in (20, 80):
        fig.add_hline(y=lvl, line=dict(width=1, dash="dot", color="#9ca3af"), row=4, col=1)

    if days != "全部":
        n = 126 if days == "近6個月" else 252
        start = df.index[max(0, len(df) - n)]
        fig.update_xaxes(range=[start, df.index[-1] + pd.Timedelta(days=3)])
        seg = df[df.index >= start]
        lo = seg[["Low", "BB_LOW"]].min().min() if show_bb else seg["Low"].min()
        hi = seg[["High", "BB_UP"]].max().max() if show_bb else seg["High"].max()
        fig.update_yaxes(range=[lo * 0.97, hi * 1.03], row=1, col=1)
        fig.update_yaxes(range=[0, seg["Volume"].max() * 1.1], row=2, col=1)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(height=900, xaxis_rangeslider_visible=False, hovermode="x unified",
                      legend=dict(orientation="h", y=1.04, x=0), margin=dict(t=60, l=10, r=10, b=10),
                      bargap=0.1)
    return fig


# ---------------- 高成交量選股 ----------------
DEFAULT_UNIVERSE = """
AAPL MSFT NVDA AMZN META GOOGL TSLA AMD AVGO NFLX INTC MU QCOM TXN ARM SMCI MRVL ON
PLTR SOFI HOOD COIN RIVN LCID NIO XPEV LI F GM UBER LYFT SNAP PINS RBLX U DKNG
PYPL SQ AFRM UPST SHOP BABA JD PDD BIDU CSCO ORCL CRM ADBE IBM DELL HPQ HPE WBD
T VZ CMCSA DIS KO PEP WMT TGT COST NKE SBUX MCD CVS WBA PFE MRK BMY ABBV JNJ
GILD MRNA BAC WFC C JPM GS MS SCHW KEY RF HBAN USB XOM CVX OXY SLB HAL DVN
BP KMI AAL DAL UAL CCL NCLH RCL BA GE CAT FCX NEM KGC GOLD AA CLF X VALE
MARA RIOT CLSK IONQ RKLB SOUN BBAI ACHR JOBY LUMN PLUG CHPT RUN ENPH
SPY QQQ IWM SOXL TQQQ SQQQ TLT XLF XLE
"""


def screener_page(st, yf):
    st.title("高成交量選股")
    st.caption("找出成交量大、股價在預算內的股票，並顯示策略 D 目前的訊號狀態。僅供研究與教學，不構成投資建議。")

    with st.sidebar:
        src = st.radio("股票池", ["內建熱門股 (約140檔)", "Yahoo 今日最活躍", "自己輸入"])
        custom = st.text_area("自己輸入代號 (空格或逗號分隔)", "", disabled=src != "自己輸入")
        pmin, pmax = st.slider("股價範圍 (美元)", 1, 1000, (5, 100))
        min_vol = st.number_input("20日均量至少 (萬股)", 0, 100000, 500, 100)
        min_dv = st.number_input("20日均成交額至少 (百萬美元)", 0, 100000, 50, 10)
        top_n = st.number_input("最多列出幾檔", 5, 200, 30, 5)
        confirm_n = st.number_input("策略D: KDJ 交叉後幾天內等 MACD", 1, 20, 5, 1)
        vol_mult = st.number_input("策略D: 帶量倍數", 1.0, 3.0, 1.2, 0.1)
        only_up = st.checkbox("只看多頭排列 (收盤>MA50>MA200)", False)
        run = st.button("開始篩選", type="primary")

    if run:
        if src.startswith("Yahoo"):
            try:
                q = yf.screen("most_actives", count=200)
                universe = [x["symbol"] for x in q.get("quotes", [])]
            except Exception as e:
                st.error(f"抓不到 Yahoo 最活躍清單: {e}，改用內建清單")
                universe = DEFAULT_UNIVERSE.split()
        elif src == "自己輸入":
            universe = custom.replace(",", " ").upper().split()
        else:
            universe = DEFAULT_UNIVERSE.split()
        universe = list(dict.fromkeys(universe))
        if not universe:
            st.warning("股票池是空的")
            return

        with st.spinner(f"下載 {len(universe)} 檔股票資料中…"):
            data = yf.download(universe, period="14mo", group_by="ticker", auto_adjust=True,
                               progress=False, threads=True)
        rows = []
        for t in universe:
            try:
                d = data[t] if len(universe) > 1 else data
                d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(d) < 210:
                    continue
                d = add_indicators(d.copy(), confirm_n=confirm_n, vol_mult=vol_mult)
                r = d.iloc[-1]
                avg_v = d["Volume"].tail(20).mean()
                avg_dv = (d["Close"] * d["Volume"]).tail(20).mean()
                if r["D_BUY"]:
                    sig = "今日買進訊號"
                elif r["D_SELL"]:
                    sig = "今日賣出訊號"
                elif d["KDJ_GOLD"].tail(confirm_n).any() and not d["MACD_GOLD"].tail(confirm_n).any():
                    sig = "KDJ已金叉，等MACD確認"
                else:
                    sig = ""
                rows.append({"代號": t, "收盤價": r["Close"],
                             "20日均量(萬股)": avg_v / 1e4, "20日均成交額(百萬$)": avg_dv / 1e6,
                             "今日量比": r["Volume"] / r["VOL_MA20"] if r["VOL_MA20"] else np.nan,
                             "多頭排列": bool(r["Close"] > r["MA50"] > r["MA200"]),
                             "策略D訊號": sig, "資料日期": d.index[-1].date()})
            except Exception:
                continue
        st.session_state["scr"] = pd.DataFrame(rows)
        st.session_state["scr_filters"] = (pmin, pmax, min_vol, min_dv, top_n, only_up)

    if "scr" not in st.session_state:
        st.info("在左側設定條件後按「開始篩選」。")
        return
    df = st.session_state["scr"]
    if df.empty:
        st.warning("沒有抓到任何資料，請稍後再試。")
        return
    pmin, pmax, min_vol, min_dv, top_n, only_up = st.session_state["scr_filters"]
    f = df[(df["收盤價"] >= pmin) & (df["收盤價"] <= pmax)
           & (df["20日均量(萬股)"] >= min_vol) & (df["20日均成交額(百萬$)"] >= min_dv)]
    if only_up:
        f = f[f["多頭排列"]]
    f = f.sort_values("20日均成交額(百萬$)", ascending=False).head(int(top_n))

    st.subheader(f"符合條件: {len(f)} 檔 (共檢查 {len(df)} 檔)")
    st.dataframe(f.style.format({"收盤價": "{:.2f}", "20日均量(萬股)": "{:,.0f}",
                                 "20日均成交額(百萬$)": "{:,.0f}", "今日量比": "{:.2f}"}),
                 use_container_width=True, hide_index=True)

    hot = f[f["策略D訊號"] != ""]
    if len(hot):
        st.subheader("策略 D 有動靜的股票")
        st.dataframe(hot[["代號", "收盤價", "今日量比", "策略D訊號"]].style.format(precision=2),
                     use_container_width=True, hide_index=True)

    if len(f):
        tick = " ".join(f["代號"])
        st.subheader("複製使用")
        st.caption("貼到「策略回測」的股票代號欄 (先驗證策略 D 在這些股票上的表現)")
        st.code(", ".join(f["代號"]), language=None)
        st.caption("貼到終端機執行自動下單程式")
        st.code(f".venv/bin/python ibkr_auto_trader.py {tick}", language=None)


# ---------------- 策略適配分類 ----------------
def efficiency_ratio(c, n=120):
    seg = c.tail(n + 1)
    move = abs(seg.iloc[-1] - seg.iloc[0])
    path = seg.diff().abs().sum()
    return move / path if path else np.nan


def classify_shape(tt, er, adx, atrp):
    if atrp < 1.5:
        return "波動太小"
    if atrp > 6:
        return "波動過大"
    if tt >= 7:
        return "趨勢領頭型 → VCP+ATR"
    if tt <= 5 and er < 0.15 and adx < 25 and atrp >= 2:
        return "規律擺盪型 → MACD+KDJ"
    return "混合型"


def classify_bt(d_ret, d_n, v_ret, v_n):
    # VCP 突破本來就少見，門檻設 2 筆；MACD+KDJ 訊號多，門檻 3 筆
    d_ok, v_ok = d_n >= 3, v_n >= 2
    if not d_ok and not v_ok:
        return "樣本不足"
    if d_ok and v_ok:
        if d_ret <= 0 and v_ret <= 0:
            return "兩者都虧"
        return "MACD+KDJ 較佳" if d_ret > v_ret else "VCP+ATR 較佳"
    if d_ok:
        return "MACD+KDJ 較佳" if d_ret > 0 else "兩者都虧"
    return "VCP+ATR 較佳" if v_ret > 0 else "兩者都虧"


def final_call(shape, bt):
    if "VCP" in shape and bt == "VCP+ATR 較佳":
        return "✅ VCP+ATR"
    if "MACD" in shape and bt == "MACD+KDJ 較佳":
        return "✅ MACD+KDJ"
    if bt in ("VCP+ATR 較佳", "MACD+KDJ 較佳"):
        return "△ " + bt.replace(" 較佳", "") + " (型態不一致)"
    if "→" in shape:
        return "△ " + shape.split("→ ")[1] + " (回測未證實)"
    return "✕ 暫不適用"


def strategy_fit_page(st, yf):
    st.title("策略適配分類")
    st.caption("判斷每檔股票比較適合 MACD+KDJ 二次確認，還是 VCP+ATR。"
               "同時看「股性型態」和「實際回測」，兩者一致才打 ✅。僅供研究與教學，不構成投資建議。")

    with st.sidebar:
        src = st.radio("股票池", ["內建熱門股 (約140檔)", "自己輸入"])
        custom = st.text_area("自己輸入代號 (空格或逗號分隔)", "", disabled=src != "自己輸入")
        years = st.number_input("回測最近幾年", 1, 10, 5, 1)
        cost = st.number_input("單邊成本 %", 0.0, 1.0, 0.10, 0.05) / 100
        confirm_n = st.number_input("MACD+KDJ: KDJ 交叉後幾天內等 MACD", 1, 20, 5, 1)
        vol_mult = st.number_input("MACD+KDJ: 帶量倍數", 1.0, 3.0, 1.2, 0.1)
        atr_init = st.number_input("VCP: 初始停損 ATR 倍數", 0.5, 5.0, 2.0, 0.5)
        atr_trail = st.number_input("VCP: 移動停損 ATR 倍數", 0.5, 6.0, 3.0, 0.5)
        run = st.button("開始分類", type="primary")

    with st.expander("判斷規則說明"):
        st.markdown("""
**股性型態**（看最近約半年）
- 趨勢效率 ER：淨漲跌幅 ÷ 每日漲跌絕對值加總。越接近 1 代表走得越直，越接近 0 代表來回震盪。
- ADX：大於 25 代表趨勢明確。
- ATR%：每日平均波動占股價比例。
- 趨勢模板：Minervini 8 條，RS 以本次股票池內的相對強度排名計算。
- 趨勢模板 ≥ 7 → 趨勢領頭型，適合 VCP+ATR（整理打底期間 ER 本來就低，所以不看 ER）
- 趨勢模板 ≤ 5、ER < 0.15、ADX < 25、ATR% ≥ 2 → 規律擺盪型，適合 MACD+KDJ
- ATR% < 1.5 太小，短線不划算；ATR% > 6 波動過大

**實際回測**：同一期間分別跑兩個策略。MACD+KDJ 至少 3 筆、VCP+ATR 至少 2 筆才列入比較（VCP 突破本來就少，所以預設回測 5 年）。

**VCP+ATR 為簡化版**：前一天通過趨勢模板前 7 條，且波動收縮（突破前 10 天內 ATR10/ATR50 曾 < 0.85、前一天 15 日振幅 < 15%），
當天收盤突破 20 日高點並帶量 1.4 倍進場；初始停損與移動停損都用 ATR 倍數。
真正的 VCP 還要看收縮次數與形態，請再用你的 VCP 工具人工確認。

這些門檻是經驗值，不是定律，可以依你的觀察調整。
""")

    if run:
        universe = (custom.replace(",", " ").upper().split() if src == "自己輸入"
                    else DEFAULT_UNIVERSE.split())
        universe = list(dict.fromkeys(universe))
        if not universe:
            st.warning("股票池是空的")
            return
        with st.spinner(f"下載 {len(universe)} 檔、{years + 2} 年資料中…"):
            data = yf.download(universe, period=f"{int(years) + 2}y", group_by="ticker",
                               auto_adjust=True, progress=False, threads=True)
        frames = {}
        for t in universe:
            try:
                d = data[t] if len(universe) > 1 else data
                d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(d) >= 300:
                    frames[t] = d
            except Exception:
                continue
        # RS 排名 (IBD 風格加權報酬)
        rs_raw = {}
        for t, d in frames.items():
            c = d["Close"]
            rs_raw[t] = sum(w * (c.iloc[-1] / c.iloc[-1 - n] - 1)
                            for w, n in ((0.4, 63), (0.2, 126), (0.2, 189), (0.2, 252)))
        rs_rank = pd.Series(rs_raw).rank(pct=True) * 99

        rows, prog = [], st.progress(0.0)
        for j, (t, d) in enumerate(frames.items()):
            prog.progress((j + 1) / len(frames))
            try:
                d = add_indicators(d.copy(), confirm_n=confirm_n, vol_mult=vol_mult)
                r = d.iloc[-1]
                tt = int(sum([r["Close"] > r["MA150"] and r["Close"] > r["MA200"], r["MA150"] > r["MA200"],
                              bool(r["MA200_UP"]), r["MA50"] > r["MA150"] and r["MA50"] > r["MA200"],
                              r["Close"] > r["MA50"], r["Close"] >= 1.3 * r["LO52"],
                              r["Close"] >= 0.75 * r["HI52"], rs_rank[t] >= 70]))
                er = efficiency_ratio(d["Close"])
                atrp = r["ATR"] / r["Close"] * 100
                start = d.index[-1] - pd.DateOffset(years=int(years))
                w = d[d.index >= start]
                eq_d, tr_d, _ = backtest(w, "D", cost, atr_init, atr_trail, 1.0)
                eq_v, tr_v, _ = backtest(w, "V", cost, atr_init, atr_trail, 1.0)
                d_ret, v_ret = (eq_d.iloc[-1] - 1) * 100, (eq_v.iloc[-1] - 1) * 100
                shape = classify_shape(tt, er, r["ADX"], atrp)
                bt = classify_bt(d_ret, len(tr_d), v_ret, len(tr_v))
                rows.append({"代號": t, "綜合建議": final_call(shape, bt), "股性型態": shape, "回測判定": bt,
                             "收盤價": r["Close"], "趨勢模板(/8)": tt, "RS": rs_rank[t], "ER": er,
                             "ADX": r["ADX"], "ATR%": atrp,
                             "MACD+KDJ報酬%": d_ret, "MACD+KDJ次數": len(tr_d),
                             "VCP+ATR報酬%": v_ret, "VCP+ATR次數": len(tr_v)})
            except Exception:
                continue
        prog.empty()
        st.session_state["fit"] = pd.DataFrame(rows)

    if "fit" not in st.session_state:
        st.info("在左側設定後按「開始分類」。")
        return
    df = st.session_state["fit"]
    if df.empty:
        st.warning("沒有抓到任何資料，請稍後再試。")
        return
    if len(df) < 20:
        st.warning("股票池少於 20 檔，RS 排名的參考性較低。")

    fmt = {"收盤價": "{:.2f}", "RS": "{:.0f}", "ER": "{:.2f}", "ADX": "{:.1f}", "ATR%": "{:.2f}",
           "MACD+KDJ報酬%": "{:.1f}", "VCP+ATR報酬%": "{:.1f}"}
    order = {"✅": 0, "△": 1, "✕": 2}
    df = df.assign(_o=df["綜合建議"].str[0].map(order)).sort_values(["_o", "綜合建議"]).drop(columns="_o")

    for label, key in (("適合 MACD+KDJ 二次確認", "MACD+KDJ"), ("適合 VCP+ATR", "VCP+ATR")):
        sub = df[df["綜合建議"] == f"✅ {key}"]
        st.subheader(f"{label}：{len(sub)} 檔")
        if len(sub):
            st.dataframe(sub.style.format(fmt), use_container_width=True, hide_index=True)
            st.code(", ".join(sub["代號"]), language=None)
        else:
            st.caption("目前沒有兩項判斷都一致的股票。")

    with st.expander(f"全部結果 ({len(df)} 檔，含 △ 待觀察與 ✕ 不適用)"):
        st.dataframe(df.style.format(fmt), use_container_width=True, hide_index=True)


# ---------------- 介面 ----------------
def main():
    import streamlit as st
    import yfinance as yf
    import plotly.graph_objects as go

    st.set_page_config(page_title="MACD+KDJ 短線回測", layout="wide")
    mode = st.sidebar.radio("功能", ["策略回測", "高成交量選股", "策略適配分類"])
    if mode == "策略適配分類":
        strategy_fit_page(st, yf)
        return
    if mode == "高成交量選股":
        screener_page(st, yf)
        return
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

    if go_btn:
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
        st.session_state["bt"] = (rows, all_res, capital)

    if "bt" not in st.session_state:
        st.info("在左側輸入股票代號後按「開始回測」。")
        return
    rows, all_res, capital = st.session_state["bt"]
    if not rows:
        return
    summary = pd.DataFrame(rows)
    st.subheader("總覽")
    st.dataframe(summary.style.format(precision=2), use_container_width=True, hide_index=True)

    st.subheader("各策略平均 (跨所有股票)")
    avg = summary.groupby("策略", sort=False).mean(numeric_only=True)
    st.dataframe(avg.style.format(precision=2), use_container_width=True)

    st.subheader("單檔細看")
    pick = st.selectbox("選擇股票", list(all_res))
    df, res = all_res[pick]

    fig = go.Figure()
    for k, v in res.items():
        fig.add_trace(go.Scatter(x=v["eq"].index, y=v["eq"], name=STRATS[k]))
    fig.update_layout(title=f"{pick} 資金曲線 (美元)", height=420, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    sk = st.radio("顯示買賣點", ["A", "B", "C", "D", "V"], format_func=lambda k: STRATS[k], horizontal=True)
    tr = res[sk]["trades"]
    c1, c2, c3 = st.columns(3)
    show_bb = c1.checkbox("布林通道", True)
    show_ma = c2.checkbox("MA50 / MA200", True)
    days = c3.selectbox("顯示區間", ["近6個月", "近1年", "全部"], index=0)
    st.plotly_chart(kline_chart(df, tr, pick, show_bb, show_ma, days), use_container_width=True)

    with st.expander("交易明細"):
        st.dataframe(tr.style.format(precision=2), use_container_width=True, hide_index=True)

    # ---------- 最下方: 最終損益加總 ----------
    st.divider()
    n = len(all_res)
    invested = capital * n
    st.subheader("最終損益加總")
    st.caption(f"共 {n} 檔股票，每檔起始 ${capital:,.0f}，總投入 ${invested:,.0f}")
    tot = summary.groupby("策略", sort=False).agg(**{"期末資金$": ("期末資金$", "sum"),
                                                     "損益$": ("損益$", "sum"),
                                                     "交易次數": ("交易次數", "sum")})
    cols = st.columns(len(tot))
    for col, (name, r) in zip(cols, tot.iterrows()):
        pct = r["損益$"] / invested * 100
        col.metric(name, f"${r['損益$']:,.0f}", f"{pct:+.1f}%")
        col.caption(f"期末 ${r['期末資金$']:,.0f}｜交易 {int(r['交易次數'])} 次")


if __name__ == "__main__":
    main()
