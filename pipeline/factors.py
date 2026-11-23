"""확장 요인 데이터셋: 미국 장 마감 시점(US day d)에 알 수 있는 정보만으로 한국 다음 세션 예측"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
X = str(ROOT / "data") + "/"; F = X
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

Y = pd.read_pickle(F + "yahoo.pkl")
def fred(path, s):
    d = pd.read_csv(path, parse_dates=[0]).set_index("observation_date")[s]; return pd.to_numeric(d, errors="coerce").dropna()
CNY = fred(F + "DEXCHUS.csv", "DEXCHUS"); WTI = fred(F + "DCOILWTICO.csv", "DCOILWTICO"); Y10 = fred(F + "DGS10.csv", "DGS10")
EPU = fred(F + "USEPUINDXD.csv", "USEPUINDXD")
Y2 = fred(X + "DGS2.csv", "DGS2"); KRW = fred(X + "DEXKOUS.csv", "DEXKOUS")
EXKR = fred(F + "XTEXVA01KRM667S.csv", "XTEXVA01KRM667S"); EXCN = fred(F + "XTEXVA01CNM667S.csv", "XTEXVA01CNM667S")

us_days = Y["SPX"].dropna().index; K = Y["KOSPI"].dropna()
def lr(s): return np.log(s).diff() * 100
def at(s, d):   # 날짜 d까지 관측된 마지막 값
    s = s[s.index <= d]; return s.iloc[-1] if len(s) else np.nan
rows = []
r = {c: lr(Y[c].dropna()) for c in ["SPX", "NASDAQ", "SOX", "EWY", "COPPER", "DXY", "HSI", "SHCOMP", "TWII", "N225", "KOSPI", "KOSDAQ"]}
vix = Y["VIX"].dropna(); rcny = lr(CNY); rwti = lr(WTI); rkrw = lr(KRW)
for i, d in enumerate(us_days[260:], start=260):
    bk = (K.index <= d).sum() - 1                          # 한국: 날짜 d 장 마감(미국 d 세션 이전)
    if bk + 1 >= len(K): break
    o = dict(d=d, y=(K.iloc[bk + 1] / K.iloc[bk] - 1) * 100, kday=K.index[bk + 1])
    # 미국 d 세션(한국 장 마감 이후 정보)
    for c in ["SPX", "NASDAQ", "SOX", "EWY", "COPPER", "DXY"]:
        o[c] = r[c].get(d, np.nan)
    o["VIXch"] = vix.get(d, np.nan) - at(vix, d - pd.Timedelta(days=1))
    o["Y2ch"] = (at(Y2, d) - at(Y2, d - pd.Timedelta(days=1))) * 100
    o["Y10ch"] = (at(Y10, d) - at(Y10, d - pd.Timedelta(days=1))) * 100
    o["WTI"] = rwti.get(d, np.nan); o["CNY"] = rcny.get(d, np.nan)
    # 아시아 같은 날(한국과 같은 시간대, 이미 마감)
    for c in ["HSI", "SHCOMP", "TWII", "N225", "KOSPI"]:
        s = r[c]; o[c + "_d"] = s[s.index <= d].iloc[-1] if (s.index <= d).any() else np.nan
    # 국면(사전 상태)
    o["VIX"] = at(vix, d); o["EPU"] = at(EPU, d - pd.Timedelta(days=1))
    o["K20"] = np.log(K.iloc[bk] / K.iloc[bk - 20]) * 100; o["Ksig60"] = lr(K).iloc[bk - 60:bk].std()
    sox = Y["SOX"].dropna(); bs = (sox.index <= d).sum() - 1; o["SOX20"] = np.log(sox.iloc[bs] / sox.iloc[bs - 20]) * 100
    hs = Y["SHCOMP"].dropna(); bh = (hs.index <= d).sum() - 1; o["SH60"] = np.log(hs.iloc[bh] / hs.iloc[bh - 60]) * 100
    cu = Y["COPPER"].dropna(); bc = (cu.index <= d).sum() - 1; o["CU60"] = np.log(cu.iloc[bc] / cu.iloc[bc - 60]) * 100
    c60 = CNY[CNY.index <= d]; o["CNY60"] = np.log(c60.iloc[-1] / c60.iloc[-61]) * 100
    k60 = KRW[KRW.index <= d]; o["KRW60"] = np.log(k60.iloc[-1] / k60.iloc[-61]) * 100
    # 수출(발표 시차 반영: 2개월 전 수치까지 사용)
    lag = (d - pd.DateOffset(months=2)).to_period("M").to_timestamp()
    ek = EXKR[EXKR.index <= lag]; ec = EXCN[EXCN.index <= lag]
    o["EXKR_yoy"] = (ek.iloc[-1] / ek.iloc[-13] - 1) * 100 if len(ek) > 13 else np.nan
    o["EXCN_yoy"] = (ec.iloc[-1] / ec.iloc[-13] - 1) * 100 if len(ec) > 13 else np.nan
    rows.append(o)
D = pd.DataFrame(rows).set_index("d")
# 정책 이벤트일 표시(미국 반응일 R 기준)
L = pd.read_pickle(X + "lib_v1.pkl")
ev = L.dropna(subset=["R"]).groupby("R").agg(n=("name", "count"), clean=("clean", "all"), types=("type", lambda x: ",".join(sorted(set(x)))),
                                              cn=("cn", "max"), kr=("kr", "max"), esc=("esc", "max"))
D = D.join(ev, how="left"); D["event"] = D.n.notna(); D["clean"] = D.clean.fillna(False)
D.to_pickle(F + "daily.pkl")
print(D.shape, D.event.sum(), (D.event & D.clean).sum()); print(D.isna().mean().round(2)[D.isna().mean() > 0.02])
