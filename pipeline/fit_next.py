"""과제 1: 미국 장 마감 정보 → 한국 다음 세션. 연도별 확장 창(과거만 학습)"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
X = str(ROOT / "data") + "/"; F = X
import pandas as pd, numpy as np, json, warnings
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
D = pd.read_pickle(X + "daily.pkl")
D = D[D.y.abs() < 15]                       # 상·하한 이상치 방지
G = {
 "미국 지수": ["SPX"],
 "+미국 기술·반도체": ["SPX", "NASDAQ", "SOX"],
 "+EWY(미 장중 한국 ETF)": ["SPX", "NASDAQ", "SOX", "EWY"],
 "+중국·아시아·원자재": ["SPX", "NASDAQ", "SOX", "EWY", "HSI_d", "SHCOMP_d", "TWII_d", "N225_d", "KOSPI_d", "CNY", "COPPER", "WTI"],
 "+금리·달러·변동성": ["SPX", "NASDAQ", "SOX", "EWY", "HSI_d", "SHCOMP_d", "TWII_d", "N225_d", "KOSPI_d", "CNY", "COPPER", "WTI", "Y2ch", "Y10ch", "DXY", "VIXch"],
 "+국면(수출·중국·불확실성)": ["SPX", "NASDAQ", "SOX", "EWY", "HSI_d", "SHCOMP_d", "TWII_d", "N225_d", "KOSPI_d", "CNY", "COPPER", "WTI", "Y2ch", "Y10ch", "DXY", "VIXch",
                       "VIX", "EPU", "K20", "Ksig60", "SOX20", "SH60", "CU60", "CNY60", "KRW60", "EXKR_yoy", "EXCN_yoy"],
}
D = D.copy(); allf = sorted({f for v in G.values() for f in v})
D[allf] = D[allf].ffill()
res, pred = [], {}
for name, cols in G.items():
    for mdl_name, mk in [("릿지", lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
                         ("부스팅", lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40))]:
        if mdl_name == "부스팅" and name not in ("+EWY(미 장중 한국 ETF)", "+국면(수출·중국·불확실성)"): continue
        p = pd.Series(np.nan, index=D.index)
        for yr in range(2019, 2027):
            tr = D[(D.index.year < yr)].dropna(subset=cols + ["y"]); te = D[D.index.year == yr].dropna(subset=cols)
            if len(tr) < 250 or te.empty: continue
            m = mk().fit(tr[cols], tr.y); p.loc[te.index] = m.predict(te[cols])
        pred[(name, mdl_name)] = p
        for seg, mask in [("전체 거래일", D.index == D.index), ("정책 이벤트일", D.event.values), ("이벤트(교란 제외)", (D.event & D.clean).values),
                          ("중국 대상 이벤트", (D.event & (D.cn == 1)).values), ("미 변동 상위20%", (D.SPX.abs() > D.SPX.abs().quantile(.8)).values)]:
            s = D[mask & p.notna().values]; ps = p[s.index]
            nz = s.y.abs() > 0.1
            res.append(dict(features=name, model=mdl_name, segment=seg, n=int(len(s)),
                            hit=round(float((np.sign(ps[nz]) == np.sign(s.y[nz])).mean()), 3),
                            rho=round(float(spearmanr(ps, s.y).correlation), 3),
                            mae=round(float((ps - s.y).abs().mean()), 3)))
R = pd.DataFrame(res)
pv = R.pivot_table(index=["features", "model"], columns="segment", values="hit").reindex(columns=["전체 거래일", "정책 이벤트일", "이벤트(교란 제외)", "중국 대상 이벤트", "미 변동 상위20%"])
order = [(k, m) for k in G for m in ("릿지", "부스팅") if (k, m) in pv.index]
print(pv.loc[order].round(2)); print(R[R.segment == "이벤트(교란 제외)"].set_index(["features", "model"])[["n", "rho", "mae"]].loc[order])
R.to_csv(X + "next_results.csv", index=False, encoding="utf-8-sig")
# 최종 모델 계수(해석): 전체 기간 릿지
cols = G["+국면(수출·중국·불확실성)"]; tr = D.dropna(subset=cols + ["y"])
m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(tr[cols], tr.y)
coef = pd.Series(m[-1].coef_, index=cols).sort_values(key=abs, ascending=False)
print("표준화 계수(상위)\n", coef.head(12).round(3))
coef.to_csv(X + "next_coef.csv", encoding="utf-8-sig")
pd.to_pickle(pred, X + "next_pred.pkl")
