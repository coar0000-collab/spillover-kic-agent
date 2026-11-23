"""과제 2: 충격 크기 예측(국면 요인 추가 효과) / 과제 3: 한국 반응의 경로 분해(미국·중국·아시아)"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
X = str(ROOT / "data") + "/"; F = X
import pandas as pd, numpy as np, json, warnings
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
warnings.filterwarnings("ignore")

L = pd.read_pickle(X + "lib_v1.pkl"); D = pd.read_pickle(F + "daily.pkl"); Yh = pd.read_pickle(F + "yahoo.pkl")
REG = ["VIX", "EPU", "K20", "Ksig60", "SOX20", "SH60", "CU60", "CNY60", "KRW60", "EXKR_yoy", "EXCN_yoy"]
def state(d):   # 이벤트 직전 상태: d 이전 마지막 미국 거래일 행
    s = D[D.index < d]; return s.iloc[-1][REG] if len(s) else pd.Series(index=REG, dtype=float)
E = L.copy()
st = pd.DataFrame([state(d) for d in E.date], index=E.index); E = pd.concat([E, st], axis=1)
E["absK"] = E.KOSPI_ret.abs(); E["ratio"] = E.absK / E.Ksig60
E["tariffish"] = E.type.isin(["tariff", "export_ctrl", "industrial", "legal", "geopolitics"]).astype(int)
E["cnkr"] = E.cn + E.kr
out = {}
# ---------- 과제 2: 크기 ----------
C = E[E.clean].sort_values("date").reset_index(drop=True)
C["EXKR_weak"] = (C.EXKR_yoy < 0).astype(int); C["x_tar_ex"] = C.tariffish * C.EXKR_weak
C["x_cn_sh"] = C.cn * (-C.SH60)
feats = {"직전 변동성×유사사례 배수(기준)": None,
         "+국면(VIX·불확실성·추세)": ["VIX", "EPU", "K20", "SOX20"],
         "+중국(상하이·위안·구리)": ["VIX", "EPU", "K20", "SOX20", "SH60", "CNY60", "CU60"],
         "+수출(한·중 수출 증가율)": ["VIX", "EPU", "K20", "SOX20", "SH60", "CNY60", "CU60", "EXKR_yoy", "EXCN_yoy", "x_tar_ex", "x_cn_sh"]}
preds = {k: [] for k in feats}; act = []
for i, r in C.iterrows():
    past = C[C.date < r.date - pd.Timedelta(days=7)]
    if len(past) < 30: continue
    same = past[(past.type == r.type) & (past.kr == r.kr)]
    if len(same) < 3: same = past[past.type == r.type]
    if len(same) < 3: same = past
    base = same.ratio.median() * r.Ksig60
    act.append(r.absK)
    for k, cols in feats.items():
        if cols is None: preds[k].append(base); continue
        tr = past.dropna(subset=cols + ["ratio"])
        m = make_pipeline(StandardScaler(), Ridge(alpha=5.0)).fit(tr[["tariffish", "kr", "cn"] + cols], np.log(tr.ratio + 0.05))
        xr = r[["tariffish", "kr", "cn"] + cols].astype(float).values.reshape(1, -1)
        if np.isnan(xr).any(): preds[k].append(base); continue
        preds[k].append(float(np.exp(m.predict(xr)[0])) * r.Ksig60)
act = np.array(act)
thr = np.quantile(act, 2 / 3)
size = {}
for k, v in preds.items():
    v = np.array(v); top = v >= np.quantile(v, 2 / 3)
    size[k] = dict(rho=round(float(spearmanr(v, act).correlation), 3), top_prec=round(float((act[top] >= thr).mean()), 3), n=len(act))
print("== 충격 크기(시간순)\n", pd.DataFrame(size).T); out["size"] = size
# 해석용: 전체 표본 회귀(국면 변수의 부호)
cols = ["tariffish", "kr", "cn", "VIX", "EPU", "K20", "SOX20", "SH60", "CNY60", "CU60", "EXKR_yoy", "EXCN_yoy"]
cc = C.dropna(subset=cols)
ols = sm.OLS(np.log(cc.ratio + 0.05), sm.add_constant((cc[cols] - cc[cols].mean()) / cc[cols].std())).fit(cov_type="HC1")
print(ols.summary().tables[1]); out["size_ols"] = {c: dict(b=round(ols.params[c], 3), t=round(ols.tvalues[c], 2)) for c in cols}
# 수출 둔화기 × 무역 이벤트 크기 비교
g = C.groupby([C.tariffish, C.EXKR_yoy < 0]).absK.agg(["median", "count"]).round(2); print("== 한국 수출 감소기 여부별 |KOSPI 5일| 중앙값\n", g)
out["export_regime"] = {f"{'무역' if a else '기타'}|{'수출감소' if b else '수출증가'}": dict(median=float(v["median"]), n=int(v["count"])) for (a, b), v in g.iterrows()}

# ---------- 과제 3: 경로 분해 ----------
def win_ret(s, d, kr=True, tm=""):
    s = s.dropna()
    b = (s.index < d).sum() - 1 if (tm == "kr") else (s.index <= d).sum() - 1
    if b + 5 >= len(s) or b < 0: return np.nan
    return np.log(s.iloc[b + 5] / s.iloc[b]) * 100
rows = []
for r in E.itertuples():
    o = dict(name=r.name, date=r.date, type=r.type, cn=r.cn, kr=r.kr, clean=r.clean, esc=r.esc, K=win_ret(Yh.KOSPI, r.date, tm=r.tm))
    for c in ["HSI", "SHCOMP", "TWII", "N225"]: o[c] = win_ret(Yh[c], r.date, tm=r.tm)
    sp = Yh.SPX.dropna(); b = (sp.index < r.R).sum() - 1 if pd.notna(r.R) else None
    o["SPX"] = np.log(sp.iloc[b + 5] / sp.iloc[b]) * 100 if b is not None and b + 5 < len(sp) else np.nan
    o["SOX"] = np.log(Yh.SOX.dropna().iloc[b + 5] / Yh.SOX.dropna().iloc[b]) * 100 if b is not None and b + 5 < len(Yh.SOX.dropna()) else np.nan
    rows.append(o)
W = pd.DataFrame(rows).dropna(subset=["K", "SPX", "HSI", "SHCOMP", "TWII", "N225", "SOX"])
W["CHINA"] = W[["HSI", "SHCOMP"]].mean(axis=1); W["ASIA"] = W[["TWII", "N225"]].mean(axis=1)
def r2(df, cols): return sm.OLS(df.K, sm.add_constant(df[cols])).fit().rsquared
dec = {}
for gname, gdf in {"전체": W, "중국 대상": W[W.cn == 1], "한국 직접 대상": W[W.kr == 1], "통화정책": W[W.type.str.startswith("mon")],
                   "무역·통상(중국 외)": W[W.type.isin(["tariff", "export_ctrl", "industrial", "legal"]) & (W.cn == 0)]}.items():
    if len(gdf) < 12: continue
    a = r2(gdf, ["SPX"]); b = r2(gdf, ["SPX", "SOX"]); c = r2(gdf, ["SPX", "SOX", "CHINA"]); d = r2(gdf, ["SPX", "SOX", "CHINA", "ASIA"])
    dec[gname] = dict(n=len(gdf), 미국=round(a, 2), 반도체추가=round(b - a, 2), 중국추가=round(c - b, 2), 아시아추가=round(d - c, 2), 설명안됨=round(1 - d, 2))
print("== 한국 5일 반응 분산 분해(R² 증분)\n", pd.DataFrame(dec).T); out["decomp"] = dec
json.dump(out, open(F + "event_results.json", "w"), ensure_ascii=False, indent=1, default=str)
