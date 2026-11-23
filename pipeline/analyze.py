from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
X = str(ROOT / "data") + "/"; F = X
import pandas as pd, numpy as np, json, warnings
from scipy.stats import spearmanr, mannwhitneyu
import statsmodels.api as sm
warnings.filterwarnings("ignore")

L = pd.read_pickle(X + "lib_v1.pkl")
SECT = ["반도체", "자동차", "철강", "2차전지", "조선중공업", "기계장비", "에너지화학", "바이오헬스", "IT", "필수소비재", "경기소비재", "은행", "증권", "건설", "운송"]
R = {}

# ========== D. 노출도: 섹터 횡단면 ==========
rows = []
for _, e in L.iterrows():
    if e.type not in ("tariff", "export_ctrl", "industrial", "legal", "geopolitics") or e.esc == 0: continue
    tgt = dict(x.split(":") for x in e.sect.split(",") if x) if e.sect else {}
    for s in SECT:
        a = e.get(f"S_{s}_abn")
        if pd.isna(a): continue
        ex = e[f"cn_{s}"] if e.type == "export_ctrl" else e[f"us_{s}"]
        rows.append(dict(event=e["name"], date=e.date, type=e.type, esc=e.esc, kr=e.kr, clean=e.clean, sector=s,
                         abn=a, abnz=e[f"S_{s}_abnz"], expo=ex, tsign=int(tgt.get(s, 0))))
S = pd.DataFrame(rows)
S["x_expo"] = -S.esc * S.expo          # 강화(+1)면 노출 클수록 음(-) 기대
S["x_tgt"] = S.tsign                    # 표적 섹터 부호(피해 -1, 수혜 +1)
def ols(df, cols, y="abn"):
    m = sm.OLS(df[y], sm.add_constant(df[cols])).fit(cov_type="cluster", cov_kwds={"groups": df.event.astype("category").cat.codes})
    return m
m_all = ols(S, ["x_expo", "x_tgt"]); m_cl = ols(S[S.clean], ["x_expo", "x_tgt"])
R["expo_reg"] = {k: dict(b=round(m.params[c], 2), t=round(m.tvalues[c], 2), n=int(m.nobs), events=int(d.event.nunique()))
                 for k, (m, d) in {"all": (m_all, S), "clean": (m_cl, S[S.clean])}.items() for c in ["x_expo", "x_tgt"]
                 for _ in [0] if True} if False else None
R["expo_reg"] = {}
for k, (m, d) in {"all": (m_all, S), "clean": (m_cl, S[S.clean])}.items():
    R["expo_reg"][k] = {c: dict(b=round(m.params[c], 3), t=round(m.tvalues[c], 2)) for c in ["x_expo", "x_tgt"]}
    R["expo_reg"][k]["n"] = int(m.nobs); R["expo_reg"][k]["events"] = int(d.event.nunique())
# 표적 섹터 방향 적중
T = S[S.tsign != 0]
R["target_hit"] = dict(n=len(T), hit=round((np.sign(T.abn) == T.tsign).mean(), 2), mean_abn_harm=round(T[T.tsign < 0].abn.mean(), 2),
                       mean_abn_benefit=round(T[T.tsign > 0].abn.mean(), 2))
# 광범위 관세(표적 없음, 한국 직접): 고노출(상위 1/3) vs 저노출
B = S[(S.type == "tariff") & (S.tsign == 0) & (S.kr == 1)]
q = B.groupby("event").expo.transform(lambda x: x.rank(pct=True))
B = B.assign(grp=np.where(q > 2 / 3, "고노출", np.where(q <= 1 / 3, "저노출", "중간")))
g = B.groupby(["esc", "grp"]).abn.mean().unstack().round(2)
R["broad_tariff"] = g.to_dict(); R["broad_tariff_n"] = int(B.event.nunique())
print("== 노출도 회귀\n", json.dumps(R["expo_reg"], ensure_ascii=False, indent=0))
print("== 표적 섹터", R["target_hit"]); print("== 광범위 관세 고/저 노출(KOSPI 대비 +5일 비정상수익률 %)\n", g, B.event.nunique())
S.to_csv(X + "sector_panel.csv", index=False, encoding="utf-8-sig")

# ========== E. 시장 기대치 ==========
C = L[L.clean].copy()
C["absK"] = C.KOSPI_z.abs(); C["absF"] = C.FX_z.abs()
C["sched"] = C.type.str.startswith("mon") & ~C.name.str.contains("긴급|무제한|스와프|잭슨홀") | (C.type == "election")
r1 = C.groupby("sched")[["absK", "absF"]].median().round(2)
print("== 예정(스케줄) vs 비예정 |z| 중앙값\n", r1)
R["sched"] = r1.to_dict()
# 사전 반영: 발표 전 5일 움직임 방향과 이후 반응 방향(반전 여부)
pre = C[C.esc != 0]
rho_pre = spearmanr(pre.KOSPI_pre * pre.esc, pre.KOSPI_ret * pre.esc)
print("== 사전 반영(위험방향 정렬) 전후 상관", rho_pre)
R["pre_rho"] = round(rho_pre.correlation, 2); R["pre_p"] = round(rho_pre.pvalue, 3); R["pre_n"] = len(pre)
rv = spearmanr(C.vix, C.absK); print("== VIX vs |KOSPI z|", rv); R["vix_rho"] = round(rv.correlation, 2); R["vix_p"] = round(rv.pvalue, 3)
# FOMC: 2년물 당일 변화(서프라이즈) 크기 vs 동결/변경
F = L[L.type.str.startswith("mon") & L.clean]
print("== FOMC 2년물 서프라이즈 |bp| 중앙값", F.s_y2.abs().median(), "KOSPI|z| 상관", spearmanr(F.s_y2.abs(), F.KOSPI_z.abs()))

# ========== B. 방향 예측: 시간순(과거 사례만 학습) ==========
TYPES = sorted(L.type.unique())
def f_type(r): return np.r_[[1.0 if r.type == t else 0 for t in TYPES], r.esc, r.kr, r.cn]
L["sched"] = L.type.str.startswith("mon") & ~L.name.str.contains("긴급|무제한|스와프|잭슨홀") | (L.type == "election")
FEATS = {
    "유형 유사사례": None,
    "미국 당일반응": ["s_spx", "s_y2", "s_usd"],
    "+기대치": ["s_spx", "s_y2", "s_usd", "s_ewy0", "KOSPI_pre", "vix"],
    "+기대치+노출도": ["s_spx", "s_y2", "s_usd", "s_ewy0", "KOSPI_pre", "vix", "kr_esc", "expo_w"],
}
# 이벤트 단위 노출도: 표적 섹터 노출(없으면 한국 직접 대상 여부)
def expo_w(r):
    if not r.sect: return 0.0
    tg = dict(x.split(":") for x in r.sect.split(","))
    return sum(int(v) * r[f"us_{k}"] for k, v in tg.items())
L["expo_w"] = L.apply(expo_w, axis=1); L["kr_esc"] = L.kr * L.esc
res = []
targets = {"KOSPI": "T_KOSPI_z", "원달러": "T_FX_z", "EWY고유": "T_EWYabn_z"}
Ls = L.sort_values("date").reset_index(drop=True)
for i, r in Ls.iterrows():
    if not r.clean: continue
    past = Ls[(Ls.date < r.date - pd.Timedelta(days=7))]
    pc = past[past.clean]
    if len(pc) < 30: continue
    for tn, tc in targets.items():
        y = r[tc]
        if pd.isna(y) or abs(y) < 0.5: continue
        rec = dict(date=r.date, name=r["name"], target=tn, y=y)
        # 기준선: 과거 평균 부호
        rec["기준선"] = np.sign(pc[tc].mean())
        # 유형 kNN
        Xp = np.vstack([f_type(p) for p in pc.itertuples()]); xi = f_type(r)
        d = np.linalg.norm(Xp - xi, axis=1); nb = np.argsort(d)[:7]
        rec["유형 유사사례"] = np.sign(pc[tc].values[nb].mean())
        for name, cols in FEATS.items():
            if cols is None: continue
            tr = pc.dropna(subset=cols + [tc])
            if pd.isna(r[cols]).any() or len(tr) < 30: rec[name] = np.nan; continue
            mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
            A = sm.add_constant(((tr[cols] - mu) / sd).values, has_constant="add")
            mdl = sm.OLS(tr[tc].values, A).fit_regularized(alpha=0.05, L1_wt=0.0)  # 릿지
            xr = np.r_[1, ((r[cols] - mu) / sd).values.astype(float)]
            rec[name] = np.sign(xr @ mdl.params)
        res.append(rec)
P = pd.DataFrame(res)
models = ["기준선", "유형 유사사례", "미국 당일반응", "+기대치", "+기대치+노출도"]
hit = P.groupby("target").apply(lambda g: pd.Series({m: (g[m] == np.sign(g.y)).mean() for m in models} | {"n": len(g)})).round(2)
print("== 시간순 방향 적중률\n", hit)
R["hit"] = hit.reset_index().to_dict("records")
# 2025년 이후(최근 구간) 성능
hit_recent = P[P.date >= "2025-01-01"].groupby("target").apply(lambda g: pd.Series({m: (g[m] == np.sign(g.y)).mean() for m in models} | {"n": len(g)})).round(2)
print("== 2025년 이후\n", hit_recent); R["hit_recent"] = hit_recent.reset_index().to_dict("records")
P.to_csv(X + "pred_oos.csv", index=False, encoding="utf-8-sig")

# ========== C. 회복 ==========
import yfinance as yf
kos = yf.download(["^KS11", "EWY"], start="2017-06-01", end="2026-11-10", progress=False, auto_adjust=True)["Close"]
rec = []
for _, e in L.iterrows():
    for a, kr in (("^KS11", 1), ("EWY", 0)):
        s = kos[a].dropna(); D = e.date
        b = (s.index <= D).sum() - 1 if (kr and e.tm != "kr") else (s.index < D).sum() - 1
        if e.tm == "ah" and not kr: b = (s.index <= D).sum() - 1
        w = s.iloc[b + 1: b + 121]; base = s.iloc[b]
        if len(w) < 20: continue
        ti = int(np.argmin(w.iloc[:20].values)); dd = (w.iloc[ti] / base - 1) * 100
        if dd > -4: continue
        hit_ = np.where(w.iloc[ti:].values >= base)[0]
        days = ti + hit_[0] + 1 if len(hit_) else np.nan
        if len(w) < 120 and np.isnan(days): continue  # 관측기간 부족
        # 후속 완화: 이후 20영업일 내 같은 영역(무역·수출통제·법원) 완화 이벤트
        later = L[(L.date > D) & (L.date <= s.index[min(b + 20, len(s) - 1)]) & (L.esc == -1)]
        dom = {"tariff", "export_ctrl", "legal", "industrial"}
        follow = int(((later.type.isin(dom)) if e.type in dom else later.type.str.startswith("mon") | later.type.isin(["fiscal", "fin_stress"])).any())
        rec.append(dict(asset="KOSPI" if kr else "EWY", date=D, name=e["name"], type=e.type, clean=e.clean, dd=dd, trough=ti + 1,
                        days=days, fast=(days <= 30) if not np.isnan(days) else False, regime=e.regime60 > 0,
                        structural=e.type in ("industrial", "election", "credit") or (e.type == "tariff" and e.esc == 1 and "발효" not in e["name"] and ("232" in e["name"] or "301" in e["name"])),
                        follow=follow))
RC = pd.DataFrame(rec)
print("== 회복 표본", len(RC), "중앙값", RC.days.median(), "미회복", RC.days.isna().sum())
for k in ["structural", "regime", "follow"]:
    print(k, RC.groupby(k).fast.agg(["mean", "count"]).round(2).to_dict())
R["rec"] = dict(n=len(RC), events=int(RC.name.nunique()), median=float(RC.days.median()), unrec=int(RC.days.isna().sum()),
                by={k: RC.groupby(k).fast.agg(["mean", "count"]).round(2).reset_index().to_dict("records") for k in ["structural", "regime", "follow"]})
# 시간순 분류: (구조적, 레짐, 후속완화) 로지스틱 vs 기준선
from sklearn.linear_model import LogisticRegression
RC = RC.sort_values("date").reset_index(drop=True); acc, base = [], []
for i, r in RC.iterrows():
    tr = RC[RC.date < r.date - pd.Timedelta(days=30)]
    if len(tr) < 25 or tr.fast.nunique() < 2: continue
    Xf = ["structural", "regime", "follow"]; m = LogisticRegression(C=1.0).fit(tr[Xf].astype(float), tr.fast)
    acc.append(m.predict(pd.DataFrame([r[Xf].astype(float)]))[0] == r.fast); base.append((tr.fast.mean() >= .5) == r.fast)
print("== 회복 분류 시간순 정확도", np.mean(acc).round(2), "기준선", np.mean(base).round(2), "n", len(acc))
R["rec_acc"] = dict(model=round(float(np.mean(acc)), 2), base=round(float(np.mean(base)), 2), n=len(acc))
RC.to_csv(X + "recovery_v1.csv", index=False, encoding="utf-8-sig")
json.dump(R, open(X + "results.json", "w"), ensure_ascii=False, default=str, indent=1)
