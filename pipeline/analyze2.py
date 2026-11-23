"""정당한 예측 과제 재정의:
 (1) 시차 전이: 미국 반응일(R) 종가 정보 → 한국 다음 세션(R+1) 방향/크기
 (2) 크기(위험 규모) 예측: 사전 정보로 |반응| 크기 순위 예측
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
X = str(ROOT / "data") + "/"; F = X
import pandas as pd, numpy as np, yfinance as yf, json, warnings
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

L = pd.read_pickle(X + "lib_v1.pkl")
px = yf.download(["^KS11", "^GSPC", "EWY"], start="2017-01-01", end="2026-11-10", progress=False, auto_adjust=True)["Close"]
K, S = px["^KS11"].dropna(), px["^GSPC"].dropna()
R = json.load(open(X + "results.json"))

# ---- (1) 시차 전이: 이벤트일 vs 전체일 ----
def nxt_kospi(d):
    """미국 거래일 d 종가 이후 첫 한국 세션 수익률(직전 한국 종가=날짜 d의 한국 종가 기준)"""
    b = (K.index <= d).sum() - 1
    if b + 1 >= len(K): return np.nan
    return (K.iloc[b + 1] / K.iloc[b] - 1) * 100
spx_r = S.pct_change() * 100
allk = pd.Series({d: nxt_kospi(d) for d in spx_r.index[200:]}); alls = spx_r.loc[allk.index]
ok = allk.notna() & alls.notna(); allk, alls = allk[ok], alls[ok]
big = alls.abs() > alls.abs().quantile(0.8)
ev = L[L.clean & L.s_spx.notna()].copy()
ev["k1"] = ev.R.map(nxt_kospi); ev = ev.dropna(subset=["k1"])
def stats(k, s):
    m = k.abs() > 0.1
    return dict(n=int(len(k)), hit=round(float((np.sign(k[m]) == np.sign(s[m])).mean()), 2), rho=round(float(spearmanr(s, k).correlation), 2),
                beta=round(float(np.polyfit(s, k, 1)[0]), 2))
r1 = {"전체 거래일": stats(allk, alls), "미 증시 큰 변동일(상위20%)": stats(allk[big], alls[big]), "정책 이벤트(교란 제외)": stats(ev.k1, ev.s_spx)}
# 이벤트 중 미 장후 발표(ah)·주말: 한국이 먼저 반응 → 이벤트 내용만으로 첫 세션 예측해야 함
ah = L[L.clean & (L.tm == "ah")].copy()
def first_k(r):
    b = (K.index <= r.date).sum() - 1
    return (K.iloc[b + 1] / K.iloc[b] - 1) * 100
ah["k_first"] = ah.apply(first_k, axis=1)
dir_rule = ah[ah.esc != 0]
r1["장후·주말 발표: 위험방향(esc) 규칙 → 첫 한국 세션"] = dict(n=int(len(dir_rule)), hit=round(float((np.sign(dir_rule.k_first) == -dir_rule.esc).mean()), 2))
print(json.dumps(r1, ensure_ascii=False, indent=1))
R["lag"] = r1

# ---- (2) 크기 예측: |반응| 순위 ----
C = L[L.clean].sort_values("date").reset_index(drop=True)
C["sigK"] = np.nan
for i, r in C.iterrows():
    b = (K.index <= r.date).sum() - 1
    C.loc[i, "sigK"] = K.pct_change().iloc[b - 60:b].std() * 100
C["absK"] = C.KOSPI_ret.abs()
C["sched"] = C.type.str.startswith("mon") & ~C.name.str.contains("긴급|무제한|스와프|잭슨홀") | (C.type == "election")
pred_hist, pred_vol, pred_combo, act = [], [], [], []
for i, r in C.iterrows():
    past = C[C.date < r.date - pd.Timedelta(days=7)]
    if len(past) < 30: continue
    same = past[(past.type == r.type) & (past.kr == r.kr)]
    if len(same) < 3: same = past[past.type == r.type]
    if len(same) < 3: same = past
    ratio = (same.absK / same.sigK).median()      # 유사 사례의 '평소 변동성 대비 배수'
    pred_hist.append(same.absK.median()); pred_vol.append(r.sigK); pred_combo.append(ratio * r.sigK); act.append(r.absK)
act = np.array(act)
r2 = {k: round(float(spearmanr(v, act).correlation), 2) for k, v in
      {"유사사례 평균 크기": pred_hist, "직전 변동성": pred_vol, "유사사례 배수×직전 변동성": pred_combo}.items()}
# 상위 위험(실제 |반응| 상위 1/3) 포착률
thr = np.quantile(act, 2 / 3); pc = np.array(pred_combo); top = pc >= np.quantile(pc, 2 / 3)
r2["상위1/3 경보 정밀도"] = round(float((act[top] >= thr).mean()), 2); r2["n"] = int(len(act))
r2["무작위 기대 정밀도"] = 0.33
# 예정 vs 비예정 / 한국 직접 대상 여부 크기 차이 (%)
r2["비예정 |KOSPI| 중앙값"] = round(float(C[~C.sched].absK.median()), 2); r2["예정 |KOSPI| 중앙값"] = round(float(C[C.sched].absK.median()), 2)
r2["한국 직접 |KOSPI|"] = round(float(C[C.kr == 1].absK.median()), 2); r2["간접 |KOSPI|"] = round(float(C[C.kr == 0].absK.median()), 2)
print(json.dumps(r2, ensure_ascii=False, indent=1))
R["size"] = r2
json.dump(R, open(X + "results.json", "w"), ensure_ascii=False, default=str, indent=1)
