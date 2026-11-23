"""SPILLOVER 사례 라이브러리 v1 구축: 이벤트 스터디 + 서프라이즈 + 기대치 + 노출도"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
X = str(ROOT / "data") + "/"; F = X
import pandas as pd, numpy as np, yfinance as yf, json, re, warnings
warnings.filterwarnings("ignore")
from events import E, FOMC

# ---------------- 시계열 ----------------
def fred(s):
    d = pd.read_csv(X + f"{s}.csv", parse_dates=[0]).set_index("observation_date")[s]
    return pd.to_numeric(d, errors="coerce").dropna()
SECT_T = {"반도체": "091160.KS", "자동차": "091180.KS", "철강": "117680.KS", "2차전지": "305720.KS",
          "조선중공업": "139230.KS", "기계장비": "102960.KS", "에너지화학": "117460.KS", "바이오헬스": "244580.KS",
          "IT": "266370.KS", "필수소비재": "266410.KS", "경기소비재": "266390.KS",
          "은행": "091170.KS", "증권": "102970.KS", "건설": "117700.KS", "운송": "140710.KS"}
tick = ["^KS11", "EWY", "^GSPC", "KRW=X"] + list(SECT_T.values())
px = yf.download(tick, start="2017-01-01", end="2026-11-10", progress=False, auto_adjust=True)["Close"]
fx = fred("DEXKOUS"); krw = px["KRW=X"].dropna()
tail = krw[krw.index > fx.index[-1]]
fx = pd.concat([fx, tail / krw[krw.index <= fx.index[-1]].iloc[-1] * fx.iloc[-1]])
P = {"FX": fx, "KOSPI": px["^KS11"].dropna(), "EWY": px["EWY"].dropna(), "SPX": px["^GSPC"].dropna()}
for k, t in SECT_T.items(): P["S_" + k] = px[t].dropna()
Y2, USD, VIX, FFR = fred("DGS2"), fred("DTWEXBGS"), fred("VIXCLS"), fred("DFEDTARU")
KR = {"KOSPI"} | {"S_" + k for k in SECT_T}
expo = pd.read_csv(X + "exposure.csv")

# ---------------- 이벤트 테이블 ----------------
rows = [dict(date=d, name=n, type=t, esc=e, kr=k, cn=c, sect=s, tm=tm, src=src) for d, n, t, e, k, c, s, tm, src in E]
for d in FOMC:
    D = pd.Timestamp(d)
    before = FFR[FFR.index < D].iloc[-1]; after = FFR[FFR.index > D].iloc[0] if (FFR.index > D).any() else before
    ch = round((after - before) * 100)
    t = "mon_tight" if ch > 0 else "mon_ease" if ch < 0 else "mon_hold"
    nm = f"FOMC {'인상' if ch > 0 else '인하' if ch < 0 else '동결'}" + (f" {abs(ch)}bp" if ch else "")
    rows.append(dict(date=d, name=nm, type=t, esc=int(np.sign(ch)), kr=0, cn=0, sect="", tm="", src="연준 FOMC 성명"))
ev = pd.DataFrame(rows); ev["date"] = pd.to_datetime(ev.date); ev = ev.sort_values("date").reset_index(drop=True)
# 출처 정리: 실제 FR 문서번호만 남김
fr = {d["document_number"]: d for d in json.load(open(X + "fr_docs.json"))}
def clean_src(s):
    nums = re.findall(r"20\d\d-\d{5}", s)
    bad = [n for n in nums if n not in fr]
    for n in bad: s = s.replace(n, "")
    s = re.sub(r"FR\s*(외)?\s*(;|$)", "", s); s = re.sub(r";\s*;", ";", s).strip(" ;")
    return s or "Federal Register"
ev["src"] = ev.src.map(clean_src)

def last_le(s, d): i = (s.index <= d).sum() - 1; return i
def last_lt(s, d): i = (s.index < d).sum() - 1; return i
def first_ge(s, d): return (s.index < d).sum()
def first_gt(s, d): return (s.index <= d).sum()

def zret(s, b, H, est=(130, 10)):
    if b - est[0] < 1 or b + H >= len(s): return (np.nan, np.nan)
    r = np.log(s).diff()
    ret = np.log(s.iloc[b + H] / s.iloc[b]) * 100
    sd = r.iloc[b - est[0]: b - est[1]].std() * 100
    return ret, ret / (sd * np.sqrt(H))

def abn(y, m, b_y, H, est=250):
    """y, m 공통 영업일로 정렬 후 시장모형 비정상수익률 CAR(b+1..b+H)"""
    j = pd.concat([np.log(y).diff(), np.log(m).diff()], axis=1, keys=["y", "m"]).dropna()
    bd = y.index[b_y]; b = (j.index <= bd).sum() - 1
    if b - est < 1 or b + H >= len(j): return (np.nan, np.nan)
    e_ = j.iloc[b - est: b - 10]; beta, alpha = np.polyfit(e_.m, e_.y, 1)
    res = e_.y - (alpha + beta * e_.m); w = j.iloc[b + 1: b + 1 + H]
    car = (w.y - (alpha + beta * w.m)).sum() * 100
    return car, car / (res.std() * 100 * np.sqrt(H))

H = 5; out = []
for r in ev.itertuples():
    D = r.date; o = dict(r._asdict()); o.pop("Index")
    # 미국 반응일 R
    spx = P["SPX"]; iR = first_gt(spx, D) if r.tm == "ah" else first_ge(spx, D)
    if iR >= len(spx): out.append(o); continue
    R = spx.index[iR]; o["R"] = R
    # ---- 충격 측정(서술용): 이벤트 직전 종가 기준 ----
    for k in ["FX", "EWY", "SPX"]:
        s = P[k]; b = last_lt(s, R)
        o[f"{k}_ret"], o[f"{k}_z"] = zret(s, b, H)
        o[f"{k}_pre"] = np.log(s.iloc[b] / s.iloc[b - 5]) * 100 if b >= 5 else np.nan
    for k in ["KOSPI"] + ["S_" + x for x in SECT_T]:
        s = P[k]; b = last_lt(s, D) if r.tm == "kr" else last_le(s, D)
        o[f"{k}_ret"], o[f"{k}_z"] = zret(s, b, H)
        if k == "KOSPI": o["KOSPI_pre"] = np.log(s.iloc[b] / s.iloc[b - 5]) * 100; o["KOSPI_b"] = s.index[b]
        if k.startswith("S_"):  # 섹터 비정상수익률(KOSPI 대비)
            o[f"{k}_abn"], o[f"{k}_abnz"] = abn(s, P["KOSPI"], b, H)
    bE = last_lt(P["EWY"], R); o["EWYabn"], o["EWYabn_z"] = abn(P["EWY"], P["SPX"], bE, H)
    # ---- 미국 당일 반응(서프라이즈) ----
    def d0(s, diff=False):
        pre = s[s.index < R]; at = s[s.index >= R]
        if pre.empty or at.empty or at.index[0] != R: return np.nan
        return (at.iloc[0] - pre.iloc[-1]) * 100 if diff else (at.iloc[0] / pre.iloc[-1] - 1) * 100
    o["s_spx"], o["s_y2"], o["s_usd"] = d0(spx), d0(Y2, True), d0(USD)
    o["s_ewy0"] = d0(P["EWY"])
    v = VIX[VIX.index < R]; o["vix"] = v.iloc[-1] if len(v) else np.nan
    # ---- 예측 타깃: 미국 반응일 마감 이후 정보로 다음 구간 예측 ----
    k = P["KOSPI"]; bK = last_le(k, R)          # 한국 R일 장 마감(미국 R일 장 시작 전)
    o["T_KOSPI"], o["T_KOSPI_z"] = zret(k, bK, H)
    f = P["FX"]; bF = first_gt(f, R)            # R+1일 뉴욕 정오 환율부터
    o["T_FX"], o["T_FX_z"] = zret(f, bF, H)
    bE2 = last_le(P["EWY"], R); o["T_EWYabn"], o["T_EWYabn_z"] = abn(P["EWY"], P["SPX"], bE2, H)
    # ---- 레짐 ----
    bk = last_le(k, D); o["regime60"] = np.log(k.iloc[bk] / k.iloc[bk - 60]) * 100
    # ---- 노출도(전년도 무역) ----
    yr = min(max(D.year - 1, 2018), 2025)
    ex = expo[expo.year == yr].set_index("sector")
    o["exp_year"] = yr
    for s_ in SECT_T:
        o[f"us_{s_}"] = ex.us_share.get(s_, 0.0); o[f"cn_{s_}"] = ex.cn_share.get(s_, 0.0)
    out.append(o)
lib = pd.DataFrame(out)

# ---------------- 교란(동시 이벤트) 플래그 ----------------
kdays = P["KOSPI"].index
pos = lib.KOSPI_b.map(lambda d: kdays.get_loc(d) if pd.notna(d) else np.nan)
flags = []
for i, r in lib.iterrows():
    f = []
    near = lib[(abs(pos - pos[i]) <= 3) & (lib.index != i)]
    if len(near): f.append("동시 이벤트: " + ", ".join(near.name.str.slice(0, 18).tolist()[:3]))
    if pd.Timestamp("2020-02-20") <= r.date <= pd.Timestamp("2020-04-30"): f.append("코로나 급락기")
    if pd.Timestamp("2024-11-28") <= r.date <= pd.Timestamp("2024-12-10"): f.append("국내 비상계엄(12/3)")
    flags.append("; ".join(f))
lib["교란"] = flags
lib["clean"] = lib.교란 == ""
lib.to_pickle(X + "lib_v1.pkl")
cols = ["date", "name", "type", "esc", "kr", "tm", "KOSPI_ret", "KOSPI_z", "FX_ret", "FX_z", "EWYabn", "EWYabn_z", "s_spx", "s_y2", "s_usd", "vix", "교란"]
lib[cols].round(2).to_csv(X + "case_library_v1.csv", index=False, encoding="utf-8-sig")
print(len(lib), "events;", lib.clean.sum(), "clean;", lib.KOSPI_z.notna().sum(), "with KOSPI window")
print(lib.groupby("type").agg(n=("name", "count"), clean=("clean", "sum")))
