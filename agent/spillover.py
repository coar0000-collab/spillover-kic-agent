"""SPILLOVER 프로토타입 v0 — LangGraph 상태 그래프
감지(Sentinel) → 구조화(Policy Analyst) → 전이경로(Transmission) → 정량화(Quant) → 위험(Risk) → 검증(Verifier) → 리포트
- 모든 수치는 Tool(API·계산)이 산출하고, 각 주장에 출처를 붙인다.
- as_of 시점 이후 데이터는 사용하지 않는다(시점 절단).
- v0의 원문 구조화는 규칙 기반이며, 본선에서 LLM(JSON 스키마) 구조화 + 규칙 교차검증으로 확장한다.
"""
from __future__ import annotations
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
X = str(ROOT / "data") + "/"; F = X
import re, html, json, time, sys
from datetime import datetime, timedelta
from typing import TypedDict, Any
import numpy as np, pandas as pd, requests, yfinance as yf
from langgraph.graph import StateGraph, END


FR_API = "https://www.federalregister.gov/api/v1/documents.json"
UA = {"User-Agent": "spillover-prototype/0.1 (research)"}

class S(TypedDict, total=False):
    as_of: str; window_days: int
    candidates: list; doc: dict; relevance: float
    policy: dict; transmission: dict; quant: dict; risk: dict
    verify: dict; retries: int; log: list; claims: list; report: str

def _log(st, node, t0, msg):
    st.setdefault("log", []).append(dict(node=node, sec=round(time.time() - t0, 2), msg=msg))

def _claim(st, text, src):
    st.setdefault("claims", []).append(dict(text=text, src=src))

# ------------------------------------------------------------------ ① Sentinel
KW = {"tariff": 2, "duties": 2, "duty": 1, "import": 1, "section 232": 2, "section 301": 2, "reciprocal": 2,
      "export control": 2, "semiconductor": 1, "steel": 1, "automobile": 1, "machinery": 1, "battery": 1,
      "korea": 3, "adjusting imports": 2}
def sentinel(st: S) -> S:
    t0 = time.time(); asof = pd.Timestamp(st["as_of"])
    lo = (asof - timedelta(days=st.get("window_days", 3))).strftime("%Y-%m-%d")
    q = [("per_page", 100), ("order", "newest"), ("conditions[type][]", "PRESDOCU"),
         ("conditions[signing_date][gte]", lo), ("conditions[signing_date][lte]", asof.strftime("%Y-%m-%d"))]
    q += [("fields[]", f) for f in ["document_number", "title", "signing_date", "publication_date", "html_url", "subtype"]]
    docs = requests.get(FR_API, params=q, timeout=60, headers=UA).json().get("results", [])
    cands = []
    for d in docs:
        t = d["title"].lower(); sc = sum(w for k, w in KW.items() if k in t)
        cands.append(dict(d, score=sc, relevance=round(min(sc / 6, 1.0), 2)))
    cands.sort(key=lambda d: -d["score"])
    st["candidates"] = cands; st["doc"] = cands[0] if cands else {}; st["relevance"] = st["doc"].get("relevance", 0)
    _log(st, "① Sentinel", t0, f"연방관보 대통령 문서 {len(docs)}건 조회 → 최상위 '{st['doc'].get('title','')[:60]}' 관련성 {st['relevance']}")
    if st["doc"]: _claim(st, f"감지 문서: {st['doc']['title']} (서명 {st['doc']['signing_date']})", st["doc"]["html_url"])
    return st

# ------------------------------------------------------------------ ② Policy Analyst
SECT_KW = {"기계장비": ["industrial machinery", "robot", "machine tool"], "자동차": ["automobile", "vehicle", "auto part"],
           "철강": ["steel", "aluminum", "copper"], "반도체": ["semiconductor"], "바이오헬스": ["pharmaceutical"],
           "조선중공업": ["wind turbine", "ship", "vessel"], "2차전지": ["battery", "lithium"], "IT": ["smartphone", "computer"]}
def policy_analyst(st: S) -> S:
    t0 = time.time(); d = st["doc"]
    url = f"https://www.govinfo.gov/content/pkg/FR-{d['publication_date']}/html/{d['document_number']}.htm"
    raw = requests.get(url, timeout=60, headers=UA).text
    txt = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", raw)))
    body = txt[txt.find("By the President"):] if "By the President" in txt else txt
    low = body.lower()
    rates = sorted({float(x) for x in re.findall(r"(\d{1,3}(?:\.\d+)?) percent", body)})
    advalorem = "ad valorem" in low
    nego = bool(re.search(r"(pursue|continue)[^.]{0,80}negotiations", low))
    future = bool(re.search(r"may consider alternative remedies|in the future", low))
    if advalorem and rates: action = "관세 부과"
    elif nego: action = "협상 지시(관세 미부과)"
    else: action = "기타 조치"
    deadline = re.search(r"within (\d+) days", low)
    korea = bool(re.search(r"\bkorea\b", low))
    m_obj = re.search(r"imports of (.+?) into the united states", d["title"].lower())
    obj = m_obj.group(1) if m_obj else d["title"].lower()
    sects = [s for s, ks in SECT_KW.items() if any(k in obj for k in ks)]          # 직접 대상 품목
    users = [s for s, ks in SECT_KW.items() if s not in sects and any(k in low for k in ks)]  # 본문 언급(수요 산업)
    def quote(pat):
        m = re.search(pat, body, re.I); return body[max(0, m.start() - 60): m.end() + 120].strip() if m else ""
    st["policy"] = dict(doc=d["document_number"], source=url, action=action, rates=rates, ad_valorem=advalorem,
                        negotiation=nego, future_tariff_possible=future, deadline_days=int(deadline.group(1)) if deadline else None,
                        korea_named=korea, sectors=sects, user_industries=users, target_phrase=obj, chars=len(body),
                        evidence=dict(action=quote(r"pursue or continue pursuing negotiations|ad valorem"),
                                      future=quote(r"may consider alternative remedies")))
    _log(st, "② Policy Analyst", t0, f"원문 {len(body):,}자 구조화 → 조치='{action}', 관세율={rates or '없음'}, 한국 명시={korea}, 대상 섹터={sects}, 본문 언급 수요산업={users}")
    _claim(st, f"조치 유형: {action} / 명시 관세율 {rates or '없음'} / 협상 기한 {st['policy']['deadline_days']}일", url)
    return st

# ------------------------------------------------------------------ ③ Transmission Mapper
HS = {"기계장비": ["8428", "8429", "8456", "8457", "8458", "8459", "8460", "8461", "8462", "8466", "8479", "8483", "8486"],
      "자동차": ["8703", "8708"], "철강": ["72", "73"], "반도체": ["8542"], "바이오헬스": ["3004"], "조선중공업": ["8502", "8503"]}
def transmission(st: S) -> S:
    t0 = time.time(); p = st["policy"]; asof = pd.Timestamp(st["as_of"])
    yr = asof.year - 1; out = {}
    for s in p["sectors"]:
        codes = HS.get(s)
        if not codes: continue
        rows, tries = [], 0
        while not rows and tries < 5:   # 재시도·예외 처리
            tries += 1
            try:
                r = requests.get("https://comtradeapi.un.org/public/v1/preview/C/A/HS", timeout=60, params=dict(
                    reporterCode=410, partnerCode="842,0", period=yr, cmdCode=",".join(codes), flowCode="X")).json()
                rows = [x for x in r.get("data", []) or [] if x.get("partner2Code") == 0 and x.get("motCode") == 0 and x.get("customsCode") == "C00"]
            except Exception: rows = []
            if not rows: time.sleep(4 * tries)
        if not rows:
            out[s] = dict(hs=codes, year=yr, us_usd_bn=None, world_usd_bn=None, us_share=None, note="Comtrade 응답 없음 → 확인 불가"); continue
        us = sum(x["primaryValue"] for x in rows if x["partnerCode"] == 842); wd = sum(x["primaryValue"] for x in rows if x["partnerCode"] == 0)
        out[s] = dict(hs=codes, year=yr, us_usd_bn=round(us / 1e9, 2), world_usd_bn=round(wd / 1e9, 2), us_share=round(us / wd, 3) if wd else None)
        _claim(st, f"{s} 대상 HS {len(codes)}개 품목 한국 대미 수출 {us/1e8:,.0f}억 달러 ({yr}년, 해당 품목 수출의 {us/wd:.0%})",
               f"UN Comtrade reporter=410 partner=842 period={yr} cmd={','.join(codes)}")
        time.sleep(1.5)
    channels = ["무역(대미 수출)"] + (["외국인 자금·심리"] if p["action"] == "관세 부과" else [])
    # 경로 가중치: 사례 164건 경로 분해(R² 증분) 결과를 이벤트 유형에 맞춰 적용
    dec = json.load(open(X + "event_results.json"))["decomp"]
    grp = "중국 대상" if "china" in st["doc"]["title"].lower() else ("한국 직접 대상" if p.get("korea_named") else "무역·통상(중국 외)")
    st.setdefault("_tmp", {})["channel_weights"] = dict(group=grp, **{k: dec[grp][k] for k in ["미국", "반도체추가", "중국추가", "아시아추가", "설명안됨"]})
    st["transmission"] = dict(exposure=out, channels=channels, channel_weights=st.get("_tmp", {}).get("channel_weights"),
                              path=" → ".join(["232조 포고문", "/".join(p["sectors"]) + " 대미 수출", "기업 실적 기대", "섹터 주가·외국인 수급"]))
    _log(st, "③ Transmission", t0, f"Comtrade 실시간 조회 → " + ", ".join(f"{k} 대미 {v['us_usd_bn']}십억$({v['us_share']})" for k, v in out.items()))
    return st

# ------------------------------------------------------------------ ④ Quant Analyst
LIB = pd.read_pickle(X + "lib_v1.pkl")
SECT_T = {"기계장비": "102960.KS", "조선중공업": "139230.KS", "자동차": "091180.KS", "철강": "117680.KS", "반도체": "091160.KS", "바이오헬스": "244580.KS"}
def quant(st: S) -> S:
    t0 = time.time(); p = st["policy"]; asof = pd.Timestamp(st["as_of"])
    past = LIB[(LIB.date < asof - timedelta(days=1)) & LIB.clean]
    sim = past[(past.type == "tariff") & (past.kr == 1)]
    if p["action"] == "관세 부과":
        base, basis = sim[(sim.esc == 1) & ~sim.name.str.contains("예고|조사|위협|검토")], "한국 직접 대상 관세 부과"
    else:   # 즉시 관세 없이 향후 조치 가능성만 있는 발표 = 예고·조사 개시 사례와 비교
        base, basis = sim[sim.name.str.contains("예고|조사|위협|검토")], "한국 관련 관세 예고·조사 개시(즉시 부과 없음)"
    if len(base) < 3: base, basis = sim, "한국 직접 대상 관세 전체(조치 성격 일치 사례 부족)"
    absK = base.KOSPI_ret.abs()
    tick = ["^KS11", "^GSPC"] + [SECT_T[s] for s in p["sectors"] if s in SECT_T]
    px = yf.download(tick, start=(asof - timedelta(days=200)).strftime("%Y-%m-%d"),
                     end=(asof + timedelta(days=10)).strftime("%Y-%m-%d"), progress=False, auto_adjust=True)["Close"]
    k = px["^KS11"].dropna(); spx = px["^GSPC"].dropna()
    kk = k[k.index <= asof]; sig = kk.pct_change().iloc[-60:].std() * 100
    # 변동성 국면 보정: 유사 사례의 '직전 60일 변동성 대비 5일 반응 배수' × 현재 변동성 (사전 검증 방식)
    kh = yf.download("^KS11", start="2017-06-01", end=(asof + timedelta(days=1)).strftime("%Y-%m-%d"), progress=False, auto_adjust=True)["Close"].squeeze()
    def sig_at(d):
        b = (kh.index <= d).sum() - 1; return kh.pct_change().iloc[b - 60:b].std() * 100
    mult_cases = (base.KOSPI_ret.abs() / base.date.map(sig_at)).dropna()
    mult = float(mult_cases.median())
    allc = past.dropna(subset=["KOSPI_ret"]); allm = (allc.KOSPI_ret.abs() / allc.date.map(sig_at)).dropna()
    terc = [float(allm.quantile(1 / 3)), float(allm.quantile(2 / 3))]
    # 미국 반응일: 발표(장후) 다음 미 거래일
    R = spx.index[spx.index > asof][0] if (spx.index > asof).any() else None
    s0 = float((spx.loc[R] / spx[spx.index < R].iloc[-1] - 1) * 100) if R is not None else None
    st["quant"] = dict(similar_basis=basis, similar_n=int(len(base)), similar_names=base.sort_values("date").name.tail(3).tolist(),
                       similar_absK_median=round(float(absK.median()), 2), similar_absK_p75=round(float(absK.quantile(.75)), 2),
                       kospi_sigma60=round(float(sig), 2), multiplier=round(mult, 2), mult_terciles=[round(t, 2) for t in terc],
                       expected_5d=round(mult * float(sig), 1), us_reaction_day=str(R.date()) if R is not None else None,
                       spx_day0=round(s0, 2) if s0 is not None else None,
                       next_session_call=("상승 우세" if s0 > 0 else "하락 우세") if s0 is not None else None,
                       next_session_hit_hist=0.75, px=px)
    _log(st, "④ Quant", t0, f"유사 사례 {len(base)}건({basis}) 반응 배수 {mult:.2f}×현재 일변동성 {sig:.2f}% → 5일 예상 변동 ±{mult*sig:.1f}% / 미 반응일 {R.date() if R is not None else '-'} S&P {s0:+.2f}%")
    _claim(st, f"유사 사례 {len(base)}건({basis})의 KOSPI 5일 변동폭 중앙값 {absK.median():.1f}%: " + ", ".join(base.sort_values("date").name.tail(3)), "SPILLOVER 사례 라이브러리 v1")
    if s0 is not None:
        _claim(st, f"미 반응일({R.date()}) S&P500 {s0:+.2f}% → 한국 다음 세션 {st['quant']['next_session_call']} (과거 정책 이벤트 98건 적중 75%)", "Yahoo Finance ^GSPC; 사전 검증")
    return st

# ------------------------------------------------------------------ ⑤ Risk Scorer
def risk(st: S) -> S:
    t0 = time.time(); p, q, tr = st["policy"], st["quant"], st["transmission"]
    size = q["expected_5d"]; m, (t1, t2) = q["multiplier"], q["mult_terciles"]
    level = "높음" if m >= t2 else "보통" if m >= t1 else "낮음"     # 이벤트 강도(배수)의 과거 분포 3분위 기준
    ranks = sorted(((s, v["us_share"]) for s, v in tr["exposure"].items() if v["us_share"] is not None), key=lambda x: -x[1])
    direct = "작음(즉시 관세 없음 → 섹터 고유 충격 제한적)" if p["action"] != "관세 부과" else "큼(대상 섹터 직접 비용 발생)"
    st["risk"] = dict(kospi_5d_range=f"±{size:.1f}%", level=level, event_intensity=f"평소 변동성의 {m:.2f}배(과거 분포 {'상위' if m>=t2 else '중위' if m>=t1 else '하위'} 1/3)",
                      direct_sector_impact=direct, sector_rank=ranks,
                      watch=[f"{p['deadline_days']}일 협상 시한({(pd.Timestamp(st['as_of']) + timedelta(days=p['deadline_days'] or 0)).date()})" if p.get("deadline_days") else ""],
                      unpredictable="향후 5일 방향은 예측하지 않음(사전 검증 기준선 수준)")
    _log(st, "⑤ Risk", t0, f"이벤트 강도 '{level}', KOSPI 5일 예상 변동 {st['risk']['kospi_5d_range']}(현재 고변동 국면 반영), 섹터 직접 영향 {direct[:2]}, 노출 순위 {ranks}")
    return st

# ------------------------------------------------------------------ ⑥ Verifier
def verifier(st: S) -> S:
    t0 = time.time(); asof = pd.Timestamp(st["as_of"]); issues, flags = [], []
    for c in st.get("claims", []):
        if not c.get("src"): issues.append("출처 없는 주장: " + c["text"][:40])
    # 재계산: 조치 유형을 원문 키워드로 재확인
    p = st["policy"]
    if p["action"] == "관세 부과" and not p["rates"]: issues.append("관세율 미확인")
    if p["action"].startswith("협상") and not p["evidence"]["action"]: issues.append("협상 근거 문장 미확보")
    # 라이브러리와 대조(제목 기반 사전 분류와 원문 판정 불일치 탐지)
    lib_row = LIB[LIB.date == pd.Timestamp(st["doc"]["signing_date"])]
    title_guess = "관세 부과" if "adjusting imports" in st["doc"]["title"].lower() else "기타"
    if title_guess == "관세 부과" and p["action"] != "관세 부과":
        flags.append("제목('Adjusting Imports')만으로는 관세로 오분류됨 → 원문 기준 '협상 지시'로 정정")
    # 교란: ±3영업일 내 다른 이벤트
    sd = pd.Timestamp(st["doc"]["signing_date"])
    self_row = (LIB.date == sd) & LIB.type.isin(["tariff", "export_ctrl", "industrial", "legal"])   # 라이브러리 내 동일 이벤트
    near = LIB[(abs((LIB.date - asof).dt.days) <= 4) & ~self_row]
    for r in near.itertuples(): flags.append(f"동시 이벤트: {r.date.date()} {r.name}")
    st["verify"] = dict(passed=not issues, issues=issues, flags=flags, claims_checked=len(st.get("claims", [])))
    st["retries"] = st.get("retries", 0) + (0 if not issues else 1)
    _log(st, "⑥ Verifier", t0, f"주장 {len(st.get('claims', []))}건 출처 확인, 문제 {len(issues)}건, 경고 {len(flags)}건")
    return st

def route_after_sentinel(st: S): return "policy" if st.get("relevance", 0) >= 0.3 else END
def route_after_verify(st: S): return END if st["verify"]["passed"] or st.get("retries", 0) >= 2 else "policy"

def build():
    g = StateGraph(S)
    for n, f in [("sentinel", sentinel), ("policy", policy_analyst), ("transmission", transmission),
                 ("quant", quant), ("risk", risk), ("verifier", verifier)]: g.add_node(n, f)
    g.set_entry_point("sentinel")
    g.add_conditional_edges("sentinel", route_after_sentinel, {"policy": "policy", END: END})
    g.add_edge("policy", "transmission"); g.add_edge("transmission", "quant"); g.add_edge("quant", "risk"); g.add_edge("risk", "verifier")
    g.add_conditional_edges("verifier", route_after_verify, {"policy": "policy", END: END})
    return g.compile()

if __name__ == "__main__":
    as_of = sys.argv[1] if len(sys.argv) > 1 else "2026-10-28"
    T0 = time.time()
    st = build().invoke({"as_of": as_of, "window_days": 2})
    st["total_sec"] = round(time.time() - T0, 1)
    px = st["quant"].pop("px")
    json.dump({k: v for k, v in st.items() if k not in ("candidates",)} | {"top_candidates": [dict(title=c["title"], relevance=c["relevance"]) for c in st["candidates"][:5]]},
              open(str(ROOT / "runs") + "/run_" + as_of + ".json", "w"), ensure_ascii=False, indent=1, default=str)
    px.to_pickle(str(ROOT / "runs") + "/px_" + as_of + ".pkl")
    for l in st["log"]: print(f"[{l['sec']:>5}s] {l['node']}: {l['msg']}")
    print("TOTAL", st["total_sec"], "s | verify", st["verify"])
