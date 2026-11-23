"""사후 검증: 에이전트 판단(as_of 시점) vs 실제 시장 결과 (as_of 이후 데이터 사용)"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
X = str(ROOT / "data") + "/"; F = X
import sys, json, numpy as np, pandas as pd, yfinance as yf, warnings
warnings.filterwarnings("ignore")
SECT_T = {"기계장비": "102960.KS", "조선중공업": "139230.KS", "자동차": "091180.KS", "철강": "117680.KS", "반도체": "091160.KS", "바이오헬스": "244580.KS"}
as_of = sys.argv[1]; run = json.load(open(f"{ROOT}/runs/run_{as_of}.json"))
D = pd.Timestamp(as_of); p, q, r = run["policy"], run["quant"], run["risk"]
tick = ["^KS11"] + [SECT_T[s] for s in p["sectors"] if s in SECT_T]
px = yf.download(tick, start=(D - pd.Timedelta(days=260)).strftime("%Y-%m-%d"), end=(D + pd.Timedelta(days=20)).strftime("%Y-%m-%d"), progress=False, auto_adjust=True)["Close"]
k = px["^KS11"].dropna(); b = (k.index <= D).sum() - 1     # 미 장후 서명 → 한국 첫 반응은 b+1
o = dict(as_of=as_of, title=run["doc"]["title"], action=p["action"], rates=p["rates"])
o["first_session"], o["first_ret"] = str(k.index[b + 1].date()), round((k.iloc[b + 1] / k.iloc[b] - 1) * 100, 2)
o["next_session"], o["next_ret"] = str(k.index[b + 2].date()), round((k.iloc[b + 2] / k.iloc[b + 1] - 1) * 100, 2)
o["next_call"] = q["next_session_call"]; o["next_hit"] = (o["next_ret"] < 0) == (q["next_session_call"] == "하락 우세")
o["kospi_5d"] = round((k.iloc[b + 5] / k.iloc[b] - 1) * 100, 2); o["range"] = r["kospi_5d_range"]
o["in_range"] = abs(o["kospi_5d"]) <= float(r["kospi_5d_range"].strip("±%"))
sec = {}
for s in p["sectors"]:
    if s not in SECT_T: continue
    m = px[SECT_T[s]].dropna(); j = pd.concat([np.log(m).diff(), np.log(k).diff()], axis=1, keys=["y", "x"]).dropna()
    bj = (j.index <= k.index[b]).sum() - 1; est = j.iloc[max(0, bj - 150): bj - 5]; beta, alpha = np.polyfit(est.x, est.y, 1)
    w = j.iloc[bj + 1: bj + 6]; car = (w.y - (alpha + beta * w.x)).sum() * 100; sd = (est.y - (alpha + beta * est.x)).std() * 100
    sec[s] = dict(abn_5d=round(car, 2), z=round(car / (sd * np.sqrt(5)), 2))
o["sector_abn"] = sec; o["direct_call"] = r.get("direct_sector_impact")
print(json.dumps(o, ensure_ascii=False, default=str))
json.dump(o, open(f"{ROOT}/runs/posthoc_{as_of}.json", "w"), ensure_ascii=False, indent=1, default=str)
