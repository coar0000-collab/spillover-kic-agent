"""UN Comtrade: 한국 섹터별 수출의 미국·중국 비중 (연도별)"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
X = str(ROOT / "data") + "/"; F = X
import requests, json, time, pandas as pd
SECT = {
 "반도체": ["8542", "8541", "8486"], "자동차": ["8703", "8708", "8704"], "철강": ["72", "73"],
 "2차전지": ["8507", "2841"], "조선중공업": ["8901", "8905", "8502", "8504", "8411", "8406"],
 "기계장비": ["8429", "8431", "8481", "8479", "8483", "8414", "8413"], "에너지화학": ["2710", "39", "29"],
 "바이오헬스": ["3002", "3004", "9018"], "IT": ["8517", "8528", "8524", "8471", "8473"],
 "필수소비재": ["3304", "2106", "1902", "2202"], "경기소비재": ["4011", "8418", "8450", "8516"]}
codes = sorted({c for v in SECT.values() for c in v})
rows = []
for y in range(2018, 2026):
    for attempt in range(4):
      r = requests.get("https://comtradeapi.un.org/public/v1/preview/C/A/HS", timeout=60, params={
          "reporterCode": 410, "partnerCode": "842,156,0", "period": y, "cmdCode": ",".join(codes), "flowCode": "X"})
      j = r.json(); d = j.get("data") or []
      if d: break
      time.sleep(8)
    print(y, len(d), j.get("error"))
    for x in d:
        if x.get("partner2Code") == 0 and x.get("motCode") == 0 and x.get("customsCode") == "C00":
            rows.append(dict(year=y, partner=x["partnerCode"], hs=x["cmdCode"], value=x["primaryValue"]))
    time.sleep(4)
df = pd.DataFrame(rows); df.to_csv(X + "comtrade_raw.csv", index=False)
out = []
for s, cs in SECT.items():
    g = df[df.hs.isin(cs)].groupby(["year", "partner"]).value.sum().unstack()
    for y, r in g.iterrows():
        out.append(dict(sector=s, year=y, world=r.get(0), us=r.get(842), cn=r.get(156),
                        us_share=r.get(842) / r.get(0), cn_share=r.get(156) / r.get(0)))
ex = pd.DataFrame(out); ex.to_csv(X + "exposure.csv", index=False, encoding="utf-8-sig")
print(ex.pivot(index="sector", columns="year", values="us_share").round(3))
print(ex.pivot(index="sector", columns="year", values="cn_share").round(3))
