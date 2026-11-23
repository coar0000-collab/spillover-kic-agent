"""시장·거시 데이터 수집: FRED(CSV) + Yahoo Finance(가격) → data/"""
from pathlib import Path
import requests, yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)

FRED = ["DEXKOUS", "DGS2", "DGS10", "DTWEXBGS", "VIXCLS", "DFEDTARU",       # 미국 금리·달러·변동성·기준금리, 원달러
        "DEXCHUS", "DCOILWTICO", "USEPUINDXD",                             # 위안화, 유가, 미 정책불확실성
        "XTEXVA01KRM667S", "XTEXVA01CNM667S"]                              # 한국·중국 수출(OECD)
for s in FRED:
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": s, "cosd": "2016-01-01"}, timeout=60)
    r.raise_for_status(); (DATA / f"{s}.csv").write_text(r.text); print("FRED", s)

YAHOO = {"^SOX": "SOX", "^N225": "N225", "^TWII": "TWII", "^HSI": "HSI", "000001.SS": "SHCOMP", "HG=F": "COPPER",
         "^KQ11": "KOSDAQ", "^VIX": "VIX", "^KS11": "KOSPI", "^GSPC": "SPX", "EWY": "EWY", "^IXIC": "NASDAQ", "DX-Y.NYB": "DXY"}
d = yf.download(list(YAHOO), start="2016-06-01", progress=False, auto_adjust=True)["Close"].rename(columns=YAHOO)
d.to_pickle(DATA / "yahoo.pkl"); print("Yahoo", d.shape)
