"""Federal Register API에서 관세·무역·수출통제 관련 문서 수집"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
X = str(ROOT / "data") + "/"; F = X
import requests, json, time, sys
BASE = "https://www.federalregister.gov/api/v1/documents.json"
FIELDS = ["title", "type", "subtype", "document_number", "signing_date", "publication_date",
          "html_url", "abstract", "agencies", "executive_order_number", "presidential_document_number"]

def pull(params, tag):
    out, page = [], 1
    while True:
        p = dict(params); p.update({"per_page": 1000, "page": page, "order": "oldest"})
        q = [(k, v) for k, v in p.items() if not isinstance(v, list)]
        for k, v in p.items():
            if isinstance(v, list): q += [(k, x) for x in v]
        q += [("fields[]", f) for f in FIELDS]
        r = requests.get(BASE, params=q, timeout=60); r.raise_for_status(); j = r.json()
        res = j.get("results", []);
        for d in res: d["_query"] = tag
        out += res
        print(tag, "page", page, len(res), "of", j.get("count"), file=sys.stderr)
        if page >= j.get("total_pages", 1): break
        page += 1; time.sleep(0.5)
    return out

common = {"conditions[publication_date][gte]": "2018-01-01", "conditions[publication_date][lte]": "2026-11-08"}
docs = []
for term in ["tariff", "duties", "\"Section 232\"", "\"Section 301\"", "reciprocal", "\"export control\"", "semiconductor", "sanctions"]:
    docs += pull({**common, "conditions[type][]": ["PRESDOCU"], "conditions[term]": term}, f"pres:{term}")
# BIS 수출통제 규칙 (반도체·첨단컴퓨팅·중국)
for term in ["semiconductor", "\"advanced computing\"", "\"Entity List\" China"]:
    docs += pull({**common, "conditions[agencies][]": ["industry-and-security-bureau"], "conditions[type][]": ["RULE"],
                  "conditions[term]": term}, f"bis:{term}")
# USTR 301 조치
docs += pull({**common, "conditions[agencies][]": ["trade-representative-office-of-united-states"],
              "conditions[term]": "\"Section 301\" China action"}, "ustr:301")
# 상무부 232 품목 조사·조치 (철강·알루미늄·자동차·반도체 등)
docs += pull({**common, "conditions[agencies][]": ["industry-and-security-bureau"], "conditions[type][]": ["NOTICE"],
              "conditions[term]": "\"Section 232\" investigation"}, "bis:232")
uniq = {}
for d in docs:
    k = d["document_number"]
    if k in uniq: uniq[k]["_query"] += "|" + d["_query"]
    else: uniq[k] = d
json.dump(list(uniq.values()), open(X + "fr_docs.json", "w"), ensure_ascii=False)
print("unique docs", len(uniq))
