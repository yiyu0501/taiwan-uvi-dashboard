#!/usr/bin/env python3
"""Aggregate the 2023-11 household-registration village table to 368 townships.

Source: Ministry of the Interior, Department of Household Registration, ODRP014
(village population), period 11211, four API pages:
  https://www.ris.gov.tw/rs-opendata/api/v1/datastore/ODRP014/11211?page=1 ... page=4
Save them as input/ris_ODRP014_11211_page1.json ... page4.json.

Usage: python code/build_population_202311.py input input/population_2023-11_town.csv
"""
import sys, json, pathlib
import pandas as pd

src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
rows = []
for i in range(1, 5):
    j = json.loads((src / f"ris_ODRP014_11211_page{i}.json").read_text(encoding="utf-8"))
    assert int(j["page"]) == i and int(j["totalPage"]) == 4 and int(j["totalDataSize"]) == 7748
    rows += j["responseData"]
p = pd.DataFrame(rows)
assert len(p) == 7748 and p.district_code.nunique() == 7748
assert set(p.statistic_yyymm) == {"11211"}
p["town_code"] = p.district_code.str[:8]
p["population_total"] = pd.to_numeric(p.people_total)
pop = p.groupby("town_code", as_index=False).population_total.sum()
assert len(pop) == 368 and int(pop.population_total.sum()) == 23415008
pop.to_csv(out, index=False, encoding="utf-8-sig")
print("villages", len(p), "towns", len(pop), "total", int(pop.population_total.sum()), "->", out)
