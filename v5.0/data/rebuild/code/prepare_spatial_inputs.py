#!/usr/bin/env python3
"""Prepare spatial-analysis inputs (spatial/data/00_input_scores_368.csv) from the rebuilt township scores in output/."""
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[1]
src=root/'output'/'01_重建鄉鎮資料368.csv'
df=pd.read_csv(src,encoding='utf-8-sig',dtype={'town_code':'string'})
assert len(df)==368 and df.town_code.nunique()==368
fields=['town_code','county_name','town_name','population_total','area_km2','F_rebuilt','C_tax_rebuilt']+[f'S{i}_rebuilt' for i in range(1,7)]
for f in fields: assert f in df.columns, f'Missing column {f}'
d=df[fields].rename(columns={'F_rebuilt':'activity_subindex_S1','C_tax_rebuilt':'consumption_subindex',**{f'S{i}_rebuilt':f'S{i}_score' for i in range(1,7)}}).copy()
for i in range(1,7):d[f'S{i}_rank']=d[f'S{i}_score'].rank(ascending=False,method='min')
out=root/'spatial'/'data'/'00_input_scores_368.csv'
out.parent.mkdir(parents=True,exist_ok=True)
d.to_csv(out,index=False,encoding='utf-8-sig')
print('OK: refreshed 368-town official map input',out)
