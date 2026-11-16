#!/usr/bin/env python3
# coding: utf-8
"""Rebuild township activity (F), consumption (C) and scoring schemes S1-S6
from public inputs: SEGIS telecom township statistics (2023-11), MOF village
consumption heat (2023, food and retail), household-registration population
(2023-11, aggregated from village records) and the NLSC 2023 township boundary.

Also writes source-coverage tables, S1/S6 common-sample ranks and the
KMO/Bartlett/PCA/VIF diagnostics.

Example:
 python code/rebuild_uvi.py --telecom input/SEGIS_11211_town.csv \
   --invoice input/MOF_consumption_heat_download.csv \
   --population input/population_2023-11_town.csv \
   --boundary input/TOWN_MOI_1120317.shp \
   --population-provenance RIS_ODRP014_11211_VILLAGE_AGGREGATED --out output

Optional: --reference <csv> compares the rebuilt F, C and S1-S6 with a table
that has columns activity_subindex_S1, consumption_subindex, S1_score ... S6_score.
"""
import argparse, hashlib, json, math, pathlib, sys, unicodedata, re
import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

FIELDMAP={
 'town_code': ['TOWN_ID','TOWNCODE','TOWN_CODE','鄉鎮市區代碼','鄉鎮市區代號','town_code'],
 'town': ['TOWN','TOWNNAME','鄉鎮市區名稱','鄉鎮市區','town_name'],
 'county': ['COUNTY','COUNTYNAME','縣市名稱','縣市','county_name'],
 'year': ['年度','year','YEAR'],
 'population': ['P_CNT','人口數','總人口數','人口總數','population_total','POPULATION'],
 'wd': ['DAY_WORK','平日日間活動人數'],
 'we': ['DAY_WEEKEND','假日日間活動人數'],
 'info_time':['INFO_TIME','資料時間','資料時間_統計'],
 'village':['村里代碼','VILLCODE','村里','村里名稱'],
 'industry':['主行業別代碼','主行業別','行業名稱','industry'],
 'source':['消費熱度計算來源','資料類別','source','來源'],
 'count':['張數指標','發票張數指標','count_heat'],
 'sales':['銷售額指標','銷售金額指標','sales_heat'],
}
# Telecom signal columns: weekday/weekend four time periods.
TRIP_ALIASES=[
 ['MORNING_WORK','平日早晨旅次','平日早晨旅次_統計'],
 ['MIDDAY_WORK','平日中午旅次','平日中午旅次_統計'],
 ['AFTERNOON_WORK','平日午後旅次','平日午後旅次_統計'],
 ['EVENING_WORK','平日晚上旅次','平日晚上旅次_統計'],
 ['MORNING_WEEKEND','假日早晨旅次','假日早晨旅次_統計'],
 ['MIDDAY_WEEKEND','假日中午旅次','假日中午旅次_統計'],
 ['AFTERNOON_WEEKEND','假日午後旅次','假日午後旅次_統計'],
 ['EVENING_WEEKEND','假日晚上旅次','假日晚上旅次_統計']]

def clean_str(x):
    if pd.isna(x): return ''
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC',str(x)).replace('台','臺')).strip()

def code8(s):
    v=str(s).strip().replace('.0','')
    if v.isdigit() and 7<=len(v)<=8: return v.zfill(8)
    return None

def pick(df, keys, required=True):
    for k in keys:
        if k in df.columns:return k
    if required:raise ValueError('找不到必要欄位 '+repr(keys)+'; 實際='+repr(list(df.columns)))
    return None

def read_csv(p):
    last=None
    for e in ['utf-8-sig','utf-8','cp950','big5']:
        try:
            d=pd.read_csv(p,encoding=e,dtype=str,low_memory=False)
            d.columns=[str(c).lstrip('\ufeff').strip() for c in d.columns]
            return d
        except UnicodeDecodeError as er:last=er
    raise last

def num(s):return pd.to_numeric(s.astype(str).str.replace(',','',regex=False).str.strip(),errors='coerce')
def pct(s):return s.rank(method='average',pct=True,na_option='keep')*100

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def area_from_boundary(boundary):
    import geopandas as gpd
    p=pathlib.Path(boundary)
    if p.suffix.lower()=='.zip':
        g=gpd.read_file('zip://'+str(p.resolve()))
    else:g=gpd.read_file(p)
    cc=pick(g,['TOWNCODE','TOWN_ID','TOWNCODE1','town_code','鄉鎮市區代碼'])
    g['town_code']=g[cc].astype(str).map(code8)
    if g['town_code'].isna().any():raise ValueError('官方圖資有不合法行政碼')
    g=g.to_crs('EPSG:3826')
    g['area_km2']=g.geometry.area/1e6
    a=g.groupby('town_code',as_index=False)['area_km2'].sum()
    if len(a)!=368:raise ValueError(f'官方圖資合併後為{len(a)}區，不是368；不能直接繼續')
    return a,g

def telecom_load(path):
    d=read_csv(path)
    cd=pick(d,FIELDMAP['town_code']);cn=pick(d,FIELDMAP['county']);tn=pick(d,FIELDMAP['town'])
    wd=pick(d,FIELDMAP['wd']);we=pick(d,FIELDMAP['we'])
    tm=pick(d,FIELDMAP['info_time'])
    # Some SEGIS CSVs contain Chinese descriptive row immediately below English header.
    d=d[d[tm].astype(str).str.fullmatch(r'112Y11M',case=False,na=False)].copy()
    if d.empty:raise ValueError('信令原始檔不含112Y11M：拒絕使用其他年月')
    d['town_code']=d[cd].map(code8)
    d['county_name']=d[cn].map(clean_str);d['town_name']=d[tn].map(clean_str)
    if d.town_code.isna().any() or d.town_code.duplicated().any():raise ValueError('信令行政碼遺漏或重複')
    d['day_work']=num(d[wd]);d['day_weekend']=num(d[we])
    for i,aliases in enumerate(TRIP_ALIASES,1):
        k=pick(d,aliases)
        d[f'trip_{i}']=num(d[k])
    if len(d)!=368:raise ValueError(f'信令資料應為368區，實際{len(d)}')
    if d[['day_work','day_weekend']+[f'trip_{i}' for i in range(1,9)]].isna().any().any():
        raise ValueError('信令主要指標有缺值，先核對官方缺值規則')
    keep=['town_code','county_name','town_name','day_work','day_weekend']+[f'trip_{i}' for i in range(1,9)]
    return d[keep]

def population_load(path):
    d=read_csv(path)
    cc=pick(d,FIELDMAP['town_code']);p=pick(d,FIELDMAP['population'])
    tm=pick(d,FIELDMAP['info_time'],required=False)
    if tm is not None:
        d=d[d[tm].astype(str).str.fullmatch(r'112Y(11|12)M',case=False,na=False)].copy()
        if d.empty:raise ValueError('人口資料非112年11或12月，拒絕混入不同年份')
        if d[tm].nunique()>1:raise ValueError('人口資料包含超過一個月份，必須明確指定基期')
    d['town_code']=d[cc].map(code8);d['population_total']=num(d[p]);d=d[d.town_code.notna()].copy()
    if d.town_code.duplicated().any():raise ValueError('人口同一鄉鎮有重複資料')
    if len(d)!=368:raise ValueError(f'人口預期368區，實際{len(d)}區')
    if d.population_total.isna().any() or (d.population_total<=0).any():raise ValueError('人口缺值/非正')
    return d[['town_code','population_total']]

def finance_load(path,area_names):
    d=read_csv(path)
    yr=pick(d,FIELDMAP['year']); co=pick(d,FIELDMAP['county']);tn=pick(d,FIELDMAP['town'])
    vi=pick(d,FIELDMAP['village']);ii=pick(d,FIELDMAP['industry'])
    sc=pick(d,FIELDMAP['source']);tc=pick(d,FIELDMAP['count']);sa=pick(d,FIELDMAP['sales'])
    d=d[d[yr].astype(str).str.strip().isin(['2023','112'])].copy()
    if d.empty:raise ValueError('財政部原始檔沒有2023年，拒絕用2024/2026資料')
    d['county_name']=d[co].map(clean_str);d['town_name']=d[tn].map(clean_str)
    # MOF's own county/town code system (letters and short codes) is NOT MOI 8-digit town code.
    lookup=area_names[['town_code','county_name','town_name']].copy()
    if lookup[['county_name','town_name']].duplicated().any():raise ValueError('基準縣市+鄉鎮名稱重複，無法安全合併')
    d=d.merge(lookup,on=['county_name','town_name'],how='left',validate='many_to_one',indicator=True)
    if (d['_merge']!='both').any():
        bad=d.loc[d['_merge']!='both',['county_name','town_name']].drop_duplicates()
        raise ValueError('財政部和信令的地名有無法對應資料，請先建立明確對照表：'+bad.to_string(index=False))
    d=d.drop(columns=['_merge'])
    d['village_key']=d[vi].map(clean_str)
    if d['village_key'].eq('').any():raise ValueError('財政部村里代碼缺值，會影響覆蓋率')
    d['industry']=d[ii].map(lambda s: '餐飲' if ('餐飲' in str(s) or str(s).strip() in ['56','056']) else ('零售' if ('零售' in str(s) or str(s).strip() in ['47','047']) else None))
    d['source']=d[sc].map(lambda s:'電子發票' if '電子' in str(s) else ('營業稅' if ('營業稅' in str(s) or '營所' in str(s)) else None))
    d=d[d.industry.notna()&d.source.notna()].copy()
    d['heat_count']=num(d[tc]);d['heat_sales']=num(d[sa])
    if d.empty:raise ValueError('財政部2023餐飲零售來源過濾後為空；請檢查產業代碼及來源欄位')
    if (~d.heat_count.between(0,100) & d.heat_count.notna()).any() or (~d.heat_sales.between(0,100) & d.heat_sales.notna()).any():
        raise ValueError('消費熱度超過0–100，可能選錯來源檔或欄位')
    keys=['town_code','village_key','industry','source']
    dups=d.duplicated(keys,keep=False)
    if dups.any():
        # Exact duplicates sometimes happen due to download concatenation; never silently average.
        dup=d.loc[dups,keys+['heat_count','heat_sales']]
        if dup.drop_duplicates().duplicated(keys).any():
            raise ValueError(f'村里×行業×來源重複且分數不同：{int(dups.sum())}筆，須人工查核')
        d=d.drop_duplicates(keys)
    pairs=d.groupby(['town_code','village_key','industry'])['source'].agg(lambda x:frozenset(x)).reset_index()
    pairs['tax']=pairs.source.map(lambda x:'營業稅' in x)
    pairs['einvoice']=pairs.source.map(lambda x:'電子發票' in x)
    ct=pairs.groupby('town_code').agg(tax_village_industry=('tax','sum'),einvoice_village_industry=('einvoice','sum')).reset_index()
    ct['coverage_einvoice_div_tax']=ct.einvoice_village_industry/ct.tax_village_industry.replace(0,np.nan)
    byind=pairs.groupby(['town_code','industry']).agg(tax_village_industry=('tax','sum'),einvoice_village_industry=('einvoice','sum')).reset_index()
    byind['coverage_einvoice_div_tax']=byind.einvoice_village_industry/byind.tax_village_industry.replace(0,np.nan)
    # In general the reporting unit is town × village × industry × source.
    agg=d.groupby(['town_code','source','industry'])[['heat_count','heat_sales']].agg(['median','mean','count'])
    agg.columns=['_'.join(z) for z in agg.columns];agg=agg.reset_index()
    wide=agg.pivot_table(index='town_code',columns=['source','industry'],values=['heat_count_median','heat_count_mean','heat_sales_median','heat_sales_mean'],aggfunc='first')
    wide.columns=[f'{t}_{s}_{i}' for (t,s,i) in wide.columns]
    wide=wide.reset_index()
    return wide,ct,byind,d

def rank_formula(df,cols):
    # Rank each indicator among all townships with observed values; average available indicators.
    # Completely missing source -> NaN; missing one industry is NOT a zero.
    return pd.concat([pct(df[c]) for c in cols],axis=1).mean(axis=1,skipna=True)

def pca_pc1(df,cols):
    use=df[cols].dropna().index
    out=pd.Series(np.nan,index=df.index,dtype=float)
    if len(use)<10:return out,None
    x=StandardScaler().fit_transform(df.loc[use,cols])
    p=PCA(n_components=1,svd_solver='full').fit(x)
    v=p.transform(x)[:,0]
    if float(p.components_.sum())<0: v=-v
    out.loc[use]=v
    return out, {'columns':cols,'components':p.components_[0].tolist(),'explained_variance_ratio':p.explained_variance_ratio_[0].item(),'n':len(use)}

def score_models(master):
    d=master.copy()
    d['mean_day_activity']=(d.day_work+d.day_weekend)/2
    d['mean_trip_8periods']=d[[f'trip_{i}' for i in range(1,9)]].mean(axis=1)
    d['activity_per_resident_mean']=d.mean_day_activity/d.population_total
    d['activity_density_mean']=d.mean_day_activity/d.area_km2
    # Full four-period trips/day = (sum weekdays 4 + sum weekends 4)/2 = 4*mean of 8 period values.
    # All within-variable ranks stay unchanged under this positive constant multiplication,
    # but log1p PCA diagnostics DO change: factor 4 is crucial for exact replication.
    d['trip_per_resident_mean']=4*d.mean_trip_8periods/d.population_total
    fcols=['activity_per_resident_mean','activity_density_mean','trip_per_resident_mean']
    taxcols=[f'{m}_營業稅_{i}' for i in ['餐飲','零售'] for m in ['heat_count_median','heat_sales_median']]
    eicols=[f'{m}_電子發票_{i}' for i in ['餐飲','零售'] for m in ['heat_count_median','heat_sales_median']]
    taxmean=[c.replace('_median_','_mean_') for c in taxcols]
    for cols in [taxcols,eicols,taxmean]:
        for c in cols:
            if c not in d.columns:d[c]=np.nan
    d['F_rebuilt']=rank_formula(d,fcols)
    d['C_tax_rebuilt']=rank_formula(d,taxcols)
    d['C_einvoice_rebuilt']=rank_formula(d,eicols)
    d['C_tax_village_mean_rebuilt']=rank_formula(d,taxmean)
    d['S1_rebuilt']=(d.F_rebuilt+d.C_tax_rebuilt)/2
    d['F_per_resident_only']=rank_formula(d,['activity_per_resident_mean','trip_per_resident_mean'])
    d['S2_rebuilt']=(d.F_per_resident_only+d.C_tax_rebuilt)/2
    d['S3_rebuilt']=(d.F_rebuilt+d.C_tax_village_mean_rebuilt)/2
    logcols=['log1p_'+c for c in fcols]
    for original,transformed in zip(fcols,logcols):
        d[transformed]=np.log1p(d[original])
    def zmean(cols):
        vals=d[cols].apply(lambda s:(s-s.mean())/s.std(ddof=0) if s.std(ddof=0)>0 else np.nan)
        return vals.mean(axis=1,skipna=True)
    # S4 z-normalize seven indicators: log1p for each activity indicator, raw heat for consumption.
    # Standardize each column on available full-country samples, domain-average, then percentile rank.
    d['S4_rebuilt']=pct((zmean(logcols)+zmean(taxcols))/2)
    # S5 two-stage domain PCA = first-stage PC1 from log1p F (all 368), and PC1
    # from raw C (366 complete). Rank both PCs ONLY in their common 366-case sample,
    # then mean the two percentile ranks; do NOT run a second global PCA.
    fpc,fpd=pca_pc1(d,logcols)
    cpc,cpd=pca_pc1(d,taxcols)
    d['F_PC1']=fpc; d['C_PC1']=cpc
    common=fpc.notna()&cpc.notna()
    d['S5_rebuilt']=(pct(fpc.where(common))+pct(cpc.where(common)))/2
    d['S6_rebuilt']=(d.F_rebuilt+d.C_einvoice_rebuilt)/2
    desc={
      'F_columns':fcols,'C_tax_columns':taxcols,'C_einvoice_columns':eicols,
      'F_trip_definition':'mean of 4 daily trip periods averaged between weekday and holiday = sum(8 reported periods)/2',
      'F':'mean of percentiles of daily activity/residents, daily activity/sqkm, daily four-period trips/residents',
      'C':'mean of available percentile indicators: village median heat for food+retail count+sales; never substitute 0 for missing',
      'S1':'(F+C_tax)/2',
      'S2':'(mean(percentile of activity/resident, percentile of trips/resident)+C_tax)/2',
      'S3':'(F+consumption where village heat is town mean instead of town median)/2',
      'S4':'percentile rank of ((mean column z of log1p(F3))+(mean column z of raw C4))/2; z uses ddof=0',
      'S5':'PCA1(log1p F3) on 368, PCA1(raw C4) on 366, percentiles among common 366, then mean two percentile scores',
      'S6':'(F+C_einvoice)/2; invoice source is switched, missing indicators skipped, all-missing -> NA',
      'PCA_first_F':fpd,'PCA_first_C':cpd,
      'SOURCE_NOTE':'Population is the 2023-11 household-registration village table (RIS ODRP014/11211, 7,748 villages) aggregated to 368 townships by the first eight digits of the village code. Telecom (SEGIS), consumption heat (MOF) and boundary (NLSC) are the official public files listed in README.'
    }
    return d,desc,logcols+taxcols

def kmo_bartlett_vif(d,cols):
    df=d[cols].dropna()
    if len(df)<(len(cols)+5):raise ValueError('PCA完整樣本過少')
    x=df.to_numpy(dtype=float)
    z=StandardScaler().fit_transform(x)
    r=np.corrcoef(z,rowvar=False)
    eig=np.linalg.eigvalsh(r)
    # KMO: partial corr from inverse correlation
    inv=np.linalg.inv(r)
    partial=-inv/np.sqrt(np.outer(np.diag(inv),np.diag(inv)))
    np.fill_diagonal(partial,0)
    rr=r.copy();np.fill_diagonal(rr,0)
    kmo=float(np.sum(rr*rr)/(np.sum(rr*rr)+np.sum(partial*partial)))
    n=len(df);p=len(cols)
    chi=float(-(n-1-(2*p+5)/6)*np.log(np.linalg.det(r)))
    dof=p*(p-1)//2;pv=float(chi2.sf(chi,dof))
    vif={cols[i]:float(variance_inflation_factor(z,i)) for i in range(p)}
    pca=PCA(n_components=p,svd_solver='full').fit(z)
    out={'complete_cases':n,'columns':cols,'KMO':kmo,'Bartlett_chisq':chi,'Bartlett_df':dof,'Bartlett_p':pv,
      'PCA_eigenvalues_population_cov':eig[::-1].tolist(),
      'PCA_eigenvalues_sample_cov':pca.explained_variance_.tolist(),
      'PCA_PC1_sample_cov_loadings':dict(zip(cols,(pca.components_[0]*np.sqrt(pca.explained_variance_[0])).tolist())),
      'PCA_variance_ratio':pca.explained_variance_ratio_.tolist(),
      'PCA_PC1_loadings':dict(zip(cols,pca.components_[0].tolist())),'VIF':vif}
    return out,df

def reference_check(d,ref):
    a=read_csv(ref)
    ac=pick(a,FIELDMAP['town_code'])
    a['town_code']=a[ac].map(code8)
    vals=[];out=a[['town_code']].merge(d[['town_code','F_rebuilt','C_tax_rebuilt']+[f'S{x}_rebuilt' for x in range(1,7)]],on='town_code',how='left',validate='one_to_one')
    targets={'F_rebuilt':'activity_subindex_S1','C_tax_rebuilt':'consumption_subindex',**{f'S{x}_rebuilt':f'S{x}_score' for x in range(1,7)}}
    for new,old in targets.items():
        if old not in a:continue
        ar=a[['town_code',old]].copy();ar[old]=pd.to_numeric(ar[old],errors='coerce')
        both=out[['town_code',new]].merge(ar,on='town_code',how='inner').dropna()
        rho=float(spearmanr(both[new],both[old]).statistic) if len(both)>3 and both[new].nunique()>1 else None
        vals.append({'comparison':new+' vs reference '+old,'n_common':len(both),'spearman':rho,
             'mae':float((both[new]-both[old]).abs().mean()) if len(both) else None,
             'max_abs_error':float((both[new]-both[old]).abs().max()) if len(both) else None,
             'exact_match_at_1e_6':int(((both[new]-both[old]).abs()<1e-6).sum())})
    return pd.DataFrame(vals)

def main():
    parser=argparse.ArgumentParser()
    for k in ['telecom','invoice','population','boundary']:
        parser.add_argument('--'+k,required=True)
    parser.add_argument('--reference',required=False)
    parser.add_argument('--out',default='output')
    parser.add_argument('--population-provenance',default='USER_PROVIDED_OFFICIAL_MONTHLY_CSV')
    args=parser.parse_args()
    dest=pathlib.Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    src={k:getattr(args,k) for k in ['telecom','invoice','population','boundary']}
    for k,v in src.items():
        if not pathlib.Path(v).is_file():raise FileNotFoundError('缺少'+k+'：'+v)
    manifest={'population_provenance':args.population_provenance,'inputs':{k:{'file':pathlib.Path(v).name,'sha256':sha256(v),'bytes':pathlib.Path(v).stat().st_size} for k,v in src.items()},'status':'REBUILT_FROM_REAL_SEGIS_MOF_NLSC_INPUTS_AND_DECLARED_POPULATION_SOURCE'}
    signal=telecom_load(args.telecom)
    pop=population_load(args.population)
    area,_=area_from_boundary(args.boundary)
    d=signal.merge(pop,on='town_code',validate='one_to_one',how='left').merge(area,on='town_code',validate='one_to_one',how='left')
    if d.population_total.isna().any() or d.area_km2.isna().any() or len(d)!=368:raise ValueError('人口或正式圖資對不上全部368鄉鎮')
    finance,coverage,byind,raw=finance_load(args.invoice,d[['town_code','county_name','town_name']])
    merged=d.merge(finance,on='town_code',how='left',validate='one_to_one')
    merged,formula,seven=score_models(merged)
    stats,df7=kmo_bartlett_vif(merged,seven)
    merged=merged.merge(coverage,on='town_code',how='left',validate='one_to_one')
    # Compare common valid sample; always recompute ranking within that sample.
    mm=merged[merged.S1_rebuilt.notna()&merged.S6_rebuilt.notna()].copy()
    mm['rank_S1']=mm.S1_rebuilt.rank(ascending=False,method='min')
    mm['rank_S6']=mm.S6_rebuilt.rank(ascending=False,method='min')
    mm['rank_delta']=mm.rank_S1-mm.rank_S6
    mm['rank_delta_abs']=mm.rank_delta.abs()
    correlations=[]
    for y in ['rank_delta','rank_delta_abs']:
        z=mm[['coverage_einvoice_div_tax',y]].dropna()
        res=spearmanr(z.iloc[:,0],z.iloc[:,1]) if len(z)>3 and z.iloc[:,0].nunique()>1 else None
        correlations.append({'outcome':y,'n':len(z),'Spearman_rho':float(res.statistic) if res else None,'naive_p':float(res.pvalue) if res else None,'note':'descriptive; not causal; p-values lack spatial dependence correction'})
    merged.to_csv(dest/'01_重建鄉鎮資料368.csv',index=False,encoding='utf-8-sig')
    coverage.to_csv(dest/'02_發票觀測覆蓋率.csv',index=False,encoding='utf-8-sig')
    byind.to_csv(dest/'03_發票餐飲零售分業覆蓋.csv',index=False,encoding='utf-8-sig')
    mm.to_csv(dest/'04_S1_S6共同樣本與排名.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(correlations).to_csv(dest/'05_覆蓋率排名相關.csv',index=False,encoding='utf-8-sig')
    (dest/'06_PCA_KMO_Bartlett_VIF.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    (dest/'07_候選公式與主張範圍.json').write_text(json.dumps(formula,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.reference:
        compare=reference_check(merged,args.reference)
        compare.to_csv(dest/'08_與參照表逐方案核對.csv',index=False,encoding='utf-8-sig')
        manifest['reference_exact_match']=bool(len(compare) and ((compare['exact_match_at_1e_6'] == compare['n_common']) & (compare['n_common']>0)).all())
    manifest.update({'telecom_towns':len(signal),'finance_rows_2023_food_retail':len(raw),'finance_town_count':raw.town_code.nunique(),
      'n_S1_valid':int(merged.S1_rebuilt.notna().sum()),'n_S6_valid':int(merged.S6_rebuilt.notna().sum()),'n_common':len(mm),
      'coverage_observation_count_definition':'unique town×village×industry with e-invoice / with business-tax; NOT merchant coverage rate',
      'note': 'Population source is recorded in population_provenance; scores are percentile-based candidates, not a vitality ranking.'})
    (dest/'00_來源資訊與品質摘要.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
