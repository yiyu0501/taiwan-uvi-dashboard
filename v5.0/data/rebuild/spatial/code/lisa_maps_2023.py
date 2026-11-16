#!/usr/bin/env python3
# coding: utf-8
"""Spatial analysis on the NLSC 2023 official township boundary: conditional
Queen LISA with Benjamini-Hochberg FDR (islands without land neighbours are
reported separately; no cross-sea links), activity-consumption types, scheme
rank ranges and S1/S6 common-sample ranks, with maps.

Input: spatial/data/00_input_scores_368.csv (from code/prepare_spatial_inputs.py)
and spatial/boundary/TOWN_MOI_1120317.shp. Random permutations use a fixed seed.

Dependencies: geopandas, shapely, numpy, pandas, scipy, statsmodels, matplotlib, numba.
"""
import os, io, sys, json, time, math, argparse, hashlib
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely import make_valid
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib import patheffects
from numba import njit

OUT=Path(__file__).resolve().parents[1]
FIG=OUT/'figures'; DAT=OUT/'data'; NOTES=OUT/'notes'; BND=OUT/'boundary'
for d in (FIG,DAT,NOTES,BND):d.mkdir(parents=True,exist_ok=True)
SEED=20261113
PERM_L=9999;PERM_G=9999
SRC='資料：內政部2023/11 SEGIS、財政部2023年餐飲零售熱度、戶政司2023/11村里人口彙總、國土測繪中心 TOWN_MOI_1120317 官方界線；本研究重算。'
from matplotlib import font_manager
available_fonts={f.name for f in font_manager.fontManager.ttflist}
TCFONT=[next((f for f in ['Noto Sans CJK JP','Noto Sans CJK TC','Heiti TC','PingFang TC','AR PL UMing TW'] if f in available_fonts),'DejaVu Sans')]
plt.rcParams.update({'font.family':TCFONT[0], 'axes.unicode_minus':False, 'savefig.dpi':220,
                     'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.titlesize':13,'figure.facecolor':'white'})
COL={'HH':'#0f766e','LL':'#c9d4df','HL':'#e09a41','LH':'#6077aa','NA':'#e8e8e8','ISL':'#606060'}

# TopoJSON 2 geometry, preserving actual polygon geometry (not rasterized/faked).
def read_topojson(path):
    obj=json.loads(Path(path).read_text('utf-8'))
    sc=np.array(obj['transform']['scale']);tr=np.array(obj['transform']['translate'])
    arcs=[]
    for arc in obj['arcs']:
        xy=np.cumsum(np.array(arc,dtype=np.int64),axis=0).astype(float)
        arcs.append(xy*sc+tr)
    def stitch(indices):
        parts=[]
        for j,k in enumerate(indices):
            a=arcs[k] if k>=0 else arcs[~k][::-1]
            parts.append(a if j==0 else a[1:])
        pts=np.vstack(parts)
        if len(pts)>0 and np.any(pts[0]!=pts[-1]):pts=np.vstack([pts,pts[0]])
        return [tuple(p) for p in pts]
    features=[]
    for item in obj['objects']['towns']['geometries']:
        arcs_obj=item['arcs']; polys=[]
        if item['type']=='Polygon':arcs_obj=[arcs_obj]
        for p in arcs_obj:
            rings=[stitch(k) for k in p]
            try:
                geom=Polygon(rings[0],rings[1:]);geom=make_valid(geom)
                if not geom.is_empty:polys.append(geom)
            except Exception as e:print('Invalid Polygon',item['properties'].get('TOWNCODE'),e)
        if len(polys)==1:geom=polys[0]
        else:geom=MultiPolygon([p for p in polys if p.geom_type=='Polygon'])
        features.append({**item['properties'],'geometry':geom})
    g=gpd.GeoDataFrame(features,geometry='geometry',crs='EPSG:4326')
    return g

def load():
    d=pd.read_csv(DAT/'00_input_scores_368.csv',encoding='utf-8-sig',dtype={'town_code':'string'})
    assert len(d)==368 and d.town_code.nunique()==368
    g=gpd.read_file(BND/'TOWN_MOI_1120317.shp')
    g['town_code']=g['TOWNCODE'].astype('string').str.zfill(8)
    assert len(g)==368 and g.town_code.nunique()==368
    m=g.merge(d,on='town_code',how='outer',validate='one_to_one',indicator=True)
    assert m._merge.eq('both').sum()==368
    m=m.drop(columns=['_merge']).to_crs('EPSG:4326')
    assert m.town_name.notna().all()
    return m

# Queen adjacency from geometry point/line contact, not minimum distance.
def queen_weights(m):
    ix=m.sindex
    pairs=ix.query(m.geometry,predicate='intersects',output_format='indices')
    ns=[set() for _ in range(len(m))]
    for i,j in zip(*pairs):
        i=int(i);j=int(j)
        if j<=i:continue
        gi=m.geometry.iloc[i];gj=m.geometry.iloc[j]
        if gi.touches(gj) or gi.boundary.intersects(gj.boundary):
            ns[i].add(j);ns[j].add(i)
    return [sorted(x) for x in ns]

@njit
# Select k DISTINCT towns from the n-1 excluding target i; local conditional
# randomization. No faux neighbour assigned to islands.
def local_sim_test(z,positions,ptrs,perms,seed):
    np.random.seed(seed)
    n=len(z);m2=np.sum(z*z)/n
    ii=np.empty(n); p=np.empty(n); lag=np.empty(n)
    for i in range(n):
        k=ptrs[i+1]-ptrs[i]
        if k==0:
            ii[i]=np.nan;p[i]=np.nan;lag[i]=np.nan
            continue
        s=0.
        for x in range(ptrs[i],ptrs[i+1]):s+=z[positions[x]]
        lag[i]=s/k
        obs=(z[i]*lag[i])/m2
        ii[i]=obs
        # E under conditional permutations: - z_i^2 / ((n-1)*m2)
        E=-z[i]*z[i]/((n-1)*m2)
        extreme=0
        for rep in range(perms):
            chosen=np.empty(k,dtype=np.int32)
            ssim=0.
            for l in range(k):
                while True:
                    r=np.random.randint(0,n-1)
                    dup=False
                    for h in range(l):
                        if chosen[h]==r:
                            dup=True;break
                    if not dup:break
                chosen[l]=r
                target=r if r<i else r+1
                ssim+=z[target]
            sim=(z[i]*(ssim/k))/m2
            if abs(sim-E)>=abs(obs-E)-1e-13:extreme+=1
        p[i]=(extreme+1)/(perms+1)
    return ii,p,lag

@njit
def global_perm_test(z,positions,ptrs,perms,seed):
    np.random.seed(seed)
    n=len(z);s0=0
    for i in range(n):
        if ptrs[i+1]>ptrs[i]:s0+=1
    denom=np.sum(z*z)
    numer=0.
    for i in range(n):
        k=ptrs[i+1]-ptrs[i]
        if k:
            lag=0.
            for a in range(ptrs[i],ptrs[i+1]):lag+=z[positions[a]]
            numer+=z[i]*(lag/k)
    obs=(n/s0)*numer/denom
    pcount=0
    for r in range(perms):
        zp=z[np.random.permutation(n)]
        s=0.
        for i in range(n):
            k=ptrs[i+1]-ptrs[i]
            if k:
                lag=0.
                for a in range(ptrs[i],ptrs[i+1]):lag+=zp[positions[a]]
                s+=zp[i]*(lag/k)
        val=(n/s0)*s/denom
        if abs(val)>=abs(obs)-1e-13:pcount+=1
    return obs,(pcount+1)/(perms+1)


def compute_lisa(m):
    v=m[m.S1_score.notna()].reset_index(drop=True)
    ns=queen_weights(v)
    isl=[i for i,s in enumerate(ns) if not s]
    z=v.S1_score.to_numpy(float)-v.S1_score.mean()
    ptrs=np.zeros(len(v)+1,dtype=np.int32)
    for i,s in enumerate(ns):ptrs[i+1]=ptrs[i]+len(s)
    positions=np.array([j for s in ns for j in s],dtype=np.int32)
    t=time.time();print('LISA 9999 conditional local permutations for',len(v),'towns...')
    I,p,lag=local_sim_test(z,positions,ptrs,PERM_L,SEED)
    print('LISA elapsed',round(time.time()-t,1))
    valid=np.isfinite(p)
    rej,padj,_,_=multipletests(p[valid],alpha=.05,method='fdr_bh')
    q=np.full(len(v),np.nan);q[valid]=padj
    sig=np.zeros(len(v),bool);sig[valid]=rej
    cls=np.empty(len(v),dtype=object)
    for i in range(len(v)):
        if i in isl:cls[i]='孤立單元'
        elif not sig[i]:cls[i]='不顯著'
        elif z[i]>0 and lag[i]>0:cls[i]='HH 高高'
        elif z[i]<0 and lag[i]<0:cls[i]='LL 低低'
        elif z[i]>0 and lag[i]<0:cls[i]='HL 高低'
        else:cls[i]='LH 低高'
    out=pd.DataFrame({'town_code':v.town_code,'county_name':v.county_name,'town_name':v.town_name,
                      'S1':v.S1_score,'num_queen_neighbors':[len(x) for x in ns],
                      'local_I':I,'lag_centered':lag,'p_conditional_two_sided':p,
                      'q_BH_FDR':q,'significant_p05_uncorrected':p<.05,
                      'significant_FDR05':sig,'FDR_cluster':cls})
    out.to_csv(DAT/'03_Queen_LISA_BH_FDR_367.csv',index=False,encoding='utf-8-sig')
    full=m.merge(out[['town_code','num_queen_neighbors','local_I','p_conditional_two_sided','q_BH_FDR','FDR_cluster']],on='town_code',how='left')
    full['FDR_cluster']=full['FDR_cluster'].fillna('S1缺值')
    # Comparison KNN, no islands, but cross-strait geometric contacts are not true
    mp=v.to_crs(3826);cen=np.array([[g.centroid.x,g.centroid.y] for g in mp.geometry]);
    from scipy.spatial import cKDTree
    _,nb=cKDTree(cen).query(cen,k=5)
    ns_k=[list(map(int,nb[i,1:])) for i in range(len(v))]
    pp=np.zeros(len(v)+1,dtype=np.int32)
    for i,s in enumerate(ns_k):pp[i+1]=pp[i]+len(s)
    jjs=np.array([j for s in ns_k for j in s],dtype=np.int32)
    # use 9999 for global Queen and 4NN
    Ig,pg=global_perm_test(z,positions,ptrs,PERM_G,SEED+1)
    Ik,pk=global_perm_test(z,jjs,pp,PERM_G,SEED+1)
    out_meta={'boundary': 'National Land Surveying and Mapping Center TOWN_MOI_1120317 (official 2023-03-17 shp)',
      'boundary_code_match':368,'scores':len(v),'missing_S1':'金門縣烏坵鄉',
      'queen_isolate_count':len(isl),'queen_isolate_names':v.iloc[isl].apply(lambda x: str(x.county_name)+str(x.town_name),axis=1).tolist(),
      'local_valid_tests':int(valid.sum()),'local_p_uncorrected_lt05':int(np.sum(p[valid]<.05)),
      'local_significant_FDR05':int(sig.sum()),'local_FDR_breakdown':out.loc[out.significant_FDR05,'FDR_cluster'].value_counts().to_dict(),
      'method': 'conditional local Monte Carlo; 9,999 permutations; two-sided centering on conditional E; BH correction across non-isolated S1 observations',
      'seed':SEED,'global_queen_I':float(Ig),'global_queen_p_two_sided':float(pg),
      'global_knn4_I':float(Ik),'global_knn4_p_two_sided':float(pk),
      'assumptions':'Queen isolates excluded from local testing; NO cross-sea artificial contiguity; 4NN is distance-based robustness only, not physical access',
      'official_boundary_required_before_final_publication':False}
    (NOTES/'spatial_QC.json').write_text(json.dumps(out_meta,ensure_ascii=False,indent=2),'utf-8')
    return full,out_meta


def style_map(ax,df,col=None, cmap=None, norm=None, fillcolors=None, edgecolor='#c4cdd3',linewidth=.21, special=None):
    if fillcolors is not None:
        ax.set_facecolor('white')
        for color in sorted(set(fillcolors)):
            s=df[np.asarray(fillcolors)==color]
            if len(s):s.plot(ax=ax,color=color,edgecolor=edgecolor,linewidth=linewidth)
    else:
        df.plot(column=col,ax=ax,cmap=cmap,norm=norm,edgecolor=edgecolor,linewidth=linewidth,missing_kwds={'color':'#efefef'})
    ax.set_xlim(118.0,122.18);ax.set_ylim(21.76,26.47)
    ax.set_aspect('equal');ax.axis('off')

def footer(fig,source=SRC,y=.026,fontsize=8):
    fig.text(.5,y,source,ha='center',va='bottom',fontsize=fontsize,color='#64748b')

def fsav(fig,name):
    fig.savefig(FIG/name,bbox_inches='tight',pad_inches=.22,dpi=220,facecolor='white')
    plt.close(fig);print('SAVED',name)

def plot_maps(m,lisa,meta):
    # Density uses the township population and official area carried in the input table.
    m['population_density']=m.population_total/m.area_km2
    # Main double panel
    fig,axs=plt.subplots(1,2,figsize=(15,8),constrained_layout=False)
    # 5 bins for S1 as equal-interval? quantiles
    s=m.S1_score.dropna(); edges=np.unique(np.quantile(s,[0,.2,.4,.6,.8,1]))
    norm=matplotlib.colors.BoundaryNorm(edges,256)
    style_map(axs[0],m,'S1_score','YlGnBu',norm)
    h1=plt.cm.ScalarMappable(norm=norm,cmap='YlGnBu');h1.set_array([])
    cax1=fig.add_axes([.11,.17,.32,.025]);cb=fig.colorbar(h1,cax=cax1,orientation='horizontal');cb.set_label('S1 候選分數（分位級距）')
    density=np.log10(m.population_density.where(m.population_density>0));end=np.quantile(density.dropna(),[0,.2,.4,.6,.8,1]);end=np.unique(end)
    dn=matplotlib.colors.BoundaryNorm(end,256)
    style_map(axs[1],m.assign(log_density=density),'log_density','OrRd',dn)
    cax2=fig.add_axes([.57,.17,.32,.025]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=dn,cmap='OrRd'),cax=cax2,orientation='horizontal');cb.set_label('人口密度 log10(人／km²)')
    axs[0].set_title('A  S1 候選活力分數',fontweight='bold',pad=12)
    axs[1].set_title('B  戶籍人口密度',fontweight='bold',pad=12)
    fig.suptitle('圖 1｜鄉鎮活力與人口密度之空間對照',fontsize=19,y=.96,fontweight='bold')
    fig.text(.5,.115,'人口密度採主資料表之人口／面積；兩圖顏色各依自身分布，不可直接比較色階數值。',fontsize=9,ha='center',color='#475569')
    footer(fig,y=.07,fontsize=8)
    fig.subplots_adjust(left=.03,right=.97,top=.90,bottom=.23,wspace=.08)
    fsav(fig,'01_S1與人口密度對照圖.png')

    # LISA
    ml=m.merge(lisa[['town_code','FDR_cluster','p_conditional_two_sided','q_BH_FDR']],on='town_code',how='left')
    cats=['HH 高高','LL 低低','HL 高低','LH 低高','不顯著','孤立單元','S1缺值']
    colors={'HH 高高':'#c34242','LL 低低':'#377bb3','HL 高低':'#f0a04b','LH 低高':'#67a776','不顯著':'#e5e8ed','孤立單元':'#424b57','S1缺值':'#b9b9b9'}
    ml['FDR_cluster']=ml.FDR_cluster.fillna('S1缺值')
    fig,ax=plt.subplots(figsize=(10.8,8.8));style_map(ax,ml,fillcolors=ml.FDR_cluster.map(colors).tolist(),edgecolor='#a6adb6',linewidth=.24)
    ax.set_title('圖 2｜S1 的 Local Moran’s I 群聚分布（BH-FDR 5%）',fontsize=17,pad=15,fontweight='bold')
    handles=[Patch(facecolor=colors[x],edgecolor='white',label=f'{x}（{(ml.FDR_cluster==x).sum()}）') for x in cats]
    ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(.0,.01),frameon=False,ncol=2,title='全臺鄉鎮類型')
    fig.text(.5,.09,f'Queen相鄰權重；有效檢定 {meta["local_valid_tests"]}；FDR顯著 {meta["local_significant_FDR05"]}；無相鄰 {meta["queen_isolate_count"]}；烏坵S1缺值。',fontsize=9,ha='center')
    footer(fig,y=.064,fontsize=8)
    fig.subplots_adjust(left=.08,right=.98,top=.9,bottom=.12)
    fsav(fig,'02_LISA_BH_FDR_群聚圖.png')

    # quadrant type median split actually based on source csv
    d=m.copy(); F=d.activity_subindex_S1; C=d.consumption_subindex
    # compute medians among 367 jointly nonmissing
    medF=float(F.dropna().median()); medC=float(C.dropna().median())
    typ=np.where(F.isna()|C.isna(),'無資料',np.where(F>=medF,np.where(C>=medC,'HH 活動高消費高','HL 活動高消費低'),np.where(C>=medC,'LH 活動低消費高','LL 活動低消費低')))
    d['type4']=typ
    d[['town_code','county_name','town_name','activity_subindex_S1','consumption_subindex','type4']].to_csv(DAT/'04_活動消費四象限_368.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(10.8,8.8))
    tycolors={'HH 活動高消費高':'#116e69','HL 活動高消費低':'#e2a13a','LH 活動低消費高':'#5974ae','LL 活動低消費低':'#c5cfda','無資料':'#eeeeee'}
    style_map(ax,d,fillcolors=d.type4.map(tycolors).tolist())
    ax.set_title('圖 3｜活動（F）與消費（C）雙構面四象限類型',fontsize=17,pad=15,fontweight='bold')
    ax.legend(handles=[Patch(facecolor=c,edgecolor='white',label=f'{k}（{(d.type4==k).sum()}）') for k,c in tycolors.items()],
              loc='lower left',bbox_to_anchor=(0,.01),frameon=False,ncol=2)
    fig.text(.5,.087,f'分組準則：F≥全臺中位數 {medF:.2f} 為高活動；C≥全臺中位數 {medC:.2f} 為高消費。',fontsize=9,ha='center')
    footer(fig,y=.062,fontsize=8)
    fig.subplots_adjust(left=.08,right=.98,top=.9,bottom=.12)
    fsav(fig,'03_活動與消費四象限地圖.png')

    # Rank map
    rcols=[f'S{i}_rank' for i in range(1,7)]
    ranks=d[rcols].apply(pd.to_numeric,errors='coerce')
    d['n_valid_schemes']=ranks.notna().sum(axis=1)
    d['rank_min_recalc']=ranks.min(axis=1)
    d['rank_max_recalc']=ranks.max(axis=1)
    d['rank_range_recalc']=d['rank_max_recalc']-d['rank_min_recalc']
    d[['town_code','county_name','town_name','n_valid_schemes','rank_min_recalc','rank_max_recalc','rank_range_recalc',*rcols]].to_csv(DAT/'05_六情境名次全距_368.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(10.8,8.8))
    rn=matplotlib.colors.Normalize(vmin=0,vmax=float(d.rank_range_recalc.quantile(.99)))
    style_map(ax,d,'rank_range_recalc','magma_r',rn)
    ax.set_title('圖 4｜六情境下各鄉鎮名次全距',fontsize=17,pad=15,fontweight='bold')
    cax=fig.add_axes([.25,.115,.50,.029]);cb=fig.colorbar(plt.cm.ScalarMappable(cmap='magma_r',norm=rn),cax=cax,orientation='horizontal');cb.set_label('最高－最低名次差（名）')
    fig.text(.5,.076,'深色表示排名較依賴測量選擇；排名使用原情境儲存名次，缺值方案不計入。',fontsize=9,ha='center')
    footer(fig,y=.051,fontsize=8)
    fig.subplots_adjust(left=.08,right=.98,top=.9,bottom=.15)
    fsav(fig,'04_六情境名次不確定性地圖.png')

    # Labeled top25 interval chart
    top=d[d.n_valid_schemes>=2].sort_values('rank_range_recalc',ascending=False).head(25).iloc[::-1]
    fig,ax=plt.subplots(figsize=(11.5,10.5))
    yy=np.arange(len(top))
    ax.hlines(yy,top.rank_min_recalc,top.rank_max_recalc,color='#bdcad7',lw=6,zorder=2)
    ax.scatter(top.rank_min_recalc,yy,c='#2a6496',s=36,zorder=4,label='最佳名次')
    ax.scatter(top.rank_max_recalc,yy,c='#c35b3d',s=36,zorder=4,label='最差名次')
    ax.set_yticks(yy,labels=(top.county_name+top.town_name).tolist());ax.set_xlabel('名次（1為分數最高）')
    ax.set_title('圖 5｜六情境名次差異最大的25個鄉鎮',fontsize=17,pad=14,fontweight='bold')
    ax.grid(axis='x',alpha=.17);ax.legend(loc='lower right',frameon=False)
    for yv,row in zip(yy,top.itertuples()):
        ax.text(row.rank_max_recalc+3,yv,str(int(row.rank_range_recalc)),va='center',fontsize=8,color='#404e5d')
    ax.set_xlim(0,405);ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=9))
    fig.text(.53,.052,'線段跨越越長，代表同一鄉鎮在六種測量方案下排名差異越大。',fontsize=9,ha='center',color='#475569')
    footer(fig,y=.032,fontsize=8)
    fig.subplots_adjust(left=.23,right=.96,top=.91,bottom=.11)
    fsav(fig,'05_前25名次區間標示圖.png')

    # S1/S6 ranks within the COMMON VALID sample (n=365).
    ss=d[d.S1_score.notna() & d.S6_score.notna()].copy()
    ss['rank_S1_common']=ss.S1_score.rank(ascending=False,method='min')
    ss['rank_S6_common']=ss.S6_score.rank(ascending=False,method='min')
    ss['delta_rank_S6_minus_S1']=ss.rank_S6_common-ss.rank_S1_common
    ss['absolute_rank_change']=ss.delta_rank_S6_minus_S1.abs()
    ss=ss.sort_values(['absolute_rank_change','town_code'],ascending=[False,True])
    ss[['town_code','county_name','town_name','S1_score','S6_score','rank_S1_common','rank_S6_common','delta_rank_S6_minus_S1','absolute_rank_change']].to_csv(DAT/'06_S1_S6共同樣本排名_365.csv',index=False,encoding='utf-8-sig')
    top10=ss.head(10)
    fig=plt.figure(figsize=(15.0,9.5))
    ax=fig.add_axes([.08,.17,.65,.72])
    ax.scatter(ss.rank_S1_common,ss.rank_S6_common,s=17,alpha=.43,c='#8ba3b3')
    ax.plot([0,366],[0,366],linestyle='--',lw=1.2,color='#333',alpha=.6)
    # Numbered marks refer to right-hand printed name list: no label overlap.
    for j,row in enumerate(top10.itertuples()):
        ax.scatter(row.rank_S1_common,row.rank_S6_common,s=255,color='#c15c3d',edgecolor='white',lw=1.2,zorder=4)
        ax.text(row.rank_S1_common,row.rank_S6_common,str(j+1),fontsize=9,fontweight='bold',color='white',ha='center',va='center',zorder=5)
    ax.set_xlim(380,0); ax.set_ylim(380,0)
    ax.set_xlabel('S1 名次（共同樣本，1為最高）'); ax.set_ylabel('S6 名次（共同樣本，1為最高）')
    ax.set_title('共同有效樣本 n=365 的名次對照',fontsize=12,pad=10)
    ax.grid(alpha=.15)
    txt=fig.add_axes([.77,.17,.22,.72]);txt.axis('off')
    txt.text(0,1.01,'名次變動最大的10個鄉鎮',fontsize=13,fontweight='bold',va='top')
    for j,row in enumerate(top10.itertuples()):
        y=.935-j*.087
        txt.text(.01,y,f'{j+1:02}',color='#c15c3d',fontsize=10,fontweight='bold',va='center')
        txt.text(.145,y,f'{row.county_name}{row.town_name}',fontsize=10,va='center')
        sg='↑' if row.delta_rank_S6_minus_S1<0 else '↓'
        txt.text(.99,y,f'{sg}{int(row.absolute_rank_change)}',fontsize=10,color='#536577',va='center',ha='right')
    fig.suptitle('圖 6｜更換消費來源後的鄉鎮名次變化：S1 與 S6',fontsize=18,y=.96,fontweight='bold')
    fig.text(.5,.103,f'共同樣本 {len(ss)} 鄉鎮｜名次差逾20名：{int((ss.absolute_rank_change>20).sum())}（{((ss.absolute_rank_change>20).mean()*100):.1f}%）｜最大移動 {int(top10.iloc[0].absolute_rank_change)} 名',fontsize=10,ha='center')
    fig.text(.5,.075,'表內 ↑ 表示在 S6 的名次提升；↓ 表示下降。點上數字與右側地名一一對應。',fontsize=9,ha='center',color='#475569')
    footer(fig,y=.04,fontsize=8)
    fsav(fig,'06_S1_S6排名散布_標記變動前10.png')

    return d,ss,top10

def main():
    # stage in standardized standalone script: accompanying files in data and boundary
    m=load();print('368 matched.')
    q,meta=compute_lisa(m)
    merged,ss,top10=plot_maps(m,q,meta)
    merged.drop(columns=['geometry'],errors='ignore').to_csv(DAT/'07_全部鄉鎮統合表.csv',index=False,encoding='utf-8-sig')
    print('META',json.dumps(meta,ensure_ascii=False,indent=2))
    print('TOP10',top10[['county_name','town_name','rank_S1_common','rank_S6_common','absolute_rank_change']].to_string(index=False))

if __name__=='__main__': main()
