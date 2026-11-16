"""儀表板資料包：鄉鎮總表（S1–S7、共同365區名次、四象限、LISA、夜光兩期、跨期活動）＋簡化鄉鎮界線GeoJSON。"""
import sys, json, struct, pathlib
import numpy as np, pandas as pd
ROOT = pathlib.Path(sys.argv[1]); TS = pathlib.Path(sys.argv[2]); OUT = pathlib.Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
P = lambda x: x.rank(method="average", pct=True) * 100
R = lambda x: x.rank(ascending=False, method="min")
rd = lambda p: pd.read_csv(p, dtype={"town_code": str})
d368 = rd(ROOT / "接管驗證_2026-11-14/output_official_Nov_independent/01_重建鄉鎮資料368.csv")
d365 = rd(ROOT / "論文V8_研究展望修訂_2026-11-15/數據/六方案共同365區名次範圍.csv")
# S7 on full sample (367 valid) for score display
A, T, L = d368.mean_day_activity, d368.mean_trip_8periods * 4, d368.area_km2
d368["F_area"] = (P(A / L) + P(T / L)) / 2
d368["S7"] = (d368.F_area + d368.C_tax_rebuilt) / 2
# common-365 ranks exactly as in paper
A, T, L = d365.mean_day_activity, d365.mean_trip_8periods * 4, d365.area_km2
d365["S7"] = (((P(A / L) + P(T / L)) / 2) + d365.C_tax_rebuilt) / 2
sch = ["S1_rebuilt", "S2_rebuilt", "S3_rebuilt", "S4_rebuilt", "S5_rebuilt", "S6_rebuilt", "S7"]
rk = pd.DataFrame({f"rank365_S{i+1}": R(d365[s]) for i, s in enumerate(sch)}); rk["town_code"] = d365.town_code
r6 = rk.iloc[:, :6]; r7 = rk.iloc[:, :7]
rk["rank_range_6"] = r6.max(axis=1) - r6.min(axis=1); rk["rank_range_7"] = r7.max(axis=1) - r7.min(axis=1)
d71 = (rk.rank365_S7 - rk.rank365_S1).abs()
chk = dict(range6_median=rk.rank_range_6.median(), range6_max=rk.rank_range_6.max(), range7_median=rk.rank_range_7.median(),
           range7_max=rk.rank_range_7.max(), S7S1_median=d71.median(), S7S1_max=d71.max(), S7S1_over20=int((d71 > 20).sum()))
assert chk["range6_median"] == 40 and chk["range6_max"] == 170 and chk["range7_median"] == 56 and chk["range7_max"] == 213
assert chk["S7S1_median"] == 26 and chk["S7S1_max"] == 133 and chk["S7S1_over20"] == 216
s2 = d365[["town_code", "S4_rebuilt", "S5_rebuilt", "S6_rebuilt", "coverage_einvoice_div_tax"]]
base = d368[["town_code", "county_name", "town_name", "population_total", "area_km2", "mean_day_activity", "mean_trip_8periods",
             "F_rebuilt", "F_area", "C_tax_rebuilt", "C_einvoice_rebuilt", "S1_rebuilt", "S2_rebuilt", "S3_rebuilt", "S7"]]
t = base.merge(s2, on="town_code", how="left").merge(rk, on="town_code", how="left")
# S4–S6 for towns outside the common 365 come from the 368 file if present
q4 = rd(ROOT / "接管驗證_2026-11-14/UVI_2023_reproducible/spatial_official/data/04_活動消費四象限_368.csv")[["town_code", "type4"]]
li = rd(ROOT / "接管驗證_2026-11-14/UVI_2023_reproducible/spatial_official/data/03_Queen_LISA_BH_FDR_367.csv")[["town_code", "local_I", "q_BH_FDR", "FDR_cluster"]]
nt = rd(TS / "out/鄉鎮_夜光與活動變化_2020_2023.csv")[["town_code", "ntl_qc_flag_20", "ntl_qc_flag_23", "ntl_mean_cf1_20", "ntl_mean_cf1_23",
                                                     "ntl_median_cf1_20", "ntl_median_cf1_23", "ntl_p90_cf1_20", "ntl_p90_cf1_23", "dlog_ntl_mean"]]
ac = rd(TS / "out/鄉鎮_活動構面變化_2020_2023.csv")[["town_code", "N20", "N23", "A20", "A23", "T20", "T23", "dlog_N", "dlog_A", "dlog_T",
                                                  "share_change_A", "share_change_T", "F20", "F23", "Frank20", "Frank23", "dFrank"]]
t = t.merge(q4, on="town_code", how="left").merge(li, on="town_code", how="left").merge(nt, on="town_code", how="left").merge(ac, on="town_code", how="left")
t["day_over_registered_2023"] = t.A23 / t.N23
t["NNK"] = t.county_name.isin(["臺北市", "新北市", "基隆市"])
t["outer_island"] = t.county_name.isin(["金門縣", "連江縣", "澎湖縣"])
t = t.rename(columns={"F_rebuilt": "F", "C_tax_rebuilt": "C_tax", "C_einvoice_rebuilt": "C_einvoice", "coverage_einvoice_div_tax": "Q_coverage",
                      **{f"S{i}_rebuilt": f"S{i}" for i in range(1, 7)}})
assert len(t) == 368 and t.town_code.is_unique
t.to_csv(OUT / "鄉鎮總表_儀表板用_v1.0_20261116.csv", index=False, encoding="utf-8-sig")
# simplified boundary
shp = ROOT / "接管驗證_2026-11-14/UVI_2023_reproducible/spatial_official/boundary/TOWN_MOI_1120317"
def read_shp(p):
    out = []
    with open(p, "rb") as f:
        f.seek(100)
        while True:
            h = f.read(8)
            if len(h) < 8: break
            ln = struct.unpack(">i", h[4:8])[0] * 2; cb = f.read(ln)
            if struct.unpack("<i", cb[:4])[0] == 0: out.append([]); continue
            npart, npt = struct.unpack("<2i", cb[36:44])
            parts = list(struct.unpack(f"<{npart}i", cb[44:44 + 4 * npart])) + [npt]
            pts = np.frombuffer(cb[44 + 4 * npart:44 + 4 * npart + 16 * npt], dtype="<f8").reshape(-1, 2)
            out.append([pts[parts[i]:parts[i + 1]] for i in range(npart)])
    return out
def read_codes(p):
    with open(p, "rb") as f:
        hdr = f.read(32); n = struct.unpack("<I", hdr[4:8])[0]; hl = struct.unpack("<H", hdr[8:10])[0]; rl = struct.unpack("<H", hdr[10:12])[0]
        fields = []
        while True:
            fd = f.read(32)
            if fd[0] == 0x0D: break
            fields.append((fd[:11].split(b"\0")[0].decode(), fd[16]))
        f.seek(hl); codes = []
        for _ in range(n):
            r = f.read(rl); pos = 1
            for nm, l in fields:
                if nm == "TOWNCODE": codes.append(r[pos:pos + l].decode().strip().zfill(8))
                pos += l
    return codes
geo, codes = read_shp(str(shp) + ".shp"), read_codes(str(shp) + ".dbf")
names = t.set_index("town_code")[["county_name", "town_name"]]
DEC = float(sys.argv[4]) if len(sys.argv) > 4 else 0.002
def area(r): x, y = r[:, 0], r[:, 1]; return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
feats = []
for code, rings in zip(codes, geo):
    if code not in names.index: continue
    keep = []
    big = max(rings, key=area)
    for r in rings:
        q = np.round(r / DEC) * DEC
        q = q[np.r_[True, np.any(np.diff(q, axis=0) != 0, axis=1)]]
        if len(q) < 4 and r is not big: continue
        if len(q) < 4: q = np.round(r, 5)
        if not np.array_equal(q[0], q[-1]): q = np.vstack([q, q[:1]])
        keep.append([[round(float(a), 4), round(float(b), 4)] for a, b in q])
    feats.append({"type": "Feature", "properties": {"town_code": code, "county_name": names.loc[code, "county_name"], "town_name": names.loc[code, "town_name"]},
                  "geometry": {"type": "MultiPolygon", "coordinates": [[k] for k in keep]}})
assert len(feats) == 368, len(feats)
json.dump({"type": "FeatureCollection", "features": feats}, open(OUT / "鄉鎮界線_簡化_v1.0_20261116.geojson", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print(chk, len(feats))
