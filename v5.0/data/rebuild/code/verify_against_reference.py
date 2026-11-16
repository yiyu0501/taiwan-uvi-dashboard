#!/usr/bin/env python3
"""Compare a new rebuild (output/01_重建鄉鎮資料368.csv) with reference_outputs/.

Usage: python code/verify_against_reference.py output reference_outputs
"""
import sys, json, pathlib
import pandas as pd

new_dir, ref_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
rd = lambda p: pd.read_csv(p, encoding="utf-8-sig", dtype={"town_code": str})
a, b = rd(new_dir / "01_重建鄉鎮資料368.csv"), rd(ref_dir / "01_重建鄉鎮資料368.csv")
m = a.merge(b, on="town_code", suffixes=("_new", "_ref"), validate="one_to_one")
cols = ["area_km2", "population_total", "F_rebuilt", "C_tax_rebuilt", "C_einvoice_rebuilt"] + [f"S{i}_rebuilt" for i in range(1, 7)]
ok = len(m) == 368
for c in cols:
    x, y = m[c + "_new"], m[c + "_ref"]
    same_na = (x.isna() == y.isna()).all()
    err = float((x - y).abs().max())
    tol = 1e-6 if c == "area_km2" else 1e-9
    good = same_na and (err <= tol or pd.isna(err))
    ok &= good
    print(f"{c:22s} n={int(x.notna().sum()):3d} max_abs_diff={err:.3g} {'OK' if good else 'DIFF'}")
s1, s2 = json.loads((new_dir / "06_PCA_KMO_Bartlett_VIF.json").read_text(encoding="utf-8")), json.loads((ref_dir / "06_PCA_KMO_Bartlett_VIF.json").read_text(encoding="utf-8"))
for k in ["complete_cases", "KMO", "Bartlett_chisq"]:
    print(f"{k:22s} new={s1[k]} ref={s2[k]}"); ok &= abs(float(s1[k]) - float(s2[k])) < 1e-6
print("ALL MATCH" if ok else "CHECK DIFFERENCES")
