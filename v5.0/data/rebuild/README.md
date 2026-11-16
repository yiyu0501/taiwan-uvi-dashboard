# 臺灣鄉鎮城市活力：從公開資料重建分數的程式

對應研究報告「臺灣鄉鎮城市活力之多源測量：資料來源覆蓋、排名敏感度與空間異質性」（2026年11月）。本資料夾的程式從官方公開資料重建全臺368個鄉鎮市區的活動構面F、消費構面C、計分方案S1至S6、統計品質診斷（KMO、Bartlett、PCA、VIF）、來源覆蓋，以及2023年官方界線上的空間分析（Queen LISA與BH-FDR校正）。

分數是測量設定的比較，不是活力排行榜，也不代表因果或政策效果。

## 一、輸入資料（請自行下載，放在 input/）

| 檔名 | 內容 | 來源 | SHA-256（研究使用的檔案） |
|---|---|---|---|
| SEGIS_11211_town.csv | 2023年11月鄉鎮市區電信信令人口統計 | 內政部社會經濟資料服務平臺 https://segis.moi.gov.tw/STATCloud/QueryInterfaceView?COL=zVTg8Fn91RUFg2R9uBdHTA%3D%3D | aa59f14b5ba8c75c30434ed4c4ea90e8a31205899f770c92363424b5c99fd69d |
| MOF_consumption_heat_download.csv | 村里消費熱度（程式只取2023年、餐飲與零售） | 財政部財政資訊中心，電子發票消費熱度指標 https://data.gov.tw/dataset/36843 | cac9fa46bed22e772c921389b23d0a6d5e73c3255c8516e628ec522e7fb785ae |
| ris_ODRP014_11211_page1–4.json | 2023年11月村里人口（四頁，共7,748村里） | 內政部戶政司 https://www.ris.gov.tw/rs-opendata/api/v1/datastore/ODRP014/11211?page=1（page改為1至4） | 由下一步彙總後核對 |
| TOWN_MOI_1120317.shp 等 | 鄉鎮市區界線，2023年3月17日版 | 內政部國土測繪中心 https://data.gov.tw/dataset/7441 | 46ed31c1bd1ab963c34c068fcb82593506d0dfcf6ddd367198213fc4c47aca3a（.shp） |

官方網站日後若更新檔案，SHA-256可能不同；不同時請先比對資料期別與筆數。

## 二、執行步驟

```bash
pip install -r requirements.txt

# 1. 村里人口彙總為368區（總人口23,415,008人）
python code/build_population_202311.py input input/population_2023-11_town.csv

# 2. 重建F、C、S1–S6、來源覆蓋與統計診斷
python code/rebuild_uvi.py --telecom input/SEGIS_11211_town.csv \
  --invoice input/MOF_consumption_heat_download.csv \
  --population input/population_2023-11_town.csv \
  --boundary input/TOWN_MOI_1120317.shp \
  --population-provenance RIS_ODRP014_11211_VILLAGE_AGGREGATED --out output

# 3. 與研究結果比對
python code/verify_against_reference.py output reference_outputs

# 4. 空間分析（界線檔另放 spatial/boundary/）
python code/prepare_spatial_inputs.py
python spatial/code/lisa_maps_2023.py
```

## 三、公式

P(x) 為全國有觀測值鄉鎮中的百分位（平均秩除以有效數再乘100）。A為平日與假日日間活動平均，T為八個時段旅次合計除以2，N為戶籍人口，L為官方界線以EPSG:3826計算的面積（平方公里）。

- F = [P(A/N) + P(A/L) + P(T/N)] / 3
- C：餐飲、零售各取張數與銷售額熱度，先取鄉鎮內村里中位數，再各取全國百分位後平均可用項；缺值不補0
- S1 = (F + C_營業稅) / 2
- S2：活動只用每人指標 [P(A/N)+P(T/N)]/2
- S3：消費改用村里平均數
- S4：活動log(1+x)、消費原值，各欄z分數後分構面平均，再合成取百分位
- S5：活動三項、消費四項各取第一主成分，於共同366區取百分位後平均
- S6 = (F + C_電子發票) / 2

## 四、輸出

| 檔案 | 內容 |
|---|---|
| output/01_重建鄉鎮資料368.csv | 368區原始指標、F、C、S1–S6 |
| output/02、03 | 電子發票相對營業稅的村里×業別覆蓋 |
| output/04、05 | S1與S6共同365區重排名、覆蓋與名次差相關 |
| output/06_PCA_KMO_Bartlett_VIF.json | 統計品質診斷 |
| spatial/data/03_Queen_LISA_BH_FDR_367.csv | LISA結果（BH-FDR 5%；無陸地鄰居的10區另列，不跨海連接） |

## 五、參考結果（reference_outputs/）

研究使用的結果：01_重建鄉鎮資料368.csv、06_PCA_KMO_Bartlett_VIF.json、03_Queen_LISA_BH_FDR_367.csv，以及步驟1應得到的 population_2023-11_town.csv（SHA-256 6d91f8c9d1e8be42a67096c0d182e418774e36cc2e0034e90b6ee0d348786ccc）。

研究執行環境見 requirements.txt。有效樣本：F 368區；C與S1–S4 367區；S5 366區；S6 365區。KMO約0.8409，Bartlett χ²約3001.77（df=21），第一主成分解釋71.57%。LISA：Moran's I約0.631，高高43區、低低25區。

## 六、說明

- 只使用官方公開的彙整統計與熱度等級，未使用個人軌跡或逐筆交易。
- 消費熱度是財政部公開的相對等級，本程式重建的是公開熱度之後的鄉鎮聚合與標準化，未由交易資料重建官方熱度。
- 各資料之使用依提供機關的授權條款。
