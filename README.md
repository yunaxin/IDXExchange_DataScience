# Home Price Predictor — CRMLS Single-Family Residence Valuation

Predicts closing sale price (`ClosePrice`) for California single-family homes using CRMLS transaction data, and serves a live estimate through a Streamlit app.

## Repository Structure

```
IDXExchange_DataScience/
├── 01_exploration.ipynb        # EDA on the raw Sold data
├── 02_preprocessing.ipynb      # cleaning, feature selection, time-based train/test split
├── 03_baseline_model.ipynb     # Linear Regression baseline
├── 04_model_comparison.ipynb   # Decision Tree / Random Forest + school-district feature experiment
├── 05_advanced_models.ipynb    # XGBoost / LightGBM tuning
├── 06_evaluation.ipynb         # final metrics + price-band analysis
├── app.py                      # Streamlit price-prediction app
├── train_app_model.py          # trains the lightweight model used by app.py -> model.pkl
├── crmls_sold.py                # pulls Sold listings from the CoreLogic Trestle API
├── crmls_listed.py              # pulls active/pending listings from the CoreLogic Trestle API
├── model.pkl                    # trained model used by app.py
├── requirements.txt
├── README.md
├── data/
│   ├── CRMLSSold*.csv           # 29 monthly sold-listing extracts (2024-01–2026-05)
│   ├── CRMLSListing*.csv        # excluded from analysis — see Dataset Source
│   └── geo/
│       └── ca_school_districts_2024_25.geojson
└── output/
    ├── cleaned_data.csv         # full cleaned dataset (02_preprocessing.ipynb)
    ├── train.csv / test.csv     # time-based model-ready split (02_preprocessing.ipynb)
    ├── metrics_summary.csv      # overall + price-band metrics (06_evaluation.ipynb)
    ├── week6_model_comparison.csv / week6_old_vs_new_pivot.csv   # 04_model_comparison.ipynb
    └── week7_advanced_model_metrics.csv                          # 05_advanced_models.ipynb
```

## Dataset Source

- **Provider:** CRMLS (California Regional MLS), pulled via the CoreLogic Trestle API through an internal IDX Exchange auth proxy (`crmls_sold.py`, `crmls_listed.py`).
- **Files used:** `data/CRMLSSold*.csv` — 29 monthly extracts, **2024-01 through 2026-05** (624,754 raw rows). `CRMLSListing*.csv` files (active/pending listings) were **excluded**: `ClosePrice` coverage is only ~25% there vs. 100% in the Sold files, and Sold already carries `ListPrice`/`DaysOnMarket`, so nothing is lost by leaving Listing data out.
- **Supplementary data:** `data/geo/ca_school_districts_2024_25.geojson` — CA school district boundaries from the California GIS Hub (`gis.data.ca.gov`), used for a spatial join in the feature-engineering step.
- **Scope filter:** `PropertyType == "Residential"` and `PropertySubType == "SingleFamilyResidence"` → 314,368 rows.

## Preprocessing (`02_preprocessing.ipynb`)

1. Combine all 29 monthly Sold files, filter to single-family residential.
2. Select core columns: `CloseDate, ClosePrice, LivingArea, BedroomsTotal, BathroomsTotalInteger, LotSizeSquareFeet, City, CountyOrParish, YearBuilt, DaysOnMarket`.
3. Drop rows missing `ClosePrice` or `CloseDate`.
4. Median-impute missing numeric fields, fill missing categoricals with `"Unknown"`, and add `*_was_missing` indicator flags.
5. Remove rows with zero/negative values in price or numeric property features.
6. Clip outliers on `ClosePrice`, `LivingArea`, `LotSizeSquareFeet` using IQR (k=3).
7. **Time-based train/test split**: walk-forward validation swept training-window lengths of 1–27 months and picked the one minimizing average validation MAE → **best window = 17 months**. Final split: train on 2024-12 through 2026-04 (173,243 rows), test on the held-out most recent month, 2026-05 (12,434 rows).
8. Encode categoricals fit on train only — frequency encoding for high-cardinality `City`, one-hot for `CountyOrParish`.
9. `StandardScaler` on numeric columns, fit on train only.
10. Outputs: `output/cleaned_data.csv`, `output/train.csv` (173,243 × 72), `output/test.csv` (12,434 × 72).

**Feature engineering experiment** (`04_model_comparison.ipynb`, second half): added `BedBathRatio`, `PropertyAge`, and a `SchoolDistrict` feature via spatial join against the CA school district boundaries (geopandas `sjoin`, with `sjoin_nearest` fallback for 113 unmatched points out of 302,704). This gave a small, consistent lift on a separate all-history-vs-last-month split (Random Forest R² 0.7230 → 0.7318) but was not carried into the final `train.csv`/`test.csv` used by the rest of the pipeline.

## Models Tested

All models below were trained/evaluated on the same time-based split (`output/train.csv` / `output/test.csv`, test month = 2026-05):

| Model | Test R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression (baseline) | 0.5019 | $420,885 | $602,151 |
| Decision Tree (max_depth=8) | 0.5251 | $402,279 | $588,014 |
| Decision Tree (unconstrained) | 0.1359 | $524,750 | $793,139 |
| Random Forest (n_estimators=200, max_depth=12) | 0.5596 | $383,411 | $566,224 |
| LightGBM (tuned) | 0.8371 | $217,549 | $344,345 |
| **XGBoost (tuned)** | **0.8501** | **$205,150** | **$330,286** |

- The unconstrained Decision Tree overfits badly (train R² 0.9998 vs. test R² 0.14) and is included only to illustrate the effect.
- XGBoost/LightGBM were tuned with a light grid search (`max_depth` ∈ {3,5,7}, `learning_rate` ∈ {0.05,0.1}, `n_estimators` ∈ {100,300}) on an 85/15 validation split carved out of the training set, then refit on the full training set. Best params for both: `max_depth=7, learning_rate=0.1, n_estimators=300`.

## Best Result

**XGBoost** is the best model overall:

- **R² = 0.8501, MAE = $205,150, RMSE = $330,286, MAPE = 18.72%, MdAPE = 12.24%** on the held-out test month.
- Error by price quintile (`output/metrics_summary.csv`): the model performs best in the mid-range (Q3, ~$806K–$1.115M: MAPE 16.4%) where most training examples live. The lowest-price band (Q1, ~$26K–$580K) has the smallest dollar error ($93,847 MAE) but the largest percentage error (27.25% MAPE), since a fixed dollar error is a bigger fraction of a cheaper home. The highest-price band (Q5) has the opposite problem — moderate MAPE but the largest dollar error ($447,433 MAE), likely due to fewer, more variable luxury comps.
- Most important features (tree-based models): `BathroomsTotalInteger`, `LivingArea`, `YearBuilt`, `LotSizeSquareFeet`, `DaysOnMarket`, `BedroomsTotal`.

## The App (`app.py`)

The Streamlit app uses a **separate, simplified model**, not the tuned XGBoost above — it's trained on just 4 raw, unencoded inputs so the form maps directly to it:

- `train_app_model.py` trains a `RandomForestRegressor(n_estimators=200, max_depth=12)` on `LivingArea, BedroomsTotal, BathroomsTotalInteger, LotSizeSquareFeet` from `output/train.csv`/`output/test.csv`, and saves it to `model.pkl` via `joblib`.
- `app.py` loads `model.pkl` and serves a form (living area, bedrooms, bathrooms, lot size) that returns an estimated closing price.

## Re-running the Pipeline

**1. Environment**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Data**

The monthly CSVs already live in `data/`. To re-pull fresh extracts instead, run `python crmls_sold.py` / `python crmls_listed.py` (requires access to the IDX Exchange CoreLogic Trestle proxy/API key referenced in those scripts).

**3. Run the notebooks in order** (each stage depends on the previous one's output):

```
01_exploration.ipynb      # EDA only, no saved outputs required downstream
02_preprocessing.ipynb    # produces output/cleaned_data.csv, train.csv, test.csv
03_baseline_model.ipynb   # Linear Regression baseline
04_model_comparison.ipynb # Decision Tree / Random Forest + school-district feature experiment
05_advanced_models.ipynb  # XGBoost / LightGBM tuning -> output/week7_advanced_model_metrics.csv
06_evaluation.ipynb       # final metrics + price-band analysis -> output/metrics_summary.csv
```

Launch with `jupyter notebook` (or open in VS Code / JupyterLab) and run top to bottom. `02_preprocessing.ipynb` must be run before 03, 05, or 06, since they all load `output/train.csv` / `output/test.csv`.

**4. Train the app's model**

```bash
python train_app_model.py
```
This reads `output/train.csv`/`output/test.csv` (from step 3) and writes `model.pkl`.

**5. Launch the app**

```bash
streamlit run app.py
```
Opens a local browser tab with the price-prediction form.
