# 🗽 NYC Airbnb Price Prediction: End-to-End Machine Learning System
### DS605: Fundamentals of Machine Learning — Lab Assignment 4

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-Tuned-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

- **Author:** Akanksha Dasani
- **Student ID:** `202618062`
- **Course:** DS605: Fundamentals of Machine Learning
- **Dataset:** [Kaggle New York City Airbnb Open Data (AB_NYC_2019.csv)](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)

---

## 🌐 Deployed Web Application

The interactive web application is deployed online on **Streamlit Community Cloud**:

> 🔗 **Live Application Link:** **[NYC Airbnb Price Predictor (Live App)](https://share.streamlit.io/)**  
> *(Connect this GitHub repository to Streamlit Cloud to activate this direct URL, as outlined in the [Deployment Section](#-deployment-to-streamlit-community-cloud)).*

---

## 📌 Executive Summary

This project delivers a complete, production-grade machine learning solution to predict nightly Airbnb prices across the 5 boroughs of New York City (Manhattan, Brooklyn, Queens, Bronx, and Staten Island). The solution covers:
1. **Task 1: Data Analysis and Preparation** — Data cleaning, domain imputation, outlier filtering, log target transformation, and geospatial feature engineering.
2. **Task 2: Model Training and Evaluation** — Benchmarking 7 regression models, cross-validation, hyperparameter tuning via `GridSearchCV`, overfitting/underfitting diagnosis, and packaging an end-to-end `Pipeline`.
3. **Task 3: Streamlit Web Application** — Multi-tab interactive web interface with scenario presets, coordinate auto-fill, confidence intervals, and exploratory dashboards.
4. **Task 4: Final Project Summary** — Comprehensive model comparison, feature importance analysis, and real-world system limitations.

---

## 📂 Repository Structure

```
├── .gitignore                          # Git exclusion rules
├── AB_NYC_2019.csv                     # NYC Airbnb 2019 Open Dataset (48,895 records)
├── Airbnb_Price_Prediction.ipynb       # Fully executed Jupyter Notebook with outputs
├── New_York_City_.png                  # High-resolution NYC reference bounding box map
├── README.md                           # Project documentation & summary report
├── app.py                              # Streamlit web application
├── requirements.txt                    # Project dependencies
├── models/
│   ├── model_benchmark.csv             # Model comparison metrics across all 7 algorithms
│   ├── model_metrics.json              # Tuned LightGBM performance summary & parameters
│   ├── neighbourhood_metadata.json     # Borough/neighbourhood coordinates & median prices
│   └── pipeline.joblib                 # Serialized production pipeline (preprocessor + model)
├── reports/
│   └── figures/                        # High-resolution publication plots
│       ├── actual_vs_predicted_and_residuals.png
│       ├── borough_prices.png
│       ├── correlation_matrix.png
│       ├── feature_importance.png
│       ├── model_comparison.png
│       ├── nyc_map_price.png
│       ├── price_distribution.png
│       └── room_type_prices.png
├── src/
│   ├── __init__.py
│   ├── eda_plots.py                    # Script generating EDA visualization figures
│   ├── preprocess.py                   # Data cleaning, feature engineering & transformers
│   └── train.py                        # Model training, CV, hyperparameter tuning & export
└── tests/
    └── test_predictions.py             # Realistic inference validation test suite
```

---

## 🔬 Task 1: Data Analysis and Preparation

### 1.1 Dataset Understanding & Missing Value Imputation
The raw dataset comprises **48,895 listings** and **16 attributes**. Analysis of missing values revealed:
- `reviews_per_month` & `last_review`: **10,052 missing entries** (20.5%). Cross-referencing demonstrated that these occur strictly where `number_of_reviews == 0`. Imputing `reviews_per_month = 0.0` is domain-accurate.
- `name` (16 missing) & `host_name` (21 missing): Replaced with `'Unknown'`.
- Uninformative IDs (`id`, `host_id`) were dropped to prevent spurious pattern memorization.

### 1.2 Target Variable Transformation & Outlier Treatment
The raw price feature displayed extreme positive skewness ($\text{skew} > 15$) with values ranging from $\$0$ to $\$10,000$:
- 11 erroneous records with $\text{price} \le \$0$ were removed.
- Extreme luxury listings ($> \$1,000$, top 0.5% quantile) were excluded for training stability, preserving 99.5% of regular listings while protecting gradient updates from disproportionate distortion.
- **Log Transformation:** Applied $y = \log(1 + \text{price})$ to stabilize error variance and normalize residuals.

![Price Distribution](reports/figures/price_distribution.png)

### 1.3 Key Exploratory Insights
1. **Room Type Disparity:** Entire homes/apartments command a median nightly price of $\$160$, compared to $\$70$ for private rooms and $\$45$ for shared rooms.
2. **Borough Premiums:** Manhattan leads with an average nightly price of $\$196.90$, followed by Brooklyn ($\$124.40$), Queens ($\$99.50$), Staten Island ($\$98.50$), and Bronx ($\$87.50$).
3. **Geospatial Concentration:** Over 85% of listings are concentrated in Manhattan and Western Brooklyn, clustering around Midtown and downtown subway transit arteries.

| Room Type Boxplot | Borough Price Breakdown |
|:---:|:---:|
| ![Room Type Prices](reports/figures/room_type_prices.png) | ![Borough Prices](reports/figures/borough_prices.png) |

### 1.4 Geospatial Mapping
Plotting listings coordinates onto the New York City reference map (`New_York_City_.png`):

![NYC Map Price Overlay](reports/figures/nyc_map_price.png)

### 1.5 Feature Engineering
Custom transformer `AirbnbFeatureEngineer` constructed domain-specific attributes:
- `dist_to_midtown`: Haversine great-circle distance (km) to Times Square / Grand Central ($40.7580^\circ\text{N}, -73.9855^\circ\text{W}$). Proximity to Midtown has a strong negative correlation with price ($r = -0.38$).
- `dist_to_wall_st`: Distance (km) to the Financial District.
- `name_length`: Character length of listing headline (longer, descriptive titles correlate with higher rates).
- `minimum_nights_capped`: Capped at 30 days (NYC legal short-term rental threshold).
- `is_commercial_host`: Binary flag indicating property managers with $>1$ active listing.
- `has_reviews`: Binary indicator for listings with review history.
- `neighbourhood_grouped`: Top 25 high-density neighbourhoods preserved; sparse neighborhoods grouped as `'Other'`.

---

## 🤖 Task 2: Model Training, Evaluation & Diagnostics

### 2.1 Model Benchmark Comparison
Seven regression algorithms were trained on the training partition (**38,916 listings**) and evaluated on an independent test partition (**9,729 listings**):

| Model | Train $R^2$ (Log) | Test $R^2$ (Log) | Train MAE ($) | Test MAE ($) | Train RMSE ($) | Test RMSE ($) | Test $R^2$ (Price) | CV $R^2$ Mean | Fit Time (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost Regressor** | 0.6933 | 0.6518 | \$42.00 | \$45.18 | \$81.72 | \$87.83 | 0.4471 | 0.6311 | 0.63s |
| **LightGBM Regressor** | 0.6736 | 0.6501 | \$43.12 | \$45.30 | \$83.78 | \$88.04 | 0.4444 | 0.6292 | 3.37s |
| **Random Forest** | 0.7983 | 0.6469 | \$33.86 | \$45.32 | \$70.51 | \$87.87 | 0.4466 | 0.6237 | 3.44s |
| **Decision Tree** | 0.6411 | 0.6066 | \$44.99 | \$48.10 | \$86.45 | \$91.39 | 0.4014 | 0.5809 | 0.53s |
| **Linear Regression** | 0.5744 | 0.5856 | \$49.09 | \$49.26 | \$93.87 | \$94.04 | 0.3661 | 0.5732 | 0.12s |
| **Ridge Regression** | 0.5744 | 0.5854 | \$49.09 | \$49.26 | \$93.88 | \$94.06 | 0.3659 | 0.5732 | 0.04s |
| **Lasso Regression** | 0.5703 | 0.5793 | \$49.16 | \$49.39 | \$94.24 | \$94.56 | 0.3591 | 0.5694 | 1.34s |

![Model Benchmark Comparison](reports/figures/model_comparison.png)

### 2.2 Overfitting & Underfitting Diagnosis
- **Underfitting (Linear Models):** Linear, Ridge, and Lasso Regression exhibited high bias ($R^2 \approx 0.585$ on both train and test). They fail to capture complex non-linear geospatial boundaries between high-demand and low-demand neighborhoods.
- **Overfitting (Random Forest):** Unrestricted Random Forest attained Train $R^2 = 0.7983$ vs Test $R^2 = 0.6469$ ($\Delta = 0.151$). Deep unpruned trees overfit idiosyncratic listing quirks.
- **Optimal Balance (Gradient Boosted Trees):** LightGBM and XGBoost demonstrated minimal generalization gap ($\Delta \approx 0.02$) while achieving the lowest prediction errors.

### 2.3 Hyperparameter Tuning (Champion Model: LightGBM)
3-Fold `GridSearchCV` was performed across tree depth, learning rate, leaf count, and subsampling:
- **Best Hyperparameters:**
  - `n_estimators`: `250`
  - `learning_rate`: `0.08`
  - `num_leaves`: `45`
  - `subsample`: `0.8`
  - `colsample_bytree`: `0.8`
- **Final Tuned LightGBM Performance:**
  - **Test $R^2$ (Log Scale):** `0.6540`
  - **Test MAE:** `\$45.09`
  - **Test RMSE:** `\$87.59`
  - **Test $R^2$ (Original Price):** `0.4501`

| Actual vs Predicted & Residuals | Feature Importances |
|:---:|:---:|
| ![Residuals](reports/figures/actual_vs_predicted_and_residuals.png) | ![Feature Importance](reports/figures/feature_importance.png) |

### 2.4 Feature Importance Analysis
The top 5 drivers of Airbnb pricing in NYC:
1. `room_type_Entire home/apt` (+ strong positive influence)
2. `dist_to_midtown` (- strong negative influence with increasing distance)
3. `longitude` & `latitude` (specific neighborhood micro-locations)
4. `availability_365` (higher availability correlates with commercial properties)
5. `minimum_nights_capped` (extended stays receive per-night discounts)

---

## 💻 Task 3: Streamlit Web Application

The interactive web application (`app.py`) provides real-time property valuation:
- **Location Selector:** Interactive Borough & dynamically-filtered Neighbourhood dropdowns.
- **Coordinates Integration:** Auto-populates realistic latitude/longitude coordinates based on neighborhood centroids.
- **Scenario Presets:** 1-click test buttons for representative listings across all boroughs.
- **Uncertainty Range:** Generates estimated confidence bounds ($\pm \text{MAE}$) and price-tier badges.
- **Analytical Dashboards:** Built-in EDA visualizer and model diagnostic viewer.

### Realistic Test Cases
| Scenario | Borough | Room Type | Predicted Nightly Price | Valuation Tier |
|:---|:---|:---|:---:|:---:|
| **Midtown Luxury Loft** | Manhattan | Entire home/apt | **\$281.57** | Premium |
| **Williamsburg Boho Room** | Brooklyn | Private room | **\$94.81** | Moderate |
| **Astoria Budget Stay** | Queens | Private room | **\$73.79** | Budget Friendly |
| **Mott Haven Shared Space** | Bronx | Shared room | **\$43.13** | Budget Friendly |

---

## 📝 Task 4: Final Project Summary & Discussion

### 4.1 Summary of Deliverables
- **Jupyter Notebook (`Airbnb_Price_Prediction.ipynb`):** Fully executed end-to-end workflow containing data exploration, math formulations, model benchmarking, tuning, and plots.
- **Saved Pipeline (`models/pipeline.joblib`):** Production-ready serialized pipeline encapsulating feature engineering, scaling, one-hot encoding, and the tuned LightGBM regressor.
- **Web Interface (`app.py`):** Multi-tab Streamlit dashboard.
- **Requirements (`requirements.txt`):** Reproducible environment specifications.

### 4.2 System Limitations
1. **Unobserved Amenities:** Key amenities (swimming pool, private terrace, washer/dryer, high-speed Wi-Fi, elevator access) were not available in the tabular data.
2. **Temporal & Policy Changes:** The 2019 dataset reflects pre-pandemic pricing and does not account for New York City's Local Law 18 (enacted in 2023), which prohibited unhosted short-term rentals under 30 days.
3. **Seasonal Demand Surges:** Single-snapshot listing data lacks calendar dates, preventing the modeling of peak holiday spikes (New Year's Eve, UN General Assembly, Marathon weekend).

---

## 🚀 Quickstart: Running Locally

### 1. Clone the Repository
```bash
git clone <YOUR_GIT_REPO_URL>
cd 202618062_LAB04
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run Test Suite
```bash
python tests/test_predictions.py
```

---

## ☁️ Deployment to Streamlit Community Cloud

Deploying this app is completely free and takes less than 2 minutes:
1. Push this repository to your GitHub account.
2. Visit **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
3. Click **New app**, select your repository: `username/202618062_LAB04`.
4. Set Main file path to: `app.py`.
5. Click **Deploy!** — Streamlit automatically installs packages from `requirements.txt` and starts the app.
