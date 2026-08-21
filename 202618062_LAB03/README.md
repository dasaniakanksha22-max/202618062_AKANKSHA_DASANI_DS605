# DS605 – Lab Assignment 3: Scikit-learn Preprocessing & Model Evaluation

**Name:** Akanksha Dasani
**ID:** 202618062
**Dataset:** [Kaggle – Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (`hotel_bookings.csv`)

## Objective
Build and compare two Scikit-learn preprocessing pipelines and evaluate two classification
models (Logistic Regression, Decision Tree) on predicting booking cancellations (`is_canceled`).

## Preprocessing Choices
- **Dropped:** `company` (>90% missing values — imputing would fabricate most of the column).
- **Removed (leakage):** `reservation_status`, `reservation_status_date` — these directly
  encode the outcome of the booking and would leak the label into the features.
- **Outliers:** Extreme values in `adr` (negative or absurdly high, e.g. >5000) and `lead_time`
  (>600 days) were removed as clear data-entry outliers.
- **Numerical features:** `KNNImputer(n_neighbors=5)` for missing values, then:
  - Pipeline A → `StandardScaler`
  - Pipeline B → `MinMaxScaler`
- **Categorical features:** `SimpleImputer(strategy="most_frequent")` +
  `OneHotEncoder(handle_unknown="ignore")`.
- All preprocessing is wrapped in a `ColumnTransformer` + `Pipeline`, fit only on the training
  split (`train_test_split(test_size=0.2, stratify=y, random_state=42)`) to avoid data leakage.

## Models
- `LogisticRegression(max_iter=1000)`
- `DecisionTreeClassifier(random_state=42)`

Each trained on both Pipeline A and Pipeline B → 4 model–pipeline combinations total.

## Final Observations
See Task 6 in `lab03.ipynb` for the full write-up, based on the results table
and confusion matrices produced in the notebook.

## Repository Contents
- `lab03.ipynb` – full runnable notebook (data cleaning, both pipelines, both
  models, evaluation).
- `hotel_bookings.csv` – dataset used for modeling.
- `README.md` – this file.
