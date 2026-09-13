"""
app.py - Streamlit Web Application for Airbnb Price Prediction
DS605: Fundamentals of Machine Learning - Lab Assignment 4
Author: Akanksha Dasani (Student ID: 202618062)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Add src to system path for custom transformer unpickling
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocess import AirbnbFeatureEngineer  # Ensure class is registered for joblib unpickling

# Set Page Config
st.set_page_config(
    page_title="NYC Airbnb Price Predictor | DS605 Lab 4",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 10px;
    }
    .prediction-price {
        font-size: 3rem;
        font-weight: 800;
        margin: 10px 0;
        color: #FBBF24;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 5px;
    }
    .badge-luxury { background-color: #EF4444; color: white; }
    .badge-premium { background-color: #F59E0B; color: white; }
    .badge-moderate { background-color: #10B981; color: white; }
    .badge-budget { background-color: #3B82F6; color: white; }
    </style>
""", unsafe_allow_html=True)

# Cache Model and Metadata Loading
@st.cache_resource
def load_pipeline():
    pipeline_path = os.path.join(CURRENT_DIR, 'models', 'pipeline.joblib')
    if not os.path.exists(pipeline_path):
        st.error(f"Model pipeline not found at {pipeline_path}. Please train the model first.")
        return None
    return joblib.load(pipeline_path)

@st.cache_data
def load_metadata():
    meta_path = os.path.join(CURRENT_DIR, 'models', 'neighbourhood_metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return {}

@st.cache_data
def load_benchmark_metrics():
    metrics_path = os.path.join(CURRENT_DIR, 'models', 'model_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return {}

@st.cache_data
def load_benchmark_csv():
    csv_path = os.path.join(CURRENT_DIR, 'models', 'model_benchmark.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

pipeline = load_pipeline()
metadata = load_metadata()
metrics_summary = load_benchmark_metrics()
benchmark_df = load_benchmark_csv()

# Sidebar: Project Info & Navigation
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/airbnb.png", width=110)
    st.title("Airbnb Predictor")
    st.caption("DS605: Fundamentals of ML - Lab 4")
    st.markdown("---")
    
    st.markdown("### 📋 Quick Presets")
    st.caption("Click to load realistic NYC listing scenarios:")
    preset = st.radio(
        "Select Scenario:",
        ["Custom Input", "Midtown Manhattan Entire Apt", "Williamsburg Brooklyn Private Room", "Astoria Queens Budget Stay", "Harlem Historic Townhouse"]
    )
    
    st.markdown("---")
    st.markdown("### 👤 Author Information")
    st.markdown("**Akanksha Dasani**")
    st.markdown("Student ID: `202618062`")
    st.markdown("Course: *DS605 ML Lab Assignment 4*")
    st.markdown("Dataset: *Kaggle AB_NYC_2019*")

# Main Title Section
st.markdown('<div class="main-header">🗽 NYC Airbnb Nightly Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">End-to-end Machine Learning web application powered by a tuned LightGBM regressor pipeline trained on NYC Open Data.</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Price Prediction", 
    "📊 Exploratory Data Analysis", 
    "🤖 Model Benchmark & Performance", 
    "📖 About & Documentation"
])

# Define Presets
preset_values = {
    "Custom Input": {
        "borough": "Manhattan", "neighbourhood": "Midtown", "room_type": "Entire home/apt",
        "name": "Spacious Cozy Midtown Apartment near Times Square",
        "min_nights": 2, "reviews": 25, "rpm": 1.5, "host_count": 1, "avail": 180
    },
    "Midtown Manhattan Entire Apt": {
        "borough": "Manhattan", "neighbourhood": "Midtown", "room_type": "Entire home/apt",
        "name": "Luxury Modern Midtown Loft near Central Park",
        "min_nights": 3, "reviews": 45, "rpm": 2.1, "host_count": 2, "avail": 240
    },
    "Williamsburg Brooklyn Private Room": {
        "borough": "Brooklyn", "neighbourhood": "Williamsburg", "room_type": "Private room",
        "name": "Cozy Bohemian Sunny Room in Trendy Williamsburg",
        "min_nights": 2, "reviews": 85, "rpm": 3.4, "host_count": 1, "avail": 90
    },
    "Astoria Queens Budget Stay": {
        "borough": "Queens", "neighbourhood": "Astoria", "room_type": "Private room",
        "name": "Clean and Quiet Sunny Room 15 mins to Manhattan",
        "min_nights": 1, "reviews": 12, "rpm": 0.8, "host_count": 1, "avail": 300
    },
    "Harlem Historic Townhouse": {
        "borough": "Manhattan", "neighbourhood": "Harlem", "room_type": "Entire home/apt",
        "name": "Historic Brownstone Full Floor Flat in Central Harlem",
        "min_nights": 4, "reviews": 60, "rpm": 1.9, "host_count": 1, "avail": 120
    }
}

active_preset = preset_values[preset]

# TAB 1: PREDICTION
with tab1:
    col_left, col_right = st.columns([1.1, 0.9], gap="large")
    
    with col_left:
        st.markdown("#### 1. Location & Listing Details")
        borough_options = list(metadata.keys()) if metadata else ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
        default_borough_idx = borough_options.index(active_preset["borough"]) if active_preset["borough"] in borough_options else 0
        
        c1, c2 = st.columns(2)
        with c1:
            selected_borough = st.selectbox("Borough (Neighbourhood Group):", borough_options, index=default_borough_idx)
        
        available_neighbourhoods = list(metadata.get(selected_borough, {}).keys()) if selected_borough in metadata else ['Midtown']
        default_neigh_idx = available_neighbourhoods.index(active_preset["neighbourhood"]) if active_preset["neighbourhood"] in available_neighbourhoods else 0
        
        with c2:
            selected_neighbourhood = st.selectbox("Neighbourhood:", available_neighbourhoods, index=default_neigh_idx)

        c3, c4 = st.columns(2)
        with c3:
            room_options = ['Entire home/apt', 'Private room', 'Shared room']
            selected_room = st.selectbox("Room Type:", room_options, index=room_options.index(active_preset["room_type"]))
        with c4:
            listing_name = st.text_input("Listing Title / Headline:", value=active_preset["name"])

        st.markdown("#### 2. Booking & Host Policies")
        c5, c6, c7 = st.columns(3)
        with c5:
            min_nights = st.number_input("Minimum Nights:", min_value=1, max_value=365, value=active_preset["min_nights"])
        with c6:
            host_count = st.number_input("Host Total Listings:", min_value=1, max_value=350, value=active_preset["host_count"])
        with c7:
            avail_365 = st.slider("Availability (Days / Year):", min_value=0, max_value=365, value=active_preset["avail"])

        st.markdown("#### 3. Review History")
        c8, c9 = st.columns(2)
        with c8:
            num_reviews = st.number_input("Total Number of Reviews:", min_value=0, max_value=700, value=active_preset["reviews"])
        with c9:
            rpm = st.number_input("Reviews Per Month:", min_value=0.0, max_value=60.0, value=float(active_preset["rpm"]), step=0.1)

        # Coordinate Autofill
        neigh_info = metadata.get(selected_borough, {}).get(selected_neighbourhood, {'lat': 40.7580, 'lon': -73.9855})
        with st.expander("📍 Geographic Coordinates (Auto-populated from Neighbourhood)"):
            c_lat, c_lon = st.columns(2)
            with c_lat:
                lat = st.number_input("Latitude:", value=float(neigh_info['lat']), format="%.5f")
            with c_lon:
                lon = st.number_input("Longitude:", value=float(neigh_info['lon']), format="%.5f")

    with col_right:
        st.markdown("#### 🔮 Estimated Valuation")
        
        if pipeline is None:
            st.warning("Model pipeline is not loaded.")
        else:
            # Prepare Input DataFrame
            input_df = pd.DataFrame([{
                'neighbourhood_group': selected_borough,
                'neighbourhood': selected_neighbourhood,
                'room_type': selected_room,
                'name': listing_name,
                'latitude': lat,
                'longitude': lon,
                'minimum_nights': min_nights,
                'number_of_reviews': num_reviews,
                'reviews_per_month': rpm,
                'calculated_host_listings_count': host_count,
                'availability_365': avail_365
            }])

            # Predict
            pred_log = pipeline.predict(input_df)[0]
            pred_price = max(10.0, float(np.expm1(pred_log)))

            # Uncertainty interval based on test MAE (~$45.09)
            mae_est = metrics_summary.get('test_metrics', {}).get('MAE ($)', 45.09)
            lower_bound = max(10.0, pred_price - (0.8 * mae_est))
            upper_bound = pred_price + (0.8 * mae_est)

            # Price Tier
            if pred_price < 80:
                tier = "Budget Friendly"
                badge_class = "badge-budget"
            elif pred_price < 180:
                tier = "Moderate Comfort"
                badge_class = "badge-moderate"
            elif pred_price < 350:
                tier = "Premium / High-End"
                badge_class = "badge-premium"
            else:
                tier = "Ultra Luxury"
                badge_class = "badge-luxury"

            st.markdown(f"""
                <div class="prediction-card">
                    <div style="font-size: 1.1rem; opacity: 0.9;">Estimated Nightly Price</div>
                    <div class="prediction-price">${pred_price:.2f}</div>
                    <div style="font-size: 0.95rem; opacity: 0.85;">Expected Range: <b>${lower_bound:.0f} - ${upper_bound:.0f}</b> / night</div>
                    <div class="badge {badge_class}">{tier}</div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("##### 📍 Location Context on NYC Map")
            map_data = pd.DataFrame([{'lat': lat, 'lon': lon}])
            st.map(map_data, zoom=12)

            st.markdown("##### 💡 Key Pricing Influencers for this Listing:")
            col_inf1, col_inf2 = st.columns(2)
            with col_inf1:
                st.metric("Borough Median", f"${neigh_info.get('median_price', 120):.0f}")
                st.metric("Room Privacy Premium", "High" if "Entire" in selected_room else "Standard")
            with col_inf2:
                # Approximate distance to Midtown Times Square
                dist = np.sqrt((lat - 40.7580)**2 + (lon - (-73.9855))**2) * 111.0
                st.metric("Distance to Midtown", f"{dist:.1f} km")
                st.metric("Commercial Host", "Yes" if host_count > 1 else "No")

# TAB 2: EDA VISUALIZATIONS
with tab2:
    st.markdown("### 📊 Exploratory Data Analysis & Patterns (Task 1)")
    st.write("Summary of key patterns identified in the Kaggle NYC Airbnb dataset (48,895 listings):")

    c_eda1, c_eda2 = st.columns(2)
    with c_eda1:
        st.markdown("#### 1. Price Distribution & Log Transformation")
        st.caption(r"Raw price exhibits severe positive skewness with high outliers ($10,000). Applying $\log(1 + \text{price})$ stabilizes variance and normalizes errors.")
        dist_img_path = os.path.join(CURRENT_DIR, 'reports', 'figures', 'price_distribution.png')
        if os.path.exists(dist_img_path):
            st.image(dist_img_path, use_container_width=True)
    
    with c_eda2:
        st.markdown("#### 2. Impact of Room Type on Pricing")
        st.caption("Entire homes command more than double the median rate of private rooms, while shared rooms are the most budget-conscious.")
        room_img_path = os.path.join(CURRENT_DIR, 'reports', 'figures', 'room_type_prices.png')
        if os.path.exists(room_img_path):
            st.image(room_img_path, use_container_width=True)

    st.markdown("---")
    c_eda3, c_eda4 = st.columns(2)
    with c_eda3:
        st.markdown("#### 3. Average Price by NYC Borough")
        st.caption("Manhattan leads with an average nightly price exceeding $190, followed by Brooklyn ($124), Queens, Staten Island, and the Bronx.")
        borough_img_path = os.path.join(CURRENT_DIR, 'reports', 'figures', 'borough_prices.png')
        if os.path.exists(borough_img_path):
            st.image(borough_img_path, use_container_width=True)

    with c_eda4:
        st.markdown("#### 4. Feature Correlation Matrix")
        st.caption("Engineered distance to Midtown Manhattan shows a strong inverse correlation with listing prices.")
        corr_img_path = os.path.join(CURRENT_DIR, 'reports', 'figures', 'correlation_matrix.png')
        if os.path.exists(corr_img_path):
            st.image(corr_img_path, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 5. NYC Listings Geospatial Map Overlay")
    st.caption("Visualizing listings across all five boroughs overlaid directly on the NYC reference map:")
    map_img_path = os.path.join(CURRENT_DIR, 'reports', 'figures', 'nyc_map_price.png')
    if os.path.exists(map_img_path):
        st.image(map_img_path, use_container_width=True)

# TAB 3: MODEL PERFORMANCE & BENCHMARK
with tab3:
    st.markdown("### 🤖 Regression Model Comparison & Evaluation (Task 2)")
    
    if benchmark_df is not None:
        st.markdown("#### 🏆 Benchmark Table (Evaluated on 9,729 Independent Test Listings)")
        st.dataframe(
            benchmark_df.style.highlight_max(subset=['Test R2 (Log)', 'Test R2 (Price)', 'CV R2 Mean'], color='#D1FAE5')
                             .highlight_min(subset=['Test MAE ($)', 'Test RMSE ($)'], color='#D1FAE5'),
            use_container_width=True
        )

    c_bench1, c_bench2 = st.columns(2)
    with c_bench1:
        st.markdown("#### Model Comparison: Train vs Test $R^2$")
        comp_img_path = os.path.join(CURRENT_DIR, 'reports', 'figures', 'model_comparison.png')
        if os.path.exists(comp_img_path):
            st.image(comp_img_path, use_container_width=True)

    with c_bench2:
        st.markdown("#### Feature Importances (Tuned LightGBM)")
        feat_img_path = os.path.join(CURRENT_DIR, 'reports', 'figures', 'feature_importance.png')
        if os.path.exists(feat_img_path):
            st.image(feat_img_path, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔍 Overfitting / Underfitting Diagnostics & Residual Analysis")
    pred_res_path = os.path.join(CURRENT_DIR, 'reports', 'figures', 'actual_vs_predicted_and_residuals.png')
    if os.path.exists(pred_res_path):
        st.image(pred_res_path, use_container_width=True)

    st.markdown(r"""
    **Key Evaluation Observations:**
    1. **Underfitting in Linear Models:** Linear Regression, Ridge, and Lasso achieved $R^2 \approx 0.585$ on test data and struggled to capture complex geospatial non-linearities.
    2. **Overfitting in Deep Trees:** Unconstrained Random Forest achieved $R^2 = 0.798$ on training data but dropped to $0.647$ on test data, demonstrating significant variance.
    3. **Optimal Balance with Tuned LightGBM:** Achieved Test $R^2 = 0.6540$, Test MAE = \$45.09, and Test RMSE = \$87.59 with controlled sub-sampling and tree depth regularization.
    """)

# TAB 4: ABOUT & DOCUMENTATION
with tab4:
    st.markdown("### 📖 Project Summary & Submission Details (Task 4)")
    st.markdown("""
    #### 🎓 Course & Lab Assignment
    - **Course**: DS605: Fundamentals of Machine Learning
    - **Assignment**: Lab Assignment - 4 (End-to-End Machine Learning Project: Airbnb Price Prediction)
    - **Student**: Akanksha Dasani (ID: `202618062`)

    #### 🎯 Objectives Completed
    1. **Task 1 - Data Analysis and Preparation**: Cleaned missing values, filtered extreme outliers, handled target skewness via log transformation, and engineered domain features (distance to Midtown/Wall St, listing name length, commercial host indicator, capped minimum nights).
    2. **Task 2 - Model Training and Evaluation**: Trained 7 regression models (Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, XGBoost, LightGBM), tuned the champion LightGBM regressor using 3-fold cross-validation, analyzed overfitting/underfitting, and exported `pipeline.joblib`.
    3. **Task 3 - Streamlit Web Application**: Developed this interactive multi-tab application with realistic scenario presets, dynamic coordinates, price confidence ranges, and EDA dashboards.
    4. **Task 4 - Final Project Summary**: Documented methodology, benchmarks, and real-world system limitations.

    #### ⚠️ System Limitations & Future Scope
    - **Temporal Seasonality**: The dataset captures 2019 cross-sectional data; seasonal holiday surges and post-2020 regulatory changes in NYC (Local Law 18) are not represented.
    - **Amenity Data Absence**: Granular amenities (e.g. pool, elevator, AC, washer/dryer, view) were not present in the tabular schema but heavily influence luxury pricing.
    - **Text Sentiment**: While name length was included, full NLP processing (BERT / TF-IDF embeddings) on listing descriptions could extract additional price signals.
    """)
