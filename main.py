import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from VehiclePriceCalculation.data_preprocessor import DataPreprocessor
from VehiclePriceCalculation.MMRPredict import MMRPredictor
from VehiclePriceCalculation.price_analyzer import PriceAnalyzer
from cohort_value_retention_analysis import run_cohort_value_retention_analysis

# === 1. Dataset Overview ===
print("=== [1] Dataset Overview ===")
df = pd.read_csv("data/car_prices.csv")

print(f"Original shape: {df.shape}")
print(f"Original columns: {list(df.columns)}\n")

print("Missing values per column:")
print(df.isnull().sum())

print("\nBasic descriptive statistics:")
print(df.describe())

if 'saledate' in df.columns:
    years = df['saledate'].astype(str).str[11:15]
    print(f"\nSale year range (from saledate): {years.min()} ~ {years.max()}")

print("\nAt this stage, we do not yet know which features are most important. We will revisit this after model training.")

# === 2. Data Cleaning and Feature Engineering ===
print("\n=== [2] Cleaned and Engineered Data Overview ===")

preprocessor = DataPreprocessor(df)
df_clean = preprocessor.prepare()

print(f"Cleaned shape: {df_clean.shape}")
print(f"Cleaned columns: {list(df_clean.columns)}")

print("\nFeature Engineering Summary:")
print("- Outliers were removed using the 2nd to 98th percentiles.")
print("- Created features: 'car_age', 'market_model', and 'value_retention'")
print("- Mapped models into higher-level market categories.")
print(f"- Records retained after cleaning: {len(df_clean)} out of {len(df)} ({len(df_clean)/len(df)*100:.1f}%)")

# Create output folder if needed
os.makedirs("dataviz", exist_ok=True)

# Plot key distributions
for col in ['car_age', 'mileage', 'condition']:
    if col in df_clean.columns:
        plt.figure()
        sns.histplot(df_clean[col], bins=30, kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"dataviz/{col}_distribution.png")
        plt.close()

# Value retention
if 'value_retention' in df_clean.columns:
    plt.figure()
    sns.histplot(df_clean['value_retention'], bins=30, kde=True)
    plt.title("Distribution of Value Retention")
    plt.savefig("dataviz/value_retention_distribution.png")
    plt.close()

# === 3. MMR Model Training and Performance ===
print("\n=== [3] MMR Prediction Model Training and Performance ===")
model = MMRPredictor(df_clean, model_type='xgboost')
model.preprocess_data()
model.train_evaluate_optimize()

print("\nModel Inputs:")
print("- Features used:", model.numerical_cols + model.categorical_cols)
print("- Preprocessing: Categorical encoding and numerical standardization applied.")
print("- Train/test split: 80/20")

# Feature importance
try:
    importances = model.model.feature_importances_
    feature_names = model.df.drop(columns=['mmr']).columns

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=feature_names)
    plt.title("Feature Importance (from XGBoost)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("dataviz/feature_importance.png")
    plt.close()

    print("Feature importance plot saved to 'dataviz/feature_importance.png'")
except Exception as e:
    print(f"Could not generate feature importance plot: {e}")

# === 4. Single Vehicle Price Analysis ===
print("\n=== [4] Example: Single Vehicle Price Analysis ===")

example_vehicles = [
    {
        'year': 2015,
        'brand': 'Toyota',
        'model': 'Camry',
        'condition': 38,
        'mileage': 60000,
        'body_type': 'sedan',
        'transmission': 'automatic',
        'state': 'CA',
        'interior': 'black',
        'color': 'white',
        'sellingprice': 12000
    },
    {
        'year': 2014,
        'brand': 'BMW',
        'model': '3 Series',
        'condition': 40,
        'mileage': 45000,
        'body_type': 'sedan',
        'transmission': 'automatic',
        'state': 'NY',
        'interior': 'beige',
        'color': 'black',
        'sellingprice': 16000
    }
]

analyzer = PriceAnalyzer(df_clean)

for idx, vehicle in enumerate(example_vehicles, start=1):
    vehicle_df = preprocessor.prepare_single_vehicle(vehicle)
    predicted_mmr = model.predict(vehicle_df)[0]
    result = analyzer.analyze_vehicle(vehicle_df.iloc[0], predicted_mmr)

    print(f"\n--- Vehicle #{idx} ---")
    for k, v in vehicle.items():
        print(f"  {k}: {v}")

    print(f"  Predicted MMR: ${result['predicted_mmr']:.2f}")
    print(f"  Value Retention: {result['value_retention']:.2f}")
    print(f"  Status: {result['status']}")
    print(f"  Recommended Price Range: ${result['recommended_range'][0]:.2f} - ${result['recommended_range'][1]:.2f}")

# === 5. Inventory-Level Analysis ===
print("\n=== [5] Inventory-Wide Price Status Analysis ===")

# Copy and encode categorical columns using trained encoders
X_all = df_clean[model.numerical_cols + model.categorical_cols].copy()

for col in model.categorical_cols:
    if col in X_all.columns:
        le = model.encoders.get(col)
        if le:
            X_all[col] = X_all[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# Scale numerical columns using trained scaler
X_all[model.numerical_cols] = model.scaler.transform(X_all[model.numerical_cols])

# Predict all at once
y_pred_all = model.model.predict(X_all)

# Analyze vehicle status for each row
status_list = []
for row, pred in zip(df_clean.itertuples(index=False), y_pred_all):
    row_dict = row._asdict()
    analysis = analyzer.analyze_vehicle(pd.Series(row_dict), pred)
    status_list.append(analysis['status'])

# Count and plot
status_series = pd.Series(status_list)
status_counts = status_series.value_counts()
print("\nPrice Status Distribution:")
print(status_counts)

plt.figure(figsize=(8, 5))
sns.barplot(x=status_counts.index, y=status_counts.values)
plt.title("Price Status Distribution in Inventory")
plt.xlabel("Status")
plt.ylabel("Vehicle Count")
plt.tight_layout()
plt.savefig("dataviz/inventory_price_status_distribution.png")
plt.close()

# === 6. Cohort-Based Retention Module ===
print("\n=== [6] Cohort-Based Retention Module ===")
run_cohort_value_retention_analysis()

# === 7. Summary and Observations ===
print("\n=== [7] Summary and Observations ===")
print("- The dataset included various missing and extreme values, which were cleaned through filtering and quantile-based clipping.")
print("- Key engineered features include car_age (from year and saledate), market_model category, and value_retention.")
print("- The MMR prediction model (XGBoost) was trained with encoded categorical and standardized numeric inputs, and tested on 20% of the data.")
print("- RMSE and R² were used to evaluate performance. Feature importance suggests condition, mileage, and car_age are top predictors.")
print("- Individual vehicle pricing can be assessed for fairness based on value retention against predicted MMR.")
print("- Full inventory analysis reveals the distribution of underpriced, fair, and overpriced listings.")
print("- Visual summaries and charts are available in the 'dataviz/' folder.")

