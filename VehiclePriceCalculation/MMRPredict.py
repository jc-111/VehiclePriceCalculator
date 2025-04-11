import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class MMRPredictor:
    def __init__(self, data, model_type='xgboost'):
        self.df = data.copy()

        self.numerical_cols = ['condition', 'car_age', 'mileage']
        self.categorical_cols = ['brand', 'market_model', 'body_type', 'transmission', 'state', 'interior', 'color']
        self.mmr = 'mmr'

        self.model_type = model_type.lower()
        self.model = None
        self.encoders = {}  # Each categorical column has its own encoder
        self.scaler = StandardScaler()

        if self.model_type in ['xgboost', 'gradient_boosting']:
            self.model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError("Invalid model_type. Choose from: 'xgboost', 'gradient_boosting', 'random_forest'")

    def preprocess_data(self):
        # Drop rows with missing mmr
        self.df = self.df.dropna(subset=[self.mmr])

        # Encode categorical columns (one LabelEncoder per column)
        for col in self.categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le

        # Scale numerical columns
        self.df[self.numerical_cols] = self.scaler.fit_transform(self.df[self.numerical_cols])

        # Keep necessary columns
        self.df = self.df[self.numerical_cols + self.categorical_cols + [self.mmr]]

    def train_evaluate_optimize(self):
        X = self.df.drop(columns=[self.mmr])
        y = self.df[self.mmr]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training model...")
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"Initial RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.2f}")

    def predict(self, vehicle_df):
        df = vehicle_df.copy()

        # Handle categorical columns safely using stored encoders
        for col in self.categorical_cols:
            if col in df.columns:
                le = self.encoders.get(col)
                if le:
                    df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        # Scale numerical columns
        df[self.numerical_cols] = self.scaler.transform(df[self.numerical_cols])

        return self.model.predict(df[self.numerical_cols + self.categorical_cols])
