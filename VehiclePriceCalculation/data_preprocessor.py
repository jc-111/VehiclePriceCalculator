import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from VehiclePriceCalculation.config import numerical_cols, categorical_cols, market_map

class DataPreprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.market_map = market_map

    def _map_market_model(self, df):
        df['market_model'] = 'Other'
        if 'model' in df.columns:
            for category, models in self.market_map.items():
                df.loc[df['model'].isin(models), 'market_model'] = category
        return df

    def clean(self):
        # Remove unnecessary columns
        if 'vin' in self.df.columns:
            self.df.drop(columns=['vin'], inplace=True)

        # Standard column renaming
        rename_map = {'make': 'brand', 'body': 'body_type', 'odometer': 'mileage'}
        self.df.rename(columns={k: v for k, v in rename_map.items() if k in self.df.columns}, inplace=True)

        # calculate the sales year
        if 'saledate' in self.df.columns:
            self.df['sale_year'] = self.df['saledate'].astype(str).str[11:15]
            self.df['sale_year'] = pd.to_numeric(self.df['sale_year'], errors='coerce').astype('Int64')
            print(f"Extracted sale_year from saledate. Range: {self.df['sale_year'].min()} ~ {self.df['sale_year'].max()}")

        # remove extreme mmr values
        if 'mmr' in self.df.columns:
            mmr_bounds = self.df['mmr'].quantile([0.02, 0.98])
            self.df = self.df[(self.df['mmr'] >= mmr_bounds.iloc[0]) & (self.df['mmr'] <= mmr_bounds.iloc[1])]
            print(f"MMR bounds: {mmr_bounds.iloc[0]} to {mmr_bounds.iloc[1]}")

        # remove extreme mileage values
        if 'mileage' in self.df.columns:
            mileage_bounds = self.df['mileage'].quantile([0.02, 0.98])
            self.df = self.df[(self.df['mileage'] >= mileage_bounds.iloc[0]) & (self.df['mileage'] <= mileage_bounds.iloc[1])]
            print(f"Mileage bounds: {mileage_bounds.iloc[0]} to {mileage_bounds.iloc[1]}")

        # remove extreme price values
        price_col = 'sellingprice' if 'sellingprice' in self.df.columns else 'selling_price'
        if price_col in self.df.columns:
            price_bounds = self.df[price_col].quantile([0.02, 0.98])
            self.df = self.df[(self.df[price_col] >= price_bounds.iloc[0]) & (self.df[price_col] <= price_bounds.iloc[1])]
            print(f"Price bounds: {price_bounds.iloc[0]} to {price_bounds.iloc[1]}")

        # remove unrealistic year/sale_year
        if 'year' in self.df.columns and 'sale_year' in self.df.columns:
            current_year = pd.Timestamp.now().year
            self.df = self.df[(self.df['sale_year'] >= self.df['year']) & (self.df['sale_year'] <= current_year)]
            print(f"Filtered records between year and {current_year}")

        # calculate car age
        if 'year' in self.df.columns and 'sale_year' in self.df.columns:
            self.df['car_age'] = self.df['sale_year'] - self.df['year']
            print(f"Created car_age column. Range: {self.df['car_age'].min()} ~ {self.df['car_age'].max()}")

        print(f"Data shape after cleaning: {self.df.shape}")
        return self

    def map_market_category(self):
        self.df = self._map_market_model(self.df)
        print(f"Mapped models to {self.df['market_model'].nunique()} market categories")
        return self

    def calculate_value_retention(self):
        if 'mmr' in self.df.columns and 'sellingprice' in self.df.columns:
            self.df['value_retention'] = self.df['sellingprice'] / self.df['mmr']
        elif 'mmr' in self.df.columns and 'selling_price' in self.df.columns:
            self.df['value_retention'] = self.df['selling_price'] / self.df['mmr']
        return self

    def feature_engineer(self):
        return self.map_market_category()

    def prepare(self):
        return self.clean().feature_engineer().calculate_value_retention().df

    def prepare_single_vehicle(self, vehicle_dict):
        df = pd.DataFrame([vehicle_dict])

        df['sale_year'] = pd.Timestamp.now().year
        df['car_age'] = df['sale_year'] - df['year']

        df = self._map_market_model(df)

        for col in self.numerical_cols:
            if col not in df.columns:
                df[col] = 0
        for col in self.categorical_cols:
            if col not in df.columns:
                df[col] = 'unknown'

        return df
