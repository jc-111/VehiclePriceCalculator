#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

sns.set(style="whitegrid")



# In[2]:


def run_cohort_value_retention_analysis():
    print("\n=== Cohort-Based Value Retention Analysis ===")

    np.random.seed(42)
    n_samples = 1000
    years = np.random.randint(2010, 2020, n_samples)
    sale_years = np.array([min(2023, y + np.random.randint(0, 6)) for y in years])
    car_ages = sale_years - years

    base_prices = 30000 - 3000 * car_ages + np.random.normal(0, 2000, n_samples)
    mmr = base_prices * (1 + np.random.normal(0, 0.05, n_samples))
    retention_factor = np.random.normal(1.0, 0.1, n_samples)
    selling_prices = mmr * retention_factor
    brands = np.random.choice(['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes'], n_samples)

    models = []
    for brand in brands:
        if brand == 'Toyota':
            models.append(np.random.choice(['Camry', 'Corolla', 'RAV4']))
        elif brand == 'Honda':
            models.append(np.random.choice(['Civic', 'Accord', 'CR-V']))
        elif brand == 'Ford':
            models.append(np.random.choice(['F-150', 'Explorer', 'Focus']))
        elif brand == 'BMW':
            models.append(np.random.choice(['3 Series', '5 Series', 'X5']))
        else:
            models.append(np.random.choice(['C-Class', 'E-Class', 'GLC']))

    body_types = np.random.choice(['Sedan', 'SUV', 'Truck', 'Coupe'], n_samples)
    mileage = car_ages * 10000 + np.random.normal(0, 2000, n_samples)

    df = pd.DataFrame({
        'make': brands,
        'model': models,
        'year': years,
        'body': body_types,
        'odometer': mileage,
        'saledate': [f"01/01/{year}" for year in sale_years],
        'mmr': mmr,
        'sellingprice': selling_prices
    })

    df = df.rename(columns={'make': 'brand', 'odometer': 'mileage', 'body': 'body_type'})
    df['sale_year'] = df['saledate'].astype(str).str[6:10].astype(int)
    df['car_age'] = df['sale_year'] - df['year']
    df = df.dropna(subset=['mmr', 'sellingprice', 'car_age', 'year'])

    df['value_retention_ratio'] = (df['sellingprice'] / df['mmr']).round(4)
    df['cohort'] = df['year']

    cohort_data = df.groupby(['cohort', 'car_age'])['value_retention_ratio'].agg(['mean', 'count']).reset_index()
    cohort_pivot = cohort_data.pivot(index='cohort', columns='car_age', values='mean')
    count_pivot = cohort_data.pivot(index='cohort', columns='car_age', values='count')

    valid_cohorts = count_pivot.index[count_pivot.sum(axis=1) >= 30]
    valid_ages = count_pivot.columns[count_pivot.sum() >= 30]
    filtered_pivot = cohort_pivot.loc[valid_cohorts, valid_ages]

    # Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(filtered_pivot, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Vehicle Value Retention by Manufacturing Year Cohort')
    plt.xlabel('Vehicle Age (years)')
    plt.ylabel('Manufacturing Year Cohort')
    plt.tight_layout()
    plt.show()

    # Line chart for recent cohorts
    plt.figure(figsize=(12, 6))
    for year in sorted(filtered_pivot.index)[-5:]:
        if year in filtered_pivot.index:
            plt.plot(filtered_pivot.columns, filtered_pivot.loc[year], marker='o', label=f'Year {year}')
    plt.title('Value Retention Over Time by Manufacturing Year')
    plt.xlabel('Vehicle Age (years)')
    plt.ylabel('Value Retention Ratio')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Print average by age
    avg_retention = filtered_pivot.mean(axis=0)
    print("\nAverage Value Retention Ratio by Vehicle Age:")
    for age, retention in avg_retention.items():
        print(f"Age {age} years: {retention:.2f}")

    best_years = filtered_pivot.mean(axis=1).sort_values(ascending=False)
    print("\nTop Cohort Years by Retention:")
    for year, retention in best_years.head(5).items():
        print(f"Year {year}: {retention:.2f}")

    # Model summary
    df['meter_value_ratio'] = df['mileage'] / df['sellingprice']
    model_summary = df.groupby(['brand', 'model']).agg(
        avg_retention=('value_retention_ratio', 'mean'),
        avg_meter_value=('meter_value_ratio', 'mean'),
        count=('value_retention_ratio', 'count')
    ).reset_index()
    
    filtered_models = model_summary[model_summary['count'] >= 30]
    top_models = filtered_models.sort_values(by='avg_retention', ascending=False)
    print("\nTop 10 Value-Retaining Models:")
    print(top_models.head(10))

    # Body type comparison
    body_retention = df.groupby('body_type')['value_retention_ratio'].agg(['mean', 'count']).reset_index()
    body_retention = body_retention[body_retention['count'] >= 30].sort_values('mean', ascending=False)
    print("\nAverage Value Retention by Vehicle Type:")
    print(body_retention)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=body_retention, x='body_type', y='mean')
    plt.axhline(y=1.0, color='red', linestyle='--')
    plt.title('Average Value Retention by Vehicle Type')
    plt.xlabel('Vehicle Type')
    plt.ylabel('Retention Ratio')
    plt.tight_layout()
    plt.show()

    # Regression
    model_df = df[df['car_age'] <= 10]
    regression = smf.ols('value_retention_ratio ~ car_age + cohort', data=model_df).fit()
    print("\nRegression Summary: Value Retention ~ Car Age + Cohort")
    print(regression.summary())

    return df, filtered_pivot, top_models, body_retention


# In[3]:


df, filtered_pivot, top_models, body_retention = run_cohort_value_retention_analysis()


