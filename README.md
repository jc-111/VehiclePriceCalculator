# Vehicle Price Calculator

A comprehensive Python package to help vehicle dealers analyze whether cars are underpriced or overpriced based on Manheim Market Report (MMR) values and value retention factors.

## Overview

The Vehicle Price Calculator analyzes vehicles to determine if they're priced correctly relative to their market value. It uses machine learning to predict MMR values when not available and calculates value retention metrics to provide pricing recommendations.

Key features:
- Market segment for general car models & MMR prediction using machine learning models
- Vsual aanalytics including MMR distribution, car age churn analysis & value retention analysis


## Installation

### Prerequisites
- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn

### Setup
1. Clone this repository:
```bash
git clone https://github.com/jc-111/VehiclePriceCalculator
cd vehicle-price-calculator
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start
Locate file **main.py** and run provided examples for sing car and batch data analysis.

### Run your own example

```python
from VehiclePriceCalculation.data_preprocessor import DataPreprocessor
from VehiclePriceCalculation.MMRPredict import MMRPredictor
from VehiclePriceCalculation.price_analyzer import PriceAnalyzer
from cohort_value_retention_analysis import run_cohort_value_retention_analysis

# Initialize with training data
calculator = VehiclePriceCalculator(data_path='car_prices.csv')

# Train the model
preprocessor = DataPreprocessor(df)
df_clean = preprocessor.prepare()
model = MMRPredictor(df_clean, model_type='xgboost'/'randomforest')
model.preprocess_data()
model.train_evaluate_optimize()

# Analyze a single vehicle
vehicle = {
    'brand': brand_name,
    'model': model_name,
    'year': sale_year,
    'mileage': mileage,
    'condition': xxx,
    'body_type': xxx,
    'selling_price': resale_price
}

analysis = calculator.analyze_vehicle(vehicle)
print(f"Price Status: {analysis['price_status']}")
print(f"Recommended Price Range: ${analysis['min_fair_price']:.2f} - ${analysis['max_fair_price']:.2f}")

# Analyze an entire inventory
import pandas as pd
inventory = pd.read_csv('your_inventory.csv')
analyzed_inventory = calculator.analyze_inventory(inventory)
```

### Data Format

The package expects vehicle data with some or all of these columns:
- `brand` or `make`: Vehicle brands
- `model`: Vehicle model name
- `year`: Vehicle manufacturing year
- `mileage` or `odometer`: Vehicle mileage
- `condition`: Vehicle condition (numeric: 1-5)
- `body_type` or `body`: Vehicle body style
- `sellingprice` or `selling_price`: Current selling price
- `mmr`: Manheim Market Report value (if available) -- vehicle's current market value (in dollars)


## Price Status Criteria

The system classifies vehicles based on value retention:
- **Value Retention**: selling price / MMR
- **Recommended Price Range**: [value retention * 0.9, value retention * 1.1]
- **Underpriced**: Value retention < 0.9
- **Fair Price**: Value retention between 0.9 and 1.1
- **Overpriced**: Value retention > 1.1

## Project Structure

```
VehiclePriceCalculator/
├── requirements.txt
│
├── VehiclePriceCalculation/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessor.py
│   ├── MMRPredict.py
│   └── price_analyzer.py
│
├── data/
│   └── car_prices.csv
│
├── dataviz/
│   └── plots generated for visual analytics
│
├── cohort_value_retention_analysis.py
|
├── main.py
|
├── test.py
|
└── README.md
```

## Troubleshooting

### Common Issues

1. **Missing columns**: Make sure your data contains the required columns.
2. **Data format**: Ensure categorical columns are properly encoded.
3. **NaN values**: Handle missing values in your data before analysis.

### Error Messages

- "Column X not found in DataFrame": You're missing a required column.
- "Error predicting MMR": The model failed to predict an MMR value.
- "Cannot calculate value_retention": Missing required columns to calculate value retention.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements
- The original dataset (car_prices.csv) is sourced from kaggle dataset [Vehicle Sales Data](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data/data)
- This package uses Random Forest and XGBoost libraries for machine learning models
- Vehicle price analysis methodology based on automotive industry standards

## Author
Yifei Shi, Ying-Jen Chiang, Ziqi Li, Yiqian Ning

## Version History
* 0.1
    * Initial Release
