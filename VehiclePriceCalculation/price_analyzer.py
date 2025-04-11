
class PriceAnalyzer:
    def __init__(self, df):
        self.df = df
        self.brand_avg = df.groupby('brand')['value_retention'].mean().to_dict()
        self.model_avg = df.groupby('market_model')['value_retention'].mean().to_dict()
        self.age_avg = df.groupby('car_age')['value_retention'].mean().to_dict()

    def analyze_vehicle(self, vehicle_row, predicted_mmr):
        selling_price = vehicle_row.get('sellingprice', None)
        value_retention = selling_price / predicted_mmr if selling_price else None

        status = 'fair'
        if value_retention is not None:
            if value_retention < 0.9:
                status = 'underpriced'
            elif value_retention > 1.1:
                status = 'overpriced'

        return {
            'brand': vehicle_row.get('brand'),
            'model': vehicle_row.get('model'),
            'car_age': vehicle_row.get('car_age'),
            'selling_price': selling_price,
            'predicted_mmr': predicted_mmr,
            'value_retention': value_retention,
            'status': status,
            'recommended_range': (predicted_mmr * 0.9, predicted_mmr * 1.1)
        }
