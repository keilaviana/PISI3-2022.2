import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer

food = pd.read_csv("food.csv")
food['food_category_id'].fillna(0, inplace=True)
food['description'].fillna("Indisponível", inplace=True)

nutrient = pd.read_csv("nutrient.csv")
nutrient['nutrient_nbr'].fillna(0, inplace=True)

food_nutrient = pd.read_csv("food_nutrient.csv", low_memory=False)
media = food_nutrient['derivation_id'].mean()
food_nutrient['derivation_id'].fillna(round(media), inplace=True)
food_nutrient['data_points'].fillna(0, inplace=True)

food_dt = food[['fdc_id', 'description', 'food_category_id']]
nutrient_dt = nutrient[['id', 'name', 'unit_name', 'nutrient_nbr']]
food_nutrient_dt = food_nutrient[['id', 'fdc_id', 'nutrient_id', 'amount', 'derivation_id', 'data_points']]

features = ['id', 'fdc_id', 'nutrient_id', 'amount']
data_for_clustering = food_nutrient_dt[features]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_for_clustering)

model = KMeans(random_state=42)
visualizer = KElbowVisualizer(model, k=(1, 11))
visualizer.fit(scaled_data)
visualizer.show()
