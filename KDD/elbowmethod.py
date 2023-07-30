import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer

food = pd.read_csv("f.csv")
food['food_category_id'].fillna(0, inplace=True)
food['description'].fillna("Indisponível", inplace=True)

nutrient = pd.read_csv("n.csv")
nutrient['nutrient_nbr'].fillna(0, inplace=True)

food_nutrient = pd.read_csv("fn.csv", low_memory=False)
media = food_nutrient['derivation_id'].mean()
food_nutrient['derivation_id'].fillna(round(media), inplace=True)
food_nutrient['data_points'].fillna(0, inplace=True)

food_dt = food[['fdc_id', 'description', 'food_category_id']]
nutrient_dt = nutrient[['id', 'name', 'unit_name', 'nutrient_nbr']]
food_nutrient_dt = food_nutrient[['id', 'fdc_id', 'nutrient_id', 'amount', 'derivation_id', 'data_points']]

label_encoder = LabelEncoder()

nutrient_dt['name'] = label_encoder.fit_transform(nutrient_dt['name'])
nutrient_dt['unit_name'] = label_encoder.fit_transform(nutrient_dt['unit_name'])

nutrient_encoded = nutrient_dt[['name', 'unit_name']].copy()

# # Criação de um novo DataFrame com as colunas 'name' e 'unit_name' e seus respectivos Z-scores
# nutrient_zscores = nutrient_dt[['name', 'unit_name']].copy()
# nutrient_zscores['name_zscore'] = (nutrient_dt['name'] - nutrient_dt['name'].mean()) / nutrient_dt['name'].std()
# nutrient_zscores['unit_name_zscore'] = (nutrient_dt['unit_name'] - nutrient_dt['unit_name'].mean()) / nutrient_dt['unit_name'].std()

# # Identificação de outliers com base nos Z-scores (utilizando critério de 3 desvios padrão)
# nutrient_zscores['outlier'] = (abs(nutrient_zscores['name_zscore']) > 3) | (abs(nutrient_zscores['unit_name_zscore']) > 3)

features = ['name', 'unit_name']
data_for_clustering = nutrient_encoded[features]

# data_for_clustering['amount'] = data_for_clustering['amount'].str.replace('[^\d.]', '', regex=True).str.replace('.', ',', regex=True)

# # Convert the 'amount' column to numeric format
# data_for_clustering['amount'] = pd.to_numeric(data_for_clustering['amount'], errors='coerce')

# # Drop rows with any NaN values after the conversion
# data_for_clustering.dropna(inplace=True)

model = KMeans(random_state=42)
visualizer = KElbowVisualizer(model, k=(1, 11))
visualizer.fit(data_for_clustering)
visualizer.show()