import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer

food = pd.read_csv("./data/dataset/food.csv")
food['food_category_id'].fillna(0, inplace=True)
food['description'].fillna("Indisponível", inplace=True)

nutrient = pd.read_csv("./data/dataset/nutrient.csv")
nutrient['nutrient_nbr'].fillna(0, inplace=True)

food_nutrient = pd.read_csv("./data/dataset/food_nutrient.csv", low_memory=False)

food_dt = food[['fdc_id', 'description']]
nutrient_dt = nutrient[['id', 'name', 'unit_name']]
food_nutrient_dt = food_nutrient[['id', 'fdc_id', 'nutrient_id', 'amount']]

nutrient_dt.rename(columns={'id': 'nutrient_id', 'name': 'nutrient_name', 'unit_name': 'nutrient_unit'}, inplace=True)

merged_table = pd.merge(food_nutrient_dt, food_dt, on='fdc_id')
merged_table = pd.merge(merged_table, nutrient_dt, on='nutrient_id')

result_table = merged_table[['fdc_id', 'nutrient_id', 'description', 'nutrient_name', 'amount', 'nutrient_unit']]

result_table.to_parquet('./kdd/tabelafinal.parquet', index=False)