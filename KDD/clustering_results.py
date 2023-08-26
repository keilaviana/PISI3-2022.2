import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sb
import numpy as np

dataset = pd.read_parquet("tabelas.parquet")

dt = dataset[['description','nutrient_name','amount','nutrient_unit']]

dt['description'] = dt['description'].apply(lambda x: x.split(',')[0])
dt['description'] = dt['description'].apply(lambda x: x.split('-')[0])
dt['nutrient_name'] = dt['nutrient_name'].apply(lambda x: x.split(' ')[0])
dt['description'] = dt['description'].str.upper()
dt['nutrient_name'] = dt['nutrient_name'].str.upper()
dt['nutrient_unit'] = dt['nutrient_unit'].str.upper()

description_counts = dt['description'].value_counts()

chunk_size = 20
for i in range(0, len(description_counts), chunk_size):
    chunk = description_counts[i:i + chunk_size]

nutrient_name_counts = dt['nutrient_name'].value_counts()

for i in range(0, len(nutrient_name_counts), chunk_size):
    chunk = nutrient_name_counts[i:i + chunk_size]

unit_counts = dt['nutrient_unit'].value_counts()
for i in range(0, len(nutrient_name_counts), chunk_size):
    chunk = unit_counts[i:i + chunk_size]

values_to_replace_description = description_counts[description_counts < 200].index
values_to_replace_nutrients = nutrient_name_counts[nutrient_name_counts < 200].index
values_to_replace_unit = unit_counts[unit_counts < 100].index

dt['description'] = dt['description'].map(lambda x: 'OUTROS' if x in values_to_replace_description else x)
dt['nutrient_name'] = dt['nutrient_name'].map(lambda x: 'OUTROS' if x in values_to_replace_nutrients else x)
dt['nutrient_unit'] = dt['nutrient_unit'].map(lambda x: 'OUTROS' if x in values_to_replace_unit else x)

scaler = StandardScaler()
dt['amount'] = scaler.fit_transform(dt[['amount']])

one_hot_encoder = OneHotEncoder(sparse=False, dtype=int)

encoded_description = one_hot_encoder.fit_transform(dt[['description']])
encoded_nutrient_name = one_hot_encoder.fit_transform(dt[['nutrient_name']])
encoded_nutrient_unit = one_hot_encoder.fit_transform(dt[['nutrient_unit']])

one_hot_encoder = OneHotEncoder(sparse=False, dtype=int)
encoded_description = one_hot_encoder.fit_transform(dt[['description']])
encoded_nutrient_name = one_hot_encoder.fit_transform(dt[['nutrient_name']])
encoded_nutrient_unit = one_hot_encoder.fit_transform(dt[['nutrient_unit']])

encoded_description_df = pd.DataFrame(encoded_description, columns=[f"description_{i}" for i in range(encoded_description.shape[1])])
encoded_nutrient_name_df = pd.DataFrame(encoded_nutrient_name, columns=[f"nutrient_name_{i}" for i in range(encoded_nutrient_name.shape[1])])
encoded_nutrient_unit_df = pd.DataFrame(encoded_nutrient_unit, columns=[f"nutrient_unit_{i}" for i in range(encoded_nutrient_unit.shape[1])])
dt_encoded = pd.concat([dt.drop(['description', 'nutrient_name', 'nutrient_unit'], axis=1),
                        encoded_description_df, encoded_nutrient_name_df, encoded_nutrient_unit_df], axis=1)

dt_encoded.to_parquet('tabela_cod.parquet', index=False)

df = dt_encoded[dt_encoded['amount'] != 0.0]

Q1 = df['amount'].quantile(0.25)
Q3 = df['amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
max_non_outlier = df[df['amount'] <= upper_bound]['amount'].max()
df.loc[df['amount'] > upper_bound, 'amount'] = max_non_outlier

original_data = df.copy()

selected_columns = ['amount'] + [col for col in df.columns if col.startswith('description_') or col.startswith('nutrient_name_') or col.startswith('nutrient_unit_')]
data_for_clustering = df[selected_columns]

num_clusters = 17
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(data_for_clustering)

df['cluster_label'] = cluster_labels

clustered_data = pd.concat([df, dataset[['description']]], axis=1)
food_counts_by_cluster = clustered_data.groupby(['cluster_label', 'description']).size().reset_index(name='count')
cluster_x_data = food_counts_by_cluster[food_counts_by_cluster['cluster_label'] == 1]
total_elements_cluster_x = cluster_x_data['count'].sum()
print(f"Total de Elementos no Cluster 1: {total_elements_cluster_x}")

print("\nAlimentos no Cluster 1:")
for index, row in cluster_x_data.iterrows():
    print(f"  Alimento: {row['description']}, Quantidade: {row['count']}")
print()