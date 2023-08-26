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

# dt['amount'] = dt['amount'].apply(str_to_float)
dt['description'] = dt['description'].apply(lambda x: x.split(',')[0])
dt['description'] = dt['description'].apply(lambda x: x.split('-')[0])
dt['nutrient_name'] = dt['nutrient_name'].apply(lambda x: x.split(' ')[0])
dt['description'] = dt['description'].str.upper()
dt['nutrient_name'] = dt['nutrient_name'].str.upper()
dt['nutrient_unit'] = dt['nutrient_unit'].str.upper()

# Contagem de ocorrências para a coluna 'description'
description_counts = dt['description'].value_counts()

chunk_size = 20  # Número de itens por vez
for i in range(0, len(description_counts), chunk_size):
    chunk = description_counts[i:i + chunk_size]

# Contagem de ocorrências para a coluna 'nutrient_name'
nutrient_name_counts = dt['nutrient_name'].value_counts()

for i in range(0, len(nutrient_name_counts), chunk_size):
    chunk = nutrient_name_counts[i:i + chunk_size]

unit_counts = dt['nutrient_unit'].value_counts()
for i in range(0, len(nutrient_name_counts), chunk_size):
    chunk = unit_counts[i:i + chunk_size]

# Obter os valores que ocorrem menos de 100 vezes
values_to_replace_description = description_counts[description_counts < 200].index
values_to_replace_nutrients = nutrient_name_counts[nutrient_name_counts < 200].index
values_to_replace_unit = unit_counts[unit_counts < 100].index

# Substituir esses valores por 'OUTROS'
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

# Selecionar as colunas relevantes para a clusterização
selected_columns = ['amount'] + [col for col in df.columns if col.startswith('description_') or col.startswith('nutrient_name_') or col.startswith('nutrient_unit_')]
data_for_clustering = df[selected_columns]

num_clusters = 17
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(data_for_clustering)

# Adicionar os rótulos de cluster de volta ao DataFrame original
df['cluster_label'] = cluster_labels

cluster_counts = []

# Valores de k (número de clusters) que você deseja analisar
k_values = [3, 10, 17]

# Loop através dos valores de k
for num_clusters in k_values:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_for_clustering)
    cluster_count = np.bincount(cluster_labels)
    cluster_counts.append(cluster_count)

# Cria um gráfico de linhas para mostrar a variação na quantidade de elementos por cluster
plt.figure(figsize=(12, 8))
for i, k in enumerate(k_values):
    plt.plot(range(k), cluster_counts[i], label=f'k={k}')

plt.xlabel('Número de Clusters')
plt.ylabel('Quantidade de Elementos')
plt.title('Variação na Quantidade de Elementos por Cluster para Diferentes Valores de k')
plt.legend()
plt.show()