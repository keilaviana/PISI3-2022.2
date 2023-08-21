from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_parquet("./kdd/tabelafinal.parquet")

dt = dataset[['description','nutrient_name','amount','nutrient_unit']]


dt['description'] = dt['description'].apply(lambda x: x.split(',')[0])
dt['description'] = dt['description'].apply(lambda x: x.split('-')[0])
dt['description'] = dt['description'].str.upper()
dt['nutrient_name'] = dt['nutrient_name'].str.upper()

description_counts = dt['description'].value_counts()

contagem = 20
for i in range(0, len(description_counts), contagem):
    chunk = description_counts[i:i + contagem]

nutrient_name_counts = dt['nutrient_name'].value_counts()

for i in range(0, len(nutrient_name_counts), contagem):
    chunk = nutrient_name_counts[i:i + contagem]

unit_counts = dt['nutrient_unit'].value_counts()
for i in range(0, len(nutrient_name_counts), contagem):
    chunk = unit_counts[i:i + contagem]

replace_description = description_counts[description_counts < 200].index
replace_nutrients = nutrient_name_counts[nutrient_name_counts < 200].index

dt['description'] = dt['description'].map(lambda x: 'OUTROS' if x in replace_description else x)
dt['nutrient_name'] = dt['nutrient_name'].map(lambda x: 'OUTROS' if x in replace_nutrients else x)

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
subsample = data_for_clustering.sample(n=100, random_state=42)

linkage_matrix = linkage(subsample, method='ward')

altura_corte = 15 # Defina uma altura de corte
clusters = fcluster(linkage_matrix, altura_corte, criterion='distance')

plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, labels=subsample.index, color_threshold=altura_corte)
plt.xticks(rotation=90)
plt.xlabel('Amostras aleatórias')
plt.ylabel('Distâncias')
plt.title('Dendrograma da Clusterização Hierárquica')
plt.axhline(y=altura_corte, color='r', linestyle='--', label=f'Altura de Corte = {altura_corte}')
plt.legend()
plt.show()