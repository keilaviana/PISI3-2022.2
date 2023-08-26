import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sb

dataset = pd.read_csv("./Streamlit/tabelas.csv")

dt = dataset[['description','nutrient_name','amount','nutrient_unit']]

dt['description'] = dt['description'].apply(lambda x: x.split(',')[0])

label_encoder = LabelEncoder()

dt['description'] = label_encoder.fit_transform(dt['description'])
dt['nutrient_name'] = label_encoder.fit_transform(dt['nutrient_name'])
dt['nutrient_unit'] = label_encoder.fit_transform(dt['nutrient_unit'])

print(dt['amount'].plt(kind = 'bar'))

dt['amount'] = StandardScaler().fit_transform(dt[['amount']])
print(dt[['amount']])

visualizer = KElbowVisualizer(KMeans(), k_list=range(1, 11))
visualizer.fit(dt)
visualizer.show()
cluster = KMeans(n_clusters=4).fit(dt[['amount', 'description', 'nutrient_name', 'nutrient_unit']])
dt['cluster'] = cluster.predict(dt[['amount', 'description', 'nutrient_name', 'nutrient_unit']])

correlation_matrix = dt.corr()
if correlation_matrix.shape[0] > 1 and correlation_matrix.shape[1] > 1:
     plt.figure(figsize=(12, 10))
     sb.heatmap(correlation_matrix, cmap='viridis', annot=True, vmin=0.0, vmax=1.0)
     plt.title("Diagrama de Correlação")
     plt.show()
else:
     print("Correlation matrix is not valid for visualization.")