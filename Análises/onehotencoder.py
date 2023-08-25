import pandas as pd
from sklearn.preprocessing import OneHotEncoder
#dataset = pd.read_parquet('/content/tabelafinal.parquet')
df = pd.read_csv('/content/tabelas.csv')
df.dtypes
encoder = OneHotEncoder(handle_unknown='ignore')
df_encoded = encoder.fit_transform(df)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(df_encoded.toarray())
import matplotlib.pyplot as plt

plt.scatter(df_encoded[:, 0], df_encoded[:, 1], c=labels)
plt.show()
from sklearn.metrics import mahalanobis

threshold = stats.chi2.ppf(0.95, df_encoded.shape[1])
outliers = []
for i in range(len(df_encoded)):
    dist = mahalanobis(df_encoded[i, :], kmeans.cluster_centers_[labels[i]], kmeans.covariances_[labels[i]])
    if dist > threshold:
        outliers.append(i)