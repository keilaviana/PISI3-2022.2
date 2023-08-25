import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import ast
import sklearn.model_selection
df = pd.read_parquet("tabelafinal.parquet")
tabelafinal = pd.read_parquet('/content/tabelafinal.parquet')
tabelafinal
base_tabelafinal = pd.read_parquet('/content/tabelafinal.parquet')
base_tabelafinal
x_tabelafinal = base_tabelafinal.iloc[:, 0:6].values
x_tabelafinal
y_tabelafinal = base_tabelafinal.iloc[:, 5].values
y_tabelafinal
from sklearn.preprocessing import LabelEncoder
label_encoder_workclass = LabelEncoder()
label_encoder_description = LabelEncoder()
label_encoder_nutrient_name = LabelEncoder()
label_encoder_nutrient_unit = LabelEncoder()
x_tabelafinal[:,2] = label_encoder_description.fit_transform(x_tabelafinal[:,2])
x_tabelafinal[:,3] = label_encoder_nutrient_name.fit_transform(x_tabelafinal[:,3])
x_tabelafinal[:,5] = label_encoder_nutrient_unit.fit_transform(x_tabelafinal[:,5])
x_tabelafinal[0]
from sklearn.preprocessing import StandardScaler
sacaler_tabelafinal = StandardScaler()
x_tabelafinal = sacaler_tabelafinal.fit_transform(x_tabelafinal)
x_tabelafinal
from sklearn.preprocessing import StandardScaler
sacaler_tabelafinal = StandardScaler()
df = pd.read_parquet("tabelafinal.parquet")
x_tabelafinal = sacaler_tabelafinal.fit_transform(x_tabelafinal)
x_tabelafinal
X_tabelafinal = pd.read_parquet('tabelafinal.parquet')
from sklearn.model_selection import train_test_split
X_tabelafinal_treinamento, X_tabelafinal_teste, y_tabelafinal_treinamento, y_tabelafinal_teste = train_test_split(X_tabelafinal, y_tabelafinal, test_size=0.15, random_state=0)
X_tabelafinal_treinamento.shape, X_tabelafinal_teste.shape
from sklearn.decomposition import PCA
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
#sacaler_tabelafinal = StandardScaler()
#x_tabelafinal = sacaler_tabelafinal.fit_transform(x_tabelafinal)
pca = PCA(n_components=4)
x_tabelafinal[0]
import pandas as pd
from sklearn.decomposition import PCA
X_tabelafinal = pd.read_parquet('tabelafinal.parquet')
df = pd.read_parquet("tabelafinal.parquet")
#datareader = parquet.reader(datafile, delimiter=',', quotechar='"')
pca = PCA(n_components=4)
X_tabelafinal_treinamento_pca = pca.fit_transform(X_tabelafinal_treinamento)
X_tabelafinal_teste_pca = pca.transform(X_tabelafinal_teste)
X_tabelafinal_treinamento_pca.shape, X_tabelafinal_teste_pca.shape
pca.explained_variance_ratio_