import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
tabelas = pd.read_csv('/content/tabelas.csv')
tabelas
import pandas as pd
df = pd.read_parquet("/content/tabelafinal.parquet")
tabelas.isnull().sum()
# Outliers amount
grafico = px.box(tabelas, y = 'amount')
grafico.show()
# Outliers nutrient_unit
grafico = px.box(tabelas, y = 'nutrient_unit')
grafico.show()
# Outliers nutrient_name
grafico = px.box(tabelas, y = 'nutrient_name')
grafico.show()
# Outliers description
grafico = px.box(tabelas, y = 'description')
grafico.show()
# Outliers nutrient_name x nutrient_unit
grafico = px.box(tabelas, x = 'nutrient_name', y = 'nutrient_unit')
grafico.show()
# Outliers amount x nutrient_name
grafico = px.box(tabelas, x = 'amount', y = 'nutrient_name')
grafico.show()
# Outliers nutrient_unit x nutrient_name
grafico = px.box(tabelas, x = 'nutrient_unit', y = 'nutrient_name')
grafico.show()
# Outliers nutrient_name x amount
grafico = px.box(tabelas, x = 'nutrient_name', y = 'amount')
grafico.show()
# nutrient_unit x nutrient_name
grafico = px.scatter(x = tabelas['nutrient_unit'], y = tabelas['nutrient_name'])
grafico.show()
# amount x nutrient_name
grafico = px.scatter(x = tabelas['amount'], y = tabelas['nutrient_name'])
grafico.show()
# nutrient_name x amount
grafico = px.scatter(x = tabelas['nutrient_name'], y = tabelas['amount'])
grafico.show()