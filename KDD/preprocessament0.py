import pandas as pd
import numpy as np

food = pd.read_csv("./data/dataset/food.csv")

food['food_category_id'].fillna(0, inplace=True) #Substituindo 'nan' por 0 (valor que não está incluso), criando nova categoria
food['description'].fillna("Indisponível", inplace=True) #Substituindo descrições nulas para uma categoria a parte

nutrient = pd.read_csv("./data/dataset/nutrient.csv")

nutrient['nutrient_nbr'].fillna(0, inplace=True) #Assumindo 0 como valor padrão de substituição quando esse valor NÃO está já presente

food_nutrient = pd.read_csv("./data/dataset/food_nutrient.csv", low_memory=False)

media = food_nutrient['derivation_id'].mean()

food_nutrient['derivation_id'].fillna(round(media), inplace=True)
food_nutrient['data_points'].fillna(0, inplace=True)

food_dt = food[['fdc_id', 'description', 'food_category_id']]
nutrient_dt = nutrient[['id', 'name', 'unit_name', 'nutrient_nbr']]
food_nutrient_dt = food_nutrient[['id', 'fdc_id', 'nutrient_id', 'amount', 'derivation_id', 'data_points']]