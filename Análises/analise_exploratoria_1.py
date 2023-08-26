import pandas as pd;
food = pd.read_csv('data/dataset/food.csv')
food_nutrient = pd.read_csv('data/dataset/food_nutrient.csv')
nutrient = pd.read_csv('data/dataset/nutrient.csv')


# Verificando quantidade de valores ausentes
print("Tabela de alimentos")
print("************************************")
print(food.isnull().sum())
print("Tabela de nutrientes nos alimentos")
print("************************************")
print(food_nutrient.isnull().sum())
print("Tabela de nutrientes")
print("************************************")
print(nutrient.isnull().sum())


# food_nutrient_2 = food_nutrient.drop(["min", "max", "median", "foodnote", "min_year_acqured"], axis = 1)

print("_______________________________________")
print("verificar tipos")
print("_______________________________________")
# Verificar tipos de dados
print(food.dtypes)
print(food_nutrient.dtypes)
print(nutrient.dtypes)


print("_______________________________________")
print("verificar consistência")
print("_______________________________________")
# Verificar consistência dos dados
print(food['description'].unique())
print(food_nutrient['derivation_id'].unique())
print(nutrient['unit_name'].unique())

