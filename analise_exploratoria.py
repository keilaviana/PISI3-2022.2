import pandas as pd
from ydata_profiling import ProfileReport

food = pd.read_csv('data/dataset/food.csv')
food_nutrient = pd.read_csv('data/dataset/food_nutrient.csv', low_memory=False)
nutrient = pd.read_csv('data/dataset/nutrient.csv')

analise = pd.concat([food, food_nutrient, nutrient], sort=False)
analise.info()
profile = ProfileReport(analise, title="Pandas Profiling Report")
profile.to_file("analise.html")