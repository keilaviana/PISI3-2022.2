import streamlit as st
import pandas as pd

food_data = pd.read_csv('data/food.csv')
food_nutrient_data = pd.read_csv('data/food_nutrient.csv')
nutrient_data = pd.read_csv('data/nutrient.csv')

category_filter = st.selectbox('Filtrar por categoria de alimentos:', food_data['food_category_id'].unique())
filtered_food = food_data[food_data['food_category_id'] == category_filter]

nutrient_filter = st.selectbox('Filtrar por nutriente:', nutrient_data['name'])
nutrient_id = nutrient_data[nutrient_data['name'] == nutrient_filter]['id'].values[0]
filtered_food_nutrient = food_nutrient_data[food_nutrient_data['nutrient_id'] == nutrient_id]

st.subheader('Alimentos filtrados por categoria:')
st.dataframe(filtered_food)

st.subheader('Alimentos filtrados por nutriente:')
st.dataframe(filtered_food_nutrient)

