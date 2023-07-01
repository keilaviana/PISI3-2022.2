import streamlit as st
import pandas as pd
import altair as alt

food_data = pd.read_csv('./food.csv')
nutrient_data = pd.read_csv('./nutrient.csv')
food_nutrient_data = pd.read_csv('./food_nutrient.csv')

merged_data = food_nutrient_data.merge(food_data[['fdc_id', 'description']], on='fdc_id')
merged_data = merged_data.merge(nutrient_data[['id', 'name']], left_on='nutrient_id', right_on='id')

nutrient_counts = merged_data['name'].value_counts().reset_index()
nutrient_counts.columns = ['Nutriente', 'Quantidade de Alimentos']

chart = alt.Chart(nutrient_counts).mark_area().encode(
    x=alt.X('Nutriente', sort='-y'),
    y='Quantidade de Alimentos')

st.altair_chart(chart, use_container_width=True)

if __name__ == '__main__':
    st.title('Gráfico de Área: Nutrientes vs. Quantidade de Alimentos')
    st.dataframe(nutrient_counts)
    st.write('O gráfico de área abaixo mostra os nutrientes com a maior quantidade de alimentos contendo-os.')
    st.write('A área de cada nutriente representa a quantidade de alimentos que o contém.')
    st.write('')
    
    st.altair_chart(chart, use_container_width=True)
