import pandas as pd
import streamlit as st
from utils import df_name_specific, df_names, read_df
import altair as alt
import plotly.express as px


def show():
    st.title("O que possível obter a partir desses dados?")
    st.write("Nessa página será possível perceber relações importantes entre os datasets")

    food_data = pd.read_csv('./data/food.csv')
    nutrient_data = pd.read_csv('./data/nutrient.csv')
    food_nutrient_data = pd.read_csv('./data/food_nutrient.csv') 

    merged_data = food_nutrient_data.merge(food_data[['fdc_id', 'description']], on='fdc_id')
    merged_data = merged_data.merge(nutrient_data[['id', 'name']], left_on='nutrient_id', right_on='id')

    nutrient_counts = merged_data['name'].value_counts().reset_index()
    nutrient_counts.columns = ['Nutriente', 'Quantidade de Alimentos']

    chart = alt.Chart(nutrient_counts).mark_area().encode(
        x=alt.X('Nutriente', sort='-y'),
        y='Quantidade de Alimentos')

    st.subheader('Gráfico de Área: Nutrientes vs. Quantidade de Alimentos')
    st.dataframe(nutrient_counts)
    st.write('O gráfico de área abaixo mostra os nutrientes com a maior quantidade de alimentos contendo-os.')
    st.write('A área de cada nutriente representa a quantidade de alimentos que o contém.')
    st.write('')
    st.altair_chart(chart, use_container_width=True)



    st.subheader('Gráfico de barras: Agrupa os dados pela descrição do alimento e soma a quantidade de ocorrências de cada tipo de dados dos alimentos')
    df_grouped_1 = food_data.groupby('data_type')['description'].sum()
    st.bar_chart(df_grouped_1)


    st.subheader('Gráfico de linhas: Agrupa os dados pelo nome do nutriente  para obter o tamanho de cada grupo')
    df_grouped_2 = nutrient_data.groupby('name').size()
    st.line_chart(df_grouped_2)

    build_scatter_section(food_data)



def build_scatter_section(df:pd.DataFrame):
    #see: https://plotly.com/python/marker-style/
    st.markdown('<h3>Gráfico de dispersão</h3>', unsafe_allow_html=True)
    st.write('Podemos visualizar a distribuição dos alimentos em relação a duas variáveis: eixo x pode representa a categoria de alimentos e o eixo y pode representa a quantidade de um nutriente específico.')

    df.sort_values(by=['food_category_id','fdc_id'], ascending=[False, True], inplace=True)
    fig = px.scatter(df, x='food_category_id', y='fdc_id', color_discrete_sequence=['#DE7047'], 
                     symbol_sequence=['circle','circle'])
    fig.update_traces(marker_size=8, marker_line_width=1)
    st.plotly_chart(fig, use_container_width=True)





    
