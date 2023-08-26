from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
import streamlit as st
from utils import df_names, read_df



def show():
    # Lendo o conteúdo do arquivo HTML
    with open('./analise.html', 'r') as file:
        html_content = file.read()

    # Renderizando o HTML no Streamlit
    st.components.v1.html(html_content, height=600, scrolling=True)
