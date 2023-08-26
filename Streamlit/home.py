import streamlit as st
import read_df as rd
import groups as gp

st.set_page_config(
    page_title= "#πsi3",
    page_icon="🗃️",
    layout="wide"
)

menu_options = ["🍴Página Inicial", "🖇️Tabela", "📊Grupos"]
selected_option = st.sidebar.selectbox("Selecione uma página", menu_options)

if selected_option == "🍴Página Inicial":
    st.title("🍴Página Inicial")
    st.write("O projeto consiste na exploração do processo de clusterização, o qual foi aplicado em um banco de dados referente a alimentos e seus perfis nutricionais.")
    st.write("Este estudo, conduzido como parte da disciplina de Projeto Interdisciplinar para Sistemas de Informação na UFRPE, evidencia os padrões de agrupamentos. Tal análise permite um aprofundamento do conjunto de dados utilizados a partir das técnicas aplicadas.")
    st.write("Grupo: André Pereira, Caio Farias, Keila Viana, Leonardo Batista e Raquel Silva.")
elif selected_option == "🖇️Tabela":
    rd.main()
elif selected_option == "📊Grupos":
    gp.main()