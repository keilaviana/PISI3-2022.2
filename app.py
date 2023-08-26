import streamlit as st
from pages.home import show as show_home
from pages.about import show as show_about
from pages.analysis import show as show_analysis

def main():
    st.sidebar.empty()
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Selecione uma página", ("Página Inicial", "Sobre", "Análise exploratória"))
    
    if page == "Página Inicial":
        show_home()
    elif page == "Sobre":
        show_about()
    elif page == "Análise exploratória":
        show_analysis()
    

if __name__ == "__main__":
    main()