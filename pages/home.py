import streamlit as st
import pandas as pd
from utils import df_names, read_df, df_name_specific


def show():
    st.title("E ai, seja bem vindo a análise sobre Dados alimentares e Nutrientes")
    st.write("Aqui vão estar alimentos e conteúdos nutricionais úteis para elaborar uma dieta adequada")

    build_body()

def build_body():
    col1, col2 = st.columns([.3,.7])
    df_name = col1.selectbox('Dataset', df_name_specific())
    df = read_df(df_name)
    cols = list(df.columns)
    group_cols = col2.multiselect('Agrupar', options=cols, default=cols[0])
    tot_opts = [x for x in cols if x not in group_cols]
    tot_fun = col1.selectbox('Função', options=['count','nunique','sum','mean'])
    tot_cols = col2.multiselect('Totalizar', options=tot_opts, default=tot_opts[0])
    select_cols = tot_cols+group_cols
    df_grouped = df[select_cols].groupby(by=group_cols).agg(tot_fun)
    st.write('Data frame:')
    st.dataframe(df_grouped, use_container_width=True)
