import pandas as pd
import streamlit as st
import plotly.express as px
from utils import read_df

def build_dataframe_section(df:pd.DataFrame):
    st.write('<h2>Dados Dataset</h2>', unsafe_allow_html=True)
    st.dataframe(df)

def __ingest_fn_data() -> pd.DataFrame:
    df = read_df('n')
    return df

def __transform_fn_data(df:pd.DataFrame) -> pd.DataFrame:
    return df

def read_fn_df() -> pd.DataFrame:
    return __transform_fn_data(__ingest_fn_data())

def __get_color_sequence_map() -> dict[str,list[str]]:
    def filter(colors) -> list[str]:
        return [x for idx, x in enumerate(colors) if idx%2!=0]
    result = {
        'Azul (reverso)': filter(px.colors.sequential.Blues_r),
        'Azul': filter(px.colors.sequential.Blues),
        'Plasma (reverso)': filter(px.colors.sequential.Plasma_r),
        'Plasma': filter(px.colors.sequential.Plasma),
        'Vermelho (reverso)': filter(px.colors.sequential.Reds_r),
        'Vermelho': filter(px.colors.sequential.Reds),
    }
    return result

def get_color_sequence_names() -> list[str]:
    return __get_color_sequence_map().keys()

def get_color_sequence(name) -> list[str]:
    return __get_color_sequence_map()[name]