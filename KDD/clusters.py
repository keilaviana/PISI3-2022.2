import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler

color_scale = ['#CDDE47','#548640','#00ccff','#DE7047','#cc00ff','#ffcc00','#6600bb','#bb0066','#0066bb','#ff0066']
n_clusters = 4
clustering_cols_opts = ['description','nutrient_name','amount','nutrient_unit']
clustering_cols = clustering_cols_opts.copy()

def build_page():
    build_header()
    build_body()

def build_header():
    st.write('<h1>Agrupamento (<i>Clustering</i>) com a base do Food & Nutrient</h1>', unsafe_allow_html=True)

def build_body():
    global n_clusters, clustering_cols
    c1, c2 = st.columns(2)
    clustering_cols = c1.multiselect('Colunas', options=clustering_cols_opts,  default=clustering_cols_opts[0:2])
    if len(clustering_cols) < 1:
        st.error('É preciso selecionar pelo menos 1 coluna.')
        return
    n_clusters = c2.slider('Quantidade de Clusters', min_value=2, max_value=10, value=4)
    dfs = create_dfs()
    for df, title, desc in dfs.values():
        plot_dataframe(df, title, desc)
    df_clusters = dfs['df_clusters'][0]
    clusters = {
        'cluster': 'Cluster Sem Normalização',
        'cluster_standard': 'Cluster Com Normalização Standard',
        'cluster_minmax': 'Cluster Com Normalização MinMax',
    }
    for col, name in clusters.items():
        plot_cluster(df_clusters, col, name)
    plot_elbow(df_clusters)

def plot_elbow(df_clusters):
    variances = []
    for n in range(1, 11):
        kmeans = KMeans(n_clusters=n, n_init=10, random_state=4294967295)
        kmeans.fit(df_clusters[clustering_cols])
        variances.append(kmeans.inertia_)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, 11)), y=variances, mode='lines+markers'))
    fig.update_layout(title='Método do Cotovelo para determinar o número de clusters',
                      xaxis_title='Número de Clusters',
                      yaxis_title='Variância',
                      xaxis=dict(tickvals=list(range(1, 11))))
    
    st.plotly_chart(fig, use_container_width=True)

def create_dfs():
    cols = ['description','nutrient_name','amount','nutrient_unit']
    df_raw = create_df_raw(cols)
    df_clean = df_raw.dropna()
    df_enc = create_df_encoded(df_clean)
    df_clusters = create_df_clusters_norm(df_enc)
    return {
        'df_raw': (df_raw, 'Original', 'Dataframe original "Nutrient", com um subconjunto de colunas utilizados para o agrupamento.'),
        'df_clean': (df_clean, 'Sem Nulos', 'Dataframe após o tratamento de registros que possuíam valores nulos nas colunas selecionadas.'),
        'df_encoded': (df_enc, 'Aplicação de Codificação', 'Dataframe após a códificação (encoding) das colunas categóricas.'),
        'df_clusters': (df_clusters, 'Clusterização do Dataframe Codificado e com Normalizações', 'Dataframe após a execução de diferentes normalizações nas colunas.'),
    }
    
def create_df_raw(cols: list[str]):
    df_raw = pd.read_csv("tabelas.csv")
    df_raw = df_raw[cols].copy()
    return df_raw

def str_to_float(value):
    if isinstance(value, float):
        return value
    value = value.replace('.', '').replace(',', '.')
    return float(value)

def create_df_encoded(df: pd.DataFrame) -> pd.DataFrame:
    df_enc = df.copy()
    lenc = LabelEncoder()
    df_enc['amount'] = df_enc['amount'].apply(str_to_float)
    df_enc['description'] = df_enc['description'].apply(lambda x: x.split(',')[0])
    df_enc['description'] = lenc.fit_transform(df_enc['description'])
    df_enc['nutrient_name'] = lenc.fit_transform(df_enc['nutrient_name'])
    df_enc['nutrient_unit'] = lenc.fit_transform(df_enc['nutrient_unit'])
    return df_enc

def create_df_clusters_norm(df_enc:pd.DataFrame) -> pd.DataFrame:
    df_clusters = df_enc.copy()
    df_clusters['cluster'] = clusterize(df_enc)
    df_clusters['cluster_standard'] = clusterize(df_enc, StandardScaler())
    df_clusters['cluster_minmax'] = clusterize(df_enc, MinMaxScaler())
    return df_clusters


def clusterize(df: pd.DataFrame, scaler:TransformerMixin=None) -> pd.DataFrame:
    df_result = df[clustering_cols].copy()
    if scaler is not None:
        df_result = scale(df_result, scaler)
    X = df_result.values
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=4294967295)
    return kmeans.fit_predict(X)

def scale(df:pd.DataFrame, scaler:TransformerMixin):
    scaling_cols = [x for x in ['description', 'nutrient_name'] if x in clustering_cols]
    for c in scaling_cols:
        vals = df[[c]].values
        df[c] = scaler.fit_transform(vals)
    return df

def plot_dataframe(df, title, desc):
    with st.expander(title):
        st.write(f'<i>{desc}</i>', unsafe_allow_html=True)
        c1, _, c2 = st.columns([.5,.05,.45])
        c1.write('<h3>Dados*</h3>', unsafe_allow_html=True)
        c1.dataframe(df, use_container_width=True)
        c2.write('<h3>Descrição</h3>', unsafe_allow_html=True)
        c2.dataframe(df.describe())

def plot_cluster(df:pd.DataFrame, cluster_col:str, cluster_name:str):
    df.sort_values(by=[cluster_col,'nutrient_name'], inplace=True)
    df[cluster_col] = df[cluster_col].apply(lambda x: f'Cluster {x}')
    df_cluster_desc = df[['nutrient_name',cluster_col]].copy().groupby(by=cluster_col).count()
    df_cluster_desc.rename(columns={'amount':'qtd'}, inplace=True)
    expander = st.expander(cluster_name)
    expander.dataframe(df_cluster_desc)
    cols = expander.columns(len(clustering_cols))
    for c1 in clustering_cols:
        for cidx, c2 in enumerate(clustering_cols):
            fig = px.scatter(df, x=c1, y=c2, color=cluster_col, color_discrete_sequence=color_scale)
            cols[cidx].plotly_chart(fig, use_container_width=True)

build_page()
