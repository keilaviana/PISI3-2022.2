import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

dataset = pd.read_parquet("./KDD/tabelafinal.parquet")

dt = dataset[['description','nutrient_name','amount','nutrient_unit']]

dt['description'] = dt['description'].apply(lambda x: x.split(',')[0])
dt['description'] = dt['description'].apply(lambda x: x.split('-')[0])
dt['description'] = dt['description'].str.upper()
dt['nutrient_name'] = dt['nutrient_name'].str.upper()

description_counts = dt['description'].value_counts()
chunk_size = 20
for i in range(0, len(description_counts), chunk_size):
    chunk = description_counts[i:i + chunk_size]

nutrient_name_counts = dt['nutrient_name'].value_counts()
for i in range(0, len(nutrient_name_counts), chunk_size):
    chunk = nutrient_name_counts[i:i + chunk_size]

unit_counts = dt['nutrient_unit'].value_counts()
for i in range(0, len(nutrient_name_counts), chunk_size):
    chunk = unit_counts[i:i + chunk_size]

values_to_replace_description = description_counts[description_counts < 200].index
values_to_replace_nutrients = nutrient_name_counts[nutrient_name_counts < 200].index
values_to_replace_unit = unit_counts[unit_counts < 100].index

dt['description'] = dt['description'].map(lambda x: 'OUTROS' if x in values_to_replace_description else x)
dt['nutrient_name'] = dt['nutrient_name'].map(lambda x: 'OUTROS' if x in values_to_replace_nutrients else x)
dt['nutrient_unit'] = dt['nutrient_unit'].map(lambda x: 'OUTROS' if x in values_to_replace_unit else x)

scaler = StandardScaler()
dt['amount'] = scaler.fit_transform(dt[['amount']])

def codf_onehotencoder(dt):
    one_hot_encoder = OneHotEncoder(sparse=False, dtype=int)

    encoded_description = one_hot_encoder.fit_transform(dt[['description']])
    encoded_nutrient_name = one_hot_encoder.fit_transform(dt[['nutrient_name']])
    encoded_nutrient_unit = one_hot_encoder.fit_transform(dt[['nutrient_unit']])

    one_hot_encoder = OneHotEncoder(sparse=False, dtype=int)
    encoded_description = one_hot_encoder.fit_transform(dt[['description']])
    encoded_nutrient_name = one_hot_encoder.fit_transform(dt[['nutrient_name']])
    encoded_nutrient_unit = one_hot_encoder.fit_transform(dt[['nutrient_unit']])

    encoded_description_df = pd.DataFrame(encoded_description, columns=[f"description_{i}" for i in range(encoded_description.shape[1])])
    encoded_nutrient_name_df = pd.DataFrame(encoded_nutrient_name, columns=[f"nutrient_name_{i}" for i in range(encoded_nutrient_name.shape[1])])
    encoded_nutrient_unit_df = pd.DataFrame(encoded_nutrient_unit, columns=[f"nutrient_unit_{i}" for i in range(encoded_nutrient_unit.shape[1])])
    dt_encoded = pd.concat([dt.drop(['description', 'nutrient_name', 'nutrient_unit'], axis=1),
                        encoded_description_df, encoded_nutrient_name_df, encoded_nutrient_unit_df], axis=1)
                            
    return dt_encoded

dt_encoded = codf_onehotencoder(dt)

df = dt_encoded[dt_encoded['amount'] != 0.0]

def tratamento_outliers(df):
    Q1 = df['amount'].quantile(0.25)
    Q3 = df['amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    max_non_outlier = df[df['amount'] <= upper_bound]['amount'].max()
    df.loc[df['amount'] > upper_bound, 'amount'] = max_non_outlier

tratamento_outliers(df)

original_data = df.copy()

selected_columns = ['description_' + str(i) for i in range(86)]+['nutrient_name_' + str(i) for i in range(135)]+['nutrient_unit_' + str(i) for i in range(6)]+['amount']
data_for_clustering = df[selected_columns]

def clusterize(df: pd.DataFrame, clusters: int) -> pd.DataFrame:
    num_clusters = clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(df)
    df['cluster_label'] = cluster_labels
    return df

selected_columns_pca = [col for col in original_data.columns if col != 'cluster_label']

def main():
    st.title('Visualização da dispersão de grupos')
    st.sidebar.title('Opções')
    st.write("Nessa seção, explora-se a dispersão dos grupos formados como resultado do pré-processamento realizado por meio do KDD.")
    st.write("Para isso, estão aptas a visualização para três diferentes valores para k: 3, 10 e 17. Além disso, pode-se escolher aplicar a PCA, reduzindo a dimensionalidade do conjunto, como forma de visualizar a dispersão como um todo.")
    st.write("Caso contrário, a dispersão pode ser vista ao selecionar duas colunas por vez. Ressalta-se, ainda, a visualização de mais uma tabela abaixo da plotagem, como resultado do processo de clusterização, a qual informa, por linha, em que cluster o elemento foi enquadrado.")
    selected_num_clusters = st.sidebar.radio("Escolha a quantidade de clusters:", [3, 10, 17], index=2)

    show_pca = st.sidebar.checkbox("Aplicar PCA")
    clusterized_data = clusterize(data_for_clustering, selected_num_clusters)  
    
    if not show_pca:
        selected_x_feature = st.sidebar.selectbox("Selecione a coluna para o eixo X", [feature for feature in data_for_clustering.columns if feature != "cluster_label"])
        available_y_features = [feature for feature in data_for_clustering.columns if feature != selected_x_feature and feature != "cluster_label"]
        selected_y_feature = st.sidebar.selectbox("Selecione a coluna para o eixo Y", available_y_features)
    if show_pca:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaler.fit_transform(clusterized_data[selected_columns_pca]))
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
        pca_df['cluster_label'] = clusterized_data['cluster_label']

        fig = px.scatter(pca_df, x='PC1', y='PC2', color='cluster_label', 
                        title='Dispersão dos clusters com PCA')
        st.plotly_chart(fig)
    else:
        fig = px.scatter(clusterized_data, x=selected_x_feature, y=selected_y_feature, color='cluster_label',
                         color_discrete_sequence=px.colors.qualitative.Set3,
                         hover_name=clusterized_data.index,
                         title=f'Visualização da dispersão com as colunas: {selected_x_feature} vs {selected_y_feature}')
        
        fig.update_traces(marker=dict(size=10, opacity=0.7),
                          selector=dict(mode='markers'))

        for trace in fig.data:
            trace.name = f'Cluster {trace.name}'

        st.plotly_chart(fig)

    st.write(clusterized_data)

if __name__ == '__main__':
    main()

