import streamlit as st
import pandas as pd
# from sklearn.preprocessing import OneHotEncoder

def create_dfs():
    df_original = pd.read_csv("tabelas.csv")
    df_raw = create_df_raw(df_original)
    df_enc = create_df_encoded(df_original)
    return {
        'df_raw': (df_raw, 'Original', 'Dataframe original com um subconjunto de colunas utilizados para o agrupamento.'),
        'df_encoded': (df_enc, 'Aplicação de Codificação', 'Dataframe após a códificação das colunas categóricas e normalização das colunas numéricas.')
    }

def create_df_raw(df: pd.DataFrame):
    return df

def create_df_encoded(df: pd.DataFrame) -> pd.DataFrame:
    dt_encoded = pd.read_parquet("tabela_cod.parquet")
    # one_hot_encoder = OneHotEncoder(sparse=False, dtype=int)

    # encoded_description = one_hot_encoder.fit_transform(df[['description']])
    # encoded_nutrient_name = one_hot_encoder.fit_transform(df[['nutrient_name']])
    # encoded_nutrient_unit = one_hot_encoder.fit_transform(df[['nutrient_unit']])

    # one_hot_encoder = OneHotEncoder(sparse=False, dtype=int)
    # encoded_description = one_hot_encoder.fit_transform(df[['description']])
    # encoded_nutrient_name = one_hot_encoder.fit_transform(df[['nutrient_name']])
    # encoded_nutrient_unit = one_hot_encoder.fit_transform(df[['nutrient_unit']])

    # encoded_description_df = pd.DataFrame(encoded_description, columns=[f"description_{i}" for i in range(encoded_description.shape[1])])
    # encoded_nutrient_name_df = pd.DataFrame(encoded_nutrient_name, columns=[f"nutrient_name_{i}" for i in range(encoded_nutrient_name.shape[1])])
    # encoded_nutrient_unit_df = pd.DataFrame(encoded_nutrient_unit, columns=[f"nutrient_unit_{i}" for i in range(encoded_nutrient_unit.shape[1])])
    # dt_encoded = pd.concat([df.drop(['description', 'nutrient_name', 'nutrient_unit'], axis=1),
    #                         encoded_description_df, encoded_nutrient_name_df, encoded_nutrient_unit_df], axis=1)
    return dt_encoded

def main():
    st.title('🖇️Dataframe: original X codificado')
    st.write("As tabelas aqui apresentadas representam as mesmas informações. Entretanto, a fim de adequação ao problema proposto, inicialmente, percebe-se a presença de colunas categóricas, fator prejudicial ao processo de clusterização.")
    st.write("Nesse sentido, buscou-se codificar essas informações, resultando em uma segunda tabela que apresenta esses valores em formato binário.")
    dfs = create_dfs()

    for _, (df, df_title, df_description) in dfs.items():
        st.subheader(df_title)
        st.write(df_description)
        st.write(df)

if __name__ == '__main__':
    main()