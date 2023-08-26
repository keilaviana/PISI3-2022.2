import pandas as pd
import os

def df_names():
    result = []
    dir_iter = os.scandir('data')
    for f in dir_iter:
        if f.name.endswith('food.csv'):
            result.append(f.name[0:-4])
    return sorted(result)

def df_name_specific(df_name = None):
    
    search = '.csv' if df_name == None else (df_name + '.csv')
    result = []
    dir_iter = os.scandir('data')
    for f in dir_iter:
        if f.name.endswith(search):
            result.append(f.name[0:-4])
    return sorted(result)

def read_df(df_name, extension='csv', encoding='utf-8', low_memory=False):
    path = f'data/{df_name}.{extension}'
    if extension=='csv':
        return __read_csv(path, encoding=encoding, low_memory=low_memory)
    if extension=='parquet':
        return pd.read_parquet(path)
    raise Exception(f"Formato inválido: {extension}")

def __read_csv(path, encoding, low_memory=False):
    try:
        df = pd.read_csv(path, sep=',', encoding=encoding, low_memory=low_memory)
    except:
        df = pd.read_csv(path, sep=';', encoding=encoding, low_memory=low_memory)
    return df