import requests
from io import BytesIO
import pandas as pd
import yaml
import os

def get_variables_from_yaml():
    """
    Helper to read variables from YAML, same as in your ETL DAG.
    Looks first in /opt/secrets (container path), then local path for dev.
    """
    yaml_path = "/opt/secrets/variables.yaml"
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)
    else:
        yaml_path = "./airflow/secrets/variables.yaml"
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)

def load_and_get_df(url: str, alias: str) -> pd.DataFrame:
    '''
    Descarga un archivo CSV.GZ desde una URL, lo lee como pandas DataFrame y agrega la columna "city".
    '''
    print(f"Descargando datos de {alias}...")

    # Descargar archivo desde la URL en memoria
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"No se pudo descargar el archivo de {alias}")

    # Leer el contenido comprimido en memoria
    df = pd.read_csv(BytesIO(response.content), compression='gzip')

    # Agregar columna de ciudad
    df["city"] = alias

    print(f"Registros en {alias}: {len(df)}")
    return df

def eliminar_columnas_con_muchos_nulos(df, umbral=0.9):
        """
        Elimina columnas con más del `umbral` porcentaje de valores nulos.
        
        - df: DataFrame de pandas
        - umbral: proporción de nulos permitidos (por defecto 0.9 = 90%)
        - return: DataFrame sin las columnas eliminadas
        """
        porcentaje_nulos = df.isna().mean()
        columnas_a_eliminar = porcentaje_nulos[porcentaje_nulos > umbral].index.tolist()
        
        print(f"Columnas eliminadas ({len(columnas_a_eliminar)}): {columnas_a_eliminar}")
        
        return df.drop(columns=columnas_a_eliminar)

def eliminar_filas_con_nulos(df, columnas):
    """
    Elimina filas del DataFrame si alguna de las columnas especificadas tiene un valor nulo.
    
    - df: DataFrame de pandas
    - columnas: lista de nombres de columnas a verificar
    - return: DataFrame sin esas filas
    """
    df_filtrado = df.dropna(subset=columnas)
    print(f"Filas eliminadas: {len(df) - len(df_filtrado)}")
    return df_filtrado

def remove_html_tags(df):
    '''
    Elimina etiquetas HTML, caracteres no ASCII y normaliza espacios en columnas de texto.
    - Input: df de pandas
    - Output: df limpio
    '''
    for column_name in df.select_dtypes(include=["object", "string"]).columns:
        df[column_name] = (
            df[column_name]
            .astype(str)
            .str.replace(r'<.*?>', ' ', regex=True)           # eliminar etiquetas HTML
            .str.replace(r'[^\x00-\x7F]+', ' ', regex=True)   # eliminar caracteres no ASCII
            .str.replace(r'\s+', ' ', regex=True)             # normalizar espacios
            .str.strip()                                      # quitar espacios en extremos
        )
    return df

def obtener_tasas_cambio(fecha):
    """
    Obtiene tasas de cambio aproximadas para convertir precios a USD.
    
    Returns:
    --------
    dict
        Diccionario con tasas de cambio por ciudad.
    """
    return {
        "Buenos Aires": 1155,  # ARS a USD 2025-01-29
    }

# Función para convertir precio a USD
def convert_currency(tasas_cambio, price, city):
    rate = tasas_cambio.get(city, 1)
    return price / rate