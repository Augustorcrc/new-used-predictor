import pandas as pd
import json
import ast
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_object_dtype
from utils import get_ordered_columns, extract_id_from_string_list

import sys
import os
sys.path.append(os.path.abspath('..'))


def preprocess(data) -> pd.DataFrame:
    """
    Preprocesa el archivo JSON y realiza el feature engineering necesario.
    Args:
        data
    """


    # Convertir a DataFrame aplanado
    df = pd.json_normalize(data)

    # Ordenar columnas
    ordered_cols = get_ordered_columns(data)
    df = df[[col for col in ordered_cols if col in df.columns]]

    # Reemplazar puntos por guiones bajos
    df.columns = [col.replace('.', '_') for col in df.columns]


    # Cambio a str el valor de descriptions ya que viene en lista de str
    df['descriptions'] = df['descriptions'].apply(extract_id_from_string_list)

    # Eliminar columnas con alto porcentaje de listas vacías, nulos o valores constantes y que no aportan información
    drop_cols = [
        'sub_status',                # >80% listas vacías
        'deal_ids',                  # >80% listas vacías
        'shipping_methods',          # todas listas vacías o NaN
        'shipping_tags',             # >80% listas vacías
        'coverage_areas',            # todas listas vacías
        'listing_source',            # todas string vacías
        'international_delivery_mode',  # todas "none"
        'shipping_dimensions',       # < 10% no nulos
        'official_store_id',         # < 10% de valores NO nulos
        'differential_pricing',      # < 10% de valores NO nulos
        'original_price',            # < 10% de valores NO nulos
        'video_id',                  # < 10% de valores NO nulos
        'catalog_product_id',        # < 10% de valores NO nulos
        'subtitle',                  # < 10% de valores NO nulos
        'variations',                # contiene como valor listas de diccionarios. A priori no contiene informacion "relevante"
        'attributes',                # contiene como valor listas de diccionarios. A priori no contiene informacion "relevante"
        'descriptions',   # pocos nulos pero en primera iteración no se usa
        'parent_item_id',  # pocos nulos pero en primera iteración no se usa
        'tags',
        'site_id',  # todas las filas tienen el mismo valor
        'seller_address_country_id', # colineal con seller_address_country_name
        'available_quantity', # colineal con initial_quantity
    ]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
 

    ### FEATURE ENGINEERING ###

    # Crear columnas de cantidad de elementos en listas
    def count_if_list(x):
        return len(x) if isinstance(x, list) else 0

    df['non_mp_methods_count'] = df['non_mercado_pago_payment_methods'].apply(count_if_list)
    df['pictures_count'] = df['pictures'].apply(count_if_list)
    df.drop(columns=['non_mercado_pago_payment_methods', 'pictures'], inplace=True, errors='ignore')
    
    #  Crear columna de cantidad de palabras en el título
    df['title_word_count'] = df['title'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    df['has_nuevo_in_title'] = df['title'].apply(lambda x: 1 if isinstance(x, str) and 'nuevo' in x.lower() else 0)


    #  Codificar binarios útiles
    df['has_warranty'] = df['warranty'].notna().astype(int)
    df['is_free_shipping'] = df['shipping_free_shipping'].astype(int)
    df['is_active'] = (df['status'] == 'active').astype(int)
    df.drop(columns=['warranty', 'shipping_free_shipping', 'status'], inplace=True)

    # Encodeo columnas categoricas
    le = LabelEncoder()
    label_encoders = {}

    for col in df.columns:
        if is_object_dtype(df[col]) and df[col].nunique() < 31:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col])
            label_encoders[col] = le  # para desencodear posteriormente si es necesario

    if 'condition' in df.columns:
        df['condition_enc'] = df['condition'].map({'new': 1, 'used': 0}) # para asegurar que el encodeo es correcto
        
    print(f"Columnas  codificadas como 'Nombre_Columna_enc'.")

    # Encodear booleanos
    bool_cols = df.select_dtypes(include='bool').columns
    for col in bool_cols:
        df[col + '_enc'] = df[col].astype(int)

    print(f"Columnas booleanas codificadas: {list(bool_cols)}")

    return df


if __name__ == "__main__":

    # Cargar el dataset
    path = "../data/MLA_100k_checked_v3.jsonlines"
    data = [json.loads(x) for x in open(path)]
    
    # Preprocesar el dataset
    df = preprocess(data)

    # Guardar el DataFrame preprocesado en un archivo parquet
    df.to_parquet('../data/preprocessed_data.parquet', index=False)