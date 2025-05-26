import pandas as pd
import ast
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_ordered_columns(data):
    """
    Extrae el orden original de las columnas de la data
    """
    def flatten_keys(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_keys(v, new_key))
            else:
                items.append(new_key)
        return items
    
    return flatten_keys(data[0])



def extract_id_from_string_list(entry):
    """
    Extrae el valor del campo 'id' desde una lista que contiene un string 
    representando un diccionario con clave 'id'.
    """
    if isinstance(entry, list) and len(entry) > 0:
        try:
            first_item = ast.literal_eval(entry[0])  # convierte el string a dict
            return first_item.get('id')
        except Exception:
            return None
    return None


def expand_list_column(df, col_name, prefix, max_items=None, fields=None):
    """
    Aplana una columna que contiene listas de diccionarios, extrayendo campos específicos.

    - col_name: nombre de la columna a expandir.
    - prefix: prefijo para las nuevas columnas.
    - max_items: máximo de ítems a considerar (por defecto, se usa el máximo real).
    - fields: campos del diccionario que se desean extraer (por defecto, todos los que aparecen en el primero).
    """
    df = df.copy()
    df[col_name] = df[col_name].apply(lambda x: x if isinstance(x, list) else [])

    if max_items is None:
        max_items = df[col_name].apply(len).max()

    # Determinar campos si no se pasan
    if fields is None:
        for entry in df[col_name]:
            if entry:  # Lista no vacía
                fields = list(entry[0].keys())
                break
        if fields is None:
            fields = []

    for i in range(int(max_items)):
        for f in fields:
            new_col = f"{prefix}_{i}_{f}"
            df[new_col] = df[col_name].apply(
                lambda x: x[i].get(f) if i < len(x) else None
            )

    return df.drop(columns=[col_name])



def tune_model(model_name, X_train, y_train, params=None, n_iter=30, cv=5):
    """
    Ajusta el modelo especificado con RandomizedSearchCV para encontrar los mejores hiperparámetros.
    model_name: nombre del modelo a ajustar ('xgboost' o 'random_forest').
    X_train: conjunto de entrenamiento.
    y_train: etiquetas de entrenamiento.
    """
    if model_name == "xgboost":
        model = XGBClassifier(random_state=42)
    elif model_name == "random_forest":
        model = RandomForestClassifier(random_state=42)
    else:
        raise ValueError("model_name debe ser 'xgboost' o 'random_forest'")
    


    search = RandomizedSearchCV(
        model,
        param_distributions=params,
        n_iter=n_iter,
        scoring='accuracy',
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)
    print(f"\n Mejor score ({model_name}): {search.best_score_:.4f}")
    print(f"📌 Mejores parámetros: {search.best_params_}")
    
    return search.best_estimator_


def scale_data(X_train, X_test, method="minmax"):
    """
    Escala X_train y X_test con el método especificado.

    Parámetros:
    -----------
    X_train : DataFrame
        Conjunto de entrenamiento (numérico).
    X_test : DataFrame
        Conjunto de test (numérico).
    method : str
        Método de escalado: "minmax" o "standard".

    Retorna:
    --------
    X_train_scaled : np.ndarray
    X_test_scaled : np.ndarray
    scaler : fitted Scaler object
    """
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("Método no válido. Usar 'minmax' o 'standard'.")

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def filtrar_y_seleccionar_features(X_train_pp, X_test_pp, y_train, y_test, logger=None, threshold_corr=0.0, select_cols=None):
    """
    Prepara los datos para modelado: codifica el target, revisa nulos,
    selecciona columnas numéricas y filtra las más correlacionadas con el target.

    Parámetros:
    -----------
    X_train_pp, X_test_pp : pd.DataFrame
        Datos preprocesados de entrenamiento y test.
    y_train, y_test : list
        Targets crudos (como strings 'new'/'used').
    logger : logging.Logger (opcional)
        Para imprimir trazas de forma controlada.
    threshold_corr : float
        Umbral mínimo de correlación para conservar una columna.

    Retorna:
    --------
    X_train_top, X_test_top : pd.DataFrame (escalado no aplicado aún)
    y_train_bin, y_test_bin : list[int] (target binario)
    top_features : list[str] (nombres de columnas seleccionadas)
    """
    
    # 1. Target binario
    y_train_bin = list(map({'new': 1, 'used': 0}.get, y_train))
    y_test_bin = list(map({'new': 1, 'used': 0}.get, y_test))

    # 2. Chequeo de nulos
    if logger:
        logger.info('Hay nulos en X_test? %s', X_test_pp.isna().sum().any())
        logger.info('Hay nulos en X_train? %s', X_train_pp.isna().sum().any())
        if X_train_pp.isna().sum().any() or X_test_pp.isna().sum().any():
            raise ValueError("Hay nulos en los datos. Asegúrate de que estén tratados antes de continuar.")


    # 3. Selección de columnas numéricas
    numeric_cols = X_train_pp.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols_test = X_test_pp.select_dtypes(include=['int64', 'float64']).columns
    X_train_numeric = X_train_pp[numeric_cols]
    X_test_numeric = X_test_pp[numeric_cols_test]

    # 4. Selección por correlación
    df_matriz = X_train_numeric.copy()
    if 'condition_enc' not in df_matriz.columns:
        raise ValueError("La columna 'condition_enc' no está en los datos. Asegurate de que esté codificada.")

    correlations = df_matriz.corr()['condition_enc'].drop('condition_enc')
    top_corr = correlations.reindex(correlations.abs().sort_values(ascending=False).index)

    if logger:
        logger.info('Top de correlaciones con el target:')
        logger.info(top_corr.head(10).round(2))

    mask = top_corr.abs() > threshold_corr
    top_features = top_corr[mask].index.tolist()

    if select_cols is not None:
        top_features = select_cols

    X_train_top = X_train_numeric[top_features]
    X_test_top = X_test_numeric[top_features]

    
    # verifico que efectivamnete no este target ni en train ni test
    if logger:
        logger.info('Hay columna condition en X_train? %s', 'condition_enc' in X_train_top.columns)
        logger.info('Hay columna condition en X_test? %s', 'condition_enc' in X_test_top.columns)
        if 'condition_enc' in X_train_top.columns:
            raise ValueError("La columna 'condition_enc' no debe estar en los datos de entrenamiento o test. Asegúrate de que esté eliminada.")
        
        logger.info('Columnas utilizadas para entrenar:')
        logger.info(X_train_top.columns)

    return X_train_top, X_test_top, y_train_bin, y_test_bin, top_features


if __name__ == "__main__":
    pass