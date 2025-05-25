import pandas as pd
import ast
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


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
    print(f"\n✅ Mejor score ({model_name}): {search.best_score_:.4f}")
    print(f"📌 Mejores parámetros: {search.best_params_}")
    
    return search.best_estimator_


if __name__ == "__main__":
    pass