"""
Exercise description
--------------------

Description:
In the context of Mercadolibre's Marketplace an algorithm is needed to predict if an item listed in the markeplace is new or used.

Your tasks involve the data analysis, designing, processing and modeling of a machine learning solution 
to predict if an item is new or used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k_checked_v3.jsonlines` and a function to read that dataset in `build_dataset`.

For the evaluation, you will use the accuracy metric in order to get a result of 0.86 as minimum. 
Additionally, you will have to choose an appropiate secondary metric and also elaborate an argument on why that metric was chosen.

The deliverables are:
--The file, including all the code needed to define and evaluate a model.
--A document with an explanation on the criteria applied to choose the features, 
  the proposed secondary metric and the performance achieved on that metrics. 
  Optionally, you can deliver an EDA analysis with other formart like .ipynb



"""

import json
from utils import scale_data, filtrar_y_seleccionar_features
from preprocessing import preprocess
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import logging



# You can safely assume that `build_dataset` is correctly implemented
def build_dataset():
    data = [json.loads(x) for x in open("../data/MLA_100k_checked_v3.jsonlines")]
    target = lambda x: x.get("condition")
    N = -10000
    X_train = data[:N]
    X_test = data[N:]
    y_train = [target(x) for x in X_train]
    y_test = [target(x) for x in X_test]
    for x in X_test:
        del x["condition"]
    return X_train, y_train, X_test, y_test


def funcion_principal(X_train, y_train, X_test, y_test):
    """
    Esta función es la principal del script. Se encarga de cargar el dataset, 
    preprocesarlo, entrenar el modelo y evaluar su rendimiento.
    """
    ######## PREPROCESSING ########

    X_train_pp = preprocess(X_train)
    X_test_pp = preprocess(X_test)

    cols_a_utilizar = ['start_time',
    'non_mp_methods_count',
    'is_active',
    'currency_id_enc',
    'has_warranty',
    'shipping_mode_enc',
    'listing_type_id_enc',
    'price',
    'buying_mode_enc',
    'automatic_relist_enc',
    'has_original_price',
    'pictures_count',
    'seller_address_state_name_enc',
    'sold_quantity',
    'stop_time',
    'seller_id',
    'has_nuevo_in_title',
    'initial_quantity']

    X_train_top, X_test_top, y_train, y_test, top_features = filtrar_y_seleccionar_features(X_train_pp, X_test_pp, y_train, y_test, logger=logger, select_cols=cols_a_utilizar)


    ## Escaleo los datos
    X_train_top, X_test_top = scale_data(X_train_top, X_test_top)


    # Entrenar el modelo con hiperparámetros óptimos
    model = XGBClassifier(
            n_estimators=900,
            max_depth=12,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.7,
            min_child_weight=1,
            gamma=0,
            scale_pos_weight=1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
    )

    logger.info("\nEntrenando modelo XGBoost con hiperparámetros ajustados...")
    model.fit(X_train_top, y_train)

    # Predicciones
    y_pred = model.predict(X_test_top)

    # Evaluación
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.info("\n Resultados del modelo:")
    logger.info(f"Accuracy:  {accuracy:.2f}")
    logger.info(f"Precision: {precision:.2f}")
    logger.info(f"Recall:    {recall:.2f}")
    logger.info(f"F1 Score:  {f1:.2f}")



    ############## Guardar metricas y exportar
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("../output/metricas_modelo.csv", index=False)

    logger.info(" Archivo 'metricas_modelo.csv' guardado.")
    

    # Guardar dataset de test con predicciones
    # Mapeo de clase binaria a string
    label_map = {1: "new", 0: "used"}

    # Agregar predicción en formato string
    for i, item in enumerate(X_test):
        item['predicted_condition'] = label_map[int(y_pred[i])]
        item['real_condition'] = label_map[int(y_test[i])]

    # Guardar como .jsonlines
    with open("../output/X_test_with_predictions.jsonlines", "w", encoding="utf-8") as f:
        for item in X_test:
            f.write(json.dumps(item) + "\n")

    logger.info(" Archivo 'X_test_with_predictions.jsonlines' guardado.")



# Configurar logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Imprime en consola
        # logging.FileHandler("pipeline.log")  # Descomentar si se quiere guardar en archivo, paso util en caso de escalar el script mas alla del challenge
    ]
)

logger = logging.getLogger(__name__)
 

if __name__ == "__main__":

    logger.info("Loading dataset...")
    # Train and test data following sklearn naming conventions
    # X_train (X_test too) is a list of dicts with information about each item.
    # y_train (y_test too) contains the labels to be predicted (new or used).
    # The label of X_train[i] is y_train[i].
    # The label of X_test[i] is y_test[i].
    X_train, y_train, X_test, y_test = build_dataset()


    # Insert your code below this line:
    # ...

    # Llamar a la función principal
    funcion_principal(X_train, y_train, X_test, y_test)
    

    