
# 🛍️ Clasificador de Productos: ¿Nuevo o Usado?

Este proyecto construye un modelo de machine learning para predecir si una publicación en el Marketplace de Mercado Libre corresponde a un producto **nuevo** o **usado**, utilizando información estructurada provista por la plataforma.

---

##  Estructura del proyecto

```
project-root/
│
├── data/
│   └── MLA_100k_checked_v3.jsonlines       # Dataset original
│
├── output/
│   ├── metricas_modelo.csv                 # Métricas del modelo final
│   └── X_test_with_predictions.jsonlines   # Conjunto de test con predicciones
│
├── src/
│   ├── eda.ipynb                           # Notebook opcional de análisis exploratorio
│   ├── new_or_used.py                      # Script principal ejecutable
│   ├── preprocessing.py                    # Preprocesamiento y feature engineering
│   └── utils.py                            # Funciones auxiliares (scaling, tuning, helpers)
│
└── requirements.txt                        # Dependencias del proyecto
```

---

##  Cómo correr el proyecto

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Ejecutar el modelo

```bash
python src/new_or_used.py
```

Esto entrenará el modelo, evaluará su rendimiento y generará dos archivos en la carpeta `output/`:

- `metricas_modelo.csv`
- `X_test_with_predictions.jsonlines`


---

##  Decisiones de diseño

- **Target**: Se codificó como `1 = new`, `0 = used`.
- **Preprocesamiento**: Se eliminan columnas con más del 80% de valores nulos, con mayoria de valores string vacios o listas vacias, irrelevantes y colineales.

Se extraen features como:
  - `pictures_count`, `non_mp_methods_count`, `title_word_count`, `has_nuevo_in_title`.

Se codifican binarias utiles como:
  - `is_free_shipping`, `is_active`, `has_warranty` y `has_original_price`.
- **Encoding**: Se codifican columnas categóricas de baja cardinalidad con `LabelEncoder`. Las booleanas se convierten a `int`.
- **Selección de variables**: Se seleccionan las features numéricas más correlacionadas con el target (`condition_enc`).
- **Escalado**: `MinMaxScaler` aplicado sólo a las features seleccionadas.
- **Modelo**: Se usó `XGBoostClassifier` con hiperparámetros ajustados vía `RandomizedSearchCV`.
- **Exportación**: El conjunto `X_test` se guarda con columnas `predicted_condition` y `real_condition` en formato `.jsonlines`.

---

##  Resultados del modelo

| Métrica      | Valor alcanzado |
|--------------|------------------|
| Accuracy     | ≥ 0.88           |
| Precision    | ✓ Calculada      |
| Recall       | ✓ Calculada      |
| F1 Score     | ✓ Calculada      |

> Todas las métricas se pueden consultar en `output/metricas_modelo.csv`.

###  Métricas utilizadas

- **Métrica principal: Accuracy**  
  En problemas de clasificación balanceados, la **accuracy** es una métrica adecuada ya que representa la proporción de predicciones correctas del modelo. Fue requerida explícitamente en la consigna y es fácilmente interpretable.

- **Métrica secundaria: Precision**  
  Se seleccionó la **precisión** como métrica secundaria porque nos indica **qué tan confiable es el modelo cuando predice que un producto es nuevo**.  
  En este contexto, **minimizar falsos positivos** es crítico: no queremos que el modelo prediga "nuevo" cuando en realidad el producto es usado, ya que eso podría generar una **mala experiencia para el cliente** y tener implicancias comerciales serias.  
  Por esta razón, se privilegia la precisión por sobre el recall como segunda métrica clave.

---

##  Hiperparámetros del modelo final

```python
XGBClassifier(
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
```

---

##  Dependencias clave

- `pandas`, `scikit-learn`, `xgboost`
- `matplotlib`, `seaborn` (EDA opcional)

> Todas especificadas en `requirements.txt`

---

##  Análisis Exploratorio (ejecucion opcional)

El archivo `eda.ipynb` explora:
- Distribución de clases
- Calidad de los datos
- Correlaciones
- Posibles outliers o campos irrelevantes
- Experimentacion de modelos
- Seleccion de columnas optimas
- Ajuste de hiperparametros


---

## Próximos pasos

- Incorporar NLP o LLM sobre título y descripción para generar nuevas features.
- Profundizar análisis de columnas no tenidas en cuenta en esta primera instancia.



