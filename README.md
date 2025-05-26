
#  Product Classifier: New or Used?

This project builds a machine learning model to predict whether a product listed on Mercado Libre's Marketplace is **new** or **used**, using structured information provided by the platform.

---

##  Project Structure

```
project-root/
│
├── data/
│   └── MLA_100k_checked_v3.jsonlines       # Original dataset
│
├── output/
│   ├── metricas_modelo.csv                 # Final model metrics
│   └── X_test_with_predictions.jsonlines   # Test set with predictions
│
├── src/
│   ├── eda.ipynb                           # Optional exploratory analysis notebook
│   ├── new_or_used.py                      # Main executable script
│   ├── preprocessing.py                    # Preprocessing and feature engineering
│   └── utils.py                            # Helper functions (scaling, tuning, etc.)
│
└── requirements.txt                        # Project dependencies
```

---

##  How to run the project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the model

>  Make sure the file `MLA_100k_checked_v3.jsonlines` is located in the `data/` folder at the root of the project.

```bash
python src/new_or_used.py
```

This will train the model, evaluate its performance, and generate the following output files in the `output/` folder:

- `metricas_modelo.csv`
- `X_test_with_predictions.jsonlines`

---

##  Design Decisions

**Target**: Encoded as `1 = new`, `0 = used`.

**Preprocessing**: Columns with more than 80% missing values, mostly empty strings or empty lists, irrelevant, or highly collinear were removed.

**Engineered features**:
  - `pictures_count`: number of images in the listing.
  - `non_mp_methods_count`: number of non-MercadoPago payment methods.
  - `title_word_count`: number of words in the title.
  - `has_nuevo_in_title`: binary flag indicating if the title contains the word "nuevo" ("new").

**Useful binary flags**:
  - `is_free_shipping`: whether the product offers free shipping.
  - `is_active`: whether the listing is currently active.
  - `has_warranty`: whether a warranty is declared.
  - `has_original_price`: whether an original price is provided (few but informative).

**Encoding**: Low-cardinality categorical columns were encoded using `LabelEncoder`, identified by the `_enc` suffix. Boolean values were cast to `int`.

**Feature selection**: Numerical features most correlated with the target (`condition_enc`) or validated via experimentation were retained.

**Scaling**: Applied `MinMaxScaler` only to selected features.

**Model**: `XGBoostClassifier` was used with hyperparameters tuned via `RandomizedSearchCV`.

**Export**: The `X_test` set is saved with `predicted_condition` and `real_condition` columns in `.jsonlines` format.

---

##  Model Results

| Metric     | Value          |
|------------|----------------|
| Accuracy   | 0.89           |
| Precision  | 0.91           |
| Recall     | 0.89           |
| F1 Score   | 0.90           |

> All metrics can be found in `output/metricas_modelo.csv`.

### Metrics used

- **Primary metric: Accuracy**

  In balanced classification problems, **accuracy** is a reliable metric as it represents the overall proportion of correct predictions. It was explicitly required in the assignment.

- **Secondary metric: Precision**

  **Precision** was selected as the secondary metric because it indicates **how reliable the model is when predicting that a product is new**.
  In this context, **minimizing false positives** is critical—we want to avoid predicting “new” when a product is actually used, as this could harm customer experience and trust.
  Therefore, precision is prioritized over recall as a secondary metric.

---

##  Final Model Hyperparameters

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

Selected columns used for training:

- start_time
- non_mp_methods_count
- is_active
- currency_id_enc
- has_warranty
- shipping_mode_enc
- listing_type_id_enc
- price
- buying_mode_enc
- automatic_relist_enc
- has_original_price
- pictures_count
- seller_address_state_name_enc
- sold_quantity
- stop_time
- seller_id
- has_nuevo_in_title
- initial_quantity

---

##  Key Dependencies

- `pandas`, `scikit-learn`, `xgboost`
- `matplotlib`, `seaborn` (optional for EDA)

> All specified in `requirements.txt`.

---

##  Exploratory Analysis (optional)

The notebook `eda.ipynb` explores:
- Class distribution
- Data quality
- Feature correlations
- Outliers and irrelevant fields
- Model experimentation
- Optimal feature subset selection
- Hyperparameter tuning

---

##  Next Steps

- Integrate NLP or LLMs to extract features from title/description.
- Further analyze unused or weak features.
