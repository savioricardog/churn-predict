
#%% [markdown]
# # --- IMPORTS ---
#%%
import kagglehub
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from feature_engine import discretisation, encoding
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import mlflow

from sklearn.cluster import KMeans

sys.path.append(os.path.abspath(os.path.join('..')))
from src.eng_funcs import CleanTransformStrNum, AnalyseDataSet

#%% [markdown]
# ## -- CONFIGURING JUPYTER PAGE --
#%%
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 500)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(experiment_id='1')

%load_ext autoreload
%reload_ext autoreload
%autoreload 2

#%% [markdown]
# ## -- DOWNLOAD DATASET LATEST VERSION --
#%%
path = kagglehub.dataset_download("yeanzc/telco-customer-churn-ibm-dataset")
print("Path to dataset files:", path)


#%% [markdown]
# ## -- VARIABLES CONFIG --
#%%
path_dataset = os.path.join(path, 'Telco_customer_churn.xlsx')

#%% [markdown]
# # --- READ AND SAMPLE DATASET ---
#%%
df = pd.read_excel(path_dataset, engine = 'calamine')
df.head(5)


#%% [markdown]
# # --- FEATURE ENGINE ---

#%% [markdown]
# ## -- DF_AUX TEST --
#%%

df_aux = df.copy()

# ADJUSTING TOTAL CHARGES COLUMN
df_aux['Total Charges'] = df_aux['Total Charges'].apply(lambda x: 0 if x == ' ' else x)

# FEATURE RELATIVE PRICE UP OR DOWN
df_aux['Price Up Recently'] = np.where(
                            df_aux['Tenure Months'] > 0,
                            df_aux['Total Charges'] / df_aux['Tenure Months'],
                            df_aux['Monthly Charges'])

# PERCENT PRICE DIFFERENCE
df_aux['Price Hike'] = df_aux['Monthly Charges'] - df_aux['Price Up Recently']

# FEATURE RELATIVE PRICE ESTIMATED VS PRICE PRICE PAYED
df['Price Sensitivity'] = df['Monthly Charges'] / df['CLTV']

# SEPARATING SERVICE COLUMNS
services_offer = ['Phone Service','Multiple Lines','Internet Service',
                  'Online Security','Online Backup','Device Protection',
                  'Tech Support','Streaming TV','Streaming Movies']

# CREATING NEW FEATURES
df_aux['Score dependency'] = 0
df_aux['Lazer Products'] = 0
df_aux['Security Product'] = 0

# FLAG FEATURE ABOUT BOUGHT PRODUCTS
for score in df_aux[services_offer]:
    points = np.where(df_aux[score].astype(str).str.contains('No'), 0, 1)
    df_aux['Score dependency'] = df_aux['Score dependency'] + points

for lazer in df[['Streaming TV','Streaming Movies']]:
    points = np.where(df[lazer].astype(str).str.contains('No'), 0, 1)
    df_aux['Lazer Products'] = df_aux['Lazer Products'] + points

for security in df[['Online Security','Online Backup','Device Protection']]:
    points = np.where(df[security].astype(str).str.contains('No'), 0, 1)
    df_aux['Security Product'] = df_aux['Security Product'] + points

# SENIOR VULNERABILITY FLAG
df_aux['Senior Vulnerable'] = np.where((df_aux['Senior Citizen'] == 'Yes') & (df_aux['Tech Support'] == 'No'), 1, 0)

# FAMILY AMOUNT CONTRACT FLAG
df_aux['Family'] = (df_aux['Partner'] == 'Yes').astype(int) + (df_aux['Dependents'] == 'Yes').astype(int)

# CLUSTERING GEOSPACES
X_geo = df_aux[['Latitude','Longitude']]
kmeans = KMeans(n_clusters=30, random_state=42)
df_aux['Geo Cluster'] = kmeans.fit_predict(X_geo).astype(str)

# PAYMENT RISK FLAG
df_aux['Payment Risk'] = np.where((df_aux['Payment Method'] == 'Credit card (automatic)') 
                                    | (df_aux['Payment Method'] == 'Bank transfer (automatic)'),
                                        0, 1)

# FEATURE REMAINING TIME CONTRACT 
df_aux['Time Contract'] = df_aux['Contract'].apply(lambda x: 24 if x == 'Two year' else
                                                             12 if x == 'One year' else  1).astype(int)

# MEASURING MONTHS FOR CONTRACT RENEWAL 
df_aux['Months to Renewal'] = df_aux['Time Contract'] - (df_aux['Tenure Months'] % df_aux['Time Contract'])

# FINDING CONTRACT NEXT TO THE END
df_aux['Last Three Months'] = np.where(df_aux['Time Contract'] <= 3, 1, 0)

df['High Tech No Support'] = np.where((df['Internet Service'] == 'Fiber optic') & (df['Tech Support'] == 'No'), 1, 0)

#%% [markdown]
# ## -- JOINING DF_AUX WITH DF OFICIAL --
#%%

new_columns = df_aux.columns.difference(df.columns)
df = df.join(df_aux[new_columns])

#%% [markdown]
# # --- UNDERSTANDING DATASET - EDA ---
#%% [markdown]
# ## -- DATASET GENERAL INFOS --
#%%
print(df.info())
print(f'\n Shape df: {df.shape}')

# DESCRIBING DATASET
df.describe().T

#%% [markdown]
# ## -- ANALISING DATASET VALUES (IF HAS FALSE NULL VALUES OR TRUE NULL VALUES) --
#%%
# FUNCTION CREATED FOR ANALYSE DATASET
AnalyseDataSet(df)

#%% [markdown]
# ## -- CREATING RANGE COLUMNS FOR BETTER UNDERSTAND --
# CREATING TENURE MONTHS RANGE
df['Tenure Months Range'] = df['Tenure Months'].apply(lambda x: '00 to 05' if x <= 5 else
                                                                '06 to 10' if x <= 10 else 
                                                                '11 to 15' if x <= 15 else
                                                                '16 to 20' if x <= 20 else
                                                                '21 to 25' if x <= 25 else
                                                                '26 to 30' if x <= 30 else
                                                                '31 to 35' if x <= 35 else
                                                                '36 to 40' if x <= 40 else
                                                                '41 to 45' if x <= 45 else
                                                                '46 to 50' if x <= 50 else
                                                                '51 to 55' if x <= 55 else
                                                                '56 to 60' if x <= 60 else
                                                                '61 to 65' if x <= 65 else
                                                                '66 to 70' if x <= 70 else
                                                                '71 to 75' if x <= 75 else
                                                                '76 to 80' if x <= 80 else
                                                                '81 to 85' if x <= 85 else
                                                                '86 to 90' if x <= 90 else
                                                                '91 to 95' if x <= 95 else
                                                                '96 to 100')

df['Tenure Months Range'].value_counts().sort_values(ascending=False)

df['Churn Score Range'] = df['Churn Score'].apply(lambda x: '00 to 10' if x <= 10 else
                                                            '11 to 20' if x <= 20 else 
                                                            '21 to 30' if x <= 30 else
                                                            '31 to 40' if x <= 40 else
                                                            '41 to 50' if x <= 50 else
                                                            '51 to 60' if x <= 60 else
                                                            '61 to 70' if x <= 70 else
                                                            '71 to 80' if x <= 80 else
                                                            '81 to 90' if x <= 90 else
                                                            100
                                                            )

corr = df.corr(numeric_only=True, 
                method='pearson')['Churn Value'].sort_values(ascending=False).to_frame()
corr.columns = ['Correlation']
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(10,6))
sns.heatmap(data=corr, cmap='coolwarm', fmt='.2f', annot=True, mask=mask)
plt.title('Correlation Plot')
plt.show()


# PLOTING CHURN DISTRIBUITION
df['Churn Value'].value_counts(normalize=True)

plt.figure(figsize=(10,6))
sns.countplot(data=df, x='Churn Value', palette='viridis')
plt.title('Distribuição de Valores de Churn (1=Saiu, 0=Ficou)')
plt.show()

#%%
# LIST COLUMN TYPES
blacklist = ['CustomerID','City','Lat Long','Churn Label', 'Monthly Charges','Zip Code',
             'Latitude','Longitude','CLTV','Churn Score',
             'Total Charges', 'Tenure Months']
category_cols = df.select_dtypes(include=['object'])
num_cols = df.select_dtypes(include=['int','float'])

cat_cols = [col for col in category_cols.columns if col in category_cols and col not in blacklist]
num_cols = [col for col in df.columns if col in num_cols and col not in blacklist]

# CATEGORICAL COUNTPLOT
plt.figure(figsize=(40, 36), dpi=350)
for i, col in enumerate(cat_cols):
    plt.subplot(6, 5, i+1)
    sns.countplot(data=df, x=col, hue='Churn Value', palette='magma')
    plt.xticks(rotation=45)
    plt.title(f'Churn por {col}')

plt.tight_layout()
plt.show()


# NUMERICAL COUNTPLOT
plt.figure(figsize=(40, 36), dpi=350)
for i, col in enumerate(num_cols):
    plt.subplot(6, 5, i+1)
    sns.countplot(data=df, x=col, hue='Churn Value', palette='coolwarm')
    plt.xticks(rotation=45)
    plt.title(f'Churn por {col}')

plt.tight_layout()
plt.show()


#%% [markdown]
# # --- X,y AND Train/Test ---
#%%

target = 'Churn Value'
X, y = df.drop(columns=[target], errors='ignore'), df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    random_state=42,
                                                    test_size=0.25)


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#%% [markdown]
# # --- SEPARATING EACH ESPECIFIC TYPE VALUE (STR, NUM, DATE, CODE) ---
#%%

# UNUSED VALUES (LEAKAGE|OVERFITTING)
blacklist = ['Churn Score','Churn Label', 'CustomerID', 'Count', 
             'Total Charges', 'Tenure Months Range','Churn Score Range',
             'Lat Long', 'Churn Reason']

num_vars = [col for col in X_train.columns 
            if col not in blacklist and pd.api.types.is_numeric_dtype(X_train[col])]
cat_vars = [col for col in X_train.columns 
            if col not in blacklist and pd.api.types.is_object_dtype(X_train[col])]

#%% [markdown]
# # --- NUMBER/OBJECT PIPELINE TRANSFORMATION ---
#%%

num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

totalcharge_pipe = Pipeline([
    ('eng', CleanTransformStrNum()),
    ('scaler', StandardScaler())
])

#%% [markdown]
# # --- PREPROCESSOR DATA ---
#%%

preprocessor = ColumnTransformer(
    transformers=[
        ('tr_num', num_pipe, num_vars),
        ('tr_cat', cat_pipe, cat_vars),
        ('totalcharges', totalcharge_pipe, ['Total Charges']),
        ], 
    remainder='drop'
)

#%% [markdown]
# # --- DEFINING PARAMETERS ---
#%%

params = [
    # --- MODEL 1: RandomForestClassifier: O mais lento ---
    {
    'model': [RandomForestClassifier(n_jobs=1, random_state=42, verbose=1 )],
    'model__n_estimators': [250],
    'model__max_depth': [6, 10, None],
    'model__class_weight': ['balanced', 'balanced_subsample'],
    'model__min_samples_leaf': [1]
    },
    # --- MODEL 2: LGBMClassifier: O mais veloz ---
    {
    'model': [LGBMClassifier(n_jobs=1, force_col_wise=True, random_state=42)],
    'model__n_estimators': [100, 300],
    'model__learning_rate': [0.05, 0.1],
    'model__num_leaves': [45],
    'model__max_depth': [-1],
    'model__class_weight': ['balanced'],
    'model__min_child_samples': [20],
    'model__subsample': [0.8],
    'model__colsample_bytree': [ 0.8],
    'model__importance_type': ['gain'],
    'model__objective': ['binary']
    },
    # --- MODEL 3: XGBOOST: O mais robusto ---
    {
    'model': [XGBClassifier(n_jobs=1, force_col_wise=True, random_state=42)],
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [3, 6],
    'model__scale_pos_weight': [1, 3],
    'model__min_child_samples': [1, 5],
    'model__subsample': [0.8],
    'model__colsample_bytree': [0.8],
    'model__gamma': [0.1], # PENALITY MODEL FOR AVOID UNUSABLE LEAVES
    'model__eval_metric': ['logloss'],
    'model__gamma':[0, 0.1, 1]
    },
    # --- MODEL 4: CATBOOST: O que lida melhor com DataSet Majoritariamente categorico ---
    {
    'model': [CatBoostClassifier(allow_writing_files=False, verbose=1, random_state=42)],
    'model__n_estimators': [500, 1000],
    'model__learning_rate': [0.01, 0.05],
    'model__depth': [4, 6],
    'model__auto_class_weights': ['Balanced'],
    'model__l2_leaf_reg': [3, 5],
    'model__border_count': [128]
    }
]

#%% [markdown]
# # --- DEFINING PARAMETERS ---
#%%

model_pipe = Pipeline([
    ('model', KNeighborsClassifier())
])


#%% [markdown]
# # --- APPLYING GRIDSEARCH ---
#%%

# CONFIGURATION GRIDSERACH PARAMS
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = RandomizedSearchCV(
    estimator=model_pipe,
    param_distributions = params,
    n_iter=40,
    cv = kfold,
    scoring='f1',
    verbose=1,
    n_jobs=-1,
    random_state=42,
    refit = True
)

#%% [markdown]
# # --- FINAL PIPE ---
#%%

final_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', grid)
    ],
    memory=None
)
#%% [markdown]
# # --- FITTING GRID MODEL ---
#%%

with mlflow.start_run() as r:

    mlflow.sklearn.autolog()

    print("Fitting model!")
    model_fitted = final_pipe.fit(X_train, y_train)
    print("Model fitted!")

    print("Doing Data Predict!")
    y_pred_model = model_fitted.predict(X_test)
    y_pred_proba = model_fitted.predict_proba(X_test)[:, 1]
    print("Predict Conclued!")

    f1_score_metric = f1_score(y_test, y_pred_model)

    print('='*40)
    print(f1_score_metric)
    print('='*40)


    # 1. Converta Probabilidade para Classe (0 ou 1) usando um corte (threshold)
    # Se a probabilidade for maior que 0.5, vira 1 (Churn), senão 0.
    # Nota: Se 'proba_predicted_model' já for a classe predita, ignore essa linha.
    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
        # Se for output do .predict_proba(), pegamos a coluna 1
        y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)
    else:
        # Se for um array simples de probabilidades
        y_pred = (y_pred_proba >= 0.5).astype(int)

    # 2. Calcule o F1 (Note que mudei o nome da variável de resultado para 'f1_result')
    f1_result = f1_score(y_test, y_pred)

    print('='*40)
    print(f"F1 Score: {f1_result}")

    # 1. Relatório Detalhado (Obrigatório)
    # Veja principalmente a linha do "1" (Churn)
    print("--- Relatório de Classificação ---")
    print(classification_report(y_test, y_pred_model))

    # 2. Matriz de Confusão Visual
    # Isso mostra: "Quantos clientes que saíram o modelo disse que ficariam?"
    print("--- Matriz de Confusão ---")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_model, cmap='Blues')
    plt.show()

    # 3. Métrica de Qualidade da Probabilidade (ROC-AUC)
    # Avalia a variável 'proba_predicted_model' que você calculou mas não usou
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {auc_score:.4f}")
    print(f'{model_fitted.best_estimator_.named_steps["model"]} \n'
            f'{f1_score_metric}')

    results_df = pd.DataFrame(model_fitted.cv_results_)

    cols_keeped = ['params','mean_test_score','std_test_score','rank_test_score']
    results_df = results_df[cols_keeped]


    results_df = results_df.sort_values(by='rank_test_score')

    print('Top 10 Melhores Modelos do Grid')
    pd.set_option('display.max_colwidth', None)
    display(results_df.head(10))


    model = model_fitted.best_estimator_.named_steps['model']
    importance = model.feature_importances_/100

    preprocessor_steps = model_fitted.best_estimator_.named_steps['preprocessor']
    features_names = preprocessor_steps.get_feature_names_out()

    df_importance = pd.DataFrame({
        'Feature': features_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)


    plt.figure(figsize=(10, 8))
    sns.barplot(data=df_importance.head(10), x='Importance', y='Feature', 
                palette='viridis', hue = 'Feature', legend=False)
    plt.title('Importância das Features (Random Forest)')
    plt.show()


    print(df_importance.head(10))

    roc_test = roc_curve(y_test, y_pred_proba)

    plt.figure(dpi=350)
    plt.plot(roc_test[0], roc_test[1])
    plt.legend('Teste')
    plt.grid(True)
    plt.title(f'Curva Roc')
    plt.show()
    plt.savefig('img/curva_roc.png')

    # mlflow.log_metrics({'roc_test': roc_test})
    mlflow.log_artifact('img/curva_roc.png')



# %% [markdown]
# # # --- TESTING STACKING MODELS ---
# #%%

# # TAKING BEST RANDOM FOREST MODEL
# best_rf_model = model_fitted.best_estimator_.named_steps['model']


# # CONFIGURING GREAT XGBOOST AND LGBM
# best_lgbm = LGBMClassifier(
#     n_estimators=2000,
#     learning_rate=0.04,
#     num_leaves=50,
#     class_weight='balanced',
#     random_state=42,
#     force_col_wise = True ,
#     n_jobs=1
# )

# best_xgb = XGBClassifier(
#     n_estimators = 1200,
#     learning_rate = 0.04,
#     max_depth = 5,
#     scale_pos_weight = 3,
#     random_state=42,
#     eval_metric='logloss',
#     n_jobs = 1
# )
# #%% [markdown]
# # ## -- STACKING MODELS --
# #%%
# estimators = [
#     ('rf', best_rf_model),
#     ('lgbm', best_lgbm),
#     ('xgb', best_xgb)
# ]

# stacking_model = StackingClassifier(
#     estimators=estimators,
#     final_estimator=LogisticRegression(),
#     cv=5,
#     n_jobs=5
# )

# final_stack_pipe = Pipeline([
#     ('preprocessor', preprocessor),
#     ('model', stacking_model)
# ])

# #%%

# print('Treinando Stacking Model')
# final_stack_pipe.fit(X_train, y_train)

# print('Predict Stacking Model')
# y_pred_stack = final_stack_pipe.predict(X_test)


# print(classification_report(y_test, y_pred_stack))
#%% [markdown]
# # --- PREDICTING VALUES WITH MODEL GRID ---
#%%
# # print("Doing Data Predict!")
# # predicted_model = model_fitted.predict(X_test)
# # proba_predicted_model = model_fitted.predict_proba(X_test)[:, 1]
# # print("Predict Conclued!")

# # f1_score_metric = f1_score(y_test, predicted_model)

# # print('='*40)
# # print(f1_score_metric)
# # print('='*40)


# # 1. Converta Probabilidade para Classe (0 ou 1) usando um corte (threshold)
# # Se a probabilidade for maior que 0.5, vira 1 (Churn), senão 0.
# # Nota: Se 'proba_predicted_model' já for a classe predita, ignore essa linha.
# if len(proba_predicted_model.shape) > 1 and proba_predicted_model.shape[1] > 1:
#     # Se for output do .predict_proba(), pegamos a coluna 1
#     y_pred = (proba_predicted_model[:, 1] >= 0.5).astype(int)
# else:
#     # Se for um array simples de probabilidades
#     y_pred = (proba_predicted_model >= 0.5).astype(int)

# # 2. Calcule o F1 (Note que mudei o nome da variável de resultado para 'f1_result')
# f1_result = f1_score(y_test, y_pred)

# print('='*40)
# print(f"F1 Score: {f1_result}")

# # 1. Relatório Detalhado (Obrigatório)
# # Veja principalmente a linha do "1" (Churn)
# print("--- Relatório de Classificação ---")
# print(classification_report(y_test, predicted_model))

# # 2. Matriz de Confusão Visual
# # Isso mostra: "Quantos clientes que saíram o modelo disse que ficariam?"
# print("--- Matriz de Confusão ---")
# ConfusionMatrixDisplay.from_predictions(y_test, predicted_model, cmap='Blues')
# plt.show()

# # 3. Métrica de Qualidade da Probabilidade (ROC-AUC)
# # Avalia a variável 'proba_predicted_model' que você calculou mas não usou
# auc_score = roc_auc_score(y_test, proba_predicted_model)
# print(f"ROC-AUC Score: {auc_score:.4f}")
# print(f'{model_fitted.best_estimator_.named_steps["model"]} \n'
#         f'{f1_score_metric}')

#%%
# # analisando os 10 melhores modelos
# results_df = pd.DataFrame(model_fitted.cv_results_)

# cols_keeped = ['params','mean_test_score','std_test_score','rank_test_score']
# results_df = results_df[cols_keeped]


# results_df = results_df.sort_values(by='rank_test_score')

# print('Top 10 Melhores Modelos do Grid')
# pd.set_option('display.max_colwidth', None)
# display(results_df.head(10))

#%% [markdown]
# # --- COMPARING MODEL CLASSIFICATION VS Y_TEST ---
#%%

# # names_num = num_vars

# # ohe_step = model_fitted.named_steps['preprocessor'].named_transformers_['tr_cat'].named_steps['onehot']
# # names_cat = ohe_step.get_feature_names_out(cat_vars)

# # names_custom = ['Total Charges']

# model = model_fitted.best_estimator_.named_steps['model']
# importance = model.feature_importances_/100

# preprocessor_steps = model_fitted.best_estimator_.named_steps['preprocessor']
# features_names = preprocessor_steps.get_feature_names_out()

# df_importance = pd.DataFrame({
#     'Feature': features_names,
#     'Importance': importance
# }).sort_values(by='Importance', ascending=False)



# plt.figure(figsize=(10, 8))
# sns.barplot(data=df_importance.head(10), x='Importance', y='Feature', 
#             palette='viridis', hue = 'Feature', legend=False)
# plt.title('Importância das Features (Random Forest)')
# plt.show()


# print(df_importance.head(10))

# mlflow.log_metrics()

#%% [markdown]
# # --- ANALYSING MODEL METRICS ---
#%%


#%% [markdown]
# ## --- PLOTTING MODEL METRICS RESULTS ---
#%%


#%% [markdown]
# # --- SAVING PKL MODEL ---
#%%


#%% [markdown]
# # --- LOADING MODEL PKL ---
#%%
versions = mlflow.search_model_versions(filter_string= "name = 'model_churn'")
last_version = max([int(i.version) for i in versions])

model = mlflow.sklearn.load_model(f'models:///model_churn/{last_version}')

#%%



#%%

model.
#%% [markdown]
# ## --- PREDICTING WITH NEW VALUES ---
#%%


#%% [markdown]
# ## --- ANALYSING NEW RESULT ---
#%%