
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
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from feature_engine import discretisation, encoding
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.frozen import FrozenEstimator
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.cluster import KMeans


sys.path.append(os.path.abspath(os.path.join('..')))
from src.eng_funcs import CleanTransformStrNum, AnalyseDataSet, profit_calc

#%% [markdown]
# ## -- CONFIGURING JUPYTER PAGE --
#%%
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

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
df_aux['Price Sensitivity'] = df_aux['Monthly Charges'] / df_aux['CLTV']

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

df_aux['High Tech No Support'] = np.where((df_aux['Internet Service'] == 'Fiber optic') & (df_aux['Tech Support'] == 'No'), 1, 0)

# DISCOVERING MEAN PRICE BY SERVICE 
df_aux['Average Price P/ Service'] = df_aux['Monthly Charges'] / df_aux['Score dependency']

# DISCOVERING MEAN PRICE BY GEOLOCATION
df_aux['Average By Geo'] = df_aux.groupby(by=['Geo Cluster'])[['Monthly Charges']].transform('mean')

# DISCOVERING PRICE DIFFERENCE BY CITY
df_aux['Average By Geo'] = df_aux.groupby(by=['City'])[['Monthly Charges']].transform('mean')
df_aux['Charge Diff City Mean'] = df_aux['Monthly Charges'] - df_aux['Average By Geo']

# DISCOVERING TIME TO END CONTRACT RATIO
df_aux['Tenure Ratio Contract'] = (df_aux['Time Contract'] - df_aux['Months to Renewal']) / df_aux['Time Contract']

# DISCOVERING VALUE RATIO FOR CLTV BY MONTHLY CHARGES
df_aux['Value Ratio'] = df_aux['CLTV'] / df_aux['Monthly Charges']

# DISCOVERING HOW ISOLATED CHURN PERSON IS
df_aux['Social Isolation'] = (df_aux['Family'].astype(int) + df_aux['Senior Vulnerable'])

#%% [markdown]
# ## -- JOINING DF_AUX WITH DF OFICIAL --
#%%

new_columns = df_aux.columns.difference(df.columns)
df = df.join(df_aux[new_columns])
df.head(3)

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
plt.title('Churn Value Distribution (1=Out, 0=Stay)')
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
    plt.title(f'Churn for {col}')

plt.tight_layout()
plt.show()


# NUMERICAL COUNTPLOT
plt.figure(figsize=(40, 36), dpi=350)
for i, col in enumerate(num_cols):
    plt.subplot(6, 5, i+1)
    sns.countplot(data=df, x=col, hue='Churn Value', palette='coolwarm')
    plt.xticks(rotation=45)
    plt.title(f'Churn for {col}')

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
    # --- MODEL 1: RandomForestClassifier: The Slowest ---
    {
    'model': [RandomForestClassifier(n_jobs=1, random_state=42, verbose=1 )],
    'model__n_estimators': [250, 500, 750],
    'model__max_depth': [3,6, 10, None],
    'model__class_weight': ['balanced', 'balanced_subsample', None],
    'model__min_samples_leaf': [1, 3]
    },
    # --- MODEL 2: LGBMClassifier: The Fastest ---
    {
    'model': [LGBMClassifier(n_jobs=1, force_col_wise=True, random_state=42)],
    'model__n_estimators': [600, 1000, 2000],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.3],
    'model__num_leaves': [45, 75, 100],
    'model__max_depth': [-1],
    'model__class_weight': ['balanced', None],
    'model__min_child_samples': [4, 12, 20],
    'model__subsample': [0.8],
    'model__colsample_bytree': [ 0.8],
    'model__importance_type': ['gain'],
    'model__objective': ['binary']
    },
    # --- MODEL 3: XGBOOST: The Most Robust ---
    {
    'model': [XGBClassifier(n_jobs=1, force_col_wise=True, random_state=42)],
    'model__n_estimators': [200, 500, 900],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.3],
    'model__max_depth': [3, 6, 8],
    'model__scale_pos_weight': [1, 3, 5, None],
    'model__min_child_samples': [1, 3, 8],
    'model__subsample': [0.8],
    'model__colsample_bytree': [0.8],
    'model__gamma': [0.1], # PENALITY MODEL FOR AVOID UNUSABLE LEAVES
    'model__eval_metric': ['logloss'],
    'model__gamma':[0, 0.1, 1]
    },
    # --- MODEL 4: CATBOOST: Works Better W/ Categorical Datasets ---
    {
    'model': [CatBoostClassifier(allow_writing_files=False, verbose=1, random_state=42)],
    'model__n_estimators': [500, 1000, 1500],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.3],
    'model__depth': [4, 6, 9],
    'model__auto_class_weights': ['Balanced'],
    'model__l2_leaf_reg': [3, 5, 7],
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
    n_iter=100,
    cv = kfold,
    scoring='roc_auc',
    verbose=1,
    n_jobs=10,
    random_state=42,
    refit = True
)
#%% [markdown]
# # --- FINAL PIPE ---
#%%

final_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('grid', grid)
    ],
    memory=None
)

#%%


# break
#%% [markdown]
# # --- FITTING MODEL PIPELINE ---
#%%

with mlflow.start_run() as r:

    # RUNNING MLFLOW LOG
    mlflow.sklearn.autolog()

    #/*****************************************************************************************/

    # FITTING MODEL
    print("Fitting model!")
    model_fit = final_pipe.fit(X_train, y_train)
    print("Model fitted!")
    
    #/*****************************************************************************************/

    # PREDICTING AND METRICS
    print("Doing Train Predict!")
    y_pred_train = model_fit.predict(X_train)
    y_proba_train = model_fit.predict_proba(X_train)[:, 1]
    roc_train_score = metrics.roc_auc_score(y_train, y_pred_train)
    f1_score_train = metrics.f1_score(y_train, y_pred_train)
    prauc_score_train = metrics.average_precision_score(y_train, y_pred_train)
    print("Train Predict Conclued!")

    print("Doing Test Predict!")
    y_pred_test = model_fit.predict(X_test)
    y_proba_test = model_fit.predict_proba(X_test)[:, 1]
    roc_test_score = metrics.roc_auc_score(y_test, y_pred_test)
    f1_score_test = metrics.f1_score(y_test, y_pred_test)
    prauc_score_test = metrics.average_precision_score(y_test, y_pred_test)
    print("Test Predict Conclued!")

    
    # PLOTING ROC CURVE AND F1 SCORE
    roc_train = metrics.roc_curve(y_train, y_proba_train)
    roc_test = metrics.roc_curve(y_test, y_proba_test)

    plt.figure(dpi=350)
    plt.plot(roc_train[0], roc_train[1])
    plt.plot(roc_test[0], roc_test[1])
    plt.legend([f"Train: {roc_train_score:.2f}",
               f"Test: {roc_test_score:.2f}"])
    plt.plot([0,1],[0,1], '--', color='black')
    plt.grid(True)
    plt.title(f'Roc Curve (Train/Test)')
    plt.savefig('img/roc_curve_train_test.png')
    mlflow.log_artifact('img/roc_curve_train_test.png')
    plt.show()

    best_model = model_fit.named_steps['grid'].best_estimator_.named_steps['model']
    model_name = best_model.__class__.__name__

    # PRINT METRICS
    print('='*40)
    print(f'AUC SCORE: {roc_test_score:.2f}')
    print('='*40)
    print(f'F1 SCORE: {f1_score_test:.2f}')
    print('='*40)
    print(f'BEST ESTIMATOR: {model_name}')
    print('='*40)

    # TRIYNG IMPROVE MODL WITH THRESHOLD VALUE
    threshhold_first_test = 0.5
    if len(y_proba_test.shape) > 1 and y_proba_test.shape[1] > 1:
        y_pred_threshold = (y_proba_test[:, 1] >= threshhold_first_test).astype(int)
    else:
        y_pred_threshold = (y_proba_test >= threshhold_first_test).astype(int)
    f1_score_threshold = metrics.f1_score(y_test, y_pred_threshold)
    
    print('='*40)
    print(f"F1 Score threshold: {f1_score_threshold:.2f}")
    print('='*40)

    # PRINTING CLASSIFICATION REPORT TEST
    print("--- Classification Report - TEST ---")
    print(metrics.classification_report(y_test, y_pred_test))

    # PRINTING CONFUSION MATRIX
    print("--- Confusion Matrix ---")
    metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, cmap='Blues')
    plt.show()

    # CALCULATING PROFITS WITH CHURNERS SAVED BY MODEL PRÉ THRESHOLD
    ltv_test = X_test['CLTV'].mean()
    cost_test = ltv_test * 0.05
    sr_test = 0.5 # success rate
    threshold_profit_test = np.linspace(0, 1, 101)
    profits_test = [profit_calc(y_test, y_proba_test, ltv=ltv_test, cost=cost_test, sr=sr_test, threshold=t) for t in threshold_profit_test]

    best_idx_profit = np.argmax(profits_test)
    best_threshold_profit_test = threshold_profit_test[best_idx_profit]
    max_proft_test = profits_test[best_idx_profit]

    # PROFIT CURVE
    plt.figure(figsize=(8,6))
    plt.plot(threshold_profit_test, profits_test, label='Estimated Profit', color='green', linewidth = 2.0)

    # FIND BETTER FINANCIAL POINT
    plt.scatter(best_threshold_profit_test, max_proft_test, color='red', s=100, zorder=5)
    plt.axvline(best_threshold_profit_test, linestyle='--', color='red', alpha=0.5, label=f'Best ThresHold for Profits {best_threshold_profit_test:.2f}')

    # THRESHOLD PROFITS VS THRESHOLD MODEL
    plt.axvline(threshhold_first_test, linestyle=':', color='blue', label=f'ThresHold F1 {threshhold_first_test:.2f}' )

    plt.title(f'Profit Test Curve by ThresHold \n Profit Max {max_proft_test:,.2f} in {best_threshold_profit_test:.2f}', fontsize=14)
    plt.xlabel('Decision ThresHold (probability)', fontsize=12)
    plt.ylabel('Estimated Profit', fontsize = 12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('img/business_profit_curve_test.png')
    mlflow.log_artifact('img/business_profit_curve_test.png')
    plt.show()

    good_preds_test = (y_proba_test >= best_threshold_profit_test).astype(int)
    tn_test, fp_test, fn_test, tp_test = metrics.confusion_matrix(y_test, good_preds_test).ravel()

    total_money_risk_test = (tp_test + fn_test) * ltv_test
    percent_save_test = (max_proft_test / total_money_risk_test) * 100

    print('='*40)
    print(f'Financial Impact Analysis (In Test)')
    print(f'Total Risk Money {total_money_risk_test:.2f}')
    print(f'Model Estimated Profit {max_proft_test:.2f}')
    print(f'Percent Loss Saved {percent_save_test:.2f}')
    print('='*40)

    mlflow.log_metric("pct_revenue_saved_test", percent_save_test)
    mlflow.log_metric("total_money_at_risk_test", total_money_risk_test)
    #/*****************************************************************************************/

    # TOP 10 BEST ESTIMATORS
    cols_keeped = ['params','mean_test_score','std_test_score','rank_test_score']
    results_df = pd.DataFrame(model_fit.named_steps["grid"].cv_results_)
    results_df = results_df[cols_keeped]
    results_df = results_df.sort_values(by='rank_test_score')
    print('Top 10 Grid Models')
    pd.set_option('display.max_colwidth', None)
    display(results_df.head(5))

    #/*****************************************************************************************/

    # DF FEATURE IMPORTANCE
    importance = best_model.feature_importances_/100
    preprocessor_steps = model_fit.named_steps['preprocessor']
    features_names = preprocessor_steps.get_feature_names_out()

    df_importance = pd.DataFrame({
        'Feature': features_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=df_importance.head(10), x='Importance', y='Feature', 
                palette='viridis', hue = 'Feature', legend=False)
    plt.title(F'Feature Importance ({best_model})')
    plt.show()
    print(df_importance.head(10))

    #/*****************************************************************************************/

    # TESTING CALIBRATION CURVE FOR UNDERSTAND BETTER THRESHOLD
    prob_true, prob_pred = calibration_curve(y_test, y_proba_test, n_bins=10, strategy='uniform')

    plt.figure(figsize=(8,8))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=model_name)
    plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Calibrated Perfectly')
    plt.ylabel('Real Positive Frac (Reality)')
    plt.xlabel('Predicted Probability (Model predict)')
    plt.title('Calibration Curve (Test)')
    plt.legend()
    plt.savefig('img/calibration_curve_test.png')
    mlflow.log_artifact('img/calibration_curve_test.png')
    plt.show()

    X_calib, X_val, y_calib, y_val = train_test_split(X_test, y_test,
                                                    test_size=0.5, 
                                                    random_state=42, 
                                                    stratify=y_test
    )
    model_calibrate = final_pipe
    final_model = CalibratedClassifierCV(FrozenEstimator(model_calibrate), method='sigmoid')
    final_model.fit(X_calib, y_calib)
    y_val_calibrated = final_model.predict(X_val)
    prob_val_calibrated = final_model.predict_proba(X_val)[:, 1]
    y_true_val, y_prob_val = calibration_curve(y_val, prob_val_calibrated, n_bins=10)
    y_true_test, y_prob_test = calibration_curve(y_test, y_proba_test, n_bins=10)

    plt.figure(figsize=(10,10))
    plt.plot(y_prob_val, y_true_val, marker='s', label='Calibrated (Sigmoid)', color='green')
    plt.plot(y_prob_test, y_true_test, marker='o', label='Original (Catboost)', color='red')
    plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Calibrated Perfectly')
    plt.title('Calibration (Val): Before (Exaggerated) vs After (Realistic)')
    plt.xlabel('Predicted Probability (Model predict)')
    plt.ylabel('Real Positive Frac (Reality)')
    plt.legend()
    plt.savefig('img/calibration_curve_val.png')
    mlflow.log_artifact('img/calibration_curve_val.png')
    plt.show()

        # PRINTING CLASSIFICATION REPORT
    print("--- Classification Report - VALIDATION ---")
    print(metrics.classification_report(y_val, y_val_calibrated))

    #/*****************************************************************************************/

    # FINDING BEST THRESHOLD FOR MODEL
    threshold_val = np.arange(0.1, 0.9, 0.1)
    f1_scores = []
    precisions = []
    recalls = []
    auc_scores = []

    for t in threshold_val:
        preds = (prob_val_calibrated >= t).astype(int)

        f1_scores.append(metrics.f1_score(y_val, preds))
        precisions.append(metrics.precision_score(y_val, preds, zero_division=0))
        recalls.append(metrics.recall_score(y_val, preds))
        auc_scores.append(metrics.roc_auc_score(y_val, preds))

    best_thresh = np.argmax(f1_scores)
    best_t = threshold_val[best_thresh]
    best_f1 = f1_scores[best_thresh]
    best_precision = precisions[best_thresh]
    best_recall = recalls[best_thresh]
    best_roc_auc = auc_scores[best_thresh]


    # PRINTING METRICS WITH THRESHOLD
    print(f'💰 Result Final Prod')
    print('='*40)
    print(f'🎯 Great Threshold: {best_t:.2f}')
    print(f'🏆 Best F1-Score: {best_f1:.4f}')
    print(f'✅ Best Precision-Score: {best_precision:.4f} (Of each 100 calls, we take {int(best_precision*100)} customers)')
    print(f'🎣 Best Recall-Score: {best_recall:.4f} (Recovered {int(best_recall*100)}% of Churners)')
    print(f'🎖️ Best ROC AUC-Score: {best_roc_auc:.4f}')

    #/*****************************************************************************************/

    # CALCULATING PROFITS WITH CHURNERS SAVED BY MODEL
    ltv = X_val['CLTV'].mean()
    cost = ltv * 0.05
    sr = 0.5 # success rate
    threshold_profit = np.linspace(0, 1, 101)
    profits = [profit_calc(y_val, prob_val_calibrated, ltv=ltv, cost=cost, sr=sr, threshold=t) for t in threshold_profit]

    best_idx_profit = np.argmax(profits)
    best_threshold_profit = threshold_profit[best_idx_profit]
    max_profit = profits[best_idx_profit]

    # PROFIT CURVE
    plt.figure(figsize=(8,6))
    plt.plot(threshold_profit, profits, label='Estimated Profit', color='green', linewidth = 2.0)

    # FIND BETTER FINANCIAL POINT
    plt.scatter(best_threshold_profit, max_profit, color='red', s=100, zorder=5)
    plt.axvline(best_threshold_profit, linestyle='--', color='red', alpha=0.5, label=f'Best ThresHold for Profits {best_threshold_profit:.2f}')

    # THRESHOLD PROFITS VS THRESHOLD MODEL
    plt.axvline(best_t, linestyle=':', color='blue', label=f'ThresHold F1 {best_t:.2f}' )

    plt.title(f'Profit Val Curve by ThresHold \n Profit Max {max_profit:,.2f} in {best_threshold_profit:.2f}', fontsize=14)
    plt.xlabel('Decision ThresHold (probability)', fontsize=12)
    plt.ylabel('Estimated Profit', fontsize = 12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('img/business_profit_curve_final.png')
    mlflow.log_artifact('img/business_profit_curve_final.png')
    plt.show()

    good_preds_val = (prob_val_calibrated >= best_threshold_profit).astype(int)
    tn_val, fp_val, fn_val, tp_val = metrics.confusion_matrix(y_val, good_preds_val).ravel()

    total_money_risk_val = (tp_val + fn_val) * ltv
    percent_save_val = (max_profit / total_money_risk_val) * 100

    print('='*40)
    print(f'Financial Impact Analysis (In val)')
    print(f'Total Risk Money {total_money_risk_val:.2f}')
    print(f'Model Estimated Profit {max_profit:.2f}')
    print(f'Percent Loss Saved {percent_save_val:.2f}')
    print('='*40)

    mlflow.log_metric("pct_revenue_saved_val", percent_save_val)
    mlflow.log_metric("total_money_at_risk_val", total_money_risk_val)

    #/*****************************************************************************************/

    # DEFINING SIGNATURE MODEL
    input_sample = X_val.iloc[:5].copy().reset_index(drop=True)
    prediction_sample = final_model.predict(input_sample)

    signature = infer_signature(input_sample, prediction_sample)

    # SAVING FINAL MODEL INFOS
    mlflow.set_tag("winner_algorithm", model_name)
    mlflow.sklearn.log_model(
        sk_model=final_model,
        name='churn_model_calibrated_prod',
        signature=signature,
        input_example=input_sample,
        pip_requirements=['catboost','scikit-learn','pandas','numpy']
    )

    # SAVING FINAL MODEL METRICS
    mlflow.log_metrics({
        "auc_train": roc_train_score,
        "auc_test": roc_test_score,
        "f1_train": f1_score_train,
        "f1_test": f1_score_test,
        "f1 threshold": f1_score_threshold,
        "prauc_train": prauc_score_train,
        "prauc_test": prauc_score_test,
        "f1_val": best_f1,
        "precision_val": best_precision,
        "recall_val": best_recall,
        "auc_val": best_roc_auc
    })  # type: ignore

    print("✅ Completed!")


#%% [markdown]
# # --- LOADING MODEL ---
#%%
versions = mlflow.search_model_versions(filter_string= "name = 'model_churn'")
last_version = max([int(i.version) for i in versions])
model = mlflow.sklearn.load_model(f'models:///model_churn/{last_version}')

#%% [markdown]
# ## --- READING NEW DATA ---
#%%
new_data = pd.DataFrame({
    "Country": 'United States',
    "State": 'California',
    "City": 'Los Angeles',
    "Zip Code":	90003,
    "Latitude": 33.964131,
    "Longitude": -118.272783,
    "Gender":'Male',
    "Senior Citizen": 'No',
    "Partner": 'No',
    "Dependents": 'No',
    "Tenure Months": 5,
    "Phone Service": 'Yes'	,
    "Multiple Lines": 'No',
    "Internet Service": 'DSL',
    "Online Security": 'No',
    "Online Backup": 'Yes',
    "Device Protection": 'Yes'	,
    "Tech Support":	'Yes',
    "Streaming TV":	'No',
    "Streaming Movies":	'No',
    "Contract": 'Month-to-month',
    "Paperless Billing": 'No',
    "Payment Method": 'Bank transfer (automatic)',
    "Monthly Charges": 75.9,
    "Total Charges": 203.79,
    "CLTV": 10000,
    "Average By Geo": 62.42377,
    "Average Price P/ Service": 36.98,
    "Charge Diff City Mean": -28.377,
    "Family": 1,
    "Geo Cluster": 30,
    "High Tech No Support": 0,
    "Last Three Months": 0,
    "Lazer Products": 1,
    "Months to Renewal": 2,
    "Payment Risk": 1,
    "Price Hike": -9.733,
    "Price Sensitivity": 0.003,
    "Price Up Recently": 91.3102,
    "Score dependency": 4,
    "Security Product": 2,
    "Senior Vulnerable": 0,
    "Social Isolation": 0,
    "Tenure Ratio Contract": 0.0,
    "Time Contract": 2,
    "Value Ratio": 120.304078,
}, index=[0])

#%% [markdown]
# ## --- PREDICTING WITH NEW VALUES ---
#%%

proba = model.predict_proba(new_data)[:, 1]
threshold_great = 0.40
final_decision = (proba >= threshold_great).astype(int)

#%% [markdown]
# ## --- ANALYSING NEW RESULT ---
#%%

print(f"🎲 Probabilidade de Churn: {proba[0]*100:.2f}%")
print(f"⚖️ Decision (ThresHold {threshold_great}): {'🔴 CHURN' if final_decision[0] == 1 else '🟢 RETAIN'}")
