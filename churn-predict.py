
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
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from feature_engine import discretisation, encoding
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

sys.path.append(os.path.abspath(os.path.join('..')))
from src.eng_funcs import CleanTransformStrNum, AnalyseDataSet

#%% [markdown]
# ## -- CONFIGURING JUPYTER PAGE --
#%%
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 500)

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
df.head(3)

#%% [markdown]
# ## -- UNDERSTANDING DATASET - EDA --
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

#%%
corr = df.corr(numeric_only=True, method='pearson')['Churn Value'].sort_values(ascending=False).to_frame()
corr.columns = ['Correlation']
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(10,6))
sns.heatmap(data=corr, cmap='coolwarm', fmt='.2f', annot=True, mask=mask)
plt.title('Correlation Plot')
plt.show()


#%%
# PLOTING CHURN DISTRIBUITION
df['Churn Value'].value_counts(normalize=True)

plt.figure(figsize=(10,6))
sns.countplot(data=df, x='Churn Value', palette='viridis')
plt.title('Distribuição de Valores de Churn (1=Saiu, 0=Ficou)')
plt.show()

#%%
# LIST COLUMN TYPES
blacklist = ['CustomerID','City','Lat Long','Churn Label', 'Monthly Charges','Zip Code','Latitude','Longitude','CLTV','Churn Score','Total Charges', 'Tenure Months']
category_cols = df.select_dtypes(include=['object'])
num_cols = df.select_dtypes(include=['int','float'])

cat_cols = [col for col in category_cols.columns if col in category_cols and col not in blacklist]
num_cols = [col for col in df.columns if col in num_cols and col not in blacklist]
#%%

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
blacklist = ['Churn Score','Churn Label', 'CustomerID', 'Count', 'Total Charges']

num_vars = [col for col in X_train.columns if col not in blacklist and pd.api.types.is_numeric_dtype(X_train[col])]
cat_vars = [col for col in X_train.columns if col not in blacklist and pd.api.types.is_object_dtype(X_train[col])]

#%% [markdown]
# # --- NUMBER/OBJECT PIPELINE TRANSFORMATION ---
#%%

num_pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('imputer', SimpleImputer(strategy='median')),
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

totalcharge_pipe = Pipeline([
    ('eng', CleanTransformStrNum()),
    ('scaler', RobustScaler())
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
# # --- FINAL PIPE ---
#%%

final_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', KNeighborsClassifier())
    ],
    memory=None
)

#%% [markdown]
# # --- DEFINING PARAMETERS ---
#%%


#%% [markdown]
# # --- APPLYING GRIDSEARCH ---
#%%


#%% [markdown]
# # --- FITTING GRID MODEL ---
#%%


#%% [markdown]
# # --- PREDICTING VALUES WITH MODEL GRID ---
#%%


#%% [markdown]
# # --- COMPARING MODEL CLASSIFICATION VS Y_TEST ---
#%%


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


#%% [markdown]
# ## --- PREDICTING WITH NEW VALUES ---
#%%


#%% [markdown]
# ## --- ANALYSING NEW RESULT ---
#%%