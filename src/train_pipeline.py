# IMPORTS E CONFIGS

print('📋 Starting run!')

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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from src.eng_funcs import CleanTransformStrNum, profit_calc
    print("✅ src.eng_funs imported successfully!")

except ImportError:
    # Caso você rode de dentro da pasta src, o import muda. 
    # O ideal é sempre rodar da raiz: python src/train_pipeline.py
    print("Warning: cannot possible import src.eng_funcs. Verify path.")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(experiment_id='1')

# DOWNLOAD DATASET
local_file_path = '../data/Telco_customer_churn.xlsx' 
try:
    print("⬇️ Trying to load nearest version from kaggle...")
    path_dir = kagglehub.dataset_download("yeanzc/telco-customer-churn-ibm-dataset")
    final_path = os.path.join(path_dir, 'Telco_customer_churn.xlsx')
    print(f"✅ Successfull download! Dataset: {final_path}")

except Exception as e:
    print(f"⚠️ Error to download dataset from kaglle: {e}")
    print(f"📂 Using local archive...")
    
    final_path = local_file_path

if os.path.exists(final_path):
    df_first = pd.read_excel(final_path, engine='calamine')
    print(f"📊 Loaded successfull local archive! Shape: {df_first.shape}")
else:
    raise FileNotFoundError(f"🚨 Warning: Doesn't possible find dataset archive in web kaggle and local '{local_file_path}' storage.")

print('🚀 Starting trainning pipeline!')

def main():
    
    print("✅ Running creating features!")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, '..', 'img')
    os.makedirs(images_dir, exist_ok=True)

    # CREATE A DATASET COPY TO CREATE NEW FEATURES
    df_aux = df_first.copy()

    df_aux['Total Charges'] = df_aux['Total Charges'].apply(lambda x: 0 if x == ' ' else x)

    df_aux['Price Up Recently'] = np.where(
                                df_aux['Tenure Months'] > 0,
                                df_aux['Total Charges'] / df_aux['Tenure Months'],
                                df_aux['Monthly Charges'])

    df_aux['Price Hike'] = df_aux['Monthly Charges'] - df_aux['Price Up Recently']

    df_aux['Price Sensitivity'] = df_aux['Monthly Charges'] / df_aux['CLTV']

    services_offer = ['Phone Service','Multiple Lines','Internet Service',
                    'Online Security','Online Backup','Device Protection',
                    'Tech Support','Streaming TV','Streaming Movies']

    df_aux['Score dependency'] = 0
    df_aux['Lazer Products'] = 0
    df_aux['Security Product'] = 0

    for score in df_aux[services_offer]:
        points = np.where(df_aux[score].astype(str).str.contains('No'), 0, 1)
        df_aux['Score dependency'] = df_aux['Score dependency'] + points

    for lazer in df_aux[['Streaming TV','Streaming Movies']]:
        points = np.where(df_aux[lazer].astype(str).str.contains('No'), 0, 1)
        df_aux['Lazer Products'] = df_aux['Lazer Products'] + points

    for security in df_aux[['Online Security','Online Backup','Device Protection']]:
        points = np.where(df_aux[security].astype(str).str.contains('No'), 0, 1)
        df_aux['Security Product'] = df_aux['Security Product'] + points

    df_aux['Senior Vulnerable'] = np.where((df_aux['Senior Citizen'] == 'Yes') & (df_aux['Tech Support'] == 'No'), 1, 0)

    df_aux['Family'] = (df_aux['Partner'] == 'Yes').astype(int) + (df_aux['Dependents'] == 'Yes').astype(int)

    X_geo = df_aux[['Latitude','Longitude']]

    kmeans = KMeans(n_clusters=30, random_state=42)
    df_aux['Geo Cluster'] = kmeans.fit_predict(X_geo).astype(str)

    df_aux['Payment Risk'] = np.where((df_aux['Payment Method'] == 'Credit card (automatic)') 
                                        | (df_aux['Payment Method'] == 'Bank transfer (automatic)'),
                                            0, 1)

    df_aux['Time Contract'] = df_aux['Contract'].apply(lambda x: 24 if x == 'Two year' else
                                                                12 if x == 'One year' else  1).astype(int)

    df_aux['Months to Renewal'] = df_aux['Time Contract'] - (df_aux['Tenure Months'] % df_aux['Time Contract'])

    df_aux['Last Three Months'] = np.where(df_aux['Time Contract'] <= 3, 1, 0)

    df_aux['High Tech No Support'] = np.where((df_aux['Internet Service'] == 'Fiber optic') & (df_aux['Tech Support'] == 'No'), 1, 0)

    df_aux['Average Price P/ Service'] = df_aux['Monthly Charges'] / df_aux['Score dependency']

    df_aux['Average By Geo'] = df_aux.groupby(by=['Geo Cluster'])[['Monthly Charges']].transform('mean')

    df_aux['Average By Geo'] = df_aux.groupby(by=['City'])[['Monthly Charges']].transform('mean')

    df_aux['Charge Diff City Mean'] = df_aux['Monthly Charges'] - df_aux['Average By Geo']

    df_aux['Tenure Ratio Contract'] = (df_aux['Time Contract'] - df_aux['Months to Renewal']) / df_aux['Time Contract']

    df_aux['Value Ratio'] = df_aux['CLTV'] / df_aux['Monthly Charges']

    df_aux['Social Isolation'] = (df_aux['Family'].astype(int) + df_aux['Senior Vulnerable'])

    # JOINING NEW DATASET WITH NEW FEATURE IN BASE DATASET DF
    new_columns = df_aux.columns.difference(df_first.columns)
    df = df_first.join(df_aux[new_columns])

    print("✅ X and Y split!")

    # DEFINING X COLUMNS AND TARGET
    target = 'Churn Value'
    X, y = df.drop(columns=[target], errors='ignore'), df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        random_state=42,
                                                        test_size=0.25)

    # SELECTING FEATURES TO THE PIPELINE
    blacklist = ['Churn Score','Churn Label', 'CustomerID', 'Count', 
                'Total Charges', 'Tenure Months Range','Churn Score Range',
                'Lat Long', 'Churn Reason']

    num_vars = [col for col in X_train.columns 
                if col not in blacklist and pd.api.types.is_numeric_dtype(X_train[col])]
    cat_vars = [col for col in X_train.columns 
                if col not in blacklist and pd.api.types.is_object_dtype(X_train[col])]

    # CREATING ESPECIFY DATA TYPE PIPELINE
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

    # EMBEDDING PIPELINES IN A PREPROCESSOR
    preprocessor = ColumnTransformer(
        transformers=[
            ('tr_num', num_pipe, num_vars),
            ('tr_cat', cat_pipe, cat_vars),
            ('totalcharges', totalcharge_pipe, ['Total Charges']),
            ], 
        remainder='drop'
    )

    # DEFINING PARAMS TO TEST AND FIND BEST MODEL
    params = [
        # --- MODEL 1: CATBOOST: Works Better W/ Categorical Datasets ---
        {
        'model': [CatBoostClassifier(allow_writing_files=False, verbose=0, random_state=42)],
        'model__n_estimators': [500, 1000, 1500],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.3],
        'model__depth': [4, 6, 9],
        'model__auto_class_weights': ['Balanced'],
        'model__l2_leaf_reg': [3, 5, 7],
        'model__border_count': [128]
        }
    ]

    # EMBEDDING MODEL INTO PIPELINE
    model_pipe = Pipeline([
        ('model', KNeighborsClassifier())
    ])

    # DEFINING CV VALUE AND CREATING GRID
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid = RandomizedSearchCV(
        estimator=model_pipe,
        param_distributions = params,
        n_iter=10,
        cv = kfold,
        scoring='roc_auc',
        verbose=1,
        n_jobs=-1,
        random_state=42,
        refit = True
    )

    # EMBEDDING GRID INTO A PIPELINE
    final_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('grid', grid)
        ],
        memory=None
    )

    print("🔥 Starting Trainning model...")
    # RUNNING MLFLOW
    with mlflow.start_run() as r:

        # RUNNING MLFLOW LOG
        mlflow.sklearn.autolog(log_models=False)

        #/*****************************************************************************************/
        
        # FITTING MODEL
        model_fit = final_pipe.fit(X_train, y_train)        
        best_model = model_fit.named_steps['grid'].best_estimator_.named_steps['model']
        model_name = best_model.__class__.__name__
        print(f'✅ Best model: {model_name}')

        #/*****************************************************************************************/

        # CREATING METRICS TO ANALYSE MODEL
        y_pred_train = model_fit.predict(X_train)
        y_proba_train = model_fit.predict_proba(X_train)[:, 1]
        roc_train_score = metrics.roc_auc_score(y_train, y_pred_train)
        f1_score_train = metrics.f1_score(y_train, y_pred_train)
        prauc_score_train = metrics.average_precision_score(y_train, y_pred_train)

        y_pred_test = model_fit.predict(X_test)
        y_proba_test = model_fit.predict_proba(X_test)[:, 1]
        roc_test_score = metrics.roc_auc_score(y_test, y_pred_test)
        f1_score_test = metrics.f1_score(y_test, y_pred_test)
        prauc_score_test = metrics.average_precision_score(y_test, y_pred_test)
        
        # PLOTING ROC CURVE AND F1 SCORE
        roc_train = metrics.roc_curve(y_train, y_proba_train)
        roc_test = metrics.roc_curve(y_test, y_proba_test)

        plot_filename = 'roc_curve_train_test.png'
        save_path = os.path.join(images_dir, plot_filename)

        plt.figure(dpi=350)
        plt.plot(roc_train[0], roc_train[1])
        plt.plot(roc_test[0], roc_test[1])
        plt.legend([f"Train: {roc_train_score:.2f}",
                f"Test: {roc_test_score:.2f}"])
        plt.plot([0,1],[0,1], '--', color='black')
        plt.grid(True)
        plt.title(f'Roc Curve (Train/Test)')
        plt.savefig(save_path)
        mlflow.log_artifact(save_path)
        plt.close()

        # TESTING THRESHOLD 0.5
        threshhold_first_test = 0.5
        if len(y_proba_test.shape) > 1 and y_proba_test.shape[1] > 1:
            y_pred_threshold = (y_proba_test[:, 1] >= threshhold_first_test).astype(int)
        else:
            y_pred_threshold = (y_proba_test >= threshhold_first_test).astype(int)
        f1_score_threshold = metrics.f1_score(y_test, y_pred_threshold)

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

        plot_filename = 'business_profit_curve_test.png'
        save_path = os.path.join(images_dir, plot_filename)

        plt.title(f'Profit Test Curve by ThresHold \n Profit Max {max_proft_test:,.2f} in {best_threshold_profit_test:.2f}', fontsize=14)
        plt.xlabel('Decision ThresHold (probability)', fontsize=12)
        plt.ylabel('Estimated Profit', fontsize = 12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        mlflow.log_artifact(save_path)
        plt.close()

        good_preds_test = (y_proba_test >= best_threshold_profit_test).astype(int)
        tn_test, fp_test, fn_test, tp_test = metrics.confusion_matrix(y_test, good_preds_test).ravel()

        total_money_risk_test = (tp_test + fn_test) * ltv_test
        percent_save_test = (max_proft_test / total_money_risk_test) * 100

        mlflow.log_metric("pct_revenue_saved_test", percent_save_test)
        mlflow.log_metric("total_money_at_risk_test", total_money_risk_test)
        
        #/*****************************************************************************************/

        # ANALYSING BEST 10 MODELS
        cols_keeped = ['params','mean_test_score','std_test_score','rank_test_score']
        results_df = pd.DataFrame(model_fit.named_steps["grid"].cv_results_)
        results_df = results_df[cols_keeped]
        results_df = results_df.sort_values(by='rank_test_score')
        best_score = results_df.iloc[0]['mean_test_score']

        html_filename = "grid_search_results.html"
        save_path = os.path.join(images_dir, html_filename)
        results_df.to_html(save_path, index=False)
        
        mlflow.log_artifact(save_path)
        mlflow.log_metric("best_cv_score", best_score)
        #/*****************************************************************************************/

        plot_filename = 'feature_importance.png'
        save_path = os.path.join(images_dir, plot_filename)

        # SAVING MODEL FEATURE IMPORTANCE MODEL
        if hasattr(best_model, 'feature_importances_'):
            importance = best_model.feature_importances_ / 100
            feats = model_fit.named_steps['preprocessor'].get_feature_names_out()
            if len(importance) == len(feats):
                df_imp = pd.DataFrame({'Feature': feats, 'Importance': importance}).sort_values('Importance', ascending=False).head(10)
                plt.figure(figsize=(10, 8))
                sns.barplot(data=df_imp, x='Importance', y='Feature')
                plt.title(f'Feature Importance ({model_name})')
                plt.tight_layout()
                plt.savefig(save_path) # <--- FALTAVA AQUI
                mlflow.log_artifact(save_path)
                plt.close()

        #/*****************************************************************************************/

        # TESTING CALIBRATION CURVE FOR UNDERSTAND BETTER THRESHOLD
        prob_true, prob_pred = calibration_curve(y_test, y_proba_test, n_bins=10, strategy='uniform')

        plot_filename = 'calibration_curve_test.png'
        save_path = os.path.join(images_dir, plot_filename)

        plt.figure(figsize=(8,8))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=model_name)
        plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Calibrated Perfectly')
        plt.ylabel('Real Positive Frac (Reality)')
        plt.xlabel('Predicted Probability (Model predict)')
        plt.title('Calibration Curve (Test)')
        plt.legend()
        plt.savefig(save_path)
        mlflow.log_artifact(save_path)
        plt.close()

        print("⚖️ Calibrating model...")
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

        plot_filename = 'calibration_curve_val.png'
        save_path = os.path.join(images_dir, plot_filename)

        plt.figure(figsize=(10,10))
        plt.plot(y_prob_val, y_true_val, marker='s', label='Calibrated (Sigmoid)', color='green')
        plt.plot(y_prob_test, y_true_test, marker='o', label='Original (Catboost)', color='red')
        plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Calibrated Perfectly')
        plt.title('Calibration (Val): Before (Exaggerated) vs After (Realistic)')
        plt.xlabel('Predicted Probability (Model predict)')
        plt.ylabel('Real Positive Frac (Reality)')
        plt.legend()
        plt.savefig(save_path)
        mlflow.log_artifact(save_path)
        plt.close()

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

        plot_filename = 'business_profit_curve_final.png'
        save_path = os.path.join(images_dir, plot_filename)

        plt.title(f'Profit Val Curve by ThresHold \n Profit Max {max_profit:,.2f} in {best_threshold_profit:.2f}', fontsize=14)
        plt.xlabel('Decision ThresHold (probability)', fontsize=12)
        plt.ylabel('Estimated Profit', fontsize = 12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        mlflow.log_artifact(save_path)
        plt.close()
        
        good_preds_val = (prob_val_calibrated >= best_threshold_profit).astype(int)
        tn_val, fp_val, fn_val, tp_val = metrics.confusion_matrix(y_val, good_preds_val).ravel()

        total_money_risk_val = (tp_val + fn_val) * ltv
        percent_save_val = (max_profit / total_money_risk_val) * 100

        mlflow.log_metric("pct_revenue_saved_val", percent_save_val)
        mlflow.log_metric("total_money_at_risk_val", total_money_risk_val)

        #/*****************************************************************************************/

        # DEFINING SIGNATURE MODEL
        print("💾 Saving final model...")

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

        # CLEANNING TEMP FILES
        for f in ['../img/roc_curve_train_test.png', 
                  '../img/business_profit_curve_test.png',
                  '../img/feature_importance.png', 
                  '../img/calibration_curve_val.png', 
                  '../img/grid_search_results.html']:
            if os.path.exists(f):
                os.remove(f)

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

        print('🏁 Pipeline completed!')

if __name__ == "__main__":
    main()