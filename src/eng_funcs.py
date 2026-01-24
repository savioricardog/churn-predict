from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class CleanTransformStrNum(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()

        if isinstance(X_copy, pd.DataFrame):
            X_copy = X_copy.apply(pd.to_numeric, errors='coerce').fillna(0)
        else:
            X_copy = pd.to_numeric(X_copy, errors='coerce').fillna(0)
    
        if len(X_copy.shape) == 1:
            X_copy = X_copy.values.reshape(-1,1)

        return X_copy
    
    def get_feature_names_out(self, input_features=None):
        """
        Retorna os nomes das colunas após a transformação.
        O Scikit-Learn chama isso automaticamente.
        """
        # Se o pipeline mandou os nomes de entrada (ex: ['Total Charges']), 
        # devolvemos eles mesmos, pois só limpamos os dados, não mudamos o nome.
        if input_features is not None:
            return input_features
        
        # Fallback de segurança: se o pipeline não mandar nada, 
        # devolvemos o nome padrão que sabemos que essa classe trata.
        return ["Total Charges"]
    
    
def AnalyseDataSet(df):
    for i in df.columns:
    # Verifica se existe QUALQUER valor igual a ' ' na coluna i
        tem_espaco = (df[i] == ' ').any()
        tem_null = (df[i] == np.nan).any()
        print ('='*40)
        print(f'Coluna: {i}')
        if tem_espaco or tem_null:
            # Conta quantos espaços existem
            total_branco = (df[i] == ' ').sum()
            total_null = (df[i] == np.nan).sum()
            print(f'⚠️ Encontrado {total_branco} espaço(s) em branco e {total_null} Valores Nulos')
        else:
            print('✅ DataSet limpo')
    print ('='*40)
    print(f'\n✅ Analise Concluída!\n')
    print ('='*40)


def profit_calc(y_true, y_proba, threshold, ltv, cost, sr):
    from sklearn.metrics import confusion_matrix
    preds = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    
    total_incomes = (tp * sr * ltv)
    total_cost = (tp + fp) * cost

    profit = total_incomes - total_cost

    return profit