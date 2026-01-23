# 🚜 Classificação de Clientes que darão Churn (Catboost)

## 📋 Sobre o Projeto
Este projeto resolve um problema muito comum em toda e qualquer empresa: Churn de clientes.

O principal desafio neste caso/projeto é conseguir entender as ocilações do comportamento humano, para assim conseguir traçar um padrão de perfil de usuários que da churn.

## 📋 Entendimento de négocios.
Realizando EDA pode-se concluir alguns padrões que ajudam a basear a classifição de Churn para tal Cliente.
Clientes mais propensos a churn utilizam menos produtos que clientes menos propensos.

## 🧠 Estratégia de Modelagem

### 1. Algoritmo e Paramêtro
Utilizei o **Catboost Classifier** com a função hiperparametros como **Learning Rate** (`0.01`) e **Class_Weight** (`balanced`).
* **Por que o Catboost venceu XGB, LGBM e RF?** Geralmente modelos boosting (XGB E LGBM) se sobressaem em analises preditivas, porém neste case, o Catboost se sobressaiu por conta de uma catecteristica que ele possui. Trato com variáveis categoricas. Por conta deste dataset ser majoritariamente formado por variáveis categóricas, ele acaba sendo o tipo de base perfeita para o Catboost performar, que foi o que ocorreu nessa rodada de treinos.
* **Por que Learning Rate e Class_Weight?** Por que no caso de análise de churns o mais díficil é aprender o padrão de comportamento numa base tão desbalanceada, e neste caso, a melhor solução é fazer que o modelo se atende a todo e qualquer detalhe no treinamento, fazendo com que o modelo não passe por um comportamento ou indicio de possível churn sem detecta-lá. Em conjunto com ele, o paramêtro class_weight ajuda muito dizendo para o modelo dar mais enfoque na classe minoritária (aumenta o peso da classe churn e ajuda o modelo a "ter medo" de perder possíveis clientes churners).

### 2. Engenharia de Features
A estrutura de dados foi construída com `Scikit-Learn` incluindo:
* **Scaling:** do **"Total Charges"** e das features numéricas para menores escalas.
* **Imputer:** da mediana em features numéricas com missings e valores constantes em features categóricas.
* **Enconding:** em features catégoricas.
* **Limpeza e Transformação:** na feature **"Total Charges"** por conta de registro de espaço na coluna.


## 📊 Resultados (Test Validation Pré Threshold)

| Métricas | Valor Final |
|----------|-------------|
| **Threshold** | **0.50** (Calibrador de probabilidades) |
| **Precision** | **54%** (Assertividade percentual dos apontamentos de fraude) |
| **Recall**    | **81%** (Capacidade de detecção) |
| **F1-Score**  | **65%** (Equilíbrio entre Precision x Recall) |
| **ROC AUC**   | **78%** (Capacidade de separação entre Churn e Não churn) |


### Performance Test: Curva de Calibração
> *O gráfico linha abaixo mostra como se comportou o modelo durante o teste, e sua performance de calibração comparada ao modelo perfeito*

![Curva de Calibração](img/calibration_curve_test.png)

### Performance Financeira Test
> *O gráfico de barras abaixo mostra o resultado financeiro do modelo.
Entregando um valor de **$9.000** dolares de prejuizo evitados, que equivalem a **70%** do total das tentativas de fraude*

![Resultado Financeiro Test](img/business_profit_curve_test.png)

## 📊 Resultados (Test Validation w/ Threshold)

| Métricas | Valor Final |
|----------|-------------|
| **Threshold** | **0.40** (Calibrador de probabilidades) |
| **Precision** | **70%** (Assertividade percentual dos apontamentos de fraude) |
| **Recall**    | **60%** (Capacidade de detecção) |
| **F1-Score**  | **65%** (Equilíbrio entre Precision x Recall) |
| **ROC AUC**   | **80%** (Capacidade de separação entre Churn e Não churn) |


### Performance Calibrada: Curva de Calibração
> *O gráfico linha abaixo mostra como se comportou o modelo durante a validação com o threshold perfeito definido, e sua performance de calibração comparada ao modelo de teste e o modelo perfeito*

![Curva de Calibração Final](img/calibration_curve_val.png)

### Performance Financeira Calibrada
> *O gráfico de barras abaixo mostra o resultado financeiro do modelo calibrado.
Entregando um valor de **$14.000** dolares de prejuizo evitados, que equivalem a **X%** do total de churns ocorridos*

![Resultado Financeiro Calibrada](img/business_profit_curve_final.png)



## 🚀 Como Rodar o Projeto

1. **Clone o repositório:**
   ```bash
   git clone [git@github.com:savioricardog/fraud-detection.git](https://github.com/savioricardog/fraud-detection.git)

2. **Instale as dependências:**
   ```bash 
   pip install -r requirements.txt

3. **Execute o arquivos :**
   ```bash 
   python fraud-detection-V2.py

## 📂 Estrutura de Arquivos 

fraud-detection.py: Estrutura principal de treinamento.

fraud-detection.ipynb: Arquivo em modelo Jupyter.

requirements.txt: Dependências do ambiente.

models/model_fraud_V2.pkl: Modelo treinado.


**Desenvolvido por Savio Ricardo Garcia 👨‍💻**
