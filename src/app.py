from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import pandas as pd
import numpy as np
import mlflow.sklearn
import os
import sys

# Config enviroment
current_dir = os.path.dirname(os.path.abspath(__file__)) # searching src folder
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.eng_funcs import CleanTransformStrNum
except ImportError:
    from eng_funcs import CleanTransformStrNum

# CONFIG MLFLOW
# MLFLOW_URI = os.getenv("MLFLOW_URI", "http://localhost:5000")
# EXPERIMENT_ID = "1"
# MODEL_NAME_IN_MLFLOW = "churn_model_calibrated_prod"

MODEL_PATH_IN_DOCKER = "/app/model_prod"

# CREATING OBJECT APP
app = FastAPI()
model = None

# --- 2. FUNÇÃO INTELIGENTE DE BUSCA ---
def find_model_path(base_path="/app/model_prod"):
    """
    Procura pelo arquivo 'MLmodel' ou 'model.pkl' recursivamente
    para descobrir onde o MLflow salvou os arquivos.
    """
    print(f"🕵️‍♂️ Investigando arquivos em: {base_path}")
    
    if not os.path.exists(base_path):
        print(f"❌ Folder {base_path} not exist in DOCKER.")
        return None

    # Lista tudo que tem lá dentro para debug
    for root, dirs, files in os.walk(base_path):
        print(f"   📁 {root} contém: {files}")
        if "MLmodel" in files or "model.pkl" in files:
            print(f"✅ MODEL FOUND IN: {root}")
            return root
            
    return base_path # Retorna o padrão se não achar nada específico

# --- CLASS PAD (MODEL DATA) ---
class CustomerData(BaseModel):
    # IDS and Cats
    Gender: str
    Senior_Citizen: str
    Partner: str
    Dependents: str
    Contract: str
    Paperless_Billing: str 
    Payment_Method: str
    Country: str
    State: str
     
    # Services
    Phone_Service: str
    Multiple_Lines: str
    Internet_Service:str
    Online_Security: str
    Online_Backup: str
    Device_Protection: str
    Tech_Support: str
    Streaming_TV: str
    Streaming_Movies: str

    # Numbers
    Zip_Code: object
    Tenure_Months: int
    Monthly_Charges: float
    Total_Charges: object
    CLTV: float

    # GeoData
    Latitude: float= 0.0
    Longitude: float= 0.0
    City: str = "Unknown"

    class Config:
        json_schema_extra = {
            "example":{
                    "Gender": "Male",
                    "Senior_Citizen": "No",
                    "Partner": "No",
                    "Dependents": "No",
                    "Phone_Service": "Yes",
                    "Multiple_Lines": "No",
                    "Internet_Service":"Fiber optic",
                    "Online_Security": "No",
                    "Online_Backup": "No",
                    "Device_Protection": "No",
                    "Tech_Support": "No",
                    "Streaming_TV": "Yes",
                    "Streaming_Movies": "Yes",
                    "Contract": "Month-to-month",
                    "Paperless_Billing": "Yes",
                    "Payment_Method": "Electronic check",
                    "Tenure_Months": 12,
                    "Monthly_Charges": 70.0,
                    "Total_Charges": 840.0,
                    "CLTV": 400,
                    "Country": "United States",
                    "State": "California",
                    "Zip_Code": 90210
            }
        }

# --- FEATURE ENGINE AND DATA TRANSFORMATION IN REAL TIME
def process_input(data: CustomerData):
    try:
        df = pd.DataFrame([data.dict()]) 
    except AttributeError:
        df = pd.DataFrame([data.model_dump()]) # Fallback para Pydantic V2
    
    df.columns = [col.replace('_', ' ') for col in df.columns] # REPLACING _ TO ' '
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce').fillna(0)
    df['Monthly Charges'] = pd.to_numeric(df['Monthly Charges'], errors='coerce').fillna(0)
    df['Tenure Months'] = pd.to_numeric(df['Tenure Months'], errors='coerce').fillna(0)

    # Lógica de contrato
    def get_contract_months(x):
        if x == 'Two year': return 24
        if x == 'One year': return 12
        return 1
    # CREATING NEW FEATURES
    df['Price Up Recently'] = np.where(
                                df['Tenure Months'] > 0,
                                df['Total Charges'] / df['Tenure Months'],
                                df['Monthly Charges'])
                                
    df['Price Hike'] = df['Monthly Charges'] - df['Price Up Recently']
    df['Price Sensitivity'] = df['Monthly Charges'] / df['CLTV']
    
    services_offer = ['Phone Service','Multiple Lines','Internet Service',
                    'Online Security','Online Backup','Device Protection',
                    'Tech Support','Streaming TV','Streaming Movies']
    
    df['Score dependency'] = 0
    df['Lazer Products'] = 0
    df['Security Product'] = 0

    for score in services_offer:
        if score in df.columns:
            points = np.where(df[score].astype(str).str.contains('No'), 0, 1)
            df['Score dependency'] = df['Score dependency'] + points

    for lazer in ['Streaming TV','Streaming Movies']:
        if lazer in df.columns:
            points = np.where(df[lazer].astype(str).str.contains('No'), 0, 1)
            df['Lazer Products'] = df['Lazer Products'] + points

    for security in ['Online Security','Online Backup','Device Protection']:
        if security in df.columns:
            points = np.where(df[security].astype(str).str.contains('No'), 0, 1)
            df['Security Product'] = df['Security Product'] + points

    df['Senior Vulnerable'] = np.where((df['Senior Citizen'] == 'Yes') & (df['Tech Support'] == 'No'), 1, 0)
    df['Family'] = (df['Partner'] == 'Yes').astype(int) + (df['Dependents'] == 'Yes').astype(int)

    # --- MOCK COMPLEX FEATURES ---
    df['Geo Cluster'] = "0" 
    df['Average By Geo'] = df['Monthly Charges'] # Mock neutro

    df['Payment Risk'] = np.where((df['Payment Method'] == 'Credit card (automatic)') 
                                    | (df['Payment Method'] == 'Bank transfer (automatic)'),
                                        0, 1)
                                        
    df['Time Contract'] = df['Contract'].apply(lambda x: 24 if x == 'Two year' else
                                                12 if x == 'One year' else  1).astype(int)
                                                
    df['Months to Renewal'] = df['Time Contract'] - (df['Tenure Months'] % df['Time Contract'])
    df['Last Three Months'] = np.where(df['Time Contract'] <= 3, 1, 0)
    df['High Tech No Support'] = np.where((df['Internet Service'] == 'Fiber optic') & (df['Tech Support'] == 'No'), 1, 0)
    
    # Prevenção de Divisão por Zero no Score Dependency
    df['Average Price P/ Service'] = df['Monthly Charges'] / df['Score dependency'].replace(0, 1)
    
    df['Charge Diff City Mean'] = 0 # Mock neutro
    
    df['Tenure Ratio Contract'] = (df['Time Contract'] - df['Months to Renewal']) / df['Time Contract']
    df['Value Ratio'] = df['CLTV'] / df['Monthly Charges']
    df['Social Isolation'] = (df['Family'].astype(int) + df['Senior Vulnerable'])

    return df

# --- LOADING MODEL INTO API ---
@app.on_event("startup")
def load_model():
    global model
    try:
        print(f"Connecting to MLFLOW in {MODEL_PATH_IN_DOCKER}...")
        # mlflow.set_tracking_uri(MLFLOW_URI)
        real_model_path = find_model_path("/app/model_prod")

        if real_model_path:
            print(f"Loading model from: {real_model_path}")
            model = mlflow.sklearn.load_model(real_model_path)
            print("✅ Model Loaded...")
        else:
            print("❌ ERROR: folder or model not exist...")

        # print("Searching for best model...")

        # searching experiments into mlflow
        # runs = mlflow.search_runs(experiment_ids=[EXPERIMENT_ID])

        # if runs.empty:
        #     print("WARNING: No model found! API will works, but predict will fail!")
        #     return
        
        # last_run_id = runs.sort_values("start_time", ascending=False).iloc[0].run_id

        # model_uri = f"runs:/{last_run_id}/{MODEL_NAME_IN_MLFLOW}"

        # print(f"Loading model from: {model_uri}")
        # model = mlflow.sklearn.load_model(model_uri)
        # print("Model load successfull!✅")

    except Exception as e:
        print(f"❌ Error while loading model {e}")
        import traceback
        traceback.print_exc()

# --- Route 1: Home page ---
@app.get("/") # when access main address "/", run this func
def home():
    status = "API is On!" if model else "Online but without model loaded"
    return {"Message": status }

@app.post("/predict")
def predict_churn(dados: CustomerData):
    if not model:
        raise HTTPException(status_code=503, detail="MODEL OFFLINE. VERIFY!")
    
    print(f"1. Data received {dados}")

    df_ready = process_input(dados)

    # --- PREDICTIONS ---
    try:
        prob = float(model.predict_proba(df_ready)[:,1][0])

        threshold = 0.3
        decision = "Churn" if prob >= threshold else "Retain"

        return {
            "status": "OK!✅",
            "message": "Predictions built!",
            "prediction": decision,
            "probability": round(prob, 4),
            "risk_percent": f"{prob*100:.2f}",
            "explanation": "High risk" if prob >= 0.7 else "Medium risk" if prob >= 0.3 else "Low risk"
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error {str(e)}")


# --- Execution block ---
# If this archive run, turn on port 8000 in server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)