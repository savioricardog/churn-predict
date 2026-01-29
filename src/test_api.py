import requests
import json

# API URL
url = "http://localhost:8000/predict"

# Dados de um cliente (Simulando o sistema da empresa enviando)
cliente = {
        "Gender": "Female",
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
        "Tenure_Months": 3,
        "Monthly_Charges": 780.0,
        "Total_Charges": 8400.0,
        "CLTV": 4000,
        "Country": "United States",
        "State": "Florida",
        "Zip_Code": 90210
}

print("📡 Enviando dados para o Modelo...")
response = requests.post(url, json=cliente)

if response.status_code == 200:
    resultado = response.json()
    print("\n✅ Resposta Recebida:")
    print(f"🔮 Previsão: {resultado['prediction']}")
    print(f"📊 Risco: {resultado['risk_percent']}%")
    print(f"📝 Explicação: {resultado['explanation']}")
else:
    print("❌ Erro:", response.text)