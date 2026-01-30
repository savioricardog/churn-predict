import requests
import json

# API URL
url = "http://localhost:8000/predict" # DEFINING API URL

# DATA TESTINT TO PREDICT (IN JSON FORMAT)
client = {
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

print("📡 sENDING DATA TO MODEL...")
response = requests.post(url, json=client)

if response.status_code == 200:
    resultado = response.json()
    print("\n✅ ANSWER ARRIVED:")
    print(f"🔮 PREVISION: {resultado['prediction']}")
    print(f"📊 RISK VALUE: {resultado['risk_percent']}%")
    print(f"📝 EXPLANATION: {resultado['explanation']}")
else:
    print("❌ Error:", response.text)