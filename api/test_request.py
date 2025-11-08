import requests

url = "http://127.0.0.1:5000/predict"

# Example minimal payload. Replace keys with the actual columns present in your CSV.
payload = {
    "Flow Duration": 12345,
    "Tot Fwd Pkts": 10,
    "Tot Bwd Pkts": 8,
    "Protocol": 6,
    "Label": "normal"  # This will be ignored if present; API uses training preprocessor which drops target.
}

r = requests.post(url, json=payload)
print(r.status_code, r.text)
