# AI-Powered Cybersecurity Threat Detection (CICIDS2017)

End-to-end project: dataset → preprocessing → model training → saved model → REST API → demo.

## Quickstart

```bash
git clone <your-repo> ai_cyber_threat_detection
cd ai_cyber_threat_detection
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 1) Download the dataset (CICIDS2017)
- Source: Canadian Institute for Cybersecurity (CICIDS2017). It contains normal + multiple attack types (DDoS, PortScan, Botnet, etc.).
- Download the **combined CSV** or daily CSVs. Place files under `data/raw/`.
- Example assumed file: `data/raw/CICIDS2017_sample.csv` (you can rename).

### 2) Configure
Update `config.yaml` paths if your filenames differ.

### 3) Run preprocessing + train
```bash
python -m src.pipelines.train --config config.yaml
```

### 4) Serve the model API
```bash
python api/app.py  # dev
# or production
gunicorn -w 2 -b 0.0.0.0:5000 api.app:app
```

### 5) Test the API
```bash
python api/test_request.py
# or use Postman collection: postman_collection.json
```

### Features
- Robust preprocessing with numeric/label handling
- Train/validation split + class imbalance support
- RandomForest (default) with tunable hyperparameters
- Model persistence to `models/model.pkl` + `models/feature_columns.json`
- REST API (`/predict`) for single-record inference

### Notes
- If using multiple CSVs, you can change `config.yaml: data.input_glob` to a pattern like `data/raw/*.csv`.
- The API expects **the same features/columns used in training**. A mapping file ensures consistent order.

*Generated on: 2025-11-07T15:56:42*
