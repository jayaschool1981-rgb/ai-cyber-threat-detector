import joblib
import argparse
from pathlib import Path
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/model.pkl")
    parser.add_argument("--scaler-path", default="models/scaler.pkl")
    parser.add_argument("--output-path", default="models/model.onnx")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    scaler_path = Path(args.scaler_path)
    output_path = Path(args.output_path)

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"Model or Scaler pickles not found. Checked: model={model_path}, scaler={scaler_path}"
        )

    print(f"Loading {model_path} and {scaler_path}...")
    model = joblib.load(model_path)
    preproc = joblib.load(scaler_path)

    # Combine scaler and classifier into a single Pipeline
    print("Building Scikit-Learn Pipeline...")
    pipeline = Pipeline([
        ("preprocessor", preproc),
        ("classifier", model)
    ])

    # Determine number of input features
    input_features_count = len(preproc.feature_names_in_)
    print(f"Model expects {input_features_count} input features.")

    # Define the initial types for ONNX conversion
    # Each feature column is defined as an individual float input tensor of shape [None, 1]
    initial_types = [(str(name), FloatTensorType([None, 1])) for name in preproc.feature_names_in_]

    print("Converting pipeline to ONNX format...")
    try:
        onnx_model = convert_sklearn(
            pipeline,
            initial_types=initial_types,
            target_opset=12
        )
        
        # Save ONNX model
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"Successfully exported combined pipeline to {output_path}")
    except Exception as e:
        print(f"ONNX conversion failed: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
