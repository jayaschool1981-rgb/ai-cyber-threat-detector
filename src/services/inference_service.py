import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from src.core.config import settings

logger = logging.getLogger("inference_service")

class InferenceService:
    def __init__(self):
        self.model = None
        self.preproc = None
        self.session = None  # ONNX Runtime Session
        self.expected_cols: List[str] = []
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        onnx_path = Path(settings.ONNX_MODEL_PATH)
        scaler_path = Path(settings.SCALER_PATH)
        model_path = Path(settings.MODEL_PATH)

        if not scaler_path.exists():
            logger.critical(
                f"Scaler artifact missing. Checked: Scaler={scaler_path}"
            )
            raise RuntimeError(
                "Scaler artifact missing. Train first: python -m src.pipelines.train --config config.yaml"
            )

        try:
            logger.info("Loading scaler artifact...")
            self.preproc = joblib.load(scaler_path)
            self.expected_cols = [str(c).strip() for c in self.preproc.feature_names_in_]
            logger.info(f"Scaler loaded. Expected features: {len(self.expected_cols)}")
        except Exception as e:
            logger.critical(f"Failed to load scaler artifact: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load scaler artifact: {str(e)}") from e

        # Try loading ONNX model first
        if onnx_path.exists():
            try:
                import onnxruntime as ort
                logger.info(f"Loading ONNX model from {onnx_path}...")
                self.session = ort.InferenceSession(str(onnx_path))
                logger.info("ONNX session loaded successfully.")
                return
            except Exception as e:
                logger.warning(f"Failed to load ONNX model, falling back to pickle: {str(e)}")

        # Fallback to joblib model
        if not model_path.exists():
            logger.critical(
                f"Pickle model artifact missing. Checked: Model={model_path}"
            )
            raise RuntimeError(
                "Pickle model artifact missing."
            )

        try:
            logger.info("Loading pickle model...")
            self.model = joblib.load(model_path)
            logger.info("Pickle model loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load pickle model: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load pickle model: {str(e)}") from e

    def predict(self, records: List[Dict[str, Any]]) -> List[Tuple[str, Optional[float]]]:
        if not records:
            return []

        try:
            # Convert list of dicts to DataFrame
            df = pd.DataFrame(records)

            # Normalize incoming columns
            df.columns = df.columns.astype(str).str.strip()

            # Align with expected columns
            for col in self.expected_cols:
                if col not in df.columns:
                    df[col] = 0.0

            # Drop extras not used by the model
            df = df[self.expected_cols]

            if self.session is not None:
                # Fast ONNX inference path (contains preprocessor + model)
                onnx_inputs = self.session.get_inputs()
                feed_dict = {}
                for idx, col in enumerate(self.expected_cols):
                    input_name = onnx_inputs[idx].name
                    feed_dict[input_name] = df[col].values.astype(np.float32).reshape(-1, 1)

                labels, probs = self.session.run(None, feed_dict)
                
                results = []
                for i in range(len(labels)):
                    label = str(labels[i])
                    prob_dict = probs[i]
                    confidence = float(prob_dict.get(label, 0.0))
                    results.append((label, confidence))
                return results
            else:
                # Fallback to joblib model
                X_pre = self.preproc.transform(df)
                preds = self.model.predict(X_pre)
                
                try:
                    probs = self.model.predict_proba(X_pre).max(axis=1).tolist()
                except Exception:
                    probs = [None] * len(preds)
                
                results = []
                for i in range(len(preds)):
                    prob = float(probs[i]) if probs[i] is not None else None
                    results.append((str(preds[i]), prob))
                return results
        except Exception as e:
            logger.error(f"Prediction execution failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Prediction failed: {str(e)}") from e

# Create singleton instance
inference_service = InferenceService()
