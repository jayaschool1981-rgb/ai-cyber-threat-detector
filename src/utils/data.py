import pandas as pd
from pathlib import Path
from typing import Optional

def load_csvs(input_csv: Optional[str], input_glob: Optional[str]) -> pd.DataFrame:
    if input_glob:
        frames = []
        for f in Path().glob(input_glob):
            frames.append(pd.read_csv(f))
        if not frames:
            raise FileNotFoundError(f"No files matched glob: {input_glob}")
        return pd.concat(frames, ignore_index=True)
    if input_csv:
        return pd.read_csv(input_csv)
    raise ValueError("Provide either data.input_csv or data.input_glob in config")
