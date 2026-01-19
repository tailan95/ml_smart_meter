import gzip
import pickle

from pathlib import Path
from pandas import DataFrame
from typing import Dict

import warnings
warnings.filterwarnings("ignore")

def acquire() -> Dict[str, Dict[str, DataFrame]]:
    path = Path("/Volumes/le41/bronze/files/gzip/LVNSLoadShapes.pickle.zip")
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    return data