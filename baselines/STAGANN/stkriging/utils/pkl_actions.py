import os.path
import pickle

import sys
import torch
import numpy as np


import os.path
import pickle
import torch
import numpy as np

if np.__version__.startswith("1."):
    import numpy.core as _core
    sys.modules["numpy._core"] = _core
    if hasattr(_core, 'numeric'):
        sys.modules["numpy._core.numeric"] = _core.numeric
    if hasattr(_core, 'multiarray'):
        sys.modules["numpy._core.multiarray"] = _core.multiarray

def load_pkl(pickle_file: str) -> object:
    """Load pickle data (safely handles misnamed .npy files).

    Args:
        pickle_file (str): file path

    Returns:
        object: loaded object
    """
    # 1. Read the first 6 bytes to check the file signature
    with open(pickle_file, "rb") as f:
        header = f.read(6)
        
    # 2. If the file is actually a NumPy file
    if header == b"\x93NUMPY":
        try:
            # allow_pickle=True is required if the array contains Python objects (like lists/tuples)
            return np.load(pickle_file, allow_pickle=True)
        except Exception as e:
            print(f"Unable to load numpy data {pickle_file}: {e}")
            raise
            
    # 3. If it's a standard pickle file
    else:
        try:
            with open(pickle_file, "rb") as f:
                return pickle.load(f, encoding="latin1")
        except Exception as e:
            print(f"Unable to load pickle data {pickle_file}: {e}")
            raise


def dump_pkl(obj: object, file_path: str):
    """Dumplicate pickle data.

    Args:
        obj (object): object
        file_path (str): file path
    """

    with open(file_path, "wb") as f:
        pickle.dump(obj, f)
