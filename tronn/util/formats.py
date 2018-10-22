"""description: tools for handling json formatting
"""

import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """correctly handle numpy types"""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def write_to_json(data, out_file):
    """write to json nicely
    """
    assert out_file.endswith(".json")
    with open(out_file, "w") as fp:
        json.dump(
            data,
            fp,
            sort_keys=True,
            indent=4,
            cls=NumpyEncoder)
