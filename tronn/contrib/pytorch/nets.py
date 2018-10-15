"""description - networks
"""

import sys
import numpy as np

from tronn.util.utils import DataKeys

# load in CS273B repo
sys.path.insert(0, "/srv/scratch/dskim89/surag/CS273B")

# surag model
import wrapper


def setup_model():
    """set up the model first
    """
    # set up model
    model = wrapper.Model(
        pt_args="/srv/scratch/dskim89/surag/pt_args.joblib",
        checkpoint_folder="/srv/scratch/dskim89/surag")

    return model


net_fns = {
    "cs273b": setup_model
}

