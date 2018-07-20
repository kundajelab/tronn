# description: code for calculating various distances

import numpy as np



def generalized_jaccard_similarity(a, b):
    """Calculates a jaccard on real valued vectors
    note that for correct min/max, vectors should be
    normalized to be on same scale
    """
    assert np.all(a >= 0)
    assert np.all(b >= 0)

    # get min and max
    min_vals = np.minimum(a, b)
    max_vals = np.maximum(a, b)

    # divide sum(min) by sum(max)
    if np.sum(max_vals) != 0:
        similarity = np.sum(min_vals) / np.sum(max_vals)
    else:
        similarity = 0
        
    return similarity


def _build_distance_fn(vector, method):
    """return a distance fn to get distances from
    input vector
    """
    if method == "dot":
        def dist_fn(compare_vector):
            return np.dot(compare_vector, vector)
        
    elif method == "norm":
        def dist_fn(compare_vector):
            return np.linalg.norm(compare_vector - vector, ord=1)
        
    elif method == "jaccard":
        def dist_fn(compare_vector):
            return generalized_jaccard_similarity(compare_vector, vector)
        
    return dist_fn


def score_distances(array, mean_vector, method="jaccard"):
    """for each example, get distance from mean vector
    """
    # build a 1d distance function
    dist_fn = _build_distance_fn(mean_vector, method)

    # apply
    distances = np.apply_along_axis(dist_fn, 1, array)
    
    return distances
