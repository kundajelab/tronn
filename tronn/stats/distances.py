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
        similarity = np.sum(min_vals) / float(np.sum(max_vals))
    else:
        similarity = 0.
        
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






# check redundancy
def get_distance_matrix(
        motif_mat,
        corr_method="pearson",
        pval_thresh=0.05,
        corr_min=0.4):
    """Given a matrix, calculate a distance metric for each pair of 
    columns and look at p-val. If p-val is above threshold, keep 
    otherwise leave as 0.
    """
    num_columns = motif_mat.shape[1]
    correlation_vals = np.zeros((num_columns, num_columns))
    correlation_pvals = np.zeros((num_columns, num_columns))
    for i in xrange(num_columns):
        for j in xrange(num_columns):

            if corr_method == "pearson":
                cor_val, pval = pearsonr(motif_mat[:,i], motif_mat[:,j])
                if pval < pval_thresh and cor_val > corr_min:
                    correlation_vals[i,j] = cor_val
                    correlation_pvals[i,j] = pval
            elif corr_method == "continuous_jaccard":
                min_vals = np.minimum(motif_mat[:,i], motif_mat[:,j])
                max_vals = np.maximum(motif_mat[:,i], motif_mat[:,j])
                if np.sum(max_vals) != 0:
                    similarity = np.sum(min_vals) / np.sum(max_vals)
                elif i == j:
                    similarity = 1.0
                else:
                    similarity = 0.0
                correlation_vals[i,j] = similarity
                correlation_pvals[i,j] = similarity
            elif corr_method == "intersection_size":
                intersect = np.sum(
                    np.logical_and(motif_mat[:,i] > 0, motif_mat[:,j] > 0))
                #intersect_fract = float(intersect) / motif_mat.shape[0]
                intersect_fract = intersect
                correlation_vals[i,j] = intersect_fract
                correlation_pvals[i,j] = intersect_fract
                
    return correlation_vals, correlation_pvals


# check redundancy
def get_correlation_file(
        mat_file,
        corr_file,
        corr_method="intersection_size", # continuous_jaccard, pearson
        corr_min=0.4,
        pval_thresh=0.05):
    """Given a matrix file, calculate correlations across the columns
    """
    mat_df = pd.read_table(mat_file, index_col=0)
    mat_df = mat_df.drop(["community"], axis=1)
            
    corr_mat, pval_mat = get_significant_correlations(
        mat_df.as_matrix(),
        corr_method=corr_method,
        corr_min=corr_min,
        pval_thresh=pval_thresh)
    
    corr_mat_df = pd.DataFrame(corr_mat, index=mat_df.columns, columns=mat_df.columns)
    corr_mat_df.to_csv(corr_file, sep="\t")

    return None
