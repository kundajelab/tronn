# description: various useful helpers



class DataKeys(object):
    """standard names for features
    and transformations of features
    """

    # core data keys
    FEATURES = "features"
    LABELS = "labels"
    PROBABILITIES = "probs"
    LOGITS = "logits"
    ORIG_SEQ = "sequence"
    SEQ_METADATA = "example_metadata"

    # playing with feature transforms
    HIDDEN_LAYER_FEATURES = "{}.nn_transform".format(FEATURES)
    
    # for feature importance extraction
    IMPORTANCE_ANCHORS = "anchors"
    IMPORTANCE_GRADIENTS = "gradients"
    WEIGHTED_SEQ = "{}-weighted".format(ORIG_SEQ)

    # clipping
    ORIG_SEQ_ACTIVE = "{}.active".format(ORIG_SEQ)
    WEIGHTED_SEQ_ACTIVE = "{}.active".format(WEIGHTED_SEQ)
    
    # for shuffles (generated for null models)
    SHUFFLE_SUFFIX = "shuffles"
    ACTIVE_SHUFFLES = SHUFFLE_SUFFIX
    ORIG_SEQ_SHUF = "{}.{}".format(ORIG_SEQ, SHUFFLE_SUFFIX)
    WEIGHTED_SEQ_SHUF = "{}.{}".format(WEIGHTED_SEQ, SHUFFLE_SUFFIX)
    ORIG_SEQ_ACTIVE_SHUF = "{}.{}".format(ORIG_SEQ_ACTIVE, SHUFFLE_SUFFIX)
    WEIGHTED_SEQ_ACTIVE_SHUF = "{}.{}".format(WEIGHTED_SEQ_ACTIVE, SHUFFLE_SUFFIX)
    
    # pwm transformation keys
    PWM_SCORES_ROOT = "pwm-scores"

    ORIG_SEQ_PWM_HITS = "{}.pwm-hits".format(ORIG_SEQ_ACTIVE)
    ORIG_SEQ_PWM_SCORES = "{}.{}".format(ORIG_SEQ_ACTIVE, PWM_SCORES_ROOT)
    ORIG_SEQ_PWM_SCORES_THRESH = "{}.thresh".format(ORIG_SEQ_PWM_SCORES)
    ORIG_SEQ_PWM_SCORES_SUM = "{}.sum".format(ORIG_SEQ_PWM_SCORES_THRESH)
    ORIG_SEQ_SHUF_PWM_SCORES = "{}.{}".format(ORIG_SEQ_ACTIVE_SHUF, PWM_SCORES_ROOT)
    
    ORIG_SEQ_PWM_DENSITIES = "{}.densities".format(ORIG_SEQ_PWM_HITS)
    ORIG_SEQ_PWM_MAX_DENSITIES = "{}.max".format(ORIG_SEQ_PWM_DENSITIES)
    
    WEIGHTED_SEQ_PWM_HITS = "{}.pwm-hits".format(WEIGHTED_SEQ_ACTIVE)    
    WEIGHTED_SEQ_PWM_SCORES = "{}.{}".format(WEIGHTED_SEQ_ACTIVE, PWM_SCORES_ROOT)
    WEIGHTED_SEQ_PWM_SCORES_THRESH = "{}.thresh".format(WEIGHTED_SEQ_PWM_SCORES)
    WEIGHTED_SEQ_PWM_SCORES_SUM = "{}.sum".format(WEIGHTED_SEQ_PWM_SCORES_THRESH)
    WEIGHTED_SEQ_SHUF_PWM_SCORES = "{}.{}".format(WEIGHTED_SEQ_ACTIVE_SHUF, PWM_SCORES_ROOT)

    # significant pwms
    PWM_SIG_ROOT = "pwms.sig"
    PWM_SIG_GLOBAL = "{}.global".format(PWM_SIG_ROOT)
    PWM_SCORES_AGG_GLOBAL = "{}.agg".format(PWM_SIG_GLOBAL)
    PWM_SIG_CLUST = "{}.clusters".format(PWM_SIG_ROOT)
    PWM_SIG_CLUST_ALL = "{}.all".format(PWM_SIG_CLUST)
    PWM_SCORES_AGG_CLUST = "{}.agg".format(PWM_SIG_CLUST_ALL)
    
    # manifold keys
    MANIFOLD_ROOT = "manifold"
    MANIFOLD_CENTERS = "{}.centers".format(MANIFOLD_ROOT)
    MANIFOLD_THRESHOLDS = "{}.thresholds".format(MANIFOLD_ROOT)
    MANIFOLD_CLUST = "{}.clusters".format(MANIFOLD_ROOT)

    # manifold sig pwm keys
    MANIFOLD_PWM_ROOT = "{}.pwms".format(MANIFOLD_ROOT)
    MANIFOLD_PWM_SIG_GLOBAL = "{}.global".format(MANIFOLD_PWM_ROOT)
    MANIFOLD_PWM_SCORES_AGG_GLOBAL = "{}.agg".format(MANIFOLD_PWM_SIG_GLOBAL)
    MANIFOLD_PWM_SIG_CLUST = "{}.clusters".format(MANIFOLD_PWM_ROOT)
    MANIFOLD_PWM_SIG_CLUST_ALL = "{}.all".format(MANIFOLD_PWM_SIG_CLUST)
    MANIFOLD_PWM_SCORES_AGG_CLUST = "{}.agg".format(MANIFOLD_PWM_SIG_CLUST_ALL)
    
    # cluster keys
    CLUSTERS = "clusters"
    
    # dmim transformation keys
    DMIM_SCORES_ROOT = "dmim-scores"

    # split
    TASK_IDX_ROOT = "taskidx"
    
