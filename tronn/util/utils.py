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
    
    # for feature importance extraction
    IMPORTANCE_ANCHORS = "anchors"
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
    PWM_SCORES_SUFFIX = "pwm-scores"
    
    ORIG_SEQ_PWM_SCORES = "{}.{}".format(ORIG_SEQ_ACTIVE, PWM_SCORES_SUFFIX)
    ORIG_SEQ_PWM_SCORES_THRESH = "{}.{}.thresh".format(ORIG_SEQ_ACTIVE, PWM_SCORES_SUFFIX)
    ORIG_SEQ_SHUF_PWM_SCORES = "{}.{}".format(ORIG_SEQ_ACTIVE_SHUF, PWM_SCORES_SUFFIX)
    ORIG_SEQ_PWM_HITS = "{}.pwm-hits".format(ORIG_SEQ_ACTIVE)
    
    WEIGHTED_SEQ_PWM_SCORES = "{}.{}".format(WEIGHTED_SEQ_ACTIVE, PWM_SCORES_SUFFIX)
    WEIGHTED_SEQ_PWM_SCORES_THRESH = "{}.{}.thresh".format(WEIGHTED_SEQ_ACTIVE, PWM_SCORES_SUFFIX)
    WEIGHTED_SEQ_SHUF_PWM_SCORES = "{}.{}".format(WEIGHTED_SEQ_ACTIVE_SHUF, PWM_SCORES_SUFFIX)
    WEIGHTED_SEQ_PWM_HITS = "{}.pwm-hits".format(WEIGHTED_SEQ_ACTIVE)

    # dmim transformation keys
    DMIM_SCORES_PREFIX = "dmim-scores"

    # split
    TASK_IDX_PREFIX = "taskidx-"
    
