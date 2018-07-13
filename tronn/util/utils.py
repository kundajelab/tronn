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

    # for feature importance extraction
    IMPORTANCE_ANCHORS = "anchors"
    
    # processed data keys
    PWM_SCORES_PREFIX = "pwm-scores"
    DMIM_SCORES_PREFIX = "dmim-scores"

    # shuffles
    SHUFFLE_PREFIX = "shuffle"

    # split
    TASK_IDX_PREFIX = "taskidx-"
    
