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
    LOGITS_NORM = "{}.norm".format(LOGITS)
    ORIG_SEQ = "sequence"
    SEQ_METADATA = "example_metadata"

    # ensembles
    LOGITS_MULTIMODEL = "{}.multimodel".format(LOGITS)
    LOGITS_MULTIMODEL_NORM = "{}.norm".format(LOGITS_MULTIMODEL)
    LOGITS_CI = "{}.ci".format(LOGITS)
    LOGITS_CI_THRESH = "{}.thresh".format(LOGITS_CI)
    
    # playing with feature transforms
    HIDDEN_LAYER_FEATURES = "{}.nn_transform".format(FEATURES)
    
    # for feature importance extraction
    IMPORTANCE_ANCHORS = "anchors"
    IMPORTANCE_GRADIENTS = "gradients"
    WEIGHTED_SEQ = "{}-weighted".format(ORIG_SEQ)
    WEIGHTED_SEQ_THRESHOLDS = "{}.thresholds".format(WEIGHTED_SEQ)

    # clipping
    ORIG_SEQ_ACTIVE = "{}.active".format(ORIG_SEQ)
    ORIG_SEQ_ACTIVE_STRING = "{}.string".format(ORIG_SEQ_ACTIVE)
    WEIGHTED_SEQ_ACTIVE = "{}.active".format(WEIGHTED_SEQ)
    GC_CONTENT = "{}.gc_fract".format(ORIG_SEQ_ACTIVE)

    WEIGHTED_SEQ_ACTIVE_CI = "{}.ci".format(WEIGHTED_SEQ_ACTIVE)
    WEIGHTED_SEQ_ACTIVE_CI_THRESH = "{}.thresh".format(WEIGHTED_SEQ_ACTIVE_CI)
    WEIGHTED_SEQ_MULTIMODEL = "{}.multimodel".format(WEIGHTED_SEQ_ACTIVE)
    
    # for shuffles (generated for null models)
    SHUFFLE_SUFFIX = "shuffles"
    ACTIVE_SHUFFLES = SHUFFLE_SUFFIX
    ORIG_SEQ_SHUF = "{}.{}".format(ORIG_SEQ, SHUFFLE_SUFFIX)
    WEIGHTED_SEQ_SHUF = "{}.{}".format(WEIGHTED_SEQ, SHUFFLE_SUFFIX)
    ORIG_SEQ_ACTIVE_SHUF = "{}.{}".format(ORIG_SEQ_ACTIVE, SHUFFLE_SUFFIX)
    WEIGHTED_SEQ_ACTIVE_SHUF = "{}.{}".format(WEIGHTED_SEQ_ACTIVE, SHUFFLE_SUFFIX)
    LOGITS_SHUF = "{}.{}".format(LOGITS, SHUFFLE_SUFFIX)
    
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
    WEIGHTED_SEQ_PWM_SCORES_SUM = "{}.sum".format(WEIGHTED_SEQ_PWM_SCORES_THRESH) # CHECK some downstream change here
    WEIGHTED_SEQ_SHUF_PWM_SCORES = "{}.{}".format(WEIGHTED_SEQ_ACTIVE_SHUF, PWM_SCORES_ROOT)

    # pwm positions
    WEIGHTED_PWM_SCORES_POSITION_MAX_VAL = "{}.max.val".format(WEIGHTED_SEQ_PWM_SCORES_THRESH)
    WEIGHTED_PWM_SCORES_POSITION_MAX_IDX = "{}.max.idx".format(WEIGHTED_SEQ_PWM_SCORES_THRESH)
    WEIGHTED_PWM_SCORES_POSITION_MAX_VAL_MUT = "{}.max.val.motif_mut".format(WEIGHTED_SEQ_PWM_SCORES_THRESH)
    WEIGHTED_PWM_SCORES_POSITION_MAX_IDX_MUT = "{}.max.idx.motif_mut".format(WEIGHTED_SEQ_PWM_SCORES_THRESH)
    NULL_PWM_POSITION_INDICES = "{}.null.idx".format(PWM_SCORES_ROOT)
    
    # significant pwms
    PWM_DIFF_GROUP = "pwms.differential"
    PWM_PVALS = "pwms.pvals"
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
    MANIFOLD_SCORES = "{}.scores".format(MANIFOLD_ROOT)

    # manifold sig pwm keys
    MANIFOLD_PWM_ROOT = "{}.pwms".format(MANIFOLD_ROOT)
    MANIFOLD_PWM_SIG_GLOBAL = "{}.global".format(MANIFOLD_PWM_ROOT)
    MANIFOLD_PWM_SCORES_AGG_GLOBAL = "{}.agg".format(MANIFOLD_PWM_SIG_GLOBAL)
    MANIFOLD_PWM_SIG_CLUST = "{}.clusters".format(MANIFOLD_PWM_ROOT)
    MANIFOLD_PWM_SIG_CLUST_ALL = "{}.all".format(MANIFOLD_PWM_SIG_CLUST)
    MANIFOLD_PWM_SCORES_AGG_CLUST = "{}.agg".format(MANIFOLD_PWM_SIG_CLUST_ALL)
    
    # cluster keys
    CLUSTERS = "clusters"
    
    # dfim transformation keys
    MUT_MOTIF_ORIG_SEQ = "{}.motif_mut".format(ORIG_SEQ)
    MUT_MOTIF_WEIGHTED_SEQ = "{}.motif_mut".format(WEIGHTED_SEQ)
    MUT_MOTIF_POS = "{}.pos".format(MUT_MOTIF_ORIG_SEQ)
    MUT_MOTIF_MASK = "{}.mask".format(MUT_MOTIF_ORIG_SEQ)
    MUT_MOTIF_PRESENT = "{}.motif_mut_present".format(ORIG_SEQ)
    MUT_MOTIF_LOGITS = "{}.motif_mut".format(LOGITS)
    MUT_MOTIF_LOGITS_SIG = "{}.motif_mut.sig".format(LOGITS)
    DFIM_SCORES = "{}.delta".format(MUT_MOTIF_WEIGHTED_SEQ)
    DFIM_SCORES_DX = "{}.delta.dx".format(MUT_MOTIF_WEIGHTED_SEQ)
    
    MUT_MOTIF_WEIGHTED_SEQ_CI = "{}.ci".format(MUT_MOTIF_WEIGHTED_SEQ)
    MUT_MOTIF_WEIGHTED_SEQ_CI_THRESH = "{}.thresh".format(MUT_MOTIF_WEIGHTED_SEQ_CI)
    MUT_MOTIF_LOGITS_MULTIMODEL = "{}.multimodel".format(MUT_MOTIF_LOGITS)

    # dmim agg results keys
    DMIM_SCORES = "dmim.motif_mut.scores"
    DMIM_SCORES_SIG = "{}.sig".format(DMIM_SCORES)
    DMIM_DIFF_GROUP = "dmim.differential"
    DMIM_PVALS = "dmim.pvals"
    DMIM_SIG_ROOT = "dmim.sig"
    DMIM_SIG_ALL = "{}.all".format(DMIM_SIG_ROOT)
    DMIM_ROOT = "{}.{}".format(DFIM_SCORES, PWM_SCORES_ROOT)
    DMIM_SIG_RESULTS = "{}.agg.sig".format(DMIM_ROOT)

    # grammar keys
    GRAMMAR_ROOT = "grammar"
    GRAMMAR_LABELS = "{}.labels".format(GRAMMAR_ROOT)

    # synergy keys
    SYNERGY_ROOT = "synergy"
    SYNERGY_SCORES = "{}.scores".format(SYNERGY_ROOT)
    SYNERGY_DIFF = "{}.diff".format(SYNERGY_SCORES)
    SYNERGY_DIFF_SIG = "{}.sig".format(SYNERGY_DIFF)
    SYNERGY_DIST = "{}.dist".format(SYNERGY_ROOT)
    SYNERGY_MAX_DIST = "{}.max".format(SYNERGY_DIST)

    # variant keys
    VARIANT_ROOT = "variants"
    VARIANT_ID = "{}.id.string".format(VARIANT_ROOT)
    VARIANT_INFO = "{}.info.string".format(VARIANT_ROOT)
    VARIANT_IDX = "{}.idx".format(VARIANT_ROOT)
    VARIANT_DMIM = "{}.dmim".format(VARIANT_ROOT)
    VARIANT_SIG = "{}.sig".format(VARIANT_ROOT)
    
    # split
    TASK_IDX_ROOT = "taskidx"


class MetaKeys(object):
    """standard names for metadata keys
    """
    REGION_ID = "region"
    ACTIVE_ID = "active"
    FEATURES_ID = "features"


class ParamKeys(object):
    """standard names for params"""
    
    # tensor management
    AUX_ROOT = "{}.aux".format(DataKeys.FEATURES)
    NUM_AUX = "{}.num".format(AUX_ROOT)
