# description: code to run timeseries style analyses



def extract_importances():
    """Use this to get importances for desired tasks.
    Consider gradient*input here (vs guided backprop) for better downstream normalization

    Make it possible to throw in regions or individual examples

    First first, only keep those that predict correctly (set up separate function for this criteria, need to try different metrics)
    First, threshold these importances (stdev). this reduces noise
    Then, normalize by making it into probabilities (across 1000 bp, how is weight distributed if total weight is the final prob)
    ie, (x - min) / (max - min) * prediction val

    Then, save out into seqlets with timepoint info
    When saving out, zscore across time
    """
    


    

    return


def cluster_seqlets():
    """Just use phenograph for this
    """
    

    return


def make_motifs():
    """ hAgglom as before, figure out how to set good stopping criterion
    """

    return




