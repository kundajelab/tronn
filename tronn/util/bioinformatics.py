"""Description: contains wrappers of relevant informatics tools

Requires that these tools are installed and/or loaded
"""

import os


def run_great(positive_bed, prefix, background_bed='None'):
    """Using rGreat, runs GREAT analysis
    Need to install rGreat first in R

    """

    run_rgreat = 'run_rgreat.R {0} {1}'.format(positive_bed, prefix)
    print run_rgreat
    os.system(run_rgreat)

    return None


def run_tomtom(query_file, target_file):
    """Run tomtom from MEME suite to get matches to file
    """

    return None



def run_homer():
    """Runs homer analysis
    """

    return None


def run_spamo():
    """Run SPAMO from MEME suite? for spacing?
    """

    return None
