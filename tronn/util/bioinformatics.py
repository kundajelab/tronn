"""Description: contains wrappers of relevant informatics tools

Requires that these tools are installed and/or loaded
"""

import os


def make_bed(seq_metadata, out_bed, key="active"):
    """make a bed from a text file
    """
    convert = (
        "cat {0} | "
        "awk -F '{1}=' '{{ print $2 }}' | "
        "awk -F ';' '{{ print $1 }}' | "
        "awk -F ':' '{{ print $1\"\t\"$2 }}' | "
        "awk -F '-' '{{ print $1\"\t\"$2 }}' | "
        "sort -k1,1 -k2,2n | "
        "gzip -c > {2}").format(
            seq_metadata,
            key,
            out_bed)
    print convert
    os.system(convert)

    return None


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
