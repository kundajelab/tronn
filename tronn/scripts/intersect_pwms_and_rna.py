#!/usr/bin/env python

'''
description: script to use for those who also
have joint RNA data to help filter their results
from `tronn scanmotifs`



'''

import os
import sys
import argparse


def parse_args():
    """parser
    """
    parser = argparser.ArgumentParser(
        description="add in RNA data to filter pwms")

    # required args
    parser.add_argument(
        "--pwm_file", required=True,
        help="pwm file to filter")
    parser.add_argument(
        "--pwm_metadata_file", required=True,
        help="metadata file, requires 1 column with PWM names and the other with gene ids OR genes present")

    # for second stage
    parser.add_argument(
        "--rna_expression_file",
        help="RNA file if adding in RNA expression information")
    parser.add_argument(
        "--pwm_score_file",
        help="RNA file if adding in RNA expression information")

    # other
    parser.add_argument(
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="out directory")


    # parse args
    args = parser.parse_args()
    
    return args


def _read_pwm_metadata(metadata_file):
    """
    """
    

    
    return



def _match_pwm_to_tf_expression(pwm_list, pwm_metadata_dict):
    """create a vector (ordered) which tells whether that TF is expressed or not
    expand for those that have multiple
    """
    
    
    

    return


def main():
    """run the intersect
    """
    args = parse_args()


    # STAGE 1 - filter PWMs for those with gene expression
    # this bit just needs the pwm file and pwm metadata

    


    # STAGE 2 - correlation information
    # this requires RNA matrix (gene_id, val)
    # also pwm score matrix (pwm, val) <- this means save this out to text file
    # then can do correlation

    # save all this information together in an hdf5 file?

    # then can plot all together in R


    # make sure to return a fully filtered pwm file too

    return



if __name__ == '__main__':
    main()
