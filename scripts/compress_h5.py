#!/usr/bin/env python

# description: script to compress h5 file if needed

import os
import sys

from tronn.util.h5_utils import compress_h5_file

def main():
    """test
    """
    # set up
    in_file = sys.argv[1]
    if len(sys.argv) <= 2:
        out_file = None
    else:
        out_file = sys.argv[2]

    # logging
    print "IN FILE: {}".format(in_file)
    print "OUT FILE: {}".format(out_file)

    # compress
    compress_h5_file(in_file, output_file=out_file)

    return

main()
