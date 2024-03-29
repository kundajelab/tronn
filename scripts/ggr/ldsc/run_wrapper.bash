#!/bin/bash

# setup
REPO_DIR=/users/dskim89/git/tronn # github repo
ANNOT_DIR=/mnt/lab_data3/dskim89/ggr/annotations # mostly just location of chromsizes
IN_DIR=/mnt/lab_data3/dskim89/ggr/nn/2019-03-12.freeze/ldsc # where all the data is
GRAMMAR_DIR=/mnt/lab_data/kundaje/users/dskim89/ggr/nn/inference.2019-03-12/dmim.shuffle.OLD/grammars.annotated.manual_filt.merged.final
OUT_DIR=$IN_DIR # where to send outputs

# run
$REPO_DIR/scripts/ggr/ldsc/runall.py $REPO_DIR/scripts/ggr/ldsc/annotations.json $ANNOT_DIR $GRAMMAR_DIR $IN_DIR .
