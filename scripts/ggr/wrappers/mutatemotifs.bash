#!/bin/bash

cd /datasets/software/git/tronn;
python setup.py develop;
PREFIX="${1}/ggr.${2}.regr.pretrained.folds.testfold-"
SUFFIX="/model.json"
tronn mutatemotifs \
      --dataset_json $3/motifs.input_x_grad.dynamic.$4/dataset.scanmotifs.2.json \
      --sig_pwms_file $5/motifs.sig/motifs.adjust.diff.rna_filt/pvals.rna_filt.corr_filt.h5 \
      --prediction_sample $3/motifs.input_x_grad.dynamic.$4/ggr.scanmotifs.prediction_sample.h5 \
      --foreground_targets $6 \
      --batch_size $7 \
      --num_gpus $8 \
      --model ensemble quantile_norm models=${PREFIX}0${SUFFIX},${PREFIX}1${SUFFIX},${PREFIX}2${SUFFIX},${PREFIX}3${SUFFIX},${PREFIX}4${SUFFIX},${PREFIX}5${SUFFIX},${PREFIX}6${SUFFIX},${PREFIX}7${SUFFIX},${PREFIX}8${SUFFIX},${PREFIX}9${SUFFIX} \
      --mutate_type $9 \
      --prefix ggr \
      -o ${10} \
      >> ${10}/job.out \
      2>> ${10}/job.err;
