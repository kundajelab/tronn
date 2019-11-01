#!/bin/bash

cd /datasets/software/git/tronn;
python setup.py develop;
PREFIX="${1}/ggr.${2}.regr.pretrained.folds.testfold-"
SUFFIX="/model.json"
tronn scanmotifs \
      --data_dir /datasets/ggr/1.0.0d/h5 \
      --targets ATAC_SIGNALS.NORM H3K27ac_SIGNALS.NORM H3K4me1_SIGNALS.NORM \
      --filter_targets $3 \
      --fasta /datasets/annotations.hg19/hg19.genome.fa \
      --batch_size $4 \
      --model ensemble quantile_norm models=${PREFIX}0${SUFFIX},${PREFIX}1${SUFFIX},${PREFIX}2${SUFFIX},${PREFIX}3${SUFFIX},${PREFIX}4${SUFFIX},${PREFIX}5${SUFFIX},${PREFIX}6${SUFFIX},${PREFIX}7${SUFFIX},${PREFIX}8${SUFFIX},${PREFIX}9${SUFFIX} \
      --num_gpus $5 \
      --inference_targets 0 1 2 3 4 5 6 9 10 12 \
      --use_filtering \
      --prefix ggr \
      --pwm_file /datasets/ggr/1.0.0d/annotations/HOCOMOCOv11_core_pwms_HUMAN_mono.renamed.nonredundant.txt \
      --add_rc_pwms \
      --scratch_dir $7 \
      -o $6 \
      >> $6/job.out \
      2>> $6/job.err;

