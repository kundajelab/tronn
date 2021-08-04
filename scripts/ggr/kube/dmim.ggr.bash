cd /datasets/software/git/tronn;
python setup.py develop;
PREFIX="${1}/ggr.${2}.regr.pretrained.folds.testfold-"
SUFFIX="/model.json"
tronn dmim \
      --dataset_json $3/motifs.input_x_grad.$4/dataset.scanmotifs.json \
      --infer_json $3/motifs.input_x_grad.$4/infer.scanmotifs.json \
      --filter_targets $5 \
      --model ensemble quantile_norm models=${PREFIX}0${SUFFIX},${PREFIX}1${SUFFIX},${PREFIX}2${SUFFIX},${PREFIX}3${SUFFIX},${PREFIX}4${SUFFIX},${PREFIX}5${SUFFIX},${PREFIX}6${SUFFIX},${PREFIX}7${SUFFIX},${PREFIX}8${SUFFIX},${PREFIX}9${SUFFIX} \
      --sig_pwms_file $7/motifs.sig/motifs.adjust.diff.rna_filt.dmim/pvals.rna_filt.corr_filt.h5 \
      --foreground_targets $5 \
      --mutate_type point \
      --num_gpus 6 \
      --prefix ggr \
      --batch_size 4 \
      -o $6 \
      >> $6/job.out \
      2>> $6/job.err;


