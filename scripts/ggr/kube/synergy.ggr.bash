cd /datasets/software/git/tronn;
python setup.py develop;
PREFIX="${1}/ggr.${2}.regr.pretrained.folds.testfold-"
SUFFIX="/model.json"
tronn synergy \
      --dataset_json $3/dataset.dmim.json $4 \
      --prediction_sample $3/ggr.dmim.prediction_sample.h5 \
      --model ensemble quantile_norm models=${PREFIX}0${SUFFIX},${PREFIX}1${SUFFIX},${PREFIX}2${SUFFIX},${PREFIX}3${SUFFIX},${PREFIX}4${SUFFIX},${PREFIX}5${SUFFIX},${PREFIX}6${SUFFIX},${PREFIX}7${SUFFIX},${PREFIX}8${SUFFIX},${PREFIX}9${SUFFIX} \
      --infer_json $3/infer.dmim.json \
      --mutate_type $6 \
      --grammar $5 \
      --num_gpus 6 \
      --prefix ggr \
      --batch_size 12 \
      -o $7 \
      >> $7/job.out \
      2>> $7/job.err;


