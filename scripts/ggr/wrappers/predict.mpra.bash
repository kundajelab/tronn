#cd /datasets/software/git/tronn;
#python setup.py develop;
PREFIX="${1}/ggr.${2}.regr.pretrained.folds.testfold-"
SUFFIX="/model.json"
tronn predict \
      --data_format table \
      --data_files $4 \
      --fasta $5 \
      --fifo \
      --prediction_sample $3/ggr.scanmotifs.prediction_sample.h5 \
      --model_json ${PREFIX}4${SUFFIX} \
      --num_gpus 1 \
      --prefix ggr \
      --batch_size 64 \
      -o $6 #\
      #>> $6/job.out \
      #2>> $6/job.err;

#--model ensemble quantile_norm models=${PREFIX}0${SUFFIX},${PREFIX}1${SUFFIX},${PREFIX}2${SUFFIX},${PREFIX}3${SUFFIX},${PREFIX}4${SUFFIX},${PREFIX}5${SUFFIX},${PREFIX}6${SUFFIX},${PREFIX}7${SUFFIX},${PREFIX}8${SUFFIX},${PREFIX}9${SUFFIX} \
