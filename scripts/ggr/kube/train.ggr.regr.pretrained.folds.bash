cd /datasets/software/git/tronn;
python setup.py develop;
tronn train \
      --data_dir /datasets/ggr/1.0.0d/h5 \
      --fasta /datasets/annotations.hg19/hg19.genome.fa \
      --targets ATAC_SIGNALS.NORM H3K27ac_SIGNALS.NORM H3K4me1_SIGNALS.NORM \
      --batch_size 1024 \
      --model $2 \
      --transfer_model_json $5/ggr.$2.clf.pretrained.folds.testfold-$4/model.json \
      --transfer_skip_vars logit \
      --use_transfer_splits \
      --kfolds 10 \
      -o $1  \
      --prefix ggr \
      --valid_fold $3 \
      --test_fold $4 \
      --regression \
      >> $1/job.out \
      2>> $1/job.err;
