cd /datasets/software/git/tronn;
python setup.py develop;
PREFIX="${1}/ggr.${2}.regr.pretrained.folds.testfold-"
SUFFIX="/model.json"
tronn buildtracks \
      --data_format bed \
      --data_files $3 \
      --fasta /datasets/annotations.hg19/hg19.genome.fa \
      --chromsizes /datasets/annotations.hg19/hg19.chrom.sizes \
      --fifo \
      --model ensemble quantile_norm models=${PREFIX}0${SUFFIX},${PREFIX}1${SUFFIX},${PREFIX}2${SUFFIX},${PREFIX}3${SUFFIX},${PREFIX}4${SUFFIX},${PREFIX}5${SUFFIX},${PREFIX}6${SUFFIX},${PREFIX}7${SUFFIX},${PREFIX}8${SUFFIX},${PREFIX}9${SUFFIX} \
      --num_gpus 6 \
      --inference_targets 0 1 2 3 4 5 6 9 10 12 \
      --prefix ggr \
      --batch_size 16 \
      -o $4 \
      >> $4/job.out \
      2>> $4/job.err;

