cd /datasets/software/git/tronn;
python setup.py develop;
PREFIX="${1}/ggr.${2}.regr.pretrained.folds.testfold-"
SUFFIX="/model.json"
tronn predict \
      --data_format pwm_sims \
      --data_files $4 \
      --single_pwm \
      --count_range 1 6 \
      --background_regions /datasets/ggr/1.0.0d/annotations/ggr.atac.idr.negatives.bed.gz \
      --fasta /datasets/annotations.hg19/hg19.genome.fa \
      --fifo \
      --prediction_sample $3/ggr.scanmotifs.prediction_sample.h5 \
      --model ensemble quantile_norm models=${PREFIX}0${SUFFIX},${PREFIX}1${SUFFIX},${PREFIX}2${SUFFIX},${PREFIX}3${SUFFIX},${PREFIX}4${SUFFIX},${PREFIX}5${SUFFIX},${PREFIX}6${SUFFIX},${PREFIX}7${SUFFIX},${PREFIX}8${SUFFIX},${PREFIX}9${SUFFIX} \
      --pwm_file /datasets/ggr/1.0.0d/annotations/HOCOMOCOv11_core_pwms_HUMAN_mono.renamed.nonredundant.txt \
      --num_gpus 6 \
      --prefix ggr \
      --batch_size 12 \
      -o $5 \
      >> $5/job.out \
      2>> $5/job.err;


