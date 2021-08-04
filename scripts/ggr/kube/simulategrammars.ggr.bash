cd /datasets/software/git/tronn;
python setup.py develop;
PREFIX="${1}/ggr.${2}.regr.pretrained.folds.testfold-"
SUFFIX="/model.json"
tronn simulategrammar \
      --data_format pwm_sims \
      --dataset_pwm_file /datasets/ggr/1.0.0d/annotations/HOCOMOCOv11_core_pwms_HUMAN_mono.renamed.nonredundant.txt \
      --data_files $3 \
      --grammar_range 0 30 \
      --num_samples 30 \
      --background_regions /datasets/ggr/1.0.0d/annotations/ggr.atac.idr.reg2map_negatives.bed.gz \
      --fasta /datasets/annotations.hg19/hg19.genome.fa \
      --batch_size 12 \
      --prediction_sample $4/ggr.dmim.prediction_sample.h5 \
      --model ensemble quantile_norm models=${PREFIX}0${SUFFIX},${PREFIX}1${SUFFIX},${PREFIX}2${SUFFIX},${PREFIX}3${SUFFIX},${PREFIX}4${SUFFIX},${PREFIX}5${SUFFIX},${PREFIX}6${SUFFIX},${PREFIX}7${SUFFIX},${PREFIX}8${SUFFIX},${PREFIX}9${SUFFIX} \
      --num_gpus 6 \
      --inference_targets 0 1 2 3 4 5 6 9 10 12 \
      --prefix ggr \
      --pwm_file /datasets/ggr/1.0.0d/annotations/HOCOMOCOv11_core_pwms_HUMAN_mono.renamed.nonredundant.txt \
      -o $5 \
      >> $5/job.out \
      2>> $5/job.err

