cd /datasets/software/git/tronn;
python setup.py develop;
tronn train \
      --data_dir /datasets/ggr/1.0.0d/h5 \
      --fasta /datasets/annotations.hg19/hg19.genome.fa \
      --targets ATAC_LABELS TRAJ_LABELS H3K27ac_LABELS H3K4me1_LABELS H3K27me3_LABELS CTCF_LABELS POL2_LABELS TP63_LABELS KLF4_LABELS ZNF750_LABELS DYNAMIC_MARK_LABELS DYNAMIC_STATE_LABELS STABLE_MARK_LABELS STABLE_STATE_LABELS ATAC_LABELS::reduce_type=any ATAC_LABELS::reduce_type=all TRAJ_LABELS=0,7,8,9,10,11::reduce_type=any TRAJ_LABELS=12,13,14,1::reduce_type=any TRAJ_LABELS=2,3,4,5::reduce_type=any TRAJ_LABELS=8,9,10,11::reduce_type=any TRAJ_LABELS=3,4,5::reduce_type=any \
      --batch_size 1024 \
      --model $2 \
      --kfolds 10 \
      -o $1  \
      --prefix ggr \
      --valid_fold $3 \
      --test_fold $4 \
      >> $1/job.out \
      2>> $1/job.err;
