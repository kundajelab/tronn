cd /datasets/software/git/tronn;
python setup.py develop;
tronn evaluate \
      --dataset_json $1/dataset.test_split.json \
      --batch_size 1024 \
      --model_json $1/model.json \
      -o $2  \
      --prefix ggr \
      >> $2/job.out \
      2>> $2/job.err;
