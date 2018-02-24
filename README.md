# TRONN
Transcriptional Regulation (Optimized) Neural Nets (TRONN) - neural network tools for building integrative gene regulation models

---
### Installation

Easiest way is to first install anaconda, and set up an environment.

```
conda create -n tronn python=2 anaconda
```

Within the environment, install tensorflow according to tensorflow instructions for installing in Anaconda
(NOTE: I believe tronn is compatible with 1.5.0, but we're currently missing CUDA 9.0)

```
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp27-none-linux_x86_64.whl
```

Then, install tronn.

```
python setup.py develop
```


---
### Quick start


#### Deep learning

To set up datasets, run:
```
tronn preprocess --labels $PEAK_FILES -o datasets/ggr --prefix ggr.integrative
```

Then train your model:
```
tronn train ...
```

To evaluate your model:
```
tronn evaluate
```

To predict outputs from your model:
```
tronn predict
```

#### Interpretation

First, with a trained model, get importance scores
```
tronn extractimportances
```

Then make motifs using importances
```
tronn makemotifs
```

Bag the motifs by representation in the sequences
```
tronn bagmotifs
```

Then do in silico mutagenesis to get dependencies between motifs
```
tronn ism
```