## tronn
Transcriptional Regulation optimized Neural Nets

tronn is a toolkit for building neural network models around genomic data types and integrative analyses of the model.

---
### Installation

We recommend first installing [anaconda](https://docs.anaconda.com/anaconda/install/) and using a conda environment (e.g. `conda create -n tronn python=2 anaconda`) to manage packages and ensure functionality. A docker [container](https://hub.docker.com/r/dskim89/tronn) is also available (tensorflow 1.10.1).

Install [tensorflow](https://www.tensorflow.org/install/pip) with `pip install`. tronn currently requires tensorflow 1.9 or 1.10.

```
# example install command, adjust tensorflow version as needed
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.0-cp27-none-linux_x86_64.whl
```

Make sure that you have bedtools v2.26.0+

If you are running preprocessing, make sure to additionally have ucsc_tools.

For variants, install seqtk.

Then, install tronn.

```
# cd into tronn repo
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

First, with a trained model, scan for motifs
```
tronn scanmotifs
```

Then call differential peaks with foreground/background using scripts in folder
```
call_differential_motifs.py
```

Take these and run dmim (deep motif importance maps) to determine synergies and effects by ISM
```
tronn dmim
```

Then build grammars
```
tronn buildgrammars
```