# TRONN
Transcriptional Regulation (Optimized) Neural Nets (TRONN) - neural network tools for building integrative gene regulation models

---
### Installation

Easiest way is to first install anaconda, and set up an environment.

```
conda create -n tronn python=2 anaconda
```

Within the environment, install tensorflow according to tensorflow instructions for installing in Anaconda
(currently on tensorflow 1.7)

```
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp27-none-linux_x86_64.whl
```

Also may require seqtk, bedtools/2.26.0+, ucsc_tools

Then, install tronn.

```
python setup.py develop
```

Finally, install Phenograph (from python2 compatible package)


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