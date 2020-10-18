# TRONN
Transcriptional Regulation (Optimized) Neural Nets (TRONN) - neural network tools for building integrative gene regulation models

Models and datasets can be found at the following Zenodo accessions:

- [Classification dataset for ENCODE-Roadmap DNase-seq peaks and Transcription Factor ChIP-seq peaks](https://doi.org/10.5281/zenodo.4059038)
- [Convolutional Neural Net (CNN) models for ENCODE-Roadmap DNase-seq peaks and Transcription Factor ChIP-seq peaks - Basset architecture](https://doi.org/10.5281/zenodo.4059060)
- [Machine learning datasets for epigenomic landscapes in epidermal differentiation](https://doi.org/10.5281/zenodo.4062509)
- [Convolutional Neural Net (CNN) models for epigenomic landscapes in epidermal differentiation - Basset architecture, classification and regression](https://doi.org/10.5281/zenodo.4062726)

---
### Installation

Easiest way is to first install anaconda, and set up an environment.

```
conda create -n tronn python=2 anaconda
```

Within the environment, install tensorflow according to tensorflow instructions for installing in Anaconda. Please use tensorflow 1.9 or 1.10.

Use the `pip install` way to install tensorflow, NOT the conda install. (use `which pip` to make sure it's the pip associated with your conda environment)

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
