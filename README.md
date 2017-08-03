# TRONN
Transcriptional Regulation (Optimized) Neural Nets (TRONN) - neural network tools for building integrative gene regulation models

---
### Installation

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