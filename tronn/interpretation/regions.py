"""Contains code to handle regions and merging regions 
more cleanly
"""

import numpy as np


def split_region(region):
    """Split region into a list of [chr, start, stop]
    """
    chrom = region.split(":")[0]
    start = int(region.split(":")[1].split("-")[0])
    stop = int(region.split(":")[1].split("-")[1].split("(")[0])
    return [chrom, start, stop]


class RegionTracker(object):
    """Builds a region tracker"""

    def __init__(self):
        self.chrom = None
        self.start = -1
        self.stop = -1

        
    def check_downstream_overlap(self, region):
        """ Check if region overlaps downstream of current region
        """
        chrom, start, stop = split_region(region)
        is_overlapping = (chrom == self.chrom) and (start < self.stop) and (stop > self.stop)

        return is_overlapping


    def merge(self, region):
        """Redefine in child class
        """
        assert self.check_downstream_overlap(region)
        
        chrom, start, stop = split_region(region)
        self.stop = stop
        
        return

    
    def reset(self, region=None):
        """Reset with given region or no region
        """
        if region is not None:
            chrom, start, stop = split_region(region)
            self.chrom = chrom
            self.start = start
            self.stop = stop
        else:
            self.chrom = None
            self.start = -1
            self.stop = -1

        return

    
    def get_region(self):
        """Get out region
        """
        return "{0}:{1}-{2}".format(self.chrom, self.start, self.stop)

    

class RegionImportanceTracker(RegionTracker):
    """Specifically tracks regions with importance scores"""

    def __init__(self, importances, labels, predictions):
        """Initialize

        Args:
          importances: importances dictionary of tensors
          labels: label tensor
          predictions: prediction tensor
        """
        super(RegionImportanceTracker, self).__init__()
        
        self.importances = {}
        for importance_key in importances.keys():
            self.importances[importance_key] = np.zeros((1, 1))
        self.labels = np.zeros((labels.get_shape()[1],))
        self.predictions = np.zeros((predictions.get_shape()[1],))
            
        return

    def merge(self, region, importances, labels, predictions):
        """Merge information

        Args:
          region: region id chr:start-stop
          importances: dictionary of importance scores
          labels: 1D label vector
          predictions: 1D prediction vector

        """
        assert super(RegionImportanceTracker, self).check_downstream_overlap(region)

        chrom, start, stop = split_region(region)
        offset = start - self.start

        # merge importances
        # concat zeros to extend sequence array and add data on top
        for importance_key in importances.keys():
            zero_padding = np.zeros((4, stop - self.stop))
            self.importances[importance_key] = np.concatenate(
                (self.importances[importance_key], zero_padding),
                axis=1)
            
            self.importances[importance_key][:, offset:] += importances[importance_key]
            
        # merge labels and predictions
        self.labels += labels
        self.predictions += predictions

        # and update region
        super(RegionImportanceTracker, self).merge(region)
        
        return


    def reset(self, region=None, importances=None, labels=None, predictions=None):
        """Reset or set up a new region
        """
        if importances is not None:
            self.importances = importances
        if labels is not None:
            self.labels = labels
        if predictions is not None:
            self.predictions = predictions
            
        # update region
        super(RegionImportanceTracker, self).reset(region)

        return

    
    def get_region(self):
        """Return region info
        """
        region = super(RegionImportanceTracker, self).get_region()
        return region, self.importances, self.labels, self.predictions
