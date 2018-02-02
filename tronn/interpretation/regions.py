"""Contains code to handle regions and merging regions 
more cleanly
"""

import numpy as np


def split_region(region):
    """Split region into a list of [chr, start, stop]
    """
    chrom = region.split(":")[0]
    start = int(region.split(":")[1].split("-")[0])
    stop = int(region.split(":")[1].split("-")[1].split("(")[0].rstrip('\0'))
    return [chrom, start, stop]


class RegionObject(object):

    def __init__(self, shape, merge_type):
        """Basically a region object must be some form of numpy array
        and adding to it
        """
        self.array = np.zeros(shape)
        self.merge_type = merge_type

        
    def merge(self, new_array, offset=0):
        """merge in array
        """
        # merge by offset
        if self.merge_type == "offset":
            # note that sequence len should be dim 1
            # deal with overhangs and clip
            left_clip_idx = abs(min(0, offset))
            right_clip_idx = new_array.shape[1] - abs(
                min(0, self.array.shape[1] - (new_array.shape[1] + offset)))
            new_array_clipped = new_array[:,left_clip_idx:right_clip_idx]

            if offset < 0:
                adjusted_offset = 0
            else:
                ajusted_offset = offset
                
            left_zero_padding = np.zeros(
                (4, adjusted_offset))
            right_zero_padding = np.zeros(
                (4, self.array.shape[1] - adjusted_offset - new_array_clipped.shape[1]))
            self.array += np.concatenate(
                (left_zero_padding, new_array_clipped, right_zero_padding),
                axis=1)
        # merge by sum
        elif self.merge_type == "sum":
            self.array += new_array
        # merge by max
        elif self.merge_type == "max":
            self.array = np.maximum(self.array, new_array)

        
    def get_array(self):
        """return array
        """
        return self.array


class RegionTracker(object):
    """Builds a flexible region tracker that updates a dictionary
    """

    def __init__(self, region, region_data):
        self.region = region
        self.chrom, self.start, self.stop = split_region(region)
        self.region_data = {}
        for key in region_data.keys():

            if region_data[key][1] == "offset":
                self.region_data[key] = RegionObject(
                    (4, self.stop - self.start),
                    "offset")
            else:
                self.region_data[key] = RegionObject(
                    region_data[key][0].shape,
                    region_data[key][1])

                
    def is_same_region(self, region):
        return self.region == region

    
    def merge(self, region, region_data):
        chrom, start, stop = split_region(region)
        offset = start - self.start
        for key in self.region_data.keys():
            self.region_data[key].merge(
                region_data[key][0], offset=offset)

            
    def get_region(self):
        """Return a region element
        """
        region_data = {}
        for key in self.region_data.keys():
            region_data[key] = self.region_data[key].get_array()
        return self.region, region_data


class ExampleGenerator(object):

    def __init__(self, sess, tensor_dict, batch_size, reconstruct_regions=False):
        """Set up params for example generation
        """
        self.sess = sess
        self.tensor_dict = tensor_dict
        self.batch_size = batch_size
        self.batch_pointer = 0
        self.reconstruct_regions = reconstruct_regions

        if self.reconstruct_regions:
            self.name_idx = -1
        else:
            self.name_idx = 0
        
        # initialize region tracker with first region in batch
        self.batch_region_arrays = self.sess.run(self.tensor_dict)
        region, region_arrays = self.build_example_dict_arrays()
        self.region_tracker = RegionTracker(region, region_arrays)
        self.region_tracker.merge(region, region_arrays)

        
    def build_example_dict_arrays(self):
        """Build dictionary of arrays for 1 example
        """
        # extract data and construct into region array dict
        region_arrays = {}
	#print(self.batch_region_arrays.keys(), "--------------")
        for key in self.batch_region_arrays.keys():
            if key == "feature_metadata":
                region = self.batch_region_arrays[key][self.batch_pointer,0]
		#print(region)
                #region_name = region.split(";")[self.name_idx].split("=")[1].split("::")[0]
		region_name = region.split("::")[1]
		#print(region_name)
            elif "importance" in key:
                region_arrays[key] = (
                    np.squeeze(
                        self.batch_region_arrays[key][self.batch_pointer,:,:,:]).transpose(1,0),
                    "offset")
            else:
                region_arrays[key] = (
                    self.batch_region_arrays[key][self.batch_pointer,:],
                    "max")
        self.batch_pointer += 1
        return region_name, region_arrays
    
                
    def run(self):

        if self.reconstruct_regions:
            # go until you can yield an example
            while True:
                
                # draw a new batch if necessary
                if self.batch_pointer == self.batch_size - 1:
                    self.batch_pointer = 0
                    self.batch_region_arrays = self.sess.run(self.tensor_dict)
                    
                # go through an example
                region, region_arrays = self.build_example_dict_arrays()

                # merge if same
                if self.region_tracker.is_same_region(region):
                    self.region_tracker.merge(region, region_arrays)
                else:
                    out_region, out_region_arrays = self.region_tracker.get_region()
                    # and reset with new info
                    self.region_tracker = RegionTracker(region, region_arrays)
                    self.region_tracker.merge(region, region_arrays)
                    break
                
            return out_region, out_region_arrays
        
        else:
            # draw a new batch if necessary
            if self.batch_pointer == self.batch_size - 1:
                self.batch_pointer = 0
                self.batch_region_arrays = self.sess.run(self.tensor_dict)
            
            # go through an example
            region, region_arrays = self.build_example_dict_arrays()

            # push out old example and load in next one
            out_region, out_region_arrays = self.region_tracker.get_region()
            self.region_tracker = RegionTracker(region, region_arrays)
            self.region_tracker.merge(region, region_arrays)

            return out_region, out_region_arrays
            
    
def region_generator_old(sess, tensor_dict, stop_idx, batch_size):
    """make a generator
    """
    region_idx = 0
    
    while region_idx < stop_idx:
        
        # run session to get batched results
        # tensor dict key: tensor name, val: array
        region_batch_arrays = sess.run([tensor_dict])
        
        # go through each example in batch
        for i in range(batch_size):

            # extract data and construct into region array dict
            region_arrays = {}
            for key in region_batch_arrays.keys():
                if key == "feature_metadata":
                    region = region_batch_arrays[keys][i,0]
                elif "importance" in key:
                    region_arrays[key] = (
                        np.squeeze(
                            region_batch_arrrays[keys][i,:,:,:]).transpose(1,0),
                        "offset")
                else:
                    region_arrays[key] = (
                        region_batch_arrays[keys][i,:],
                        "max")
                    
            # set up region tracker
            if region_idx == 0:
                region_tracker = RegionTracker(region, region_arrays)

            # merge if same
            if region_tracker.is_same_region(region):
                region_tracker.merge(region, region_arrays)
            else:
                yield region_tracker.get_region()
                
                # check stop criteria
                region_idx += 1
                if region_idx == stop_idx:
                    break

                # and reset with new info
                region_tracker = RegionTracker(region, region_arrays)
                region_tracker.merge(region, region_arrays)

        
        return

            


class RegionTrackerOld(object):
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

    

class RegionImportanceTracker(RegionTrackerOld):
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
