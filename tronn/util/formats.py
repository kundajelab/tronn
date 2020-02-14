"""description: tools for handling json formatting
"""

import os
import json
import gzip

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """correctly handle numpy types"""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def write_to_json(data, out_file):
    """write to json nicely
    """
    assert out_file.endswith(".json")
    with open(out_file, "w") as fp:
        json.dump(
            data,
            fp,
            sort_keys=True,
            indent=4,
            cls=NumpyEncoder)


def array_to_bed(data, bed_file, interval_key="active", name_key="region", merge=True):
    """take an array of metadata and extract out 
    desired bed regions
    """
    assert bed_file.endswith(".gz")
    
    with gzip.open(bed_file, "w") as out:
        for region_metadata in data:
            interval_types = region_metadata.split(";")
            interval_types = dict([
                interval_type.split("=")[0:2]
                for interval_type in interval_types])
            interval_string = interval_types[interval_key]
            if name_key == "all":
                region_name = str(region_metadata)
            else:
                region_name = interval_types[name_key]
            
            chrom = interval_string.split(":")[0]
            start = interval_string.split(":")[1].split("-")[0]
            stop = interval_string.split("-")[1]
            out.write("{}\t{}\t{}\t{}\t1000\t.\n".format(
                chrom, start, stop, region_name))

    if merge:
        tmp_bed_file = "{}.tmp.bed.gz".format(bed_file.split(".bed")[0])
        os.system("mv {} {}".format(bed_file, tmp_bed_file))
        os.system((
            "zcat {} | "
            "sort -k1,1 -k2,2n | "
            "bedtools merge -i stdin | "
            "gzip -c > {}").format(
            tmp_bed_file, bed_file))
        os.system("rm {}".format(tmp_bed_file))

    return None
