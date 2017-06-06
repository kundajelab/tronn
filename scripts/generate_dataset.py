# set up and make datasets

import os
import sys
import glob
import preprocess

def main():

    SCRATCH_DIR = '/srv/scratch/shared/indra/dskim89/ggr/{}_tmp'.format(os.getcwd().split('/')[-1])
    os.system('mkdir -p {0} {0}/data'.format(SCRATCH_DIR))
    
    dnase_nn_prefix = 'roadcode.dnase.nn.idr'
    HONEYBADGER_DIR = '/mnt/data/epigenomeRoadmap/integrative/regulatoryRegions/ENCODERoadmap/reg2map/HoneyBadger2_release/'
    univ_dhs_file = '{}/DNase/p10/regions_all.bed.gz'.format(HONEYBADGER_DIR)
    ref_fasta = '/mnt/data/annotations/by_release/hg19.GRCh37/hg19.genome.fa'

    # Set up label file set
    ROADCODE_DIR = '/mnt/data/integrative/dnase'
    roadcode_peak_files = glob.glob('{}/*/*/*/idr/*/*/*IDR*filt.narrowPeak.gz'.format(ROADCODE_DIR))
    print len(roadcode_peak_files)

    CHIPSEQ_DIR = '/srv/scratch/shared/indra/dskim89/ggr/encode.chipseq.peaks'
    chipseq_peak_files = glob.glob('{}/*.bed.gz'.format(CHIPSEQ_DIR))
    print len(chipseq_peak_files)

    all_label_peak_files = sorted(roadcode_peak_files + chipseq_peak_files)

    # for testing purposes - REMOVE
    #all_label_peak_files = all_label_peak_files[0:100]
    

    # first generate master regions file
    master_regions_bed = '{0}/{1}.master.bed.gz'.format(SCRATCH_DIR, dnase_nn_prefix)
    if not os.path.isfile(master_regions_bed):
        tmp_master = '{0}/{1}.master.tmp.bed.gz'.format(SCRATCH_DIR, dnase_nn_prefix)
        for i in range(len(all_label_peak_files)):

            label_peak_file = all_label_peak_files[i]
            
            if i == 0:
                transfer = "cp {0} {1}".format(label_peak_file, master_regions_bed)
                print transfer
                os.system(transfer)
            else:
                merge_bed = ("zcat {0} {1} | "
                             "awk -F '\t' '{{ print $1\"\t\"$2\"\t\"$3 }}' | "
                             "sort -k1,1 -k2,2n | "
                             "bedtools merge -i - | "
                             "gzip -c > {2}").format(label_peak_file, master_regions_bed, tmp_master)
                print merge_bed
                os.system(merge_bed)
                transfer = "cp {0} {1}".format(tmp_master, master_regions_bed)
                print transfer
                os.system(transfer)
                
        os.system('rm {}'.format(tmp_master))

    #if not os.path.isdir('{}/data/h5'.format(SCRATCH_DIR)):
    preprocess_test.generate_nn_dataset(master_regions_bed,
                                        univ_dhs_file,
                                        ref_fasta,
                                        all_label_peak_files,
                                        '{}/data'.format(SCRATCH_DIR),
                                        dnase_nn_prefix,
                                        neg_region_num=0,
                                        reverse_complemented=False)
                                                                                                                 

    
    return None

main()
