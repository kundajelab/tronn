# description code for working with variants

import os
import gzip
import logging


# basically generate two fasta files, and then
# run a version of setup_h5_dataset that takes in two fasta files
# to generate two one hot encoded sets


def generate_new_fasta(vcf_file, fasta_file, out_fasta_file, ref=True):
    """given a variant file and choice of ref or alt, adjust the fasta at those
    positions
    """
    logging.info("WARNING: when processing the VCF, grabs the first base pair given")
    # set up the tmp snp file
    if ref:
        tmp_snp_file = "{}/{}.ref.snp".format(
            os.path.dirname(out_fasta_file),
            os.path.basename(vcf_file).split(".gz")[0])
    else:
        tmp_snp_file = "{}/{}.alt.snp".format(
            os.path.dirname(out_fasta_file),
            os.path.basename(vcf_file).split(".gz")[0])
    with open(tmp_snp_file, "w") as out:
        with open(vcf_file, "r") as fp:
            for line in fp:
                if line.startswith("#"):
                    continue
                fields = line.strip().split()
                chrom, pos, snp_id = fields[0], int(fields[1]), fields[2]
                if ref:
                    basepair = fields[3][0]
                else:
                    basepair = fields[4][0]
                out.write("chr{}\t{}\t{}\t{}\n".format(chrom, pos, snp_id, basepair))
                
    # adjust the fasta
    mutate = "seqtk mutfa {} {} > {}".format(fasta_file, tmp_snp_file, out_fasta_file)
    print mutate
    os.system(mutate)
    
    return out_fasta_file


def generate_bed_file_from_variant_file(vcf_file, out_bed_file, bin_size):
    """given a variant file and params for dataset generation, create a bed file
    that will correctly tile the snp region when dataset is generated
    """
    with gzip.open(out_bed_file, "w") as out:
        with open(vcf_file, "r") as fp:
            for line in fp:
                if line.startswith("#"):
                    continue
                fields = line.strip().split()
                chrom, snp_pos, snp_id = fields[0], int(fields[1]), fields[2]
                start = snp_pos - bin_size
                stop = snp_pos
                metadata = "snp_id={};snp_pos={}:{}".format(snp_id, chrom, snp_pos)
                out.write("{}\t{}\t{}\t{}\n".format(chrom, start, stop, metadata))
                
    return None
