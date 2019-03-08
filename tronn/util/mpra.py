
"""
description: code to easily build MPRA sequences
"""

import os
import logging

import numpy as np
import pandas as pd

from numpy.random import RandomState
from tronn.util.pwms import MotifSetManager
from tronn.util.utils import DataKeys


class MPRA_PARAMS(object):
    """all mpra design params (Khavari Lab)
    """
    LEN_FILLER = 0 # UPDATE
    LEN_BARCODE = 20
    MAX_OLIGO_LENGTH = 225
    FWD_PCR_PRIMER = 'ACTGGCCGCTTCACTG'
    REV_PCR_PRIMER = 'AGATCGGAAGAGCGTCG'
    RS_ECORI = 'GAATTC' # 5'-3'
    RS_BAMHI = 'GGATCC'
    RS_XHOI = 'CTCGAG'
    #RS_XBAI = 'TCTAGA'
    RS_NHEI = "GCTAGC"
    #RS_NCOI = 'CCATGG'
    #RS_XBAI_dam1 = 'GATCTAGA'
    #RS_XBAI_dam2 = 'TCTAGATC'
    LETTERS= ['A', 'C', 'G', 'T']
    MAX_FRAG_LEN = MAX_OLIGO_LENGTH - (
        len(FWD_PCR_PRIMER) + len(RS_XHOI) + LEN_FILLER + len(RS_NHEI) + LEN_BARCODE + len(REV_PCR_PRIMER))
    assert MAX_FRAG_LEN == 160
    

def is_rs_clean(sequence):
    """check for cut sites, should have NONE
    """
    if sequence.count(MPRA_PARAMS.RS_ECORI) != 0: return False
    if sequence.count(MPRA_PARAMS.RS_BAMHI) != 0: return False
    if sequence.count(MPRA_PARAMS.RS_XHOI) != 0: return False
    if sequence.count(MPRA_PARAMS.RS_NHEI) != 0: return False
    
    return True


def is_fragment_compatible(sequence):
    """check fragment for compatibility
    """
    # check for NO cut sites
    if not is_rs_clean(sequence): return False

    # check again when attaching FWD primer
    if not is_rs_clean(MPRA_PARAMS.FWD_PCR_PRIMER + sequence): return False

    # check after attaching XHOI
    fragment_extended = sequence + MPRA_PARAMS.RS_XHOI
    if fragment_extended.count(MPRA_PARAMS.RS_ECORI) != 0: return False
    if fragment_extended.count(MPRA_PARAMS.RS_BAMHI) != 0: return False
    if fragment_extended.count(MPRA_PARAMS.RS_XHOI) != 1: return False
    if fragment_extended.count(MPRA_PARAMS.RS_NHEI) != 0: return False
    
    # check for N
    if sequence.count("N") != 0: return False
    
    return True
    

def is_barcode_compatible(barcode):
    """check barcode for compatibility
    """
    # check when attaching REV primer
    if not is_rs_clean(barcode + MPRA_PARAMS.REV_PCR_PRIMER): return False

    # check when attaching NHEI site
    barcode_extended = MPRA_PARAMS.RS_NHEI + barcode
    if barcode_extended.count(MPRA_PARAMS.RS_ECORI) != 0: return False
    if barcode_extended.count(MPRA_PARAMS.RS_BAMHI) != 0: return False
    if barcode_extended.count(MPRA_PARAMS.RS_XHOI) != 0: return False
    if barcode_extended.count(MPRA_PARAMS.RS_NHEI) != 1: return False
    
    return True


def is_filler_compatible(filler):
    """check if filler is compatible with library design
    """
    # first check cut sites
    if not is_rs_clean(filler): return False
    
    # check overlaps
    filler_extended = MPRA_PARAMS.RS_XHOI + filler + MPRA_PARAMS.RS_NHEI
    if filler_extended.count(MPRA_PARAMS.RS_ECORI) != 0: return False
    if filler_extended.count(MPRA_PARAMS.RS_BAMHI) != 0: return False
    if filler_extended.count(MPRA_PARAMS.RS_XHOI) != 1: return False
    if filler_extended.count(MPRA_PARAMS.RS_NHEI) != 1: return False
        
    return True


def generate_compatible_filler(rand_seed, length):
    """generate filler sequence and check methylation
    """
    while True:
        # generate random sequence
        rand_state = RandomState(rand_seed)
        random_seq = rand_state.choice(
            MPRA_PARAMS.LETTERS,
            size=length)
        random_seq = "".join(random_seq)
        
        # if passes checks then break
        if is_filler_compatible(random_seq):
            break

        # otherwise change the seed and keep going
        rand_seed += 1

    # and move up one more (for next filler)
    rand_seed += 1
        
    return random_seq, rand_seed


def is_sequence_mpra_ready(sequence):
    """given a sequence, check that it's compatible 
    with current library generation strategy
    """
    # cut sites (insert to backbone) should NOT exist
    if sequence.count(MPRA_PARAMS.RS_ECORI) != 0:
        logging.info("ecori")
        logging.info(sequence)
        return False
    if sequence.count(MPRA_PARAMS.RS_BAMHI) != 0:
        logging.info("bamhi")
        logging.info(sequence)
        return False

    # cut sites (insert promoter and luc) SHOULD exist ONCE
    if sequence.count(MPRA_PARAMS.RS_XHOI) != 1:
        logging.info("xhoi")
        logging.info(sequence)
        return False
    if sequence.count(MPRA_PARAMS.RS_NHEI) != 1:
        logging.info("nhei")
        logging.info(sequence)
        return False

    # check length
    if len(sequence) > MPRA_PARAMS.MAX_OLIGO_LENGTH:
        logging.info("length")
        logging.info(sequence)
        return False

    # check for N
    if sequence.count("N") != 0:
        logging.info("found N")
        logging.info(sequence)
        return False

    return True


def trim_sequence_for_mpra(sequence, edge_indices):
    """trim sequence to fit in library
    """
    # get min possible len that contains motifs and check less than max possible len
    motif_padding_len = 10
    frag_minimal_len = edge_indices[1] - edge_indices[0] + (motif_padding_len * 2)
    assert frag_minimal_len < MPRA_PARAMS.MAX_FRAG_LEN

    # add padding and adjust start/stop
    extra_padding = int((MPRA_PARAMS.MAX_FRAG_LEN - frag_minimal_len) / 2.)
    start_position = edge_indices[0] - motif_padding_len - extra_padding
    end_position = start_position + MPRA_PARAMS.MAX_FRAG_LEN
    
    if start_position < 0:
        # if closer to start, trim off the back only
        sequence = sequence[:MPRA_PARAMS.MAX_FRAG_LEN]
    elif end_position > len(sequence):
        # if closer to back, trim off front only
        sequence = sequence[-MPRA_PARAMS.MAX_FRAG_LEN:]
    else:
        # in the middle, trim equally
        sequence = sequence[start_position:]
        sequence = sequence[:MPRA_PARAMS.MAX_FRAG_LEN]
        
    # check
    assert len(sequence) == MPRA_PARAMS.MAX_FRAG_LEN, edge_indices
    assert is_fragment_compatible(sequence)
    
    return sequence


def build_mpra_sequence(sequence, barcode, rand_seed, log):
    """attach on relevant sequence info
    """
    assert is_fragment_compatible(sequence)
    assert is_barcode_compatible(barcode)
    
    # attach FWD primer to front
    sequence = MPRA_PARAMS.FWD_PCR_PRIMER + sequence
    # attach XHOI
    sequence += MPRA_PARAMS.RS_XHOI
    # attach filler (random 20)
    filler, rand_seed = generate_compatible_filler(rand_seed, MPRA_PARAMS.LEN_FILLER)
    sequence += filler
    # attach NHEI
    sequence += MPRA_PARAMS.RS_NHEI
    # attach barcode
    sequence += barcode
    # attach reverse primer
    sequence += MPRA_PARAMS.REV_PCR_PRIMER

    # sanity check
    assert is_sequence_mpra_ready(sequence), log
    
    return sequence, rand_seed


def seq_list_compatible(seq_list, left_clip=420, right_clip=580):
    """helper function when checking sequences from h5 file
    """
    for seq in seq_list:
        if not is_fragment_compatible(seq[420:580]):
            return False
        if len(seq[420:580]) == 0:
            return False
        
    return True


def barcode_generator(barcodes):
    """generator to push out barcodes
    """
    barcode_idx = 0
    while barcode_idx < len(barcodes):
        barcode = barcodes[barcode_idx]
        if is_barcode_compatible(barcode):
            yield barcode
        barcode_idx += 1


def is_barcode_gc_compatible(barcode, min_gc=0.20, max_gc=0.80):
    """check GC content
    """
    gc_count = barcode.count("G") + barcode.count("C")
    gc_fract = gc_count / float(len(barcode))

    if gc_fract < min_gc:
        return False
    if gc_fract > max_gc:
        return False
    
    return True
        
        
def _is_barcode_compatible(barcode, prev_barcodes, min_diff=3, min_gc=0.20, max_gc=0.80):
    """check barcode 
    """
    # no restriction sites
    if not is_barcode_compatible(barcode):
        return False
    
    # GC content acceptable
    if not is_barcode_gc_compatible(barcode, min_gc=min_gc, max_gc=max_gc):
        return False
    
    # does not match previous barcodes
    for prev_barcode in prev_barcodes:
        if sum(bp1 != bp2 for bp1, bp2 in zip(barcode, prev_barcode)) < min_diff:
            return False
    
    return True
        

def barcode_generator_v2(total_barcodes=400000, length=20, rand_seed=24):
    """generate barcodes on the fly
    """
    # set up reproducible rand state
    rand_state = RandomState(rand_seed)
    rand_seeds = rand_state.choice(10000000, size=total_barcodes)
    rand_seed_idx = 0

    # make barcodes
    barcode_idx = 0
    prev_barcodes = []
    while barcode_idx < total_barcodes:
        # make reproducible random state
        rand_state = RandomState(rand_seeds[rand_seed_idx])

        # make barcode
        barcode = rand_state.choice(
            MPRA_PARAMS.LETTERS,
            size=length)
        barcode = "".join(barcode)
        
        # check
        if _is_barcode_compatible(barcode, prev_barcodes):
            prev_barcodes.append(barcode)
            barcode_idx += 1
            yield barcode 

        # always increase rand seed
        rand_seed_idx += 1

        
        
# MPRA tools for controls

def build_metadata(sequence, prefix, idx, keys, metadata_type="synergy"):
    """when adding non NN sequences, make metadata to match up sequences
    """
    # generate for all keys
    metadata = {}
    for key in keys:
        metadata[key] = 0

    # adjust some manually
    metadata[DataKeys.SEQ_METADATA] = "features={}".format(prefix)
    metadata[DataKeys.GC_CONTENT] = float(sequence.count("G") + sequence.count("C")) / len(sequence)
    metadata["sequence.nn"] = sequence
    metadata["example_id"] = "{}-{}".format(prefix, idx)
    metadata["edge_indices"] = "60.,100."

    # adjust for specific library types
    if metadata_type == "synergy":
        metadata["example_combo_id"] = "{}-{}.combo-0".format(prefix, idx)
        metadata["motifs"] = prefix

    # make into df
    metadata = pd.DataFrame(metadata, index=[0])
        
    return metadata


def build_shuffles(rand_seed=0, metadata_keys=[], metadata_type="synergy", num_shuffles=50):
    """add in shuffles
    """
    logging.info("adding {} shuffles".format(num_shuffles))
    
    # set up reproducible rand state
    rand_state = RandomState(rand_seed)
    rand_seeds = rand_state.choice(1000000, size=num_shuffles)
    
    for shuffle_idx in range(num_shuffles):
        # build random sequence and metadata
        shuffle_sequence, _ = generate_compatible_filler(
            rand_seeds[shuffle_idx], MPRA_PARAMS.MAX_FRAG_LEN)
        shuffle_example = build_metadata(
            shuffle_sequence,
            "shuffle",
            shuffle_idx,
            metadata_keys,
            metadata_type=metadata_type)

        # attach
        try:
            if shuffle_idx == 0:
                all_shuffles = shuffle_example
            else:
                all_shuffles = pd.concat([all_shuffles, shuffle_example], sort=True)
        except:
            import ipdb
            ipdb.set_trace()
            
    return all_shuffles


def _build_pwm_embedded_sequence(pwm, length, rand_seed, num_pwms=3):
    """make random background, embed pwm into it
    """
    # generate random sequence
    random_sequence, _ = generate_compatible_filler(
        rand_seed, length)

    # embed equally spaced
    stride = length / (num_pwms + 1.)
    indices = stride * np.arange(1, num_pwms+1)
    for pos_idx in indices:
        sampled_pwm = pwm.get_sampled_string(rand_seed)
        len_pwm = len(sampled_pwm)
        random_sequence = "".join([
            random_sequence[:int(pos_idx)],
            sampled_pwm,
            random_sequence[int(pos_idx+len_pwm):]])
        
    return random_sequence


def build_pwm_controls(pwm_file, rand_seed=1, metadata_keys=[], metadata_type="synergy"):
    """add sampled pwms
    """
    pwms = MotifSetManager.read_pwm_file(pwm_file)
    logging.info("adding {} pwm embedded sequences".format(len(pwms)))

    # set up reproducible rand state
    rand_state = RandomState(rand_seed)
    rand_seeds = rand_state.choice(1000000, size=len(pwms))
    
    for pwm_idx in range(len(pwms)):
        pwm = pwms[pwm_idx]
        
        # generate a triplicate pattern
        pwm_embedded_sequence = _build_pwm_embedded_sequence(
            pwm, MPRA_PARAMS.MAX_FRAG_LEN, rand_seeds[pwm_idx])

        pwm_example = build_metadata(
            pwm_embedded_sequence,
            "pwm",
            pwm_idx,
            metadata_keys,
            metadata_type=metadata_type)

        # attach
        if pwm_idx == 0:
            all_pwm_controls = pwm_example
        else:
            all_pwm_controls = pd.concat([all_pwm_controls, pwm_example], sort=True)
    
    return all_pwm_controls


def build_promoter_controls(
        tss_file,
        fasta,
        rand_seed=2,
        metadata_keys=[],
        metadata_type="synergy",
        prefix="promoter"):
    """add positive control set of regions
    """
    logging.info("adding {} regions".format(prefix))
    
    # getfasta
    tmp_fasta = "promoters.fasta"
    getfasta = "bedtools getfasta -tab -fi {} -bed {} > {}".format(fasta, tss_file, tmp_fasta)
    os.system(getfasta)
    
    # read in fasta file
    sequences = pd.read_table(tmp_fasta, header=None)
    
    # for each sequence, get a sample position (just center it for now)
    prom_total = 0
    for prom_idx in range(sequences.shape[0]):
        prom_sequence = sequences.iloc[prom_idx,1].upper()
        prom_sequence = prom_sequence.replace("N", "A")
        start_idx = (len(prom_sequence) - MPRA_PARAMS.MAX_FRAG_LEN) / 2
        prom_sequence = prom_sequence[start_idx:(start_idx+MPRA_PARAMS.MAX_FRAG_LEN)]
        if not is_fragment_compatible(prom_sequence):
            continue
        
        # add metadata
        prom_example = build_metadata(
            prom_sequence,
            prefix,
            prom_idx,
            metadata_keys,
            metadata_type=metadata_type)
        prom_example["example_metadata"] = "features={}".format(sequences.iloc[prom_idx,0])
        
        # attach
        if prom_idx == 0:
            all_prom_controls = prom_example
        else:
            all_prom_controls = pd.concat([all_prom_controls, prom_example], sort=True)
        prom_total += 1

    os.system("rm promoters.fasta")
    logging.info("added {} promoters".format(prom_total))
    
    return all_prom_controls


def build_negative_controls(
        bed_file,
        fasta,
        num_negatives=50,
        rand_seed=3,
        metadata_keys=[],
        metadata_type="synergy"):
    """sample randomly from coordinates
    """
    logging.info("adding negative regions")

    # randomly select subset of regions
    tmp_bed_file = "negatives.subset.bed.gz"
    subset = "zcat {0} | shuf --random-source {0} -n {1} | gzip -c > {2}".format(
        bed_file, num_negatives, tmp_bed_file)
    os.system(subset)
    
    # reuse promoter code
    negative_genomic_controls = build_promoter_controls(
        tmp_bed_file,
        fasta,
        rand_seed=rand_seed,
        metadata_keys=metadata_keys,
        metadata_type=metadata_type,
        prefix="genomic_negative")

    # cleanup
    os.system("rm {}".format(tmp_bed_file))
    
    return negative_genomic_controls


def build_variants(
        variants_work_dir,
        ref_fasta,
        alt_fasta,
        metadata_keys=[],
        metadata_type="synergy"):
    """given a bed file of variants and ref/alt fasta files,
    get variant sequences
    """
    # make a master variants file
    master_variants_bed_file = "{}/variants.master.bed.gz".format(variants_work_dir)
    os.system((
        "zcat {}/variants.grammar*.bed.gz | "
        "sort -k1,1 -k2,2n | "
        "bedtools merge -i stdin | "
        "gzip -c > {}").format(
            variants_work_dir,
            master_variants_bed_file))
    
    # reuse promoter code
    snp_ref = build_promoter_controls(
        master_variants_bed_file,
        ref_fasta,
        rand_seed=4,
        metadata_keys=metadata_keys,
        metadata_type=metadata_type,
        prefix="ref_snp")
    snp_alt = build_promoter_controls(
        master_variants_bed_file,
        alt_fasta,
        rand_seed=5,
        metadata_keys=metadata_keys,
        metadata_type=metadata_type,
        prefix="alt_snp")
    
    # interleave
    variants = pd.concat([snp_ref, snp_alt]).sort_values("example_metadata")
    
    return variants


def build_controls(
        metadata_keys,
        metadata_type,
        pwm_file=None,
        promoter_regions=None,
        negative_regions=None,
        variant_work_dir=None,
        ref_fasta=None,
        alt_fasta=None,
        fasta=None):
    """build all controls
    """
    # make shuffles
    controls_df = build_shuffles(
        metadata_keys=metadata_keys,
        metadata_type=metadata_type,
        num_shuffles=50)

    # make pwm controls
    if pwm_file is not None:
        pwm_controls_df = build_pwm_controls(
            pwm_file,
            metadata_keys=metadata_keys,
            metadata_type=metadata_type)
        controls_df = pd.concat(
            [controls_df, pwm_controls_df], sort=True)
        
    # make promoter regions
    if promoter_regions is not None:
        promoter_controls_df = build_promoter_controls(
            promoter_regions,
            fasta,
            metadata_keys=metadata_keys,
            metadata_type=metadata_type)
        controls_df = pd.concat(
            [controls_df, promoter_controls_df], sort=True)
        
    # make negative controls
    if negative_regions is not None:
        genomic_negatives_df = build_negative_controls(
            negative_regions,
            fasta,
            num_negatives=50,
            metadata_keys=metadata_keys,
            metadata_type=metadata_type)
        controls_df = pd.concat(
            [controls_df, genomic_negatives_df], sort=True)

    # add variants
    if variant_work_dir is not None:
        variants_df = build_variants(
            variant_work_dir,
            ref_fasta,
            alt_fasta,
            metadata_keys=metadata_keys,
            metadata_type=metadata_type)
        controls_df = pd.concat(
            [controls_df, variants_df], sort=True)
        
        
    return controls_df
