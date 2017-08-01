"""Contains run function for ISM
"""

import os
import glob
import logging

import numpy as np
import pandas as pd

from tronn.preprocess import generate_nn_dataset

from tronn.datalayer import load_data_from_filename_list, get_total_num_examples
from tronn.graphs import TronnGraph
from tronn.architectures import ism_for_grammar_dependencies
from tronn.architectures import models

from tronn.interpretation.motifs import PWM
from tronn.interpretation.motifs import get_encode_pwms
from tronn.interpretation.ism import run_ism_for_motif_pairwise_dependency
from tronn.interpretation.grammars import read_motifset_file



def grammar_to_file(
        out_file,
        indiv_scores_array,
        pairwise_scores_array,
        grammar_names):
    """Write out grammar to file
    """
    with open(out_file, 'w') as fp:
        header='# Grammar model: Linear w pairwise interactions\n\n'
        fp.write(header)
        indiv_motif_coeff_header = 'Non_interacting_coefficients\n'
        fp.write(indiv_motif_coeff_header)
        indiv_df = pd.DataFrame(
            data=indiv_scores_array, columns=grammar_names)
        indiv_df.to_csv(fp, sep='\t')
        fp.write("\n")
        synergy_motif_coeff_header = 'Pairwise_interacting_coefficients\n'
        fp.write(synergy_motif_coeff_header)
        synergy_df = pd.DataFrame(
            data=pairwise_scores_array, index=grammar_names, columns=grammar_names)
        synergy_df.to_csv(fp, sep='\t')

    return None


def ism_pipeline(
        motif_sets,
        pwms,
        bed_dir,
        model,
        model_dir,
        tasks,
        out_dir,
        prefix,
        annotations,
        batch_size):
    """Pipeline for running in silico mutagenesis
    to get pairwise motif dependencies

    Args:
      motif_sets: list of motif lists. Each elem corresponds to a 
        list of motif names that are in the pwms dict.
      pwms: dictionary of PWM class objects, key is pwm name.
      bed_dir: directory of community BED files
      model: model type
      model_dir: directory with checkpoint
      tasks: tasks (needed for TronnGraph instance)
      out_dir: output directory
      prefix: prefix for output files
      annotations: dictionary of annotation files
      batch_size: batch_size

    Returns:
      None
    """
    # generate datasets to run ISM - make BED files then run preprocess
    community_bed_sets = glob.glob('{}/*.bed.gz'.format(bed_dir))
    for community in range(len(community_bed_sets)):

        # set up directory for data and output grammar file
        community_bed = '{0}/{1}.community_{2}.bed.gz'.format(bed_dir, prefix, community)
        print community_bed
        community_dir = "{0}/{1}.community_{2}".format(out_dir, prefix, community)
        os.system("mkdir -p {}".format(community_dir))
        community_prefix = "{0}/{1}.community_{2}".format(community_dir, prefix, community)

        # preprocess data
        community_data_dir = '{}/data'.format(community_dir)
        if not os.path.isdir(community_data_dir):
            generate_nn_dataset(community_bed,
                                annotations['univ_dhs'],
                                annotations['ref_fasta'],
                                [community_bed],
                                community_data_dir,
                                '{0}.community_{1}'.format(prefix, community),
                                parallel=12,
                                neg_region_num=0)
        data_files = glob.glob('{}/h5/*.h5'.format(community_data_dir))

        # Set up motif list for community
        motif_list = motif_sets[int(community)]
        print motif_list
        num_motifs = len(motif_list)

        # for each pair of motifs, read in and run model
        synergies = np.zeros((num_motifs, num_motifs))
        indiv_motif_coeffs = np.zeros((num_motifs, num_motifs))
        for motif1_idx in range(num_motifs):
            for motif2_idx in range(num_motifs):
                if motif1_idx >= motif2_idx:
                    continue
                
                # here, run ISM tests
                print motif1_idx, motif2_idx
                pwm1 = pwms[motif_list[motif1_idx]]
                pwm2 = pwms[motif_list[motif2_idx]]
        
                # for each pair of motifs:
                model_params = {
                    "trained_net": model,
                    "pwm_a": pwm1,
                    "pwm_b": pwm2}
        
                # setup graph
                ism_graph = TronnGraph(
                    {"data": data_files},
                    tasks,
                    load_data_from_filename_list,
                    ism_for_grammar_dependencies,
                    model_params,
                    batch_size)

                # then run and get synergy, and average indiv scores to get indiv coeffs
                synergy_score, pwm1_score, pwm2_score = run_ism_for_motif_pairwise_dependency(ism_graph, model_dir, batch_size) # TODO(dk) change to evaluate more examples
                indiv_motif_coeffs[motif1_idx, motif2_idx] = pwm1_score
                indiv_motif_coeffs[motif2_idx, motif1_idx] = pwm2_score
                synergies[motif1_idx, motif2_idx] = synergy_score
                indiv_avg_scores = np.zeros((1, num_motifs))
                for i in range(num_motifs):
                    indiv_avg_scores[0,i] = np.mean(indiv_motif_coeffs[i,:])                

        # write all this out to files
        grammar_dependencies_file = "{0}/{1}.community_{2}.grammar.pairwise.txt".format(community_dir, prefix, community) # TODO fix this name
        grammar_to_file(grammar_dependencies_file, indiv_avg_scores, synergies, motif_list)

    return None


def run(args):
    """Run pipeline
    """
    logging.info("Running ISM...")    

    # read in various files
    motif_sets = read_motifset_file(args.motif_sets_file)
    pwms = get_encode_pwms(args.pwm_file, as_dict=True)

    # run ISM
    ism_pipeline(
        motif_sets,
        pwms,
        args.bed_dir,
        models[args.model['name']],
        args.model_dir,
        args.tasks,
        args.out_dir, 
        args.prefix,
        args.annotations,
        args.batch_size)

    return 
