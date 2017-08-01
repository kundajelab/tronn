"""Contains run function for ISM
"""

import glob
import logging

from tronn.preprocess import generate_nn_dataset

from tronn.datalayer import load_data_from_filename_list, get_total_num_examples
from tronn.architectures import ism_for_grammar_dependencies
from tronn.architectures import models

from tronn.interpretation.motifs import PWM
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
        prefix):
    """Pipeline for running in silico mutagenesis
    to get pairwise motif dependencies
    """

    # generate datasets to run ISM - make BED files then run preprocess
    community_bed_sets = glob.glob('{}/*.bed.gz'.format(bed_dir))
    for community in range(len(community_bed_sets)):
        community_bed = '{0}/task_{1}.community_{2}.bed.gz'.format(test_dir, task_num, community)

        # preprocess data
        # TODO modify the annotations files
        with open(args.preprocess_annotations, 'r') as fp:
            annotation_files = json.load(fp)
            
        community_data_dir = '{}/data'.format(test_dir)
        if not os.path.isdir(community_data_dir):
            generate_nn_dataset(community_bed,
                                annotation_files['univ_dhs'],
                                annotation_files['ref_fasta'],
                                [community_bed],
                                community_data_dir,
                                'task_0.community_{}'.format(community),
                                parallel=12,
                                neg_region_num=0)
        data_files = glob.glob('{}/h5/*.h5'.format(community_data_dir))

        # now focus on individual grammar that belongs to that community.
        motif_set = motif_sets[int(community)]
        print motif_set
        num_motifs = len(motif_set)

        # for each pair of motifs, read in and run model
        synergies = np.zeros((num_motifs, num_motifs))
        indiv_motif_coeffs = np.zeros((num_motifs, num_motifs))
        for motif1_idx in range(num_motifs):
            for motif2_idx in range(num_motifs):
                if motif1_idx >= motif2_idx:
                    continue
                
                # here, run ISM tests
                print motif1_idx, motif2_idx
                pwm1 = pwm_dict[grammar[motif1_idx]]
                pwm2 = pwm_dict[grammar[motif2_idx]]

        
                # for each pair of motifs:
                model_params = {"trained_net": models[args.model['name']],
                                "pwm_a": pwm1,
                                "pwm_b": pwm2}
        
                # setup graph
                ism_graph = TronnGraph({"data": args.data_files},
                                       args.tasks,
                                       load_data_from_filename_list,
                                       ism_for_grammar_dependencies,
                                       model_params,
                                       args.batch_size)

                # then run and get synergy, and average indiv scores to get indiv coeffs
                synergy_score, pwm1_score, pwm2_score = run_ism_for_motif_pairwise_dependency(ism_graph) # TODO(dk) change to evaluate more examples
                indiv_motif_coeffs[motif1_idx, motif2_idx] = pwm1_score
                indiv_motif_coeffs[motif2_idx, motif1_idx] = pwm2_score
                synergies[motif1_idx, motif2_idx] = synergy_score
                indiv_avg_scores = np.zeros((1, num_motifs))
                for i in range(num_motifs):
                    indiv_avg_scores[0,i] = np.mean(indiv_motif_coeffs[i,:])                

                # write all this out to files
                grammar_dependencies_file = "{}.grammar.pairwise.txt".format(prefix)
                grammar_to_file(grammar_dependencies_file, indiv_avg_scores, synergies, motif_set)

    return None


def run(args):
    """Run pipeline
    """
    logging.info("Running ISM...")    

    # read in various files
    motif_sets = read_motifset_file(args.motif_sets_file)
    pwms = PWM.get_encode_motifs(args.pwm_file, as_dict=True)

    # run ISM
    ism_pipeline(
        motif_sets,
        pwms,
        bed_dir,
        prefix)

    return 
