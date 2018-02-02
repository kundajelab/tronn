# description: code for doing network analyses

import os

import numpy as np
import pandas as pd
import networkx as nx

def separate_and_save_components(G, mat_file, prefix, name_to_id, grammar_file):
    """given a graph, separate by non-connected components
    """
    # read in score file and get sums (to be weights)
    score_df = pd.read_table(mat_file, index_col=0)
    signal = np.sum(score_df, axis=0)
    signal = signal / signal.max()
    id_to_signal_dict = dict(zip(signal.index, signal.as_matrix()))
    
    # TODO: format as a grammar file
    # for master file, first save out nodes
    with open(grammar_file, "w") as out:
        out.write("")
    
    # save out connected components as separate groups (just connectivity, look at cliques later)
    components = []
    component_idx = 0
    for component in nx.connected_components(G):
        print component
        components.append(list(component))

        with open(grammar_file, "a") as out:
            out.write(">grammar.{}\n".format(component_idx))
            out.write("#params type:cc;directed=no\n".format(component_idx))
            
            # first save nodes
            out.write("#nodes\n".format(component_idx))
            for name in list(component):
                node_id = name_to_id[name]
                out.write("{0}\t{1}\n".format(node_id, id_to_signal_dict[node_id]))

            # then save edges - assume all connected (not true, but for simplicity for now)
            out.write("#edges\n".format(component_idx))
            for name1 in list(component):
                for name2 in list(component):
                    if name1 == name2:
                        continue
                    node_id1 = name_to_id[name1]
                    node_id2 = name_to_id[name2]
                    out.write("{0}\t{1}\t{2}\n".format(
                        node_id1, node_id2, 1.0))

        # also save out with hgnc names (for plotting purposes)
        component_file = "{}.component_{}.pwms.txt".format(prefix, component_idx)
        with open(component_file, "w") as out:
            for name in list(component):
                out.write("{}\t{}\n".format(name_to_id[name], name))
                
        # quick back check - look for regions that have these motifs
        # full check comes when re-scanning with correct thresholds
        # TODO this selection needs to be a little smarter - may be picking up too much noise
        pwm_names = [name_to_id[name] for name in list(component)]
        pwm_hits_pwm_subset_df = score_df[pwm_names]

        pwm_hits_positive_df = pwm_hits_pwm_subset_df.loc[~(pwm_hits_pwm_subset_df==0).any(axis=1)]
        component_regions_file = "{}.component_{}.regions.txt".format(prefix, component_idx)
        pwm_hits_positive_df.to_csv(component_regions_file, columns=[], header=False)

        # make bed file
        component_regions_bed = "{}.component_{}.regions.bed".format(prefix, component_idx)
        make_bed = (
            "cat {0} | "
            "awk -F ';' '{{ print $3 }}' | "
            "awk -F '=' '{{ print $2 }}' | "
            "awk -F ':' '{{ print $1\"\t\"$2 }}' | "
            "awk -F '-' '{{ print $1\"\t\"$2 }}' | "
            "sort -k1,1 -k2,2n | "
            "bedtools merge -i stdin > "
            "{1}").format(component_regions_file, component_regions_bed)
        os.system(make_bed)

        component_idx += 1

    return None
