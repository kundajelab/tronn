# description: code for doing network analyses

import os

import pandas as pd
import networkx as nx

def separate_and_save_components(G, mat_file, prefix, name_to_id):
    """given a graph, separate by non-connected components
    """
    score_df = pd.read_table(mat_file, index_col=0)
    
    master_component_file = "{}.components.txt".format(prefix)
    
    # save out connected components as separate groups (just connectivity, look at cliques later)
    components = []
    component_idx = 0
    for component in nx.connected_components(G):
        print component
        components.append(list(component))

        # here, save out. also save out to a master file
        component_file = "{}.component_{}.txt".format(prefix, component_idx)
        with open(component_file, "w") as out:
            for name in list(component):
                out.write("{}\t{}\n".format(name_to_id[name], name))
        with open(master_component_file, "a") as out:
            out.write("> component_{}\n".format(component_idx))
            for name in list(component):
                out.write("{}\t{}\n".format(name_to_id[name], name))
                
        # quick back check - look for regions that have these motifs
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
