#!/usr/bin/env python

"""
description: script to add in functional annotations
to grammars

"""

import os
import sys
import glob
import logging
import argparse

import pandas as pd
import networkx as nx

from tronn.interpretation.networks import get_bed_from_nx_graph
from tronn.interpretation.networks import stringize_nx_graph


def parse_args():
    """parser
    """
    parser = argparse.ArgumentParser(
        description="annotate grammars with functions")

    # required args
    parser.add_argument(
        "--grammars_dir",
        help="directory of gml files")
    parser.add_argument(
        "--annotation_type", default="great",
        help="annotation type (great, etc etc)")

    # out
    parser.add_argument(
        "-o", "--out_dir", dest="out_dir", type=str,
        default="./",
        help="out directory")
    
    # parse args
    args = parser.parse_args()

    return args


def _run_great(bed_file, out_dir):
    """run great
    """
    run_cmd = "run_rgreat.R {} {}/{}".format(
        bed_file,
        out_dir,
        os.path.basename(bed_file).split(".bed")[0])
    print run_cmd
    os.system('GREPDB="{}"; /bin/bash -c "$GREPDB"'.format(run_cmd))
    
    return None


def _add_go_terms_to_graph(nx_graph, great_file, top_k=None):
    """add go terms into graph
    """
    enrichments = pd.read_table(great_file)

    # add in
    nx_graph.graph["GOids"] = enrichments["ID"].values.tolist()
    nx_graph.graph["GOnames"] = enrichments["name"].values.tolist()
    nx_graph.graph["GOqvals"] = enrichments["hyper_q_vals"].values.tolist()
    
    return None


def track_runs(args):
    """track command and github commit
    """
    # keeps track of restores (or different commands) in folder
    subcommand_name = "annotate_grammars"
    num_restores = len(glob.glob('{0}/{1}.command*'.format(args.out_dir, subcommand_name)))
    logging_file = '{0}/{1}.command_{2}.log'.format(args.out_dir, subcommand_name, num_restores)
    
    # track github commit
    git_repo_path = os.path.dirname(os.path.realpath(__file__))
    os.system('echo "commit:" > {0}'.format(logging_file))
    os.system('git --git-dir={0}/.git rev-parse HEAD >> {1}'.format(
        git_repo_path.split("/scripts")[0], logging_file))
    os.system('echo "" >> {0}'.format(logging_file))
    
    # write out the command
    with open(logging_file, 'a') as f:
        f.write(' '.join(sys.argv)+'\n\n')
    
    return logging_file


def _setup_logs(args):
    """set up logging
    """
    logging_file = track_runs(args)
    reload(logging)
    logging.basicConfig(
        filename=logging_file,
        level=logging.DEBUG, # TODO ADJUST BEFORE RELEASE
        format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    for arg in sorted(vars(args)):
        logging.info("{}: {}".format(arg, getattr(args, arg)))
    logging.info("")

    return


def main():
    """run annotation
    """
    # set up args
    args = parse_args()
    # make sure out dir exists
    os.system("mkdir -p {}".format(args.out_dir))
    _setup_logs(args)

    # get all gml files and load in
    grammar_files = glob.glob("{}/*gml".format(args.grammars_dir))
    grammars = [nx.read_gml(grammar_file) for grammar_file in grammar_files]

    # and adjust examples
    for grammar in grammars:
        grammar.graph["examples"] = set(
            grammar.graph["examples"].split(","))
        for node_name in grammar.nodes():
            grammar.nodes[node_name]["examples"] = set(
                grammar.nodes[node_name]["examples"].split(","))
        for edge_name in grammar.edges():
            for edge_idx in xrange(len(grammar[edge_name[0]][edge_name[1]])):
                grammar[edge_name[0]][edge_name[1]][edge_idx]["examples"] = set(
                    grammar[edge_name[0]][edge_name[1]][edge_idx]["examples"].split(","))

    # annotate
    if args.annotation_type == "great":
        # tricky part here is to use the BED files and then link back to gml files...
        # maybe just use the gml files and produce BED files on the fly
        for grammar_idx in xrange(len(grammars)):
            # make bed
            bed_file = "{}/{}.bed".format(
                args.out_dir,
                os.path.basename(grammar_files[grammar_idx]).split(".gml")[0])
            get_bed_from_nx_graph(grammars[grammar_idx], bed_file)
            
            # run great
            _run_great(bed_file, args.out_dir)
            
            # get terms back and save into file
            biol_terms_file = "{}.GO_Biological_Process.txt".format(bed_file.split(".bed")[0])
            _add_go_terms_to_graph(grammars[grammar_idx], biol_terms_file)

            # and write a new file out
            graph_file = "{}/{}.great_annotated.gml".format(
                args.out_dir,
                os.path.basename(grammar_files[grammar_idx]).split(".gml")[0])
            nx.write_gml(stringize_nx_graph(grammars[grammar_idx]), graph_file, stringizer=str)

        # and plot the GO table
        go_files = glob.glob("{}/*Biol*txt".format(args.out_dir))
        plot_go_table = "make_go_table.R {} {}".format(
            args.out_dir, " ".join(go_files))
        print plot_go_table
        os.system(plot_go_table)
        
    else:
        raise ValueError, "annotation type not implemented"
    
    return None



if __name__ == "__main__":
    main()
