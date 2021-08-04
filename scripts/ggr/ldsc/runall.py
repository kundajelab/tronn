#!/usr/bin/env python

# generate all results in this file

import os
import sys
import glob
import json
import gzip

import networkx as nx
import pandas as pd


def get_bed_from_nx_graph(graph, bed_file, interval_key="active", merge=True):
    """get BED file from nx examples
    """
    assert bed_file.endswith("gz")
    examples = list(graph.graph["examples"])

    with gzip.open(bed_file, "w") as fp:
        for region_metadata in examples:
            interval_types = region_metadata.split(";")
            interval_types = dict([
                interval_type.split("=")[0:2]
                for interval_type in interval_types])
            interval_string = interval_types[interval_key]

            chrom = interval_string.split(":")[0]
            start = interval_string.split(":")[1].split("-")[0]
            stop = interval_string.split("-")[1]
            fp.write("{}\t{}\t{}\n".format(chrom, start, stop))

    if merge:
        tmp_bed_file = "{}.tmp.bed.gz".format(bed_file.split(".bed")[0])
        os.system("mv {} {}".format(bed_file, tmp_bed_file))
        merge_cmd = (
            "zcat {0} | "
            "sort -k1,1 -k2,2n | "
            "bedtools merge -i stdin | "
            "gzip -c > {1}").format(
                tmp_bed_file,
                bed_file)
        os.system(merge_cmd)
        os.system("rm {}".format(tmp_bed_file))
    
    return None


def download_if_needed(annot_set, out_dir="."):
    """only download if dir/file does not exist
    """
    # check for dir or file
    annot_exists = True
    if annot_set.get("dir", None) is not None:
        annot_exists = os.path.isdir(annot_set["dir"])
    else:
        annot_exists = os.path.isfile(annot_set["file"])

    # if it doesn't exist, download and unzip
    if not annot_exists:
        url = annot_set["url"]
        download_cmd = "wget {} -P {}".format(url, out_dir)
        if url.endswith(".tgz"):
            unzip_cmd = "tar -xzvf {}/{}".format(out_dir, os.path.basename(url))
        elif url.endswith(".bz2"):
            unzip_cmd = "bunzip2 {}/{}".format(out_dir, os.path.basename(url))
        os.system(download_cmd)
        os.system(unzip_cmd)
        os.system("rm {}".format(os.path.basename(url)))
        
    return None


def setup_ldsc_annotations(bed_file, bim_prefix, hapmap_prefix, out_dir, chromsizes=None, extend_len=500):
    """set up annotations
    """
    # setup
    chroms = range(1,23)
    os.system("mkdir -p {}".format(out_dir))
    prefix = os.path.basename(bed_file).split(".bed")[0]

    # adjust bed file here as needed instead of before
    # param from Finucane 2018 is 500 for bed regions
    tmp_bed_file = "{}/{}.tmp_extend_500bp.bed.gz".format(
        out_dir, os.path.basename(bed_file).split(".bed")[0])
    if chromsizes is not None:
        slop_line = "bedtools slop -i stdin -g {} -b {} | ".format(chromsizes, extend_len)
    else:
        slop_line = ""
    merge_cmd = (
        "zcat {0} | "
        "sort -k1,1 -k2,2n | "
        "bedtools merge -i stdin | "
        "{1}"
        "gzip -c > {2}").format(
            bed_file,
            slop_line,
            tmp_bed_file)
    print merge_cmd
    os.system(merge_cmd)
    
    # go through chroms
    for chrom in chroms:
        # make annot file
        make_annot = (
            "python ~/git/ldsc/make_annot.py "
            "--bed-file {0} "
            "--bimfile {2}.{3}.bim "
            "--annot-file {4}/{1}.{3}.annot.gz").format(
                tmp_bed_file, prefix, bim_prefix, chrom, out_dir)
        print make_annot
        os.system(make_annot)
        
        # compute LD scores with annot file
        compute_ld = (
            "python ~/git/ldsc/ldsc.py "
            "--l2 "
            "--bfile {1}.{2} "
            "--ld-wind-cm 1 "
            "--annot {4}/{0}.{2}.annot.gz "
            "--thin-annot --out {4}/{0}.{2} "
            "--print-snps {3}.{2}.snp").format(
                prefix, bim_prefix, chrom, hapmap_prefix, out_dir)
        print compute_ld
        os.system(compute_ld)

    # clean up
    os.system("rm {}".format(tmp_bed_file))
    
    return


def setup_ggr_annotations(
        ldsc_annot, out_dir, custom_annot_dir, annot_table_file,
        chromsizes,
        grammar_dir="."):
    """set up GGR annotation sets
    """
    # annotation prefixes
    plink_prefix = "{}/{}/{}".format(
        out_dir, ldsc_annot["plink"]["dir"], ldsc_annot["plink"]["prefix"])
    hapmap_prefix = "{}/{}/{}".format(
        out_dir, ldsc_annot["hapmap3_snps"]["dir"], ldsc_annot["hapmap3_snps"]["prefix"])
    
    # add univ DHS regions
    REG2MAP_DIR = "/mnt/lab_data3/dskim89/ggr/annotations"
    univ_dhs_bed_file = "{}/reg2map_honeybadger2_dnase_all_p10_ucsc.bed.gz".format(REG2MAP_DIR)
    prefix = os.path.basename(univ_dhs_bed_file).split(".bed")[0]
    ldscore_file = "{}/{}.22.l2.ldscore.gz".format(
        custom_annot_dir, prefix)
    if not os.path.isfile(ldscore_file):
        setup_ldsc_annotations(
            univ_dhs_bed_file, plink_prefix, hapmap_prefix, custom_annot_dir,
            chromsizes=chromsizes)
    with open(annot_table_file, "w") as fp:
        fp.write("Reg2Map\t{}/{}.\n".format(
            custom_annot_dir, prefix))
    
    # get an unrelated cell type - HepG2
    HEPG2_DIR = "/mnt/lab_data/kundaje/users/dskim89/encode-roadmap/encode.dnase.peaks"
    hepg2_bed_file = "{}/ENCSR000ENP.HepG2_Hepatocellular_Carcinoma_Cell_Line.UW_Stam.DNase-seq_rep1-pr.IDR0.1.filt.narrowPeak.gz".format(
        HEPG2_DIR)
    prefix = os.path.basename(hepg2_bed_file).split(".bed")[0] # this is more for checking isfile
    ldscore_file = "{}/{}.22.l2.ldscore.gz".format(
        custom_annot_dir, prefix)
    if not os.path.isfile(ldscore_file):
        setup_ldsc_annotations(
            hepg2_bed_file, plink_prefix, hapmap_prefix, custom_annot_dir,
            chromsizes=chromsizes)
    with open(annot_table_file, "a") as fp:
        fp.write("HepG2\t{}/{}.\n".format(
            custom_annot_dir, prefix))
        
    # get ATAC all
    GGR_DIR = "/mnt/lab_data/kundaje/users/dskim89/ggr/integrative/v1.0.0a"
    ggr_master_bed_file = "{}/data/ggr.atac.idr.master.bed.gz".format(GGR_DIR)
    prefix = os.path.basename(ggr_master_bed_file).split(".bed")[0]
    ldscore_file = "{}/{}.22.l2.ldscore.gz".format(
        custom_annot_dir, prefix)
    if not os.path.isfile(ldscore_file):
        setup_ldsc_annotations(
            ggr_master_bed_file, plink_prefix, hapmap_prefix, custom_annot_dir,
            chromsizes=chromsizes)
    with open(annot_table_file, "a") as fp:
        fp.write("GGR_ALL\t{}/{}.\n".format(
            custom_annot_dir, prefix))
        
    # get ATAC timepoints
    timepoint_dir = "{}/results/atac/peaks.timepoints".format(GGR_DIR)
    timepoint_bed_files = sorted(glob.glob("{}/*narrowPeak.gz".format(timepoint_dir)))
    for timepoint_bed_file in timepoint_bed_files:
        prefix = os.path.basename(timepoint_bed_file).split(".bed")[0]
        ldscore_file = "{}/{}.22.l2.ldscore.gz".format(
            custom_annot_dir, prefix)
        if not os.path.isfile(ldscore_file):
            setup_ldsc_annotations(
                timepoint_bed_file, plink_prefix, hapmap_prefix, custom_annot_dir,
                chromsizes=chromsizes)
        if True:
            with open(annot_table_file, "a") as fp:
                fp.write("{1}\t{0}/{1}.\n".format(
                    custom_annot_dir, prefix))

    # get ATAC traj files
    traj_dir = "{}/results/atac/timeseries/dp_gp/reproducible/hard/reordered/bed".format(GGR_DIR)
    traj_bed_files = sorted(glob.glob("{}/*bed.gz".format(traj_dir)))
    for traj_bed_file in traj_bed_files:
        prefix = os.path.basename(traj_bed_file).split(".bed")[0]
        ldscore_file = "{}/{}.22.l2.ldscore.gz".format(
            custom_annot_dir, prefix)
        if not os.path.isfile(ldscore_file):
            setup_ldsc_annotations(
                traj_bed_file, plink_prefix, hapmap_prefix, custom_annot_dir,
                chromsizes=chromsizes)
        if True:
            with open(annot_table_file, "a") as fp:
                fp.write("{1}\t{0}/{1}.\n".format(
                    custom_annot_dir, prefix))

    # and grammars here
    grammar_dir = "/mnt/lab_data/kundaje/users/dskim89/ggr/nn/inference.2019-03-12/dmim.shuffle.OLD/grammars.annotated.manual_filt.merged.final"
    tmp_dir = "{}/tmp".format(custom_annot_dir)
    os.system("mkdir -p {}".format(tmp_dir))
    
    # get BED files from grammar files
    grammar_summary_file = "{}/grammars_summary.txt".format(grammar_dir)
    grammars = pd.read_csv(grammar_summary_file, sep="\t")
    for grammar_idx in range(grammars.shape[0]):

        # read in grammar
        grammar_file = grammars.iloc[grammar_idx]["filename"]
        grammar_file = "{}/{}".format(
            grammar_dir, os.path.basename(grammar_file))
        grammar = nx.read_gml(grammar_file)
        grammar.graph["examples"] = grammar.graph["examples"].split(",")

        # make bed file
        bed_file = "{}/{}.bed.gz".format(tmp_dir, os.path.basename(grammar_file).split(".gml")[0])
        if not os.path.isfile(bed_file):
            get_bed_from_nx_graph(grammar, bed_file) 

        # then make annotations
        prefix = os.path.basename(bed_file).split(".bed")[0]
        ldscore_file = "{}/{}.22.l2.ldscore.gz".format(
            custom_annot_dir, prefix)
        if not os.path.isfile(ldscore_file):
            setup_ldsc_annotations(
                bed_file, plink_prefix, hapmap_prefix, custom_annot_dir,
                chromsizes=chromsizes)
        if True:
            with open(annot_table_file, "a") as fp:
                fp.write("{1}\t{0}/{1}.\n".format(
                    custom_annot_dir, prefix))
            
    # setup some joint annotations
    # early, mid, late
    if True:
        early_bed_file = "{}/grouped.early.bed.gz".format(tmp_dir)
        if not os.path.isfile(early_bed_file):
            make_group = (
                "zcat {0}/*TRAJ_LABELS-0*bed.gz {0}/*TRAJ_LABELS-8*bed.gz {0}/*TRAJ_LABELS-9*bed.gz | "
                "sort -k1,1 -k2,2n | "
                "bedtools merge -i stdin | "
                "gzip -c > {1}").format(tmp_dir, early_bed_file)
            print make_group
            os.system(make_group)
        prefix = os.path.basename(early_bed_file).split(".bed")[0]
        ldscore_file = "{}/{}.22.l2.ldscore.gz".format(
            custom_annot_dir, prefix)
        if not os.path.isfile(ldscore_file):
            setup_ldsc_annotations(
                early_bed_file, plink_prefix, hapmap_prefix, custom_annot_dir,
                chromsizes=chromsizes)
        with open(annot_table_file, "a") as fp:
            fp.write("{1}\t{0}/{1}.\n".format(
                custom_annot_dir, prefix))

        mid_bed_file = "{}/grouped.mid.bed.gz".format(tmp_dir)
        if not os.path.isfile(mid_bed_file):
            make_group = (
                "zcat {0}/*TRAJ_LABELS-1*bed.gz | "
                "sort -k1,1 -k2,2n | "
                "bedtools merge -i stdin | "
                "gzip -c > {1}").format(tmp_dir, mid_bed_file)
            print make_group
            os.system(make_group)
        prefix = os.path.basename(mid_bed_file).split(".bed")[0]
        ldscore_file = "{}/{}.22.l2.ldscore.gz".format(
            custom_annot_dir, prefix)
        if not os.path.isfile(ldscore_file):
            setup_ldsc_annotations(
                mid_bed_file, plink_prefix, hapmap_prefix, custom_annot_dir,
                chromsizes=chromsizes)
        with open(annot_table_file, "a") as fp:
            fp.write("{1}\t{0}/{1}.\n".format(
                custom_annot_dir, prefix))

        late_bed_file = "{}/grouped.late.bed.gz".format(tmp_dir)
        if not os.path.isfile(late_bed_file):
            make_group = (
                "zcat {0}/*TRAJ_LABELS-2*bed.gz {0}/*TRAJ_LABELS-3*bed.gz | "
                "sort -k1,1 -k2,2n | "
                "bedtools merge -i stdin | "
                "gzip -c > {1}").format(tmp_dir, late_bed_file)
            print make_group
            os.system(make_group)
        prefix = os.path.basename(late_bed_file).split(".bed")[0]
        ldscore_file = "{}/{}.22.l2.ldscore.gz".format(
            custom_annot_dir, prefix)
        if not os.path.isfile(ldscore_file):
            setup_ldsc_annotations(
                late_bed_file, plink_prefix, hapmap_prefix, custom_annot_dir,
                chromsizes=chromsizes)
        with open(annot_table_file, "a") as fp:
            fp.write("{1}\t{0}/{1}.\n".format(
                custom_annot_dir, prefix))

    return


def setup_sumstats_file(
        sumstats_file,
        merge_alleles_file,
        out_file,
        other_params=""):
    """setup summary states file
    """
    munge_cmd = (
        "python ~/git/ldsc/munge_sumstats.py "
        "--sumstats {} "
        "--merge-alleles {} "
        "--out {} "
        "{} ").format(
            sumstats_file,
            merge_alleles_file,
            out_file.split(".sumstats")[0],
            other_params)
    print munge_cmd
    os.system(munge_cmd)
    
    return None


def setup_sumstats_files(
        sumstats_dir,
        sumstats_orig_dir,
        hapmap_snps_file):
    """build sumstats files
    """
    sumstats_files = []

    # -------------------------------
    # ukbb codes
    # -------------------------------

    # ukbb standardized, can do all in one go
    ukbb_manifest_file = "./ukbb/ukbb.gwas_imputed_3.release_20180731.tsv"
    ukbb_manifest = pd.read_csv(ukbb_manifest_file, sep="\t")

    # get the variant annotation file
    ukbb_annot_file = "./ukbb/variants.tsv.bgz"
    get_variant_annotation_file = "wget https://www.dropbox.com/s/puxks683vb0omeg/variants.tsv.bgz?dl=0 -O {}".format(
        ukbb_annot_file)
    if not os.path.isfile(ukbb_annot_file):
        os.system(get_variant_annotation_file)

    # GGR relevant codes
    ukbb_codes = [
        "20001_1003", # skin cancer
        "20001_1060", # skin cancer
        "20001_1061", # BCC
        "20001_1062", # SCC
        "20002_1371", # sarcoidosis (self report)
        "22133", # sarcoidosis (doctor dx)
        "D86", # sarcoidosis ICD
        "20002_1381", # lupus
        "20002_1382", # sjogrens
        "20002_1384", # scleroderma
        "20002_1452", # eczema/dermatitis
        "20002_1453", # psoriasis (self report)
        "L12_PSORI_NAS", # psoriasis
        "L12_PSORIASIS", # psoriasis
        "L40", # psoriasis ICD
        "20002_1454", # blistering
        "20002_1455", # skin ulcers
        "20002_1625", # cellulitis
        "20002_1660", # rosacea
        "L12_ROSACEA", # rosacea
        "L71", # rosacea
        "20002_1661", # vitiligo
        "B07", # viral warts
        "C_SKIN",
        "C_OTHER_SKIN", # neoplasm of skin
        "C3_SKIN",
        "C3_OTHER_SKIN", # neoplasm of skin
        "C44", # cancer ICD
        "D04", # carcinoma in situ of skin
        "D23", # benign neoplasms
        "L12_ACTINKERA", # actinic keratosis
        "L12_ATROPHICSKIN", # atrophic skin
        "L90", # atrophic skin ICD
        "L30", # other dermatitis
        "L12_EPIDERMALTHICKOTH", # epidermal thickening
        "L12_EPIDERMTHICKNAS", # epidermal thickening
        "L85", # epidermal thickening
        "L12_GRANULOMASKINNAS", # granulomatous
        "L12_GRANULOMATOUSSKIN", # granulomatous
        "L92", # granulomatous
        "L91", # hypertrophic disorders
        "L12_HYPERTROPHICNAS",
        "L12_HYPERTROPHICSKIN",
        "L12_HYPETROPHICSCAR",
        "L82", # seborrhoeic keratosis
        "XII_SKIN_SUBCUTAN", # skin
        "L12_SCARCONDITIONS", # scarring
        "L12_SKINSUBCUTISNAS", # other
        "20002_1548", # acne
        "20002_1549", # lichen planus
        "20002_1550", # lichen sclerosis
        "L12_NONIONRADISKIN", # skin changes from nonionizing radiation
        "L57", # skin changes from nonionizing radiation ICD
        "L12_OTHERDISSKINANDSUBCUTIS", # other
        "L98", # other
        #"21001_irnt" # BMI (non skin condition control)
    ]
        
    # for each, download and process
    for ukbb_code in ukbb_codes:

        # get ID
        id_metadata = ukbb_manifest[ukbb_manifest["Phenotype Code"] == ukbb_code]
        if id_metadata.shape[0] > 1:
            id_metadata = id_metadata[id_metadata["Sex"] == "both_sexes"]

        # download file
        filename = "{}/{}".format(sumstats_orig_dir, id_metadata["File"].iloc[0])
        if not os.path.isfile(filename):
            download_cmd = id_metadata["wget command"].iloc[0].split("-O")[0]
            download_cmd = "{} -O {}".format(download_cmd, filename)
            print download_cmd
            os.system(download_cmd)

        # paste with annot file
        w_annot_file = "{}.annot.tsv.gz".format(filename.split(".tsv")[0])
        if not os.path.isfile(w_annot_file):
            paste_cmd = (
                "paste <(zcat {}) <(zcat {}) | "
                "cut -f 1-6,10,13,17,31,34,37 | "
                #"cut -f 1-6,10,13,17,33,36,39 | " # use for BMI
                "gzip -c > {}").format(
                    ukbb_annot_file, filename, w_annot_file)
            print paste_cmd
            os.system('GREPDB="{}"; /bin/bash -c "$GREPDB"'.format(paste_cmd))
        
        # set up sumstats file
        description = str(id_metadata["Phenotype Description"].iloc[0]).split(":")
        if len(description) > 1:
            description = description[1]
        else:
            description = description[0]
        short_description = description.strip().replace(" ", "_").replace("/", "_").replace(",", "_").replace("'", "").replace("(", "").replace(")", "").lower()
        final_sumstats_file = "{}/ukbb.{}.{}.ldsc.sumstats.gz".format(
            sumstats_dir,
            ukbb_code,
            short_description)
        if not os.path.isfile(final_sumstats_file):
            setup_sumstats_file(
                w_annot_file,
                hapmap_snps_file,
                final_sumstats_file,
                other_params="--N-col n_complete_samples --a1 ref --a2 alt --frq AF")
                #other_params="--N-col n_called --a1 ref --a2 alt --frq AF") # use for BMI

        # append
        sumstats_files.append(final_sumstats_file)
    
    # -------------------------------
    # ukbb LDSC preprocessed files
    # -------------------------------
    
    # derm
    ukbb_derm_sumstats = "{}/ukbb.LDSCORE.derm.ldsc.sumstats.gz".format(sumstats_dir)
    if not os.path.isfile(ukbb_derm_sumstats):
        file_url = "https://data.broadinstitute.org/alkesgroup/LDSCORE/independent_sumstats/UKB_460K.disease_DERMATOLOGY.sumstats.gz"
        get_ukbb = "wget {} -O {}".format(file_url, ukbb_derm_sumstats)
        os.system(get_ukbb)
    sumstats_files.append(ukbb_derm_sumstats)

    # eczema
    ukbb_eczema_sumstats = "{}/ukbb.LDSCORE.eczema.ldsc.sumstats.gz".format(sumstats_dir)
    if not os.path.isfile(ukbb_eczema_sumstats):
        file_url = "https://data.broadinstitute.org/alkesgroup/LDSCORE/independent_sumstats/UKB_460K.disease_ALLERGY_ECZEMA_DIAGNOSED.sumstats.gz"
        get_ukbb = "wget {} -O {}".format(file_url, ukbb_eczema_sumstats)
        os.system(get_ukbb)
    sumstats_files.append(ukbb_eczema_sumstats)

    # bmi (as baseline control phenotype)
    ukbb_bmi_sumstats = "{}/ukbb.LDSCORE.bmi.ldsc.sumstats.gz".format(sumstats_dir)
    if not os.path.isfile(ukbb_bmi_sumstats):
        file_url = "https://data.broadinstitute.org/alkesgroup/LDSCORE/independent_sumstats/UKB_460K.body_BMIz.sumstats.gz"
        get_ukbb = "wget {} -O {}".format(file_url, ukbb_bmi_sumstats)
        os.system(get_ukbb)
    sumstats_files.append(ukbb_bmi_sumstats)
    
    # height (as baseline control phenotype)
    ukbb_bmi_sumstats = "{}/ukbb.LDSCORE.height.ldsc.sumstats.gz".format(sumstats_dir)
    if not os.path.isfile(ukbb_bmi_sumstats):
        file_url = "https://data.broadinstitute.org/alkesgroup/LDSCORE/independent_sumstats/UKB_460K.body_HEIGHTz.sumstats.gz"
        get_ukbb = "wget {} -O {}".format(file_url, ukbb_bmi_sumstats)
        os.system(get_ukbb)
    sumstats_files.append(ukbb_bmi_sumstats)

    # -------------------------------
    # other GWAS with sumstats
    # -------------------------------
            
    # acne - confirmed genome-wide genotyping array, Affy
    gwas_acne_sumstats = "{}/gwas.GCST006640.acne.ldsc.sumstats.gz".format(sumstats_dir)
    if not os.path.isfile(gwas_acne_sumstats):
        file_url = "ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/HirataT_29855537_GCST006640/Dataset_S7.txt"
        save_file = "{}/gwas.GCST006640.acne.sumstats.gz".format(sumstats_orig_dir)
        get_file = "wget -O - {} | gzip -c > {}".format(file_url, save_file)
        os.system(get_file)
        setup_sumstats_file(
            save_file,
            hapmap_snps_file,
            gwas_acne_sumstats,
            other_params="--N-cas 1115 --N-con 4619 --ignore regional.analysis")
    sumstats_files.append(gwas_acne_sumstats)
    
    # alopecia - genome-wide genotyping array
    gwas_alopecia_sumstats = "{}/gwas.GCST006661.alopecia.ldsc.sumstats.gz".format(sumstats_dir)
    if not os.path.isfile(gwas_alopecia_sumstats):
        file_url = "ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/HagenaarsSP_28196072_GCST006661/Hagenaars2017_UKB_MPB_summary_results.zip"
        unzip_dir = "{}/gwas.GCST006661.alopecia".format(sumstats_orig_dir)
        save_file = "{}/gwas.GCST006661.alopecia.zip".format(unzip_dir)
        os.system("mkdir -p {}".format(unzip_dir))
        get_file = "wget {} -O {}".format(file_url, save_file)
        os.system("unzip {} -d {}".format(save_file, unzip_dir))
        setup_sumstats_file(
            "",
            hapmap_snps_file,
            gwas_alopecia_sumstats,
            other_params="--N 52874 --snp Markername")
    sumstats_files.append(gwas_alopecia_sumstats)
        
    # dermatitis - genome-wide genotyping array, illumina
    gwas_dermatitis_sumstats = "{}/gwas.GCST003184.dermatitis.ldsc.sumstats.gz".format(sumstats_dir)
    if not os.path.isfile(gwas_dermatitis_sumstats):
        file_url = "ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/PaternosterL_26482879_GCST003184/EAGLE_AD_no23andme_results_29072015.txt"
        save_file = "{}/gwas.GCST003184.dermatitis.sumstats.gz".format(sumstats_orig_dir)
        get_file = "wget -O - {} | gzip -c > {}".format(file_url, save_file)
        os.system(get_file)
        setup_sumstats_file(
            save_file,
            hapmap_snps_file,
            gwas_dermatitis_sumstats,
            other_params="--N-col AllEthnicities_N")
    sumstats_files.append(gwas_dermatitis_sumstats)
        
    # lupus - genome-wide genotyping array, illumina
    gwas_lupus_sumstats = "{}/gwas.GCST005831.lupus.ldsc.sumstats.gz".format(sumstats_dir)
    if not os.path.isfile(gwas_lupus_sumstats):
        file_url = "ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/JuliaA_29848360_GCST005831/GWAS_SLE_summaryStats/Meta_results.txt"
        save_file = "{}/gwas.GCST005831.lupus.sumstats.gz".format(sumstats_orig_dir)
        get_file = "wget -O - {} | gzip -c > {}".format(file_url, save_file)
        setup_sumstats(
            save_file,
            hapmap_snps_file,
            gwas_lupus_sumstats,
            other_params="--N-cas 4943 --N-con 8483 --a1 A1lele1 --a2 Allele2")
    sumstats_files.append(gwas_lupus_sumstats)

    # lupus - genome-wide genotyping array, illumina
    gwas_lupus_sumstats = "{}/gwas.GCST003156.lupus.ldsc.sumstats.gz".format(sumstats_dir)
    if not os.path.isfile(gwas_lupus_sumstats):
        file_url = "ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/BenthamJ_26502338_GCST003156/bentham_2015_26502338_sle_efo0002690_1_gwas.sumstats.tsv.gz"
        save_file = "{}/gwas.GCST003156.lupus.sumstats.gz".format(sumstats_orig_dir)
        get_file = "wget {} -O {}".format(file_url, save_file)
        setup_sumstats(
            save_file,
            hapmap_snps_file,
            gwas_lupus_sumstats,
            other_params="--N 14267 --N-cas 5201 --N-con 9066 --ignore OR,OR_lower,OR_upper")
    sumstats_files.append(gwas_lupus_sumstats)
        
    # lupus - targeted array, ignore
    gwas_lupus_sumstats = "{}.gwas.GCST007400.lupus.ldsc.sumstats.gz"
        
    # psoriasis - targeted array, ignore
    gwas_psoriasis_sumstats = "{}/gwas.GCST005527.psoriasis.ldsc.sumstats.gz".format(sumstats_dir)

    # baldness - NOTE problem with pval column, do not use
    if False:
        gwas_baldness_sumstats = "{}/gwas.GCST007020.baldness.ldsc.sumstats.gz".format(sumstats_dir)
        if not os.path.isfile(gwas_baldness_sumstats):
            file_url = "ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/YapCX_30573740_GCST007020/mpb_bolt_lmm_aut_x.tab.zip"
            unzip_dir = "{}/gwas.GCST007020.baldness".format(sumstats_orig_dir)
            os.system("mkdir -p {}".format(unzip_dir))
            save_file = "{}/gwas.GCST007020.baldness.zip".format(unzip_dir)
            get_file = "wget {} -O {}".format(file_url, save_file)
            os.system("unzip {} -d {}".format(save_file, unzip_dir))
            save_file = "{}/mpb_bolt_lmm_aut_x.tab".format(unzip_dir)
            setup_sumstats(save_file, gwas_baldness_sumstats, other_params="--N 205327 --p P_BOLT_LMM_INF --a1 ALLELE1 --a2 ALLELE0")
    
    # solar lentigines - genome-wide genotyping array, Affy - NOTE more pigmentation change, do not use
    if False:
        gwas_lentigines_sumstats = "{}/gwas.GCST006096.lentigines.ldsc.sumstats.gz".format(sumstats_dir)
        if not os.path.isfile(gwas_lentigines_sumstats):
            file_url = "ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/EndoC_29895819_GCST006096/DatasetS1.txt"
            save_file = "{}/gwas.GCST006096.lentigines.sumstats.gz".format(sumstats_orig_dir)
            get_file = "wget {} -O - | gzip -c > {}".format(file_url, save_file)
            os.system(get_file)
            setup_sumstats(
                save_file,
                hapmap_snps_file,
                gwas_lentigines_sumstats,
                other_params="--N 11253 --N-cas 3815 --N-con 7438")
    
    # hyperhidrosis - genome-wide genotyping array, Affy
    gwas_hyperhidrosis_sumstats = "{}/gwas.GCST006090.hyperhidrosis.ldsc.sumstats.gz".format(sumstats_dir)
    if not os.path.isfile(gwas_hyperhidrosis_sumstats):
        file_url = "ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/EndoC_29895819_GCST006090/DatasetS7.txt"
        save_file = "{}/gwas.GCST006090.hyperhidrosis.sumstats.gz".format(sumstats_orig_dir)
        get_file = "wget {} -O - | gzip -c > {}".format(sumstats_orig_dir)
        setup_sumstats(
            save_file,
            hapmap_snps_file,
            gwas_hyperhidrosis_sumstats,
            other_params="--N 4538 --N-cas 1245 --N-con 3293")
    sumstats_files.append(gwas_hyperhidrosis_sumstats)

    # hirsutism (1) GCST006095
    gwas_hirsutism_sumstats = "{}/gwas.GCST006095.hirsutism.ldsc.sumstats.gz".format(sumstats_dir)
    if not os.path.isfile(gwas_hirsutism_sumstats):
        file_url = "ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/EndoC_29895819_GCST006095/DatasetS6.txt"
        save_file = "{}/gwas.GCST006095.hirsutism.sumstats.gz".format(sumstats_orig_dir)
        get_file = "wget {} -O - | gzip -c > {}".format(sumstats_orig_dir)
        setup_sumstats(
            save_file,
            hapmap_snps_file,
            gwas_hyperhidrosis_sumstats,
            other_params="--N 11244 --N-cas 3830 --N-con 7414")
    sumstats_files.append(gwas_hirsutism_sumstats)
    
    if False:
        # sarcoidosis (1) GCST005540
        gwas_sarcoidosis_sumstats = ""
        
    # lofgren - NOTE some error in the harmonized file - do not use
    if False:
        gwas_lofgrens_sumstats = "{}/gwas.GCST005540.lofgrens.ldsc.sumstats.gz".format(sumstats_orig_dir)
        if not os.path.isfile(gwas_lofgrens_sumstats):
            file_url = "ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/RiveraNV_26651848_GCST005540/harmonised/26651848-GCST005540-EFO_0009466.h.tsv.gz"
            save_file = "{}/gwas.GCST005540.lofgrens.sumstats.gz".format(sumstats_orig_dir)
            get_file = "wget {} -O {}".format(file_url, save_file)
            setup_sumstats(save_file, gwas_lofgrens_sumstats, other_params="")
    
    # vitiligo (4) GCST007112, GCST007111, GCST004785, GCST001509
    if False:
        gwas_vitiligo_sumstats = "{}/gwas.GCST007112.vitiligo.ldsc.sumstats.gz"
        gwas_vitiligo_sumstats = "{}/gwas.GCST007111.vitiligo.ldsc.sumstats.gz"
        gwas_vitiligo_sumstats = "{}/gwas.GCST004785.vitiligo.ldsc.sumstats.gz".format(sumstats_dir)
        gwas_vitiligo_sumstats = "{}/gwas.GCST001509.vitiligo.ldsc.sumstats.gz".format(sumstats_dir)
    
    return sumstats_files


def run_ldsc(
        annot_table_file,
        sumstats_file,
        baseline_model_prefix,
        weights_prefix,
        out_dir,
        celltype_specific=False,
        freq_prefix=None):
    """run LD score regression, either cell type specific (as recommended in tutorials) or as 
    partitioned heritability (to pick up enrichment scores)
    """
    atac_prefix = ""
    
    if celltype_specific:
        # Cell type specific analysis
        out_prefix = "{}/{}".format(out_dir, os.path.basename(sumstats_file).split(".ldsc")[0])
        out_results_file = "{}.cell_type_results.txt".format(out_prefix)
        if not os.path.isfile(out_results_file):
            run_ldsc_cmd = (
                "python ~/git/ldsc/ldsc.py "
                "--h2-cts {} "
                "--ref-ld-chr {} "
                "--w-ld-chr {} "
                "--ref-ld-chr-cts {} "
                "--out {}").format(
                    sumstats_file,
                    baseline_model_prefix,
                    weights_prefix,
                    annot_table_file,
                    out_prefix)
            print run_ldsc_cmd
            os.system(run_ldsc_cmd)

    else:
        # run full partition heritability
        # first check heritability, only use those above approx 0.07 h2
        out_prefix = "{}/{}.BASELINE.heritability".format(
            out_dir, os.path.basename(sumstats_file).split(".ldsc")[0])
        if not os.path.isfile("{}.results".format(out_prefix)):
            run_ldsc_cmd = (
                "python ~/git/ldsc/ldsc.py "
                "--h2 {} "
                "--ref-ld-chr {} "
                "--w-ld-chr {} "
                "--overlap-annot "
                "--frqfile-chr {} "
                "--out {} "
                "--print-coefficients").format(
                    sumstats_file,
                    baseline_model_prefix,
                    weights_prefix,
                    freq_prefix,
                    out_prefix)
            print run_ldsc_cmd
            os.system(run_ldsc_cmd)

        # save out heritability in an easy to read format...
        save_name = "echo -n {} >> {}/heritabilities.log".format(
            os.path.basename(out_prefix), out_dir)
        grep_h2 = (
            "cat {}.log | "
            "grep 'Total Observed scale h2' | "
            "awk -F ': ' '{{ print \"\t\"$2 }}' "
            ">> {}/heritabilities.log".format(out_prefix, out_dir))
        save_out = "{}; {}".format(save_name, grep_h2)
        os.system(save_out)
        
        if False:
            # I think actually may need to do this to pull out enrichments
            # remember enrichments are not corrected for baseline, no need to include baseline model in run
            annotations = pd.read_csv(annot_table_file, sep="\t", header=None)

            # try run all together?

            
            # for each line, run model
            for annot_i in range(annotations.shape[0]):
                annot_name_prefix = annotations[0].iloc[annot_i]
                annot_file_prefix = annotations[1].iloc[annot_i]
                out_prefix = "{}/{}.{}".format(
                    out_dir,
                    os.path.basename(sumstats_file).split(".ldsc")[0],
                    annot_name_prefix)
                out_file = "{}.results".format(out_prefix)
                if not os.path.isfile(out_file):
                    run_ldsc_cmd = (
                        "python ~/git/ldsc/ldsc.py "
                        "--h2 {} "
                        "--ref-ld-chr {} "
                        "--w-ld-chr {} "
                        "--overlap-annot "
                        "--frqfile-chr {} "
                        "--out {} "
                        "--print-coefficients").format(
                            sumstats_file,
                            ",".join([annot_file_prefix]),
                            weights_prefix,
                            freq_prefix,
                            out_prefix)
                    print run_ldsc_cmd
                    os.system(run_ldsc_cmd)
            quit()
    
    return


def main():
    """run all analyses for GGR GWAS variants
    """
    # args
    annotations_json = sys.argv[1]
    ANNOT_DIR = sys.argv[2]
    in_dir = sys.argv[3]
    grammar_dir = sys.argv[4]
    out_dir = sys.argv[5]

    # setup LDSC annotations
    with open(annotations_json, "r") as fp:
        ldsc_annotations = json.load(fp)
    for annot_set_key in sorted(ldsc_annotations.keys()):
        print annot_set_key
        download_if_needed(ldsc_annotations[annot_set_key], out_dir=out_dir)
    
    # generate annot files for custom region sets and save to table file
    custom_annot_dir = "{}/annot.custom.extend_500bp".format(out_dir)
    os.system("mkdir -p {}".format(custom_annot_dir))
    #annot_table_file = "{}/annot.table.txt".format(out_dir)
    annot_table_file = "{}/annot.table.extend_500bp.txt".format(out_dir)
    setup_ggr_annotations(
        ldsc_annotations, out_dir, custom_annot_dir, annot_table_file,
        "{}/hg19.chrom.sizes".format(ANNOT_DIR))
    
    # set up summary stats files
    sumstats_dir = "./sumstats"
    os.system("mkdir -p {}".format(sumstats_dir))
    sumstats_orig_dir = "{}/orig".format(sumstats_dir)
    os.system("mkdir -p {}".format(sumstats_orig_dir))
    sumstats_files = setup_sumstats_files(
        sumstats_dir, sumstats_orig_dir,
        "{}/{}".format(out_dir, ldsc_annotations["hapmap3_snp_list"]["file"]))
    sumstats_files = sorted(sumstats_files)

    # manual filter here for those that actually have enough heritability to partition (approx h2 > 0.01)
    filter_substrings = [
        # do not use (not really derm)
        #"gwas.GCST003156.lupus",
        #"gwas.GCST005831.lupus",
        #"gwas.GCST006090.hyperhidrosis",
        #"gwas.GCST006095.hirsutism",
        #"gwas.GCST006661.alopecia",

        # do not use (derm)
        #"ukbb.LDSCORE.derm",
        #"ukbb.LDSCORE.eczema",
        #"ukbb.C3_OTHER_SKIN.other_malignant_neoplasms_of_skin", # all non melanoma lol
        #"ukbb.C44.c44_other_malignant_neoplasms_of_skin"
        
        "gwas.GCST006640.acne",
        "ukbb.20001_1061.basal_cell_carcinoma",
        "ukbb.LDSCORE.bmi",
        "ukbb.LDSCORE.height",
        
        "ukbb.20002_1452.eczema_dermatitis",
        "ukbb.20002_1453.psoriasis",
        "ukbb.L12_ACTINKERA.actinic_keratosis",

        "gwas.GCST003184.dermatitis",                
        "ukbb.20001_1062.squamous_cell_carcinoma",
        "ukbb.20002_1548.acne_acne_vulgaris",
        "ukbb.L12_ROSACEA.rosacea",
        "ukbb.L71.l71_rosacea",
    ]
    
    # go through sumstats files and get enrichments
    results_dir = "{}/results".format(out_dir)
    os.system("mkdir -p {}".format(results_dir))
    for sumstats_file in sumstats_files:

        if True:
            # apply substrings filter
            stop_processing = True
            for filter_substring in filter_substrings:
                if filter_substring in sumstats_file:
                    stop_processing = False
            if stop_processing:
                continue
        
        print sumstats_file
        # NOTE: ldsc recommends 1.2 for cell types via pvalue, and LD 2.2 for estimating heritability enrichments
        # https://data.broadinstitute.org/alkesgroup/LDSCORE/readme_baseline_versions
        baseline_key = "baseline_1.2" # cell type specific pvals
        #baseline_key = "baselineLD_2.2" # heritability vals
        if False:
            run_ldsc(
                annot_table_file,
                sumstats_file,
                "{}/{}/{}".format(out_dir, ldsc_annotations[baseline_key]["dir"], ldsc_annotations[baseline_key]["prefix"]),
                "{}/{}/{}".format(out_dir, ldsc_annotations["weights"]["dir"], ldsc_annotations["weights"]["prefix"]),
                results_dir,
                celltype_specific=True,
                freq_prefix="{}/{}/{}".format(out_dir, ldsc_annotations["freqs"]["dir"], ldsc_annotations["freqs"]["prefix"]))

        # and plot the results
        results_prefix = "{}/{}".format(results_dir, os.path.basename(sumstats_file).split(".ldsc")[0])
        results_file = "{}.cell_type_results.txt".format(results_prefix)
        plot_file = "{}.pval_results.pdf".format(results_prefix)
        plot_file_2 = "{}.coef_results.pdf".format(results_prefix)
        #plot_cmd = "plot.ldsc.pval_results.R {} {}".format(results_file, plot_file)
        plot_cmd = "Rscript ~/git/tronn/scripts/ggr/ldsc/plot.ldsc.pval_results.R {} {} {}".format(
            results_file, plot_file, plot_file_2)
        print plot_cmd
        os.system(plot_cmd)
        
    return


main()
