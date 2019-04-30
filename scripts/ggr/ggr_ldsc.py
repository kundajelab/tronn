# generate all results in this file

import os
import glob

import networkx as nx
import pandas as pd


def get_bed_from_nx_graph(graph, bed_file, interval_key="active", merge=True):
    """get BED file from nx examples
    """
    examples = list(graph.graph["examples"])

    with open(bed_file, "w") as fp:
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
        tmp_bed_file = "{}.tmp.bed".format(bed_file.split(".bed")[0])
        os.system("mv {} {}".format(bed_file, tmp_bed_file))
        os.system("cat {} | sort -k1,1 -k2,2n | bedtools merge -i stdin | gzip -c > {}".format(
            tmp_bed_file, bed_file))
        os.system("rm {}".format(tmp_bed_file))
    
    return None


def setup_ldsc_annotations(bed_file, bim_prefix, hapmap_prefix, out_dir):
    """set up annotations
    """
    chroms = range(1,23)
    os.system("mkdir -p {}".format(out_dir))
    prefix = os.path.basename(bed_file).split(".bed")[0]
    
    for chrom in chroms:
        # make annot file
        make_annot = (
            "python ~/git/ldsc/make_annot.py "
            "--bed-file {0} "
            "--bimfile {2}.{3}.bim "
            "--annot-file {4}/{1}.{3}.annot.gz").format(
                bed_file, prefix, bim_prefix, chrom, out_dir)
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
    
    return


def setup_sumstats_file(sumstats_file, out_file, other_params=""):
    """setup summary states file
    """
    munge_cmd = (
        "python ~/git/ldsc/munge_sumstats.py "
        "--sumstats {} "        
        "--out {} "
        "{} ").format(
            sumstats_file,
            out_file.split(".sumstats")[0],
            other_params)
    print munge_cmd
    os.system(munge_cmd)
    
    return None


def main():
    """run all analyses for GGR GWAS variants
    """
    # get baseline ldsc model
    if not os.path.isdir("1000G_EUR_Phase3_baseline"):
        get_baseline_model = "wget https://data.broadinstitute.org/alkesgroup/LDSCORE/1000G_Phase3_baseline_ldscores.tgz"
        setup_baseline_model = "tar -xvzf 1000G_Phase3_baseline_ldscores.tgz"
        os.system(get_baseline_model)
        os.system(setup_baseline_model)
    baseline_model_prefix = "1000G_EUR_Phase3_baseline/baseline."

    # get model weights
    if not os.path.isdir("weights_hm3_no_hla"):
        get_weights = "wget https://data.broadinstitute.org/alkesgroup/LDSCORE/weights_hm3_no_hla.tgz"
        setup_weights = "tar -xvzf weights_hm3_no_hla.tgz"
        os.system(get_weights)
        os.system(setup_weights)
    weights_prefix = "weights_hm3_no_hla/weights."
        
    # get plink files (for setting up annotations from BED)
    if not os.path.isdir("1000G_EUR_Phase3_plink"):
        get_plink = "wget https://data.broadinstitute.org/alkesgroup/LDSCORE/1000G_Phase3_plinkfiles.tgz"
        setup_plink = "tar -xzvf 1000G_Phase3_plinkfiles.tgz"
        os.system(get_plink)
        os.system(setup_plink)
    bim_prefix = "1000G_EUR_Phase3_plink/1000G.EUR.QC"
        
    # get hapmap
    if not os.path.isdir("hapmap3_snps"):
        get_hapmap = "wget https://data.broadinstitute.org/alkesgroup/LDSCORE/hapmap3_snps.tgz"
        setup_hapmap = "tar -xzvf hapmap3_snps.tgz"
        os.system(get_hapmap)
        os.system(setup_hapmap)
    hapmap_prefix = "hapmap3_snps/hm"
    
    # ldsc annot dir
    ldsc_annot_dir = "./ldsc.annot"
    os.system("mkdir -p {}".format(ldsc_annot_dir))

    # ldsc file table
    ldsc_table_file = "./annot.table.2.ldsc"
        
    # get ATAC all
    GGR_DIR = "/mnt/lab_data/kundaje/users/dskim89/ggr/integrative/v1.0.0a"
    ggr_master_bed_file = "{}/data/ggr.atac.idr.master.bed.gz".format(GGR_DIR)
    prefix = os.path.basename(ggr_master_bed_file).split(".bed")[0]
    ldscore_file = "{}/{}.1.l2.ldscore.gz".format(
        ldsc_annot_dir, prefix)
    #if not os.path.isfile(ldscore_file):
    if False:
        setup_ldsc_annotations(
            ggr_master_bed_file, bim_prefix, hapmap_prefix, ldsc_annot_dir)
    with open(ldsc_table_file, "w") as fp:
        fp.write("GGR_ALL\t{}/{}.\n".format(
            ldsc_annot_dir, prefix))

    # get ATAC timepoints
    timepoint_dir = "{}/results/atac/peaks.timepoints".format(GGR_DIR)
    timepoint_bed_files = sorted(glob.glob("{}/*narrowPeak.gz".format(timepoint_dir)))
    for timepoint_bed_file in timepoint_bed_files:
        prefix = os.path.basename(timepoint_bed_file).split(".bed")[0]
        ldscore_file = "{}/{}.1.l2.ldscore.gz".format(
            ldsc_annot_dir, prefix)
        #if not os.path.isfile(ldscore_file):
        if False:
            setup_ldsc_annotations(
                timepoint_bed_file, bim_prefix, hapmap_prefix, ldsc_annot_dir)
        with open(ldsc_table_file, "a") as fp:
            fp.write("{1}\t{0}/{1}.\n".format(
                ldsc_annot_dir, prefix))

    # get ATAC traj files
    traj_dir = "{}/results/atac/timeseries/dp_gp/reproducible/hard/reordered/bed".format(GGR_DIR)
    traj_bed_files = sorted(glob.glob("{}/*bed.gz".format(traj_dir)))
    for traj_bed_file in traj_bed_files:
        prefix = os.path.basename(traj_bed_file).split(".bed")[0]
        ldscore_file = "{}/{}.1.l2.ldscore.gz".format(
            ldsc_annot_dir, prefix)
        #if not os.path.isfile(ldscore_file):
        if False:
            setup_ldsc_annotations(
                traj_bed_file, bim_prefix, hapmap_prefix, ldsc_annot_dir)
        with open(ldsc_table_file, "a") as fp:
            fp.write("{1}\t{0}/{1}.\n".format(
                ldsc_annot_dir, prefix))
                     
    # links
    grammar_dir = "./grammars"
    
    # download
    kube_dir = "kundajelab/dk.dev:/datasets/inference.2019-03-12/dmim.shuffle/grammars.annotated.manual_filt.merged.final"
    if not os.path.isdir(grammar_dir):
        os.system("mkdir -p {}".format(grammar_dir))
        download_cmd = "kubectl cp {}/ {}/".format(
            kube_dir, grammar_dir)
        print download_cmd
        os.system(download_cmd)

    # get BED files from grammar files
    grammar_summary_file = "{}/grammars_summary.txt".format(grammar_dir)
    grammars = pd.read_csv(grammar_summary_file, sep="\t")
    for grammar_idx in range(grammars.shape[0]):
        print grammar_idx

        # read in grammar
        grammar_file = grammars.iloc[grammar_idx]["filename"]
        grammar_file = "{}/{}".format(
            grammar_dir, os.path.basename(grammar_file))
        grammar = nx.read_gml(grammar_file)
        grammar.graph["examples"] = grammar.graph["examples"].split(",")

        # make bed file
        bed_file = "{}.bed.gz".format(grammar_file.split(".gml")[0])
        if False:
            get_bed_from_nx_graph(grammar, bed_file)

        # then make annotations
        prefix = os.path.basename(bed_file).split(".bed")[0]
        ldscore_file = "{}/{}.1.l2.ldscore.gz".format(
            ldsc_annot_dir, prefix)
        #if not os.path.isfile(ldscore_file):
        if False:
            setup_ldsc_annotations(
                bed_file, bim_prefix, hapmap_prefix, ldsc_annot_dir)
        with open(ldsc_table_file, "a") as fp:
            fp.write("{1}\t{0}/{1}.\n".format(
                ldsc_annot_dir, prefix))
    
    # pull relevant GWAS summary stats (plus UKBB), configure, and run
    sumstats_dir = "./sumstats"
    os.system("mkdir -p {}".format(sumstats_dir))
    sumstats_orig_dir = "{}/orig".format(sumstats_dir)
    os.system("mkdir -p {}".format(sumstats_orig_dir))
    
    # get UKBB derm stats
    ukbb_derm_sumstats = "{}/ukbb.none.derm.ldsc.sumstats.gz".format(sumstats_dir)
    if not os.path.isfile(ukbb_derm_sumstats):
        file_url = "https://data.broadinstitute.org/alkesgroup/UKBB/disease_DERMATOLOGY.sumstats.gz"
        save_file = "{}/ukbb.ldsc_pheno.dermatology.sumstats.gz".format(sumstats_orig_dir)
        get_ukbb = "wget {} -O {}".format(
            file_url,
            save_file)
        os.system(get_ukbb)
        setup_sumstats_file(save_file, ukbb_derm_sumstats)

    # ukbb standardized, can do all in one go
    ukbb_manifest_file = "./ukbb/ukbb.gwas_imputed_3.release_20180731.tsv"
    ukbb_manifest = pd.read_csv(ukbb_manifest_file, sep="\t")

    # get the variant annotation file
    ukbb_annot_file = "./ukbb/variants.tsv.bgz"
    get_variant_annotation_file = "wget https://www.dropbox.com/s/puxks683vb0omeg/variants.tsv.bgz?dl=0 -O {}".format(
        ukbb_annot_file)
    if not os.path.isfile(ukbb_annot_file):
        os.system(get_variant_annotation_file)
    
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
        
    ]

    # for each, download and process
    for ukbb_code in ukbb_codes:
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
        short_description = description.strip().replace(" ", "_").replace("/", "_").replace(",", "_").lower()
        final_sumstats_file = "{}/ukbb.{}.{}.ldsc.sumstats.gz".format(
            sumstats_dir,
            ukbb_code,
            short_description)
        if not os.path.isfile(final_sumstats_file):
            setup_sumstats_file(
                w_annot_file,
                final_sumstats_file,
                other_params="--N-col n_complete_samples --a1 ref --a2 alt --frq AF")

    quit()
        
        
    print ukbb_manifest
    print len(ukbb_codes)
    
    quit()
        
    # acne - confirmed genome-wide genotyping array, Affy
    gwas_acne_sumstats = "{}/gwas.GCST006640.acne.ldsc.sumstats.gz".format(sumstats_dir)
    if not os.path.isfile(gwas_acne_sumstats):
        file_url = "ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/HirataT_29855537_GCST006640/Dataset_S7.txt"
        save_file = "{}/gwas.GCST006640.acne.sumstats.gz".format(sumstats_orig_dir)
        get_file = "wget -O - {} | gzip -c > {}".format(file_url, save_file)
        os.system(get_file)
        setup_sumstats_file(save_file, gwas_acne_sumstats, n_cases=1115, n_controls=4619, other_params="--ignore regional.analysis")
        
    # alopecia - genome-wide genotyping array
    gwas_alopecia_sumstats = "{}/gwas.GCST006661.alopecia.ldsc.sumstats.gz".format(sumstats_dir)
    if not os.path.isfile(gwas_alopecia_sumstats):
        file_url = "ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/HagenaarsSP_28196072_GCST006661/Hagenaars2017_UKB_MPB_summary_results.zip"
        unzip_dir = "{}/gwas.GCST006661.alopecia".format(sumstats_orig_dir)
        save_file = "{}/gwas.GCST006661.alopecia.zip".format(unzip_dir)
        os.system("mkdir -p {}".format(unzip_dir))
        get_file = "wget {} -O {}".format(file_url, save_file)
        os.system("unzip {} -d {}".format(save_file, unzip_dir))
        setup_sumstats_file("", gwas_alopecia_sumstats, n_sample=52874, other_params="--snp Markername")
        
    # dermatitis - genome-wide genotyping array, illumina
    gwas_dermatitis_sumstats = "{}/gwas.GCST003184.dermatitis.ldsc.sumstats.gz".format(sumstats_dir)
    if not os.path.isfile(gwas_dermatitis_sumstats):
        file_url = "ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/PaternosterL_26482879_GCST003184/EAGLE_AD_no23andme_results_29072015.txt"
        save_file = "{}/gwas.GCST003184.dermatitis.sumstats.gz".format(sumstats_orig_dir)
        get_file = "wget -O - {} | gzip -c > {}".format(file_url, save_file)
        os.system(get_file)
        setup_sumstats_file(save_file, gwas_dermatitis_sumstats, other_params="--N-col AllEthnicities_N")
        
    # lupus - genome-wide genotyping array, illumina
    gwas_lupus_sumstats = "{}/gwas.GCST005831.lupus.ldsc.sumstats.gz".format(sumstats_dir)
    if not os.path.isfile(gwas_lupus_sumstats):
        file_url = "ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/JuliaA_29848360_GCST005831/GWAS_SLE_summaryStats/Meta_results.txt"
        save_file = "{}/gwas.GCST005831.lupus.sumstats.gz".format(sumstats_orig_dir)
        get_file = "wget -O - {} | gzip -c > {}".format(file_url, save_file)
        setup_sumstats(save_file, gwas_lupus_sumstats, other_params="--N-cas 4943 --N-con 8483 --a1 A1lele1 --a2 Allele2")

    # lupus - targeted array, ignore
    gwas_lupus_sumstats = "{}.gwas.GCST007400.lupus.ldsc.sumstats.gz"

    # lupus - genome-wide genotyping array, illumina
    gwas_lupus_sumstats = "{}/gwas.GCST003156.lupus.ldsc.sumstats.gz".format(sumstats_dir)
    if not os.path.isfile(gwas_lupus_sumstats):
        file_url = "ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/BenthamJ_26502338_GCST003156/bentham_2015_26502338_sle_efo0002690_1_gwas.sumstats.tsv.gz"
        save_file = "{}/gwas.GCST003156.lupus.sumstats.gz".format(sumstats_orig_dir)
        get_file = "wget {} -O {}".format(file_url, save_file)
        setup_sumstats(save_file, gwas_lupus_sumstats, other_params="--N 14267 --N-cas 5201 --N-con 9066 --ignore OR,OR_lower,OR_upper")
        
    # psoriasis - targeted array, ignore
    gwas_psoriasis_sumstats = "{}/gwas.GCST005527.psoriasis.ldsc.sumstats.gz".format(sumstats_dir)
    
    # baldness - NOTE problem with pval column
    gwas_baldness_sumstats = "{}/gwas.GCST007020.baldness.ldsc.sumstats.gz".format(sumstats_dir)
    if False:
        if not os.path.isfile(gwas_baldness_sumstats):
            file_url = "ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/YapCX_30573740_GCST007020/mpb_bolt_lmm_aut_x.tab.zip"
            unzip_dir = "{}/gwas.GCST007020.baldness".format(sumstats_orig_dir)
            os.system("mkdir -p {}".format(unzip_dir))
            save_file = "{}/gwas.GCST007020.baldness.zip".format(unzip_dir)
            get_file = "wget {} -O {}".format(file_url, save_file)
            os.system("unzip {} -d {}".format(save_file, unzip_dir))
            save_file = "{}/mpb_bolt_lmm_aut_x.tab".format(unzip_dir)
            setup_sumstats(save_file, gwas_baldness_sumstats, other_params="--N 205327 --p P_BOLT_LMM_INF --a1 ALLELE1 --a2 ALLELE0")
    
    # solar lentigines - genome-wide genotyping array, Affy
    gwas_lentigines_sumstats = "{}/gwas.GCST006096.lentigines.ldsc.sumstats.gz".format(sumstats_dir)
    if not os.path.isfile(gwas_lentigines_sumstats):
        file_url = "ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/EndoC_29895819_GCST006096/DatasetS1.txt"
        save_file = "{}/gwas.GCST006096.lentigines.sumstats.gz".format(sumstats_orig_dir)
        get_file = "wget {} -O - | gzip -c > {}".format(file_url, save_file)
        os.system(get_file)
        setup_sumstats(save_file, gwas_lentigines_sumstats, other_params="--N 11253 --N-cas 3815 --N-con 7438")
    
    # hyperhidrosis - genome-wide genotyping array, Affy
    gwas_hyperhidrosis_sumstats = "{}/gwas.GCST006090.hyperhidrosis.ldsc.sumstats.gz".format(sumstats_dir)
    if not os.path.isfile(gwas_hyperhidrosis_sumstats):
        file_url = "ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/EndoC_29895819_GCST006090/DatasetS7.txt"
        save_file = "{}/gwas.GCST006090.hyperhidrosis.sumstats.gz".format(sumstats_orig_dir)
        get_file = "wget {} -O - | gzip -c > {}".format(sumstats_orig_dir)
        setup_sumstats(save_file, gwas_hyperhidrosis_sumstats, other_params="--N 4538 --N-cas 1245 --N-con 3293")
        
    # hirsutism (1) GCST006095
    gwas_hirsutism_sumstats = "{}/gwas.GCST006095.hirsutism.ldsc.sumstats.gz".format(sumstats_dir)
    if not os.path.isfile(gwas_hirsutism_sumstats):
        file_url = "ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/EndoC_29895819_GCST006095/DatasetS6.txt"
        save_file = "{}/gwas.GCST006095.hirsutism.sumstats.gz".format(sumstats_orig_dir)
        get_file = "wget {} -O - | gzip -c > {}".format(sumstats_orig_dir)
        setup_sumstats(save_file, gwas_hyperhidrosis_sumstats, other_params="--N 11244 --N-cas 3830 --N-con 7414")

    if False:
        # sarcoidosis (1) GCST005540
        gwas_sarcoidosis_sumstats = ""

        # lofgren - NOTE some error in the harmonized file?
        gwas_lofgrens_sumstats = "{}/gwas.GCST005540.lofgrens.ldsc.sumstats.gz".format(sumstats_orig_dir)
        if not os.path.isfile(gwas_lofgrens_sumstats):
            file_url = "ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/RiveraNV_26651848_GCST005540/harmonised/26651848-GCST005540-EFO_0009466.h.tsv.gz"
            save_file = "{}/gwas.GCST005540.lofgrens.sumstats.gz".format(sumstats_orig_dir)
            get_file = "wget {} -O {}".format(file_url, save_file)
            setup_sumstats(save_file, gwas_lofgrens_sumstats, other_params="")
    
    # vitiligo (4) GCST007112, GCST007111, GCST004785, GCST001509
    gwas_vitiligo_sumstats = "{}/gwas.GCST007112.vitiligo.ldsc.sumstats.gz"
    
    gwas_vitiligo_sumstats = "{}/gwas.GCST007111.vitiligo.ldsc.sumstats.gz"


    gwas_vitiligo_sumstats = "{}/gwas.GCST004785.vitiligo.ldsc.sumstats.gz".format(sumstats_dir)

    
    gwas_vitiligo_sumstats = "{}/gwas.GCST001509.vitiligo.ldsc.sumstats.gz".format(sumstats_dir)
    
    
    quit()
    

    sumstats_files = glob.glob("{}/*ldsc.sumstats.gz".format(sumstats_dir))
    
    if False:
        run_ldsc = (
            "python ~/git/ldsc/ldsc.py "
            "--h2-cts {} "
            "--ref-ld-chr {} "
            "--out {} "
            "--ref-ld-chr-cts {} "
            "--w-ld-chr {}").format(
                sum_stats_file,
                baseline_model_prefix,
                out_prefix,
                condition_table_file,
                weights_prefix)
    

    # make plots
    
    

    return


main()
