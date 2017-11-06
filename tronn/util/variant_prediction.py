# hacky code for quick aggregation for variant prediction


import sys
import pandas as pd


def main():
    """Quick aggregation of results
    """

    allele1_file = sys.argv[1]
    allele2_file = sys.argv[2]
    out_file = sys.argv[3]
    
    # join the files together
    allele1 = pd.read_table(allele1_file, header=0, names=["name1", "allele1"])
    allele2 = pd.read_table(allele2_file, header=0, names=["name2", "allele2"])

    joint = pd.concat([allele1, allele2], axis=1)

    joint["delta"] = joint["allele2"] - joint["allele1"]
    
    # save out
    tmp_file = "{}.tmp".format(out_file)
    joint.to_csv(tmp_file, index=False, sep='\t')

    
    # and adjust
    with open(tmp_file, "r") as fp:
        with open(out_file, "w") as out:
            current_snp = ""
            current_sum = 0
            for line in fp:
                if line.startswith("name"):
                    continue
                
                fields = line.strip().split()
                
                # get the SNP
                snp_id = fields[0].split(";")[1].split("=")[1]
                snp_effect = float(fields[4])

                if snp_id == current_snp:
                    current_sum += snp_effect
                else:
                    if current_snp != "":
                        out.write("{}\t{}\n".format(current_snp, current_sum))
                    current_snp = snp_id
                    current_sum = snp_effect

    return None

main()
