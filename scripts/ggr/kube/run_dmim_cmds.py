# small wrapper to submit all dmim jobs appropriately

import os

def main():
    """
    """

    jobs = {
        "early": [0, 7, 8, 9, 10, 11],
        "mid": [1, 12, 13, 14],
        "late": [2, 3, 4, 5]
    }

    job_file = "job.dmim.ggr.yaml"
    
    for job_set in sorted(jobs.keys()):
        print job_set

        for idx in jobs[job_set]:
            adjust_job_set = "sed -i -e 's/early\|mid\|late/{}/g' {}".format(job_set, job_file)
            print adjust_job_set
            os.system(adjust_job_set)
            adjust_job_name = "sed -i -e 's/traj-[0-9]\+/traj-{}/g' {}".format(idx, job_file)
            print adjust_job_name
            os.system(adjust_job_name)
            adjust_job_foreground = "sed -i -e 's/LABELS=[0-9]\+/LABELS={}/g' {}".format(idx, job_file)
            print adjust_job_foreground
            os.system(adjust_job_foreground)
            adjust_job_foreground_name = "sed -i -e 's/LABELS-[0-9]\+/LABELS-{}/g' {}".format(idx, job_file)
            print adjust_job_foreground_name
            os.system(adjust_job_foreground_name)
            print ""

            import ipdb
            ipdb.set_trace()
    
    return


main()
