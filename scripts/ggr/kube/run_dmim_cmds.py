# small wrapper to submit all dmim jobs appropriately

import os

def main():
    """run all dmim jobs 
    """
    # setup
    START_JOBS = False
    job_file = "job.dmim.ggr.yaml"    
    jobs = {
        "early": [0, 7, 8, 9, 10, 11],
        "mid": [1, 12, 13, 14],
        "late": [2, 3, 4, 5]}

    # runs
    if START_JOBS:
        # start jobs
        for job_set in sorted(jobs.keys()):
            print job_set

            for idx in jobs[job_set]:
                adjust_job_set = "sed -i -e 's/\"early\|\"mid\|\"late/\"{}/g' {}".format(job_set, job_file)
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

                run_job = "kubectl apply -f {}".format(job_file)
                print run_job
                os.system(run_job)

    else:
        # clean up (delete jobs)
        for job_set in sorted(jobs.keys()):
            print job_set
            
            for idx in jobs[job_set]:
                job_name = "dk.job.ggr.dmim.basset.traj-{}".format(idx)
                remove_job = "kubectl delete job {}".format(job_name)
                print remove_job
                os.system(remove_job)
                
    return


main()
