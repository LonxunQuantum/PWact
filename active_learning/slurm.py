from enum import Enum
from subprocess import Popen, PIPE
import os
import sys
import time
import shutil

class JobStatus (Enum) :
    unsubmitted = 1 #
    waiting = 2 # PD
    running = 3 # R
    terminated = 4
    finished = 5
    unknown = 100
    resubmit_failed = 6
    submit_limit:int = 3

def get_slurm_sbatch_cmd(job_dir:str, job_name:str):
    cmd = "cd {} && sbatch {}".format(job_dir, job_name)
    return cmd

class SlurmJob(object):
    def __init__(self, job_id=None, status=JobStatus.unsubmitted, user=None, name=None, nodes=None, nodelist=None, partition=None) -> None:
        self.job_id = job_id
        self.status = status
        self.user = user
        self.name = name
        self.partition=partition
        self.nodes = nodes
        self.nodelist = nodelist
        self.submit_num = 0
        
    def set_cmd(self, submit_cmd, slurm_job_run_dir):
        #such as "sbatch main_MD_test.sh"
        self.slurm_job_run_dir = slurm_job_run_dir
        self.submit_cmd = submit_cmd
    
    def set_tag(self, tag):
        self.job_finish_tag = tag

    def submit(self):
        # ret = Popen([self.submit_cmd + " " + self.job_script], stdout=PIPE, stderr=PIPE, shell = True)
        ret = Popen([self.submit_cmd], stdout=PIPE, stderr=PIPE, shell = True)
        stdout, stderr = ret.communicate()
        if str(stderr, encoding='ascii') != "":
            raise RuntimeError (stderr)
        job_id = str(stdout, encoding='ascii').replace('\n','').split()[-1]
        self.job_id = job_id
        self.submit_num += 1
        status = self.update_status()
        print ("# job {} submitted, status is {}".format(self.job_id, status))

    def update_status(self):
        self.status = self.check_status()
        return self.status

    def check_status (self):
        ret = Popen (["squeue --job " + self.job_id], shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = ret.communicate()
        if (ret.returncode != 0) :
            if str("Invalid job id specified") in str(stderr, encoding='ascii') :
                if os.path.exists (self.job_finish_tag) :
                    return JobStatus.finished
                else :
                    return JobStatus.terminated
            else :
                print("status command " + "squeue" + " fails to execute")
                print("erro info: " + str(stderr, encoding='ascii'))
                print("return code: " + str(ret.returncode))
                sys.exit ()
        status_line = str(stdout, encoding='ascii').split ('\n')[-2]
        status_word = status_line.split ()[4]
        if      status_word in ["PD","CF","S"] :
            return JobStatus.waiting
        elif    status_word in ["R","CG"] :
            return JobStatus.running
        elif    status_word in ["C","E","K","BF","CA","CD","F","NF","PR","SE","ST","TO"] :
            if os.path.exists (self.job_finish_tag) :
                return JobStatus.finished
            else :
                return JobStatus.terminated
        else :
            return JobStatus.unknown

    def running_work(self):
        self.submit()
        while True:
            status = self.check_status()
            if (status == JobStatus.waiting) or \
                (status == JobStatus.running):
                time.sleep(10)
            else:
                break
        
        assert(status == JobStatus.finished)
        return status

class Mission(object):
    def __init__(self, mission_id=None) -> None:
        self.mission_id = mission_id
        self.job_list: list[SlurmJob]= []
    
    def add_job(self, job:SlurmJob):
        self.job_list.append(job)

    def pop_job(self, job_id):
        del_job, index = self.get_job(job_id)
        self.job_list.remove(del_job)

    def get_job(self, job_id):
        for i, job in enumerate(self.job_list):
            if job.job_id == job_id:
                return job, i

    def update_job_state(self, job_id, state):
        up_job, index = self.get_job(job_id)
        up_job.status = state
        self.job_list[index] = up_job
    
    def get_running_jobs(self):
        job_list: list[SlurmJob] = []
        for job in self.job_list:
            if (job.status == JobStatus.waiting) or (job.status == JobStatus.running):
                job_list.append(job)
        return job_list

    def move_slurm_log_to_slurm_work_dir(self, slurm_log_dir_source:str):
        for job in self.job_list:
            slurm_log_source = os.path.join(slurm_log_dir_source, "slurm-{}.out".format(job.job_id))
            slurm_job_log_target = os.path.join(os.path.dirname(job.slurm_job_path), os.path.basename(slurm_log_source))
            if os.path.exists(slurm_log_source):
                shutil.move(slurm_log_source, slurm_job_log_target)
                
    '''
    Description: 
    job_finish_tag does not exist means this job running in error 
    param {*} self
    Returns: 
    Author: WU Xingxing
    '''
    def get_error_jobs(self):
        job_list: list[SlurmJob] = []
        for job in self.job_list:
            if os.path.exists(job.job_finish_tag) is False:
                job_list.append(job)
        return job_list
    
    def all_job_finished(self):
        error_jobs = self.get_error_jobs()
        if len(error_jobs) >= 1:
            error_log_content = ""
            for error_job in error_jobs:
                error_log_path = os.path.join(error_job.slurm_job_run_dir, "slurm-{}.out".format(error_job.job_id))
                error_log_content += "ERRIR! The cmd '{}' failed! Please check the slurm log file for more information: {}!\n\n".format(\
                    error_job.submit_cmd, error_log_path)
            raise Exception(error_log_content)
        return True
    
    def commit_jobs(self):
        for job in self.job_list:
            if job.status == JobStatus.unsubmitted:
                    job.submit()
    
    '''
    description: 
        return all job ids, the job id is the slurm job id
    param {*} self
    return {*}
    author: wuxingxing
    '''    
    def get_all_job_ids(self):
        job_id_list = []
        for job in self.job_list:
            job_id_list.append(job.job_id)
        return job_id_list
    
    def check_running_job(self):
        while True:
            for job in self.job_list:
                if job.status == JobStatus.resubmit_failed: # For job resubmitted more than 3 times, do not check again
                    continue
                status = job.check_status()
                self.update_job_state(job.job_id, status)
            # if the job failed, resubmit it until the resubmit time more than 3 times
            self.resubmit_jobs()
            if len(self.get_running_jobs()) == 0:
                break
            time.sleep(10)
        # error_jobs = self.get_error_jobs()
        # if len(error_jobs) > 0:
        #     error_info = "job error: {}".format([_.job_id for _ in error_jobs])
        #     raise Exception(error_info)
        return True
    
    def resubmit_jobs(self):
        for job in self.job_list:
            if job.status == JobStatus.terminated:
                if job.submit_num <= JobStatus.submit_limit.value:
                    print("resubmit job: {}, the time is {}\n".format(job.submit_cmd, job.submit_num))
                    job.submit()
                else:
                    job.status = JobStatus.resubmit_failed
                    slurm_name = "slurm-{}.out".format(job.job_id)
                    print("Error! The job '{}' has been resubmitted more than {} times but still fialed, please check {} file for more information!".\
                        format(job.submit_cmd, JobStatus.submit_limit.value, slurm_name))
                     
                
    '''
    Description: 
    after some jobs finished with some jobs terminated, we should try to recover these terminated jobs.
    param {*} self
    Returns: 
    Author: WU Xingxing
    '''
    def re_submmit_terminated_jobs(self):
        error_jobs = self.get_error_jobs()
        if len(error_jobs) == 0:
            return
        self.job_list.clear()
        self.job_list.extend(error_jobs)
        self.reset_job_state()
        self.commit_jobs()
        self.check_running_job()

    def reset_job_state(self):
        for job in self.job_list:
            job.status == JobStatus.unsubmitted
