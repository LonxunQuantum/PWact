import os
import signal
import subprocess
from subprocess import Popen, PIPE
import time
from pwact.active_learning.slurm.slurm import scancle_byjobid
def kill_process(pid:str, job_id:str):
    if job_id is None:
        try:
            # 发送终止信号 (SIGTERM) 给指定的进程
            os.kill(int(pid), signal.SIGTERM)
            print(f"process {pid} has been terminated.")
        except ProcessLookupError:
            print(f"process {pid} non-existent.")
        except PermissionError:
            print(f"No permission to terminate the process {pid}.")
        except Exception as e:
            print(f"Error terminating process {pid}: {e}")
    else:
        scancle_byjobid(job_id)

def get_pid():
    pid = os.getpid()
    job_id = os.getenv('SLURM_JOB_ID')
    return "pid {} job {}".format(pid, job_id) if job_id is not None else "pid {}".format(pid)

