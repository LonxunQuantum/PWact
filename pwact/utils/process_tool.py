import os
import signal
import subprocess

def kill_process(pid:int):
    try:
        # 发送终止信号 (SIGTERM) 给指定的进程
        os.kill(pid, signal.SIGTERM)
        print(f"process {pid} has been terminated.")
    except ProcessLookupError:
        print(f"process {pid} non-existent.")
    except PermissionError:
        print(f"No permission to terminate the process {pid}.")
    except Exception as e:
        print(f"Error terminating process {pid}: {e}")
