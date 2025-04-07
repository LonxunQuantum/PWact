import subprocess
import pkg_resources
def check_envs():
    # for pwmat
    comm_info()

def comm_info():
    print("\n" + "=" * 50) 
    print("         PWACT Basic Information")
    print("=" * 50) 
    print("Version: 0.3.1")
    print("Compatible pwdata: >= 0.5.0")
    print("Compatible MatPL: >= 2025.3")
    print("Contact: support@pwmat.com")
    print("Citation: https://github.com/LonxunQuantum/MatPL")
    print("Manual online: http://doc.lonxun.com/PWMLFF/")
    print("=" * 50)  
    print("\n\n")
