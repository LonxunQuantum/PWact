
def cmd_infos(cmd_type=None):
    if cmd_type is None:
        cmd_info = ""
        cmd_info += "init_bulk".lower() + "\n"
        cmd_info += "you could use this method to do relax, super cell, scale, pertub and aimd\n"
        cmd_info += "example: pwact init_bulk init_param.json machine.json\n\n"
        
        # cmd_info += "int_surface".lower() + "\n"
        # cmd_info += ""
        cmd_info += "run".lower() + "\n"
        cmd_info += "you could use this method to do active learning sampling\n"
        cmd_info += "example: pwact run param.json machine.json\n\n"
        # cmd_info += "init_json".lower() + "\n"

        # cmd_info += "run_json".lower() + "\n"

        cmd_info += "kill".lower() + "\n"
        cmd_info += "you could use this method to kill active learning tasks currently being executed\n"
        cmd_info += "example:\n"
        cmd_info += "'pwact kill init_bulk' for 'init_bulk' tasks\n"
        cmd_info += "'pwact kill run' for 'run' tasks\n\n"

        cmd_info += "to_pwdata".lower() + "\n"
        cmd_info += "you could use this method to change outcars or movements to pwdata format.\nFor more detail for this command, you could use 'PWact to_pwdata -h'\n\n"

        cmd_info += "gather_pwdata".lower() + "\n"
        cmd_info += "you could use this method to extract pwdatas after active learing done.\nFor more detail for this command, you could use 'PWact gather_pwdata -h'\n\n"
        
        cmd_info += "\n\n"
        cmd_info += "examples of pwact are located at the dir 'pwact/example'\n"
        cmd_info += "you can alse download them from github link:\n"
        cmd_info += "https://github.com/LonxunQuantum/PWact/tree/main/pwact/example\n\n"
        cmd_info += "or download from BaiduNetdisk included the calculation results of examples:\n"
        cmd_info += "https://pan.baidu.com/s/14E0u_7cpntiBZgg-C1S5XA?pwd=pwmt\n\n"
    elif cmd_type == "init_bulk":
        cmd_info = cmd_info_init_bulk()
    elif cmd_type == "run":
        cmd_info = cmd_info_run_iter()
    elif cmd_type == "kill":
        cmd_info = cmd_info_kill()
    print(cmd_info)


def cmd_info_init_bulk():
    cmd_info = ""
    cmd_info += "init_bulk".lower() + "\n"
    cmd_info += "you could use this method to do relax, super cell, scale, pertub and aimd\n"
    cmd_info += "example: pwact init_bulk init_param.json machine.json\n\n"
    return cmd_info

def cmd_info_run_iter():
    cmd_info = ""
    cmd_info += "run" + "\n"
    cmd_info += "you could use this method to do active learning sampling\n"
    cmd_info += "example: pwact run param.json machine.json\n\n"
    return cmd_info

def cmd_info_kill():
    cmd_info = ""
    cmd_info += "kill".lower() + "\n"
    cmd_info += "you could use this method to kill active learning tasks currently being executed\n"
    cmd_info += "example:\n"
    cmd_info += "'pwact kill init_bulk' for 'init_bulk' tasks\n"
    cmd_info += "'pwact kill run' for 'run' tasks\n\n"
    return cmd_info
