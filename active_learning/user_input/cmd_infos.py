
def cmd_infos():
    cmd_info = ""
    cmd_info += "init_bulk".upper() + "\n"
    cmd_info += "you could use this method to do relax, super cell, scale, pertub and aimd\n\n"
    # cmd_info += "int_surface".upper() + "\n"
    # cmd_info += ""
    cmd_info += "run".upper() + "\n"
    cmd_info += "you could use this method to do active learning sampling\n\n"
    # cmd_info += "init_json".upper() + "\n"

    # cmd_info += "run_json".upper() + "\n"

    cmd_info += "pwdata".upper() + "\n"
    cmd_info += "you could use this method to change outcars or movements to pwdata format, for more detail for this command, you could use 'PWMLFF_AL pwdata -h'\n\n"
    
    cmd_info += "gather_pwdata".upper() + "\n"
    cmd_info += "you could use this method to extract pwdatas after active learing done, for more detail for this command, you could use 'PWMLFF_AL extract_pwdata -h'\n\n"
    
    return cmd_info