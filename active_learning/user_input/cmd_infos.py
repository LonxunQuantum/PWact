
def cmd_infos():
    cmd_info = ""
    cmd_info += "init_bulk".upper() + "\n"
    cmd_info += "you could use this method to do relax, super cell, scale, pertub and aimd\n"
    cmd_info += "example: al_pwmlff init_bulk init_param.json machine.json\n\n"
    
    # cmd_info += "int_surface".upper() + "\n"
    # cmd_info += ""
    cmd_info += "run".upper() + "\n"
    cmd_info += "you could use this method to do active learning sampling\n"
    cmd_info += "example: al_pwmlff run param.json machine.json\n\n"
    # cmd_info += "init_json".upper() + "\n"

    # cmd_info += "run_json".upper() + "\n"

    cmd_info += "to_pwdata".upper() + "\n"
    cmd_info += "you could use this method to change outcars or movements to pwdata format.\nFor more detail for this command, you could use 'al_pwmlff to_pwdata -h'\n\n"

    cmd_info += "gather_pwdata".upper() + "\n"
    cmd_info += "you could use this method to extract pwdatas after active learing done.\nFor more detail for this command, you could use 'al_pwmlff gather_pwdata -h'\n\n"
    
    return cmd_info