from pandas.core.frame import DataFrame
import os

from pwact.utils.constant import EXPLORE_FILE_STRUCTURE
from pwact.utils.file_operation import write_to_file

def select_image(save_dir:str, devi_pd:DataFrame, lower:float, higer:float, max_select:float):
    accurate_pd  = devi_pd[devi_pd[EXPLORE_FILE_STRUCTURE.devi_columns[0]] < lower]
    candidate_pd = devi_pd[(devi_pd[EXPLORE_FILE_STRUCTURE.devi_columns[0]] >= lower) & (devi_pd[EXPLORE_FILE_STRUCTURE.devi_columns[0]] < higer)]
    error_pd     = devi_pd[devi_pd[EXPLORE_FILE_STRUCTURE.devi_columns[0]] > higer]
    #4. if selected images more than number limitaions, randomly select
    remove_candi = None
    rand_candi = None
    if candidate_pd.shape[0] > max_select:
        rand_candi = candidate_pd.sample(max_select)
        remove_candi = candidate_pd.drop(rand_candi.index)
    
    #5. save select info
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    summary = "Total structures {}    accurate {} rate {:.2f}%    selected {} rate {:.2f}%    error {} rate {:.2f}%\n"\
        .format(devi_pd.shape[0], accurate_pd.shape[0], accurate_pd.shape[0]/devi_pd.shape[0]*100, \
                    candidate_pd.shape[0], candidate_pd.shape[0]/devi_pd.shape[0]*100, \
                        error_pd.shape[0], error_pd.shape[0]/devi_pd.shape[0]*100)

    accurate_pd.to_csv(os.path.join(save_dir, EXPLORE_FILE_STRUCTURE.accurate))
    candi_info = ""
    if rand_candi is not None:
        rand_candi.to_csv(os.path.join(save_dir, EXPLORE_FILE_STRUCTURE.candidate))
        remove_candi.to_csv(os.path.join(save_dir, EXPLORE_FILE_STRUCTURE.candidate_delete))
        candi_info += "Candidate configurations: {}, randomly select {}, delete {}\n        Select details in file {}\n        Delete details in file {}.\n".format(
                candidate_pd.shape[0], rand_candi.shape[0], remove_candi.shape[0],\
                EXPLORE_FILE_STRUCTURE.candidate, EXPLORE_FILE_STRUCTURE.candidate_delete)
    else:
        candidate_pd.to_csv(os.path.join(save_dir, EXPLORE_FILE_STRUCTURE.candidate))
        candi_info += "        Candidate configurations: {}\n    Select details in file {}\n".format(
                candidate_pd.shape[0], EXPLORE_FILE_STRUCTURE.candidate)
            
    error_pd.to_csv(os.path.join(save_dir, EXPLORE_FILE_STRUCTURE.failed))
    
    summary_info = ""

    summary_info += summary
    summary_info += "\nSelect by model deviation force:\n"
    summary_info += "Accurate configurations: {}, details in file {}\n".\
        format(accurate_pd.shape[0], EXPLORE_FILE_STRUCTURE.accurate)
        
    summary_info += candi_info
        
    summary_info += "Error configurations: {}, details in file {}\n".\
        format(error_pd.shape[0], EXPLORE_FILE_STRUCTURE.failed)
    
    write_to_file(os.path.join(save_dir, EXPLORE_FILE_STRUCTURE.select_summary), summary_info, "w")
    return summary_info, summary