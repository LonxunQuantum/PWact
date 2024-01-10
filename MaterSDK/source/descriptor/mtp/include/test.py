from typing import List


def calc_level(mju:int, nju:int):
    return (2 + 4*mju + nju)


def find_combination(num_M:int, max_level:int, level:int=0, combination:List[int]=None):
    if combination is None:
        combination = []
        
    if (num_M == 0):
        if (level <= max_level):
            print(f"Combination: {combination}, level = {level}")
        return 
    
    for ii in range(max_level+1):
        for jj in range(max_level+1):
            new_level = level + calc_level(ii, jj)
            if (new_level <= max_level):
                new_combination = combination + [(ii, jj)]
                find_combination(num_M-1, max_level, new_level, new_combination)



if __name__ == "__main__":
    num_M = 3
    max_level = 8
    find_combination(num_M, max_level)