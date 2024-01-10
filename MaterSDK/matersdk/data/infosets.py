import os 
import numpy as np
from typing import List, Union, Optional, Dict
from collections import Counter

from .infoset import InfoSet
from ..io.pwmat.utils.mvextractor import MVExtractor
from ..io.pwmat.utils.parameters import specie2atomic_number, atomic_number2specie


class InfoSets(object):
    def __init__(self, infoset_lst:List[InfoSet]):
        self.infoset_lst = infoset_lst
    
    
    def __str__(self):
        return self.__repr__()
    
    
    def __repr__(self):
        print("{0:*^60}".format(" InfoSets Summary "))
        print("\t+ Total number of {0:<15}: {1:<5}".format("frames", self.get_num_frames()))
        for tmp_infoset in self.infoset_lst:
            print("\t+ Total number of {0:<15}: {1:<5}".format(tmp_infoset.formula, tmp_infoset.num_frames))
        print("*" * 60)
        return ""
    
    
    @classmethod
    def from_dir(cls, dir_path:str, file_name:str, file_format:str):
        # Step 1. Search for all `file_name` in `dir_path`
        file_paths_lst:List[str] = []
        for root, dirs, files in os.walk(dir_path):
            for tmp_file in files:
                if tmp_file.upper() == file_name.upper():
                    file_paths_lst.append( os.path.join(root, tmp_file) )
        
        # Step 2. Get `infosets_atomic_numbers`
        infosets_atomic_numbers:List[int] = []
        infoset_lst:List[InfoSet] = []
        for tmp_file in file_paths_lst:
            infoset_lst.append(InfoSet.from_file(file_path=tmp_file, file_format=file_format))
        for tmp_infoset in infoset_lst:
            for tmp_element in tmp_infoset.formula_dict.keys():
                if not (specie2atomic_number[tmp_element] in infosets_atomic_numbers):
                    infosets_atomic_numbers.append(specie2atomic_number[tmp_element])
        del infoset_lst
        
        # Step 3. Get `infoset_lst`
        infoset_lst:List[InfoSet] = []
        formula_lst:List[str] = []
        for tmp_file in file_paths_lst:
            tmp_infoset = InfoSet.from_file(file_path=tmp_file, file_format=file_format, infosets_atomic_numbers=infosets_atomic_numbers)
            infoset_lst.append(tmp_infoset)
            formula_lst.append(tmp_infoset.formula)
        formula2count:Dict[str, int] = {}
        for tmp_formula, tmp_count in Counter(formula_lst).items():
            formula2count.update({tmp_formula: tmp_count})
        
        # Step 4. Get `reduced_infoset_lst`
        reduced_infoset_lst:List[InfoSet] = []
        for tmp_formula, tmp_fmcount in formula2count.items():
            if tmp_fmcount == 1:
                for tmp_infoset in infoset_lst:
                    if tmp_infoset.formula == tmp_formula:
                        reduced_infoset_lst.append(tmp_infoset)
            else:
                box_lst:List[np.ndarray] = []
                types_lst:List[np.ndarray] = []
                coord_lst:List[np.ndarray] = []
                etot_lst:List[np.ndarray] = []
                fatom_lst:List[np.ndarray] = []
                virial_mark_lst:List[bool] = []
                eatom_mark_lst:List[bool] = []
                magmom_mark_lst:List[bool] = []
                virial_lst:List[np.ndarray] = []
                eatom_lst:List[np.ndarray] = []
                magmom_lst:List[np.ndarray] = []
                # formula_dict
                # formula
                num_frames_lst:List[int] = []
                for tmp_infoset in infoset_lst:
                    if tmp_infoset.formula == tmp_formula:
                        box_lst.append(tmp_infoset.box)
                        types_lst.append(tmp_infoset.types)
                        coord_lst.append(tmp_infoset.coord)
                        etot_lst.append(tmp_infoset.etot)
                        fatom_lst.append(tmp_infoset.fatom)
                        virial_mark_lst.append(tmp_infoset.virial_mark)
                        eatom_mark_lst.append(tmp_infoset.eatom_mark)
                        magmom_mark_lst.append(tmp_infoset.magmom_mark)
                        virial_lst.append(tmp_infoset.virial)
                        eatom_lst.append(tmp_infoset.eatom)
                        magmom_lst.append(tmp_infoset.magmom)
                        num_frames_lst.append(tmp_infoset.num_frames)
                parameters:List[str, np.ndarray] = {}
                parameters.update({"box": np.concatenate(box_lst, axis=0)})
                parameters.update({"types": np.concatenate(types_lst, axis=0)})
                parameters.update({"coord": np.concatenate(coord_lst, axis=0)})
                parameters.update({"etot": np.concatenate(etot_lst, axis=0)})
                parameters.update({"fatom": np.concatenate(fatom_lst, axis=0)})
                parameters.update({"formula": tmp_formula})
                parameters.update({"formula_dict": tmp_infoset.formula_dict})
                parameters.update({"num_frames": sum(num_frames_lst)})
                if not False in virial_mark_lst:
                    parameters.update({"virial": np.concatenate(virial_lst, axis=0)})
                if not False in eatom_mark_lst:
                    parameters.update({"eatom": np.concatenate(eatom_lst, axis=0)})
                if not False in magmom_mark_lst:
                    parameters.update({"magmom": np.concatenate(magmom_lst, axis=0)})
                reduced_infoset_lst.append(InfoSet(**parameters))        
        
        infosets = cls(reduced_infoset_lst)
        # Output infosets
        print(infosets)
        
        return infosets
        
        
    def get_num_frames(self):
        num_frames = 0
        for tmp_infoset in self.infoset_lst:
            num_frames += tmp_infoset.num_frames
        return num_frames


    def to_dir(self, dir_path:str, part_size:Union[int, bool]=False):
        if part_size is False:
            part_size = self.get_num_frames()
        for tmp_infoset in self.infoset_lst:
            tmp_infoset.to_dir(dir_path=dir_path, part_size=part_size)
        # Output infosets
        print(self)