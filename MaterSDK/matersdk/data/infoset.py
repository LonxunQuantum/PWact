from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional
import numpy as np
import os
import warnings

from ..io.pwmat.utils.mvextractor import MVExtractor
from ..io.pwmat.utils.parameters import atomic_number2specie, specie2atomic_number


class InfoSet(object):
    def __init__(
                self,
                box:np.ndarray,
                types:np.ndarray,
                coord:np.ndarray,
                etot:np.ndarray,
                fatom:np.ndarray,
                formula_dict:Dict[str, int],
                formula:str,
                num_frames:int,
                virial:Union[np.ndarray, bool]=False,
                eatom:Union[np.ndarray, bool]=False,
                magmom:Union[np.ndarray, bool]=False):
        '''
        Description
        -----------
            1. 
        '''
        self.box = box
        self.types = types
        self.coord = coord
        self.etot = etot
        self.fatom = fatom
        if virial is not False:
            self.virial_mark = True
            self.virial = virial
        else:
            self.virial_mark = False
            self.virial = False
        if eatom is not False:
            self.eatom_mark = True
            self.eatom = eatom
        else:
            self.eatom_mark = False
            self.eatom = False
        if magmom is not False:
            self.magmom_mark = True
            self.magmom = magmom
        else:
            self.magmom_mark = False
            self.magmom = False
        self.formula_dict = formula_dict
        self.formula = formula
        self.num_frames = num_frames


    @classmethod
    def from_file(cls, file_path:str, file_format:str, infosets_atomic_numbers:Union[List[int], bool]=False):
        if (file_format.upper() == "PWMAT/MOVEMENT"):
            traj_extractor = MVExtractor(movement_path=file_path)
            virial_mark, eatom_mark, magmom_mark = traj_extractor._find_extra_properties();
        else:
            raise NameError("InfoSetError : No implementation for format of {0}".format(file_format))

        frames_info:List[np.ndarray] = traj_extractor.get_frames_info()
        box:np.ndarray = frames_info[0]
        types:np.ndarray = frames_info[1]  # 此处是atomic number
        coord:np.ndarray = frames_info[2]
        etot:np.ndarray = frames_info[3]
        fatoms:np.ndarray = frames_info[4]
        if virial_mark:
            virial:np.ndarray = frames_info[5]
        if eatom_mark:
            if virial_mark:
                eatoms:np.ndarray = frames_info[6]
            else:
                eatoms:np.ndarray = frames_info[5]
        if magmom_mark:
            if (virial_mark and eatom_mark):
                magmoms:np.ndarray = frames_info[7]
            elif (virial_mark or eatom_mark):
                magmoms:np.ndarray = frames_info[6]
            else:
                magmoms:np.ndarray = frames_info[5]
        
        num_frames:int = frames_info[0].shape[0]
        formula_dict:Dict[str, int] = {}
        formula = ""
        if infosets_atomic_numbers is False:
            # 1. self.formula_dict
            for tmp_type, tmp_count in zip(np.unique(types[0], return_counts=True)[0], np.unique(types[0], return_counts=True)[1]):
                formula_dict.update({atomic_number2specie[tmp_type]: tmp_count})
            infosets_atomic_numbers = [specie2atomic_number[tmp_element] for tmp_element in list(formula_dict.keys())]
        else:
            # 1. Init keys
            for tmp_an in infosets_atomic_numbers:
                formula_dict.update({atomic_number2specie[tmp_an]: 0})
            # 2. Populate 
            for tmp_an in infosets_atomic_numbers:
                for tmp_struct_an in types[0]:
                    if tmp_struct_an == tmp_an:
                        formula_dict[atomic_number2specie[tmp_an]] += 1
        # 2. self.formula
        for k, v in formula_dict.items():
            formula += "{0}{1}".format(k, v)
            
        # 3. self.types
        new_types:List[np.ndarray] = []
        for tmp_frame_idx in range(num_frames):
            new_types.append(
                np.array( [infosets_atomic_numbers.index(tmp_type) for tmp_type in types[tmp_frame_idx]] )
            )
        types = np.array(new_types)    # 此处 starts from `0`
        
        
        parameters = {}
        parameters.update({"box": box})
        parameters.update({"types": types})
        parameters.update({"coord": coord})
        parameters.update({"etot": etot})
        parameters.update({"fatom": fatoms})
        parameters.update({"formula_dict": formula_dict})
        parameters.update({"formula": formula})
        parameters.update({"num_frames": num_frames})
        if virial_mark:
            parameters.update({"virial": virial})
        if eatom_mark:
            parameters.update({"eatom": eatoms})
        if magmom_mark:
            parameters.update({"magmom": magmoms})
        
        info_set = cls(**parameters)
        return info_set

    
    def to_dir(self, dir_path:str, part_size:Union[int, bool]=False):
        if part_size is not False:
            num_parts = int(self.num_frames / part_size) + 1
        else:
            num_parts = 1
            part_size = self.num_frames
        
        if os.path.exists( os.path.join(dir_path, self.formula) ):
            warnings.warn("This dir exists: ".format(os.path.join(dir_path, self.formula)))
        os.mkdir(os.path.join(dir_path, self.formula))
        
        with open(os.path.join(dir_path, self.formula, "type_map.raw"), "w") as f:
            for tmp_element in self.formula_dict.keys():
                f.write(tmp_element)
                f.write("\n")
        
        for part_rank in range(num_parts):
            start:int = part_rank * part_size
            if part_rank == (num_parts-1):
                end:int = self.num_frames
            else:
                end:int = start + part_size
            if (end == start):
                break
                        
            ### Step 1. 
            tmp_part_dir_path:str = os.path.join(dir_path, self.formula, "part.{0:0>3}".format(part_rank))
            if os.path.exists(tmp_part_dir_path):
                warnings.warn("This dir contains part.{0:0>3}".format(part_rank))
            os.mkdir(tmp_part_dir_path)
            np.save(file=os.path.join(tmp_part_dir_path, "box.npy"), arr=self.box[start:end, :])
            np.save(file=os.path.join(tmp_part_dir_path, "types.npy"), arr=self.types[start:end, :])
            np.save(file=os.path.join(tmp_part_dir_path, "coord.npy"), arr=self.coord[start:end, :])
            np.save(file=os.path.join(tmp_part_dir_path, "etot.npy"), arr=self.etot[start:end, :])
            np.save(file=os.path.join(tmp_part_dir_path, "fatom.npy"), arr=self.fatom[start:end, :])
            if self.virial_mark:
                np.save(file=os.path.join(tmp_part_dir_path, "virial.npy"), arr=self.virial[start:end, :])
            if self.eatom_mark:
                np.save(file=os.path.join(tmp_part_dir_path, "eatom.npy"), arr=self.eatom[start:end, :])
            if self.magmom_mark:
                np.save(file=os.path.join(tmp_part_dir_path, "magmoms"), arr=self.magmom[start:end, :])
    