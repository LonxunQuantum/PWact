from pymatgen.core import Molecule


class DMolecule(Molecule):
    @classmethod
    def from_file(cls,
                file_path:str,
                file_format:str,
                coords_are_cartesian:bool=False):
        if (file_format != "pwmat"):
            molecule = Molecule.from_file(filename=file_path)
        if (file_format == "pwmat"):
            raise ValueError("Not implement!!!")
        
        return molecule 
    
    
    def to(self,
            output_file_path:str,
            output_file_format:str
            ):
        if (output_file_format != "pwmat"):
            super(DMolecule, self).to(
                                fmt=output_file_format,
                                filename=output_file_path)
        
        if (output_file_format == "pwmat"):
            raise ValueError("Not implement!!!");