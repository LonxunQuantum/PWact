from pymatgen.io.vasp import Outcar


class DOutcar(Outcar):
    def __init__(self, file_path:str):
        super(DOutcar, self).__init__(file_path)
    
    def get_energy(self):
        return self.final_energy
    
    def get_fr_energy(self):
        return self.final_fr_energy