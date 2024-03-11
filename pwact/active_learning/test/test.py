import numpy as np
import argparse

def make_kspacing_kpoints(config, kspacing):
    with open(config, "r") as fp:
        lines = fp.read().split("\n")
    box = []
    for idx, ii in enumerate(lines):
        if "LATTICE" in ii.upper():
            for kk in range(idx + 1, idx + 1 + 3):
                vector = [float(jj) for jj in lines[kk].split()[0:3]]
                box.append(vector)
            box = np.array(box)
            rbox = _reciprocal_box(box)
    kpoints = [
        round(2 * np.pi * np.linalg.norm(ii) / kspacing) for ii in rbox
    ]
    kpoints[0] = 1 if kpoints[0] == 0 else kpoints[0]
    kpoints[1] = 1 if kpoints[1] == 0 else kpoints[1]
    kpoints[2] = 1 if kpoints[2] == 0 else kpoints[2]
    ret = ""
    ret += "%d %d %d 0 0 0 " % (kpoints[0], kpoints[1], kpoints[2])
    print(ret)
    return ret
    
def _reciprocal_box(box):
    rbox = np.linalg.inv(box)
    rbox = rbox.T
    return rbox
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--config', help="specify config file path of atom.config", type=str, default='atom.config')
    parser.add_argument('-k', '--kspacing', help="specify the kspacing, the default 0.5", type=float, default=0.5)
    args = parser.parse_args()
    make_kspacing_kpoints(config=args.config, kspacing=args.kspacing)