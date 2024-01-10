import os
from pathlib import Path
import torch
import sys

matersdk_root_dir:str = Path(__file__).parent.absolute().parent.absolute().parent.absolute()
matersdk_source_build_lib_dir:str = os.path.join(matersdk_root_dir, "source", "build", "lib")

### Part 1 . nblist
nblist_bind_dir:str = os.path.join(matersdk_root_dir, "source", "nblist", "bind")
nblist_bind_gen_dir:str = os.path.join(nblist_bind_dir, "gen")
sys.path.append(nblist_bind_gen_dir)
# import nblist
import nblist


### Part 2. deepmd
deepmd_lib_dir:str = os.path.join(matersdk_source_build_lib_dir, "descriptor", "deepmd")
envMatrixOp_bind_so_path:str = os.path.join(deepmd_lib_dir, "libenvMatrixOp_bind.so")
torch.ops.load_library(envMatrixOp_bind_so_path)
# name `envMatrixOp`
envMatrixOp = torch.ops.deepmd.EnvMatrixOp
