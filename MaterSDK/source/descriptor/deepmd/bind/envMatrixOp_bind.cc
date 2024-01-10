#include <torch/torch.h>
#include <torch/extension.h>

#include "../include/envMatrixOp.h"


TORCH_LIBRARY(deepmd, m) {
    m.def(
        "EnvMatrixOp",
        [](
            at::Tensor ilist_tensor, 
            at::Tensor numneigh_tensor,
            at::Tensor firstneigh_tensor,
            at::Tensor relative_coords_tensor,
            at::Tensor types_tensor,
            at::Tensor umax_num_neigh_atoms_lst_tensor,
            double rcut,
            double rcut_smooth)
        {
            return matersdk::deepPotSE::EnvMatrixOp(
                ilist_tensor,
                numneigh_tensor,
                firstneigh_tensor,
                relative_coords_tensor,
                types_tensor,
                umax_num_neigh_atoms_lst_tensor,
                rcut,
                rcut_smooth);
        }
    );
}
