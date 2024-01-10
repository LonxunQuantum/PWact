#include <torch/torch.h>
#include "./envMatrix.h"


namespace matersdk {
namespace deepPotSE {

class EnvMatrixFunction : public torch::autograd::Function<EnvMatrixFunction> {
public: 
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor ilist_tensor,
        at::Tensor numneigh_tensor,
        at::Tensor firstneigh_tensor,
        at::Tensor relative_coords_tensor,
        at::Tensor types_tensor,
        at::Tensor umax_num_neigh_atoms_lst_tensor,
        double rcut,
        double rcut_smooth);

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs);
};  // class : EnvMatrixFunction


torch::autograd::variable_list EnvMatrixOp(
    at::Tensor ilist_tensor,
    at::Tensor numneigh_tensor,
    at::Tensor firstneigh_tensor,
    at::Tensor relative_coords_tensor,
    at::Tensor types_tensor,
    at::Tensor umax_num_neigh_atoms_lst_tensor,
    double rcut,
    double rcut_smooth);

};  // namespace : deepPotSE
};  // namespace : matersdk