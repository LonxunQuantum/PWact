#include <stdexcept>
#include "../include/envMatrixOp.h"


namespace matersdk {
namespace deepPotSE {

/**
 * @brief Process structure batch by batch. 
 *        Note: `batch_size` maybe larger than 1.
 * 
 * @param ctx torch::autograd::AutogradContext* 
 * @param ilist_tensor .size() = (batch_size, num_atoms)
 * @param numneigh_tensor .size() = (batch_size, num_atoms)
 * @param firstneigh_tensor .size() = (batch_size, num_atoms, umax_num_neigh_atoms)
 * @param relative_coords_tensor .size() = (batch_size, num_atoms, umax_num_neigh_atoms, 3)
 * @param types_tensor .size() = (batch_size, num_atoms + num_ghost)
 * @param umax_num_neigh_atoms_lst_tensor .size() = (batch_size, ntypes)
 * @param rcut 
 * @param rcut_smooth 
 * @return torch::autograd::variable_list 
 */
torch::autograd::variable_list EnvMatrixFunction::forward(
    torch::autograd::AutogradContext* ctx,
    at::Tensor ilist_tensor,
    at::Tensor numneigh_tensor,
    at::Tensor firstneigh_tensor,
    at::Tensor relative_coords_tensor,  // float
    at::Tensor types_tensor,
    at::Tensor umax_num_neigh_atoms_lst_tensor,
    double rcut,
    double rcut_smooth)
{   
    // Step 1. 
    // Step 1.1. Some assert
    assert(ilist_tensor.scalar_type() == torch::kInt32);
    assert(numneigh_tensor.scalar_type() == torch::kInt32);
    assert(firstneigh_tensor.scalar_type() == torch::kInt32);
    assert(types_tensor.scalar_type() == torch::kInt32);
    assert(umax_num_neigh_atoms_lst_tensor.scalar_type() == torch::kInt32);
    assert(
        (relative_coords_tensor.scalar_type() == torch::kFloat32) 
        | (relative_coords_tensor.scalar_type() == torch::kFloat64));
    // Step 1.2. ctx->save_for_backward({...})
    ctx->save_for_backward({
        at::Tensor(), at::Tensor(), at::Tensor(),
        at::Tensor(), at::Tensor(), at::Tensor(),
        at::Tensor(), at::Tensor()});
    // Setp 1.3. Get `batch_size`, ...
    int batch_size = (int)ilist_tensor.size(0);    // long -> int
    int inum = (int)ilist_tensor.size(1);   // long -> int
    int umax_num_neigh_atoms = umax_num_neigh_atoms_lst_tensor.sum(1)[0].item<int>();
    int ntypes = (int)umax_num_neigh_atoms_lst_tensor.size(1);  // long -> int
    
    // Step 2. Init return variables
    c10::Device device = relative_coords_tensor.device();
    c10::TensorOptions float_options;
    at::Tensor tilde_r_tensor;
    at::Tensor tilde_r_deriv_tensor;
    if (relative_coords_tensor.scalar_type() == torch::kFloat32) {
        float_options = c10::TensorOptions().dtype(torch::kFloat32).device(device);
        tilde_r_tensor = at::zeros(
            {batch_size, inum, umax_num_neigh_atoms, 4}, 
            float_options);
        tilde_r_deriv_tensor = at::zeros(
            {batch_size, inum, umax_num_neigh_atoms, 4, 3},
            float_options);
    } else {
        float_options = c10::TensorOptions().dtype(torch::kFloat64).device(device);
        tilde_r_tensor = at::zeros(
            {batch_size, inum, umax_num_neigh_atoms, 4},
            float_options);
        tilde_r_deriv_tensor = at::zeros(
            {batch_size, inum, umax_num_neigh_atoms, 4, 3},
            float_options);
    }

    // Step 3. find tilde_r_tensor, tilde_r_deriv_tensor
    for (int ii=0; ii<batch_size; ii++) {
        int* ilist = ilist_tensor[ii].data_ptr<int>();
        int* numneigh = numneigh_tensor[ii].data_ptr<int>();
        int* firstneigh = firstneigh_tensor[ii].data_ptr<int>();
        int* types = types_tensor[ii].data_ptr<int>();
        int* umax_num_neigh_atoms_lst = umax_num_neigh_atoms_lst_tensor[ii].data_ptr<int>();
        
        if (relative_coords_tensor.scalar_type() == torch::kFloat32) {
            float* tilde_r = tilde_r_tensor[ii].data_ptr<float>();
            float* tilde_r_deriv = tilde_r_deriv_tensor[ii].data_ptr<float>();
            float* relative_coords = relative_coords_tensor[ii].data_ptr<float>();
            EnvMatrix<float>::find_value_deriv(
                tilde_r,
                tilde_r_deriv,
                inum,
                ilist,
                numneigh,
                firstneigh,
                relative_coords,
                types,
                ntypes,
                umax_num_neigh_atoms_lst,
                rcut,
                rcut_smooth);            
        } else {
            double* tilde_r = tilde_r_tensor[ii].data_ptr<double>();
            double* tilde_r_deriv = tilde_r_deriv_tensor[ii].data_ptr<double>();
            double* relative_coords = relative_coords_tensor[ii].data_ptr<double>();
            EnvMatrix<double>::find_value_deriv(
                tilde_r,
                tilde_r_deriv,
                inum,
                ilist,
                numneigh,
                firstneigh,
                relative_coords,
                types,
                ntypes,
                umax_num_neigh_atoms_lst,
                rcut,
                rcut_smooth);
        }
    }

    return {tilde_r_tensor, tilde_r_deriv_tensor};
}


torch::autograd::variable_list EnvMatrixFunction::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs)
{
    torch::autograd::variable_list arr = ctx->get_saved_variables();
    return {at::Tensor(), at::Tensor(), at::Tensor(),
            at::Tensor(), at::Tensor(), at::Tensor(),
            at::Tensor(), at::Tensor()
        };
}


torch::autograd::variable_list EnvMatrixOp(
    at::Tensor ilist_tensor,
    at::Tensor numneigh_tensor,
    at::Tensor firstneigh_tensor,
    at::Tensor relative_coords_tensor,
    at::Tensor types_tensor,
    at::Tensor umax_num_neigh_atoms_lst_tensor,
    double rcut,
    double rcut_smooth)
{
    return EnvMatrixFunction::apply(
        ilist_tensor,
        numneigh_tensor,
        firstneigh_tensor,
        relative_coords_tensor,
        types_tensor,
        umax_num_neigh_atoms_lst_tensor,
        rcut,
        rcut_smooth);
}

};  // namespace : deepPotSE
};  // namespace : matersdk