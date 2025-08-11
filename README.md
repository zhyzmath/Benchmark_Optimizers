# Benchmarking Optimizers

This repository compares a set of modern optimizers on several synthetic matrix objectives. It logs and visualizes loss curves to provide a practical comparison across optimizers.

### Optimization tasks covered

- **Linear regression with SPD preconditioning**:  
  ```math
  L(X) = (X - X_*)^T H (X - X_*)
  ```
  Function: `linear_regression_loss(X, X_star, H)`

- **Low-rank matrix completion (masked Frobenius)**:  
  ```math
  L(X, Y) = \frac{\|A \odot (X Y^T - M_*)\|_F^2}{\|A\|_F^2}
  ```
  Function: `lowrank_matrix_completion_loss(X, Y, A_mask, M_star)`

- **Low-rank matrix completion with nuclear-norm residual**:  
  ```math
  L(X, Y) = \frac{\|A \odot (X Y^T - M_*)\|_{\mathrm{nuc}}^2}{\|A\|_{\mathrm{nuc}}^2}
  ```
  Function: `lowrank_matrix_completion_loss_with_nuc_norm(X, Y, A_mask, M_star)`

- **Matrix quadratic regression**:  
  ```math
  L(X) = \frac{1}{2} \|A X B - C\|_F^2
  ```
  Function: `matrix_quadratic_regression_loss(X, A, B, C)`



### Optimizers implemented
- **SOAP**: Adam-style updates in the eigen-bases of Shampoo-like preconditioners with periodic preconditioning and dimension merging options. Class: `optimizers.soap.SOAP`.
- **Shampoo (full)**: Two-sided matrix preconditioning with momentum. Class: `optimizers.optimizers.Shampoo`.
- **One-Sided Shampoo**: Shape-aware left/right preconditioning. Class: `optimizers.optimizers.OneSidedShampoo`.
- **Muon family (momentum orthogonalized by Newton–Schulz)**: An optimizer for hidden layers in neural networks.
  - Single-device: `optimizers.muon.SingleDeviceMuon`
  - Distributed: `optimizers.muon.Muon`
  - With auxiliary Adam for incompatible params: `optimizers.muon.MuonWithAuxAdam`, `optimizers.muon.SingleDeviceMuonWithAuxAdam`
- **Sign Gradient Descent**: Uses `sign(grad)` for updates. Class: `optimizers.optimizers.SignGradientDescent`.
- **Normalized Gradient Descent**: L∞-normalized gradients with momentum. Class: `optimizers.optimizers.NormalizedGradientDescent`.
- **Adam**: A standard first-order optimizer that computes adaptive learning rates for each parameter. Implemented as `torch.optim.Adam`.
- **AdamW**: A variant of Adam that fixes weight decay regularization. Implemented as `torch.optim.AdamW`.
- **Adagrad**: An optimizer that adapts the learning rate to parameters, performing smaller updates for more frequent features. Implemented as `torch.optim.Adagrad`.

### References
The implementations of SOAP and Muon are based on their official repositories:
- **SOAP**: [https://github.com/nikhilvyas/SOAP](https://github.com/nikhilvyas/SOAP)
- **Muon**: [https://github.com/KellerJordan/Muon](https://github.com/KellerJordan/Muon)
