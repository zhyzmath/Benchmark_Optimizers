import torch
import torch.optim
from muon import SingleDeviceMuon # Assuming muon.py is in the same directory or installed

class Shampoo(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, eps=1e-8, momentum=0.9):
        defaults = dict(lr=lr, eps=eps, momentum=momentum)
        super().__init__(params, defaults)
        self.G_left = None
        self.G_right = None
        self.momentum_buffers = {}

    @staticmethod
    def fractional_matrix_power(mat, power):
        eigvals, eigvecs = torch.linalg.eigh(mat)
        return eigvecs @ torch.diag(eigvals.clamp(min=1e-8).pow(power)) @ eigvecs.t()

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if self.G_left is None:
                    self.G_left = torch.eye(p.shape[0], device=p.device) * group['eps']
                    self.G_right = torch.eye(p.shape[1], device=p.device) * group['eps']
                self.G_left += grad @ grad.t()
                self.G_right += grad.t() @ grad
                inv_left = self.fractional_matrix_power(self.G_left, -0.25)
                inv_right = self.fractional_matrix_power(self.G_right, -0.25)
                update = inv_left @ grad @ inv_right
                if p not in self.momentum_buffers:
                    self.momentum_buffers[p] = torch.zeros_like(update)
                self.momentum_buffers[p] = group['momentum'] * self.momentum_buffers[p] + (1 - group['momentum']) * update
                p.add_(self.momentum_buffers[p], alpha=-group['lr'])

class OneSidedShampoo(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, eps=1e-8, momentum=0.9):
        defaults = dict(lr=lr, eps=eps, momentum=momentum)
        super().__init__(params, defaults)
        self.G = None
        self.momentum_buffers = {}

    @staticmethod
    def fractional_matrix_power(mat, power):
        eigvals, eigvecs = torch.linalg.eigh(mat)
        return eigvecs @ torch.diag(eigvals.clamp(min=1e-8).pow(power)) @ eigvecs.t()

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                m, n = grad.shape
                if m <= n:
                    # Only do Shampoo on the left
                    if self.G is None:
                        self.G = torch.eye(m, device=p.device) * group['eps']
                    self.G += grad @ grad.t()
                    inv_left = self.fractional_matrix_power(self.G, -0.5)
                    update = inv_left @ grad
                else:
                    # Only do Shampoo on the right
                    if self.G is None:
                        self.G = torch.eye(n, device=p.device) * group['eps']
                    self.G += grad.t() @ grad
                    inv_right = self.fractional_matrix_power(self.G, -0.5)
                    update = grad @ inv_right
                if p not in self.momentum_buffers:
                    self.momentum_buffers[p] = torch.zeros_like(update)
                self.momentum_buffers[p] = group['momentum'] * self.momentum_buffers[p] + (1 - group['momentum']) * update
                p.add_(self.momentum_buffers[p], alpha=-group['lr'])

class SignGradientDescent(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, eps=1e-8):
        defaults = dict(lr=lr, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: # Skip if no gradient
                    continue
                grad = p.grad
                # Use sign of gradient for update
                update = torch.sign(grad)
                # Apply learning rate and update parameters
                p.add_(update, alpha=-group['lr'])


class NormalizedGradientDescent(torch.optim.Optimizer):
    """L-infinity normalized gradient descent with momentum"""
    def __init__(self, params, lr=0.01, eps=1e-8, momentum=0.9):
        defaults = dict(lr=lr, eps=eps, momentum=momentum)
        super().__init__(params, defaults)
        self.momentum_buffers = {}

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: # Skip if no gradient
                    continue
                grad = p.grad
                # Divide by the infinity norm (max absolute value) of the gradient
                grad_norm_inf = grad.abs().max()
                if grad_norm_inf > group['eps']:
                    update = grad / grad_norm_inf
                else:
                    update = torch.zeros_like(grad) # No update for very small gradients
                if p not in self.momentum_buffers:
                    self.momentum_buffers[p] = torch.zeros_like(update)
                self.momentum_buffers[p] = group['momentum'] * self.momentum_buffers[p] + (1 - group['momentum']) * update
                p.add_(self.momentum_buffers[p], alpha=-group['lr'])