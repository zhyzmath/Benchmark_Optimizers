import torch
import numpy as np

def linear_regression_loss(X, X_star, H):
    diff = X - X_star
    return torch.sum(diff * (H @ diff))

def lowrank_matrix_completion_loss(X, Y, A_mask, M_star):
    pred = (X @ Y.t())
    return torch.norm(A_mask * (pred - M_star), p='fro')**2 / torch.norm(A_mask, p='fro')**2

def lowrank_matrix_completion_loss_with_nuc_norm(X, Y, A_mask, M_star):
    pred = (X @ Y.t())
    return torch.norm(A_mask * (pred - M_star), p='nuc')**2 / torch.norm(A_mask, p='nuc')**2

def matrix_quadratic_regression_loss(X, A, B, C):
    return 0.5 * torch.norm(A @ X @ B - C, p='fro')**2
