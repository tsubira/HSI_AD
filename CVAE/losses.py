import torch
import numpy as np
from torchmetrics.functional import spectral_angle_mapper

def kl_loss(mu, logvar):
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    if torch.isinf(kl):
        kl = 10e8
    return kl

def kl_loss_mean(mu, logvar):
    kl_mean = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if torch.isinf(kl_mean):
        kl_mean = 10e8
    return kl_mean

def l2_error_matrix(x, x_hat):
    # loss = torch.nn.MSELoss(reduction='none')
    # return loss(x_hat, x)
    return torch.mean((x_hat - x)**2)

def weight_map(x, x_hat):
    # return torch.max(l2_error_matrix(x, x_hat)) - l2_error_matrix(x, x_hat)
    return torch.max(torch.linalg.norm(x-x_hat, dim=1)) - torch.linalg.norm(x-x_hat, dim=1)

def weighted_MSE(x, x_hat, weights):
    # print(f'shape of X = {x.size()}')
    # print(f'shape of X_hat = {x_hat.size()}')
    # print(f'shape of weights = {weights.size()}')
    norm = torch.linalg.norm((x-x_hat)*weights, dim=1)
    # print(norm)
    return torch.sum(norm)

def SAD(x, x_hat):
    # product = torch.matmul(x,x_hat)
    # print(f'product = {product.size()}')
    # norm_product = (torch.linalg.norm(x, dim=0, ord=2)*torch.linalg.norm(x_hat, dim=0, ord=2))
    # print(f'norm_product = {norm_product.size()}')
    # theta = product/norm_product
    # print(f'theta = {theta.size()}')
    # clamped_theta = torch.clamp(theta, min=-1+1e-7, max=1-1e-7)
    # arccos_theta = torch.arccos(clamped_theta)
    # print(f'arccos_theta = {arccos_theta.size()}')
    # return torch.sum(arccos_theta)
    return spectral_angle_mapper(x_hat,x)

def vector_SAD(x, x_hat):
    batch_SAD = torch.zeros(x.size(0))
    # Compute the SAD for every vector in the batch
    for vector_id in np.arange(x.size(0)):
        vector = x[vector_id, :]
        vector_hat = x_hat[vector_id, :]
        theta = torch.dot(vector,vector_hat) / (torch.linalg.norm(vector) * torch.linalg.norm(vector_hat))
        batch_SAD[vector_id] = torch.arccos(theta)

    # Return the matrix with SAD values
    return torch.mean(batch_SAD)