import torch

def weighted_MSE(x, x_hat, weights):
    # print(f'shape of X = {x.size()}')
    # print(f'shape of X_hat = {x_hat.size()}')
    # print(f'shape of weights = {weights.size()}')
    norm = torch.linalg.norm((x-x_hat)*weights, dim=1)
    # print(norm)
    return torch.sum(norm)

def weight_map(x, x_hat):
    # return torch.max(l2_error_matrix(x, x_hat)) - l2_error_matrix(x, x_hat)
    return torch.max(torch.linalg.norm(x-x_hat, dim=1)) - torch.linalg.norm(x-x_hat, dim=1)