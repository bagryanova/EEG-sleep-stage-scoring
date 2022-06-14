import torch

def scale(X):
    if len(X.shape) == 1:
        X -= torch.mean(X)
        return X / torch.std(X)
    else:
        X -= torch.mean(X, axis=1).view(-1, 1)
        return X / torch.std(X, axis=1).view(-1, 1)
