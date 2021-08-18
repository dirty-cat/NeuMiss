"""
Implements the NeuMiss network
"""

import torch
import torch.nn as nn


class NeuMiss(nn.Module):

    """
    The NeuMiss layer

    Parameters
    ----------

    in_features: int
        Size of each input sample.

    mode: str
        One of:
        * 'shared': The weight matrices for the Neumann iteration are shared.
        * 'shared_accelerated': The weight matrices for the Neumann iteration
        are shared and one coefficient per residual connection can be learned
        for acceleration.

    depth: int
        The number of Neumann iterations.

    """

    def __init__(self, in_features, mode, depth):
        super().__init__()
        self.in_features = in_features
        self.mode = mode
        self.depth = depth
        self.relu = nn.ReLU()

        # Create the parameters of the network
        W = torch.empty(in_features, in_features, dtype=torch.double)
        Wc = torch.empty(in_features, in_features, dtype=torch.double)
        mu = torch.empty(in_features, dtype=torch.double)

        beta = torch.empty(1 * in_features, dtype=torch.double)
        b = torch.empty(1, dtype=torch.double)
        coefs = torch.ones(self.depth + 1, dtype=torch.double)

        # Initialize randomly the parameters of the network
        nn.init.xavier_normal_(W)
        nn.init.xavier_normal_(Wc)
        nn.init.normal_(beta)
        nn.init.normal_(mu)
        nn.init.normal_(b)

        # Make tensors learnable parameters
        self.W = torch.nn.Parameter(W)
        self.Wc = torch.nn.Parameter(Wc)
        self.beta = torch.nn.Parameter(beta)
        self.mu = torch.nn.Parameter(mu)
        self.b = torch.nn.Parameter(b)
        self.coefs = torch.nn.Parameter(coefs)

        if mode != 'shared_accelerated':
            self.coefs.requires_grad = False

    def forward(self, x, m):
        """
        Parameters:
        ----------
        x: tensor, shape (batch_size, n_features)
            The input data imputed by 0.
        m: tensor, shape (batch_size, n_features)
            The missingness indicator (0 if observed and 1 if missing).
        """

        h = x - (1 - m) * self.mu
        h = h * self.coefs[0]

        for i in range(self.depth):
            h = torch.matmul(h, self.W) * (1 - m)

        y = torch.matmul(h, self.beta)

        y = y + self.b

        return y
