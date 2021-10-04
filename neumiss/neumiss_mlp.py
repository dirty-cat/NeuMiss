"""
Implements NeuMissMLP, a scikit-learn friendly NeuMiss interface.
"""

import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator

import torch
import torch.nn as nn
import torch.optim as optim
from .pytorchtools import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .neumiss_layer import NeuMiss


class NeuMissMLP(BaseEstimator):
    """
    An interface for NeuMiss, adopting the scikit-learn API conventions
    (fit, predict, ...).

    Parameters
    ----------

    mode: str
        One of:
        * 'shared': The weight matrices for the Neumann iteration are shared.
        * 'shared_accelerated': The weight matrices for the Neumann iteration
        are shared and one coefficient per residual connection can be learned
        for acceleration.

    depth: int
        The number of Neumann iterations.

    n_epochs: int, default=1000
        The maximum number of epochs.

    batch_size: int, default=100

    lr: float, default=0.01
        The learning rate.

    early_stopping: boolean, default=False
        If True, early stopping is used based on the validation set, with a
        patience of 15 epochs.

    optimizer_name: str, default='sgd'
        One of `sgd` or `adam`.

    verbose: boolean, default=False
        If True, prints additional information
    """

    def __init__(self,
                 mode: str,
                 depth: int,
                 n_epochs: int = 1000,
                 batch_size: int = 100,
                 lr: float = 0.01,
                 early_stopping: bool = False,
                 optimizer_name: str = 'sgd',
                 verbose: bool = False,
                 ):
        self.mode = mode
        self.depth = depth
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.early_stopping = early_stopping
        self.optimizer_name = optimizer_name
        self.optimizer = None
        self.scheduler = None
        self.verbose = verbose

    def fit(self,
            X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None):

        self.r2_train_ = []
        self.mse_train_ = []
        self.r2_values_ = []
        self.mse_values_ = []

        # Get missing values mask of X
        M = np.isnan(X)
        # Impute ?
        X = np.nan_to_num(X)

        n_samples, n_features = X.shape

        if X_val is not None:
            M_val = np.isnan(X_val)
            X_val = np.nan_to_num(X_val)

            M_val = torch.as_tensor(M_val, dtype=torch.double)
            X_val = torch.as_tensor(X_val, dtype=torch.double)
            y_val = torch.as_tensor(y_val, dtype=torch.double)

        M = torch.as_tensor(M, dtype=torch.double)
        X = torch.as_tensor(X, dtype=torch.double)
        y = torch.as_tensor(y, dtype=torch.double)

        # Create the network
        self.net = NeuMiss(
            in_features=n_features, mode=self.mode, depth=self.depth
        )

        if len(list(self.net.parameters())) > 0:
            if self.optimizer_name == 'sgd':
                self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr,
                                           weight_decay=1e-4)
            elif self.optimizer_name == 'adam':
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr,
                                            weight_decay=1e-4)
            else:
                raise ValueError("Invalid optimizer specified. "
                                 "Expected any of {'sgd', 'adam'},"
                                 f"got {self.optimizer_name!r}")

            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min',
                                               factor=0.2, patience=2,
                                               threshold=1e-4)

        early_stopper = EarlyStopping(verbose=self.verbose)

        running_loss = np.inf
        criterion = nn.MSELoss()

        # Train the network
        for i_epoch in range(self.n_epochs):
            if self.verbose:
                print(f"Epoch number {i_epoch}")

            # Shuffle tensors to have different batches at each epoch
            ind = torch.randperm(n_samples)
            X = X[ind]
            M = M[ind]
            y = y[ind]

            xx = torch.split(X, split_size_or_sections=self.batch_size, dim=0)
            mm = torch.split(M, split_size_or_sections=self.batch_size, dim=0)
            yy = torch.split(y, split_size_or_sections=self.batch_size, dim=0)

            self.scheduler.step(running_loss / len(xx))

            param_group = self.optimizer.param_groups[0]
            learning_rate = param_group['lr']
            if self.verbose:
                print(f"Current learning rate is: {learning_rate}")
            if learning_rate < 5e-6:
                break

            running_loss = 0

            for bx, bm, by in zip(xx, mm, yy):

                self.optimizer.zero_grad()

                y_pred = self.net(bx, bm)

                loss = criterion(y_pred, by)
                running_loss += loss.item()
                loss.backward()

                # Take gradient step
                self.optimizer.step()

            # Evaluate the train loss
            with torch.no_grad():
                y_pred = self.net(X, M, phase='test')
                loss = criterion(y_pred, y)
                mse = loss.item()
                self.mse_train_.append(mse)

                var = ((y - y.mean())**2).mean()
                r2 = 1 - mse/var
                self.r2_train_.append(r2)

                if self.verbose:
                    print(f"Train loss - r2: {r2}, "
                          f"mse: {running_loss / len(xx)}")

            # Evaluate the validation loss
            if X_val is not None:
                with torch.no_grad():
                    y_pred = self.net(X_val, M_val, phase='test')
                    loss_value = criterion(y_pred, y_val)
                    mse_value = loss_value.item()
                    self.mse_values_.append(mse_value)

                    var = ((y_val - y_val.mean()) ** 2).mean()
                    r2_value = 1 - (mse_value / var)
                    self.r2_values_.append(r2_value)
                    if self.verbose:
                        print(f"Validation loss is: {r2_value}")

                if self.early_stopping:
                    early_stopper(mse_value, self.net)
                    if early_stopper.early_stop:
                        if self.verbose:
                            print("Early stopping")
                        break

        # load the last checkpoint with the best model
        if self.early_stopping and early_stopper.early_stop:
            self.net.load_state_dict(early_stopper.checkpoint)

    def predict(self, X):

        M = np.isnan(X)
        X = np.nan_to_num(X)

        M = torch.as_tensor(M, dtype=torch.double)
        X = torch.as_tensor(X, dtype=torch.double)

        with torch.no_grad():
            y_hat = self.net(X, M, phase='test')

        return np.array(y_hat)
