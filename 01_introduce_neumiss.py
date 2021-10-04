import torch
from typing import Tuple
from pathlib import Path

from neumiss import NeuMiss
import pytorch_lightning as pl

from torch.nn.functional import mse_loss
from torch.utils.data import random_split, DataLoader, TensorDataset


def read_dataset(dataset_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reads data from a file.
    We expect it has a CSV format and for each line, all the elements up
    to the last constitute x, and the very last is y.
    """
    with dataset_path.open(mode='r') as fl:
        xs = []
        ys = []
        for line in fl.readlines():
            line_parts = list(map(float, line.split(',')))
            y = line_parts.pop(-1)
            x = line_parts
            xs.append(x)
            ys.append(y)
        return torch.Tensor(xs), torch.Tensor(ys)


mnar_X, mnar_y = read_dataset(Path('./MNAR_DATA.csv'))
mcar_X, mcar_y = read_dataset(Path('./MCAR_DATA.csv'))
mar_X, mar_y = read_dataset(Path('./MAR_DATA.csv'))


# Next, we'll create a network for our use:

class NeuMissNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            NeuMiss(in_features=10, depth=10, mode='shared'),
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x_hat = self.model(x)
        loss = mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x_hat = self.model(x)
        loss = mse_loss(x_hat, x)
        self.log('val_loss', loss)


features_count = mnar_X.shape[0] * mnar_X.shape[1]
model = NeuMissNetwork()

test_amount = .25
train_size = round(mnar_X.size()[0] * (1 - test_amount))
test_size = round(mnar_X.size()[0] * test_amount)

mnar_train, mnar_val = random_split(TensorDataset(mnar_X), [train_size, test_size])

# Define loaders
train_loader = DataLoader(mnar_train, batch_size=10)
val_loader = DataLoader(mnar_val, batch_size=10)

# Training
trainer = pl.Trainer(num_nodes=8, limit_train_batches=0.5)
trainer.fit(model, train_loader, val_loader)
