import os
import math
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import BioactivityDataset, Collater
from smart.optimizer import Lamb
from sklearn.metrics import confusion_matrix, f1_score


class ResBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.resweight = nn.Parameter(torch.FloatTensor([0]))

    def forward(self, x):
        return x + self.block(x) * self.resweight


class System(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.num_gpus = torch.cuda.device_count()
        # Will scale the learning rate by batch size
        self.train_bsz = args.batch_size
        self.val_bsz = args.batch_size
        self.num_workers = args.num_workers

        # Percentage of each class in the dataset
        # label_prct = torch.FloatTensor(
        #     [0.02937934, 0.08206363, 0.14337856, 0.11532754, 0.05297417, 0.57687677])
        label_prct = torch.FloatTensor(
            [0.12872068, 0.2694658, 0.22057954, 0.14202641, 0.23920757])

        # MLP
        self.model = nn.Sequential(
            # Projection
            # nn.Linear(6144, 64),
            # nn.Dropout(0.5),
            # nn.Sequential(*(ResBlock(64, 64) for _ in range(10))),
            # # Output layer
            # nn.Linear(64, len(label_prct)),

            nn.Linear(6144, 10),
        )
#        self.ce_loss = nn.CrossEntropyLoss(
#            reduction='none', weight=(1 - label_prct))

	bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.forward(x)
        # loss = ((y - y_pred) ** 2).float().mean()
#        loss = self.ce_loss(y_pred, y).float().mean()
	loss = self.bce_loss(y_pred, y)
        with torch.no_grad():
            preds = y_pred.argmax(dim=-1)
            acc = (preds == y).float().mean()

        return {'loss': loss,
                'y_pred': preds,
                'y_true': y,
                'log': {
                    'loss': loss,
                    'acc': acc,
                    'f1': torch.tensor(float(f1_score(y.cpu().numpy(), preds.cpu().numpy(), average='weighted'))),
                }}

    def validation_step(self, batch, batch_idx):
        # No difference in train vs validation
        metrics = self.training_step(batch, batch_idx)
        return {'val_' + k: v for k, v in metrics['log'].items()}

    def validation_end(self, outputs):
        keys = list(outputs[0].keys())
        metrics = {k: torch.stack([x[k] for x in outputs]).mean()
                   for k in keys}

        print('val metrics', metrics)
        return {**metrics, 'log': metrics}

    def test_step(self, batch, batch_idx):
        metrics = self.training_step(batch, batch_idx)
        return {
            **{'val_' + k: v for k, v in metrics['log'].items()},
            **{'y_pred': metrics['y_pred'], 'y_true': metrics['y_true']}
        }

    def test_end(self, outputs):
        y_pred = torch.cat([o['y_pred']
                            for o in outputs], dim=0).flatten().cpu()
        y_true = torch.cat([o['y_true']
                            for o in outputs], dim=0).flatten().cpu()

        print('Confusion matrix:\n', confusion_matrix(y_true, y_pred))
        print('F1 Score:\n', f1_score(y_true, y_pred, average='weighted'))

        keys = set(outputs[0].keys())
        keys.remove('y_pred')
        keys.remove('y_true')

        metrics = {k: torch.stack([x[k] for x in outputs]).mean()
                   for k in keys}
        print('Test results', metrics)
        return {**metrics, 'log': metrics}

    def configure_optimizers(self):
        # Linear scaling rule
        # Learning rate is per batch
        effect_bsz = self.num_gpus * self.train_bsz * self.args.grad_acc
        scaled_lr = self.args.lr * \
            math.sqrt(effect_bsz) if self.args.lr else None
        print('Effective learning rate:', scaled_lr)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=scaled_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda e: max(
                1 - e / self.args.max_steps, scaled_lr / 1000)
        )
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        dataset = BioactivityDataset(self.args.data_path + '.train.pkl')
        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return DataLoader(
            dataset,
            sampler=dist_sampler,
            num_workers=self.num_workers,
            batch_size=self.train_bsz,
            collate_fn=Collater(),
            pin_memory=True)

    @pl.data_loader
    def val_dataloader(self):
        dataset = BioactivityDataset(self.args.data_path + '.valid.pkl')
        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return DataLoader(
            dataset,
            sampler=dist_sampler,
            num_workers=self.num_workers,
            batch_size=self.val_bsz,
            collate_fn=Collater(),
            pin_memory=True)

    @pl.data_loader
    def test_dataloader(self):
        dataset = BioactivityDataset(self.args.data_path + '.test.pkl')
        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return DataLoader(
            dataset,
            sampler=dist_sampler,
            num_workers=self.num_workers,
            batch_size=self.val_bsz,
            collate_fn=Collater(),
            pin_memory=True)
