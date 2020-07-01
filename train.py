import argparse
import torch
import os
import pytorch_lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import wandb
from system import System
from pytorch_lightning import loggers
from pytorch_lightning.loggers import WandbLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help='Name of the experiment')
    parser.add_argument('--data-path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Number of samples per batch')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate per batch')
    parser.add_argument('--max-epochs', type=int, default=1000)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--early-stop', type=int, default=50)
    parser.add_argument('--grad-acc', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--val-interval', type=int, default=1.0)
    parser.add_argument('--out', type=str, default='out', help='Path to save model checkpoint')
    args = parser.parse_args()

    torch.manual_seed(0)

    model = System(args)
    
    wandb_logger = WandbLogger(name=args.name, project='smart')
    trainer = Trainer(
        logger=wandb_logger,
        early_stop_callback=EarlyStopping(
            'val_loss', patience=args.early_stop),
        default_save_path=os.path.join(args.out, args.name),
        gpus=torch.cuda.device_count(),
        distributed_backend='ddp',
        precision=16,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.grad_acc,
        val_check_interval=args.val_interval,
    )
    trainer.fit(model)
