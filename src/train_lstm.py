# Necessities
import wandb
import warnings
import pandas as pd
import numpy as np
import gc
import re
import spacy
from collections import Counter
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error

# PyTorch
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

# Transformers
from transformers import AutoTokenizer, AutoModel, AdamW

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Colored Terminal Text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

# Aesthetics
warnings.simplefilter('ignore')

# Weights and Biases
wandb.login()

CONFIG = {"seed": 42,
          "vocab_size": 84422,
          "epochs": 30,
          "num_layers": 3,
          "vocab_index_path": "../input/vocab2index.txt",
          "train_file_path": "../input/folds/train_folds_score_5.csv",
          "checkpoint_directory_path": '../models/checkpoints/blstm',
          "train_batch_size": 128,
          "valid_batch_size": 128,
          "embedding_dim": 100,
          "hidden_dim": 32,
          "max_length": 512,
          "learning_rate": 1e-4,
          "scheduler": 'CosineAnnealingLR',
          "min_lr": 1e-6,
          "T_max": 500,
          "weight_decay": 1e-6,
          "n_fold": 5,
          "n_accumulate": 1,
          "num_classes": 1,
          "margin": 0.5,
          "num_workers": 8,
          "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          "infra": "Kaggle",
          "competition": 'Jigsaw Toxic Severity LSTM',
          "_wandb_kernel": 'neuracort',
          "wandb": True
          }
# Seed
pl.seed_everything(seed=42)


def fetch_scheduler(optimizer):

    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=CONFIG['T_max'],
            eta_min=CONFIG['min_lr']
        )

    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=CONFIG['T_0'],
            eta_min=CONFIG['min_lr']
        )

    elif CONFIG['scheduler'] == None:
        return None

    return scheduler


# W&B Logger
wandb_logger = WandbLogger(
    project='jigsaw-lightning-bilstm',
    job_type='train',
    anonymous='allow',
    config=CONFIG
)


class JigsawEncodedDataset(Dataset):
    def __init__(self, df):
        self.X = df["encoded"]
        self.y = df["y"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"encoding": torch.from_numpy(self.X.loc[idx].astype(np.int32)), "y": self.y[idx]}


class JigsawLSTMDataModule(pl.LightningDataModule):

    def __init__(self, df_train, df_valid):
        super().__init__()
        self.df_train = df_train
        self.df_valid = df_valid

    def setup(self, stage=None):

        self.train_dataset = JigsawEncodedDataset(
            self.df_train
        )

        self.valid_dataset = JigsawEncodedDataset(
            self.df_valid
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=CONFIG['train_batch_size'],
            num_workers=CONFIG["num_workers"],
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=CONFIG['valid_batch_size'],
            num_workers=CONFIG["num_workers"],
            shuffle=False,
            pin_memory=True
        )


class JigsawLSTMModel(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(JigsawLSTMModel, self).__init__()
        self.criterion = nn.MSELoss()
        self.embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            bidirectional=True,
                            dropout=0.2,
                            batch_first=True)
        self.linear = nn.Linear(hidden_dim*4, 32)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.out = nn.Linear(32, 1)
        self.double()

    def forward(self, x, y=None):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, _ = self.lstm(x)
        avg_pool = torch.mean(lstm_out, 1)
        max_pool, _ = torch.max(lstm_out, 1)
        conc = torch.cat((avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        pred = self.out(conc)

        loss = 0
        if y is not None:
            loss = self.criterion(pred, y)
        return loss, pred

    def training_step(self, batch, batch_idx):
        enc = batch["encoding"]
        y = batch["y"]
        loss, pred = self(enc, y)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "prediction": pred, "actual": y}

    def validation_step(self, batch, batch_idx):
        enc = batch["encoding"]
        y = batch["y"]
        loss, pred = self(enc, y)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(
        ), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        scheduler = fetch_scheduler(optimizer)
        return dict(
            optimizer=optimizer,
            lr_scheduler=scheduler
        )


if __name__ == "__main__":
    # Load Dataset
    df = pd.read_csv(CONFIG['train_file_path'])
    df['encoded'] = df['encoded'].apply(
        lambda x: np.fromstring(x, dtype=int, sep=' '))

    for fold in range(3, CONFIG['n_fold']):
        print(f'{y_}====== Fold: {fold} ======{sr_}')
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        # Declare Model Checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=CONFIG["checkpoint_directory_path"],
            filename=f"fold_{fold}_blstm",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min"
        )

        # Early Stopping based on Validation Loss
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)

        # Initialise the Trainer
        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stopping_callback],
            max_epochs=CONFIG['epochs'],
            gpus=-1,
            progress_bar_refresh_rate=30,
            precision=16,                # Activate fp16 Training
            accelerator='dp'         # Uncomment for Multi-GPU Training
        )

        data_module = JigsawLSTMDataModule(df_train, df_valid)
        model = JigsawLSTMModel(
            CONFIG['vocab_size'], CONFIG['embedding_dim'], CONFIG['hidden_dim'], CONFIG['num_layers'])

        trainer.fit(model, data_module)
