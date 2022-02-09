# Necessities
import wandb
import warnings
import pandas as pd
import numpy as np
import gc

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
          "epochs": 5,
          "hidden_size": 768,
          "dropout": 0.2,
          "model_name": "../model/roberta-base",
          "tokenizer": AutoTokenizer.from_pretrained("../model/roberta-base"),
          "train_file_path": "../input/folds/train_full_folds.csv",
          "checkpoint_directory_path": '../models/checkpoints/roberta',
          "train_batch_size": 32,
          "valid_batch_size": 128,
          "max_length": 512,
          "learning_rate": 1e-4,
          "scheduler": 'CosineAnnealingLR',
          "min_lr": 1e-6,
          "T_max": 500,
          "weight_decay": 1e-6,
          "n_fold": 5,
          "n_accumulate": 1,
          "margin": 0.5,
          "num_workers": 4,
          "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          "infra": "Kaggle",
          "competition": 'Jigsaw',
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
    project='jigsaw-lightning-rb',
    job_type='train',
    anonymous='allow',
    config=CONFIG
)


class JigsawDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index].comment_text
        inputs_text = self.tokenizer.encode_plus(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            pad_to_max_length=True,
            return_token_type_ids=False
        )

        y = self.data.iloc[index]['score']
        ids = inputs_text['input_ids']
        mask = inputs_text['attention_mask']

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'y': torch.tensor(y, dtype=torch.float)
        }


class JigsawDataModule(pl.LightningDataModule):

    def __init__(self, df_train, df_valid):
        super().__init__()
        self.df_train = df_train
        self.df_valid = df_valid

    def setup(self, stage=None):

        self.train_dataset = JigsawDataset(
            self.df_train,
            tokenizer=CONFIG['tokenizer'],
            max_length=CONFIG['max_length']
        )

        self.valid_dataset = JigsawDataset(
            self.df_valid,
            tokenizer=CONFIG['tokenizer'],
            max_length=CONFIG['max_length']
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


class AlReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return self.alrelu(input)

    def alrelu(self,x):
        alpha = 0.01
        return torch.maximum(torch.abs(alpha*x), x)


class JigsawModel(pl.LightningModule):
    def __init__(self, model_name):
        super(JigsawModel, self).__init__()
        self.criterion = nn.MSELoss()
        self.model = AutoModel.from_pretrained(model_name, return_dict=False)
        self.layer_norm = nn.LayerNorm(CONFIG['hidden_size'])
        self.dropout = nn.Dropout(CONFIG['dropout'])
        self.dense = nn.Sequential(
            nn.Linear(CONFIG['hidden_size'], 256),
            AlReLU(),
            nn.Dropout(CONFIG['dropout']),
            nn.Linear(256, 1)
        )

    def forward(self, ids, mask, labels=None):
        _, pooled_output = self.model(input_ids=ids, attention_mask=mask)
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        preds = self.dense(pooled_output)
        loss = 0
        if labels is not None:
            loss = self.criterion(preds, labels)
        return loss, preds

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        score = batch["y"]
        loss, outputs = self(input_ids, attention_mask, score)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "prediction": outputs, "score": score}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        score = batch["y"]
        loss, outputs = self(input_ids, attention_mask, score)
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

    for fold in range(4, CONFIG['n_fold']):
        print(f'{y_}====== Fold: {fold} ======{sr_}')

        # Load Dataset
        df = pd.read_csv(CONFIG['train_file_path'])
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        # Declare Model Checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=CONFIG["checkpoint_directory_path"],
            filename=f"fold_{fold}_robertabase-base",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min"
        )

        # Early Stopping based on Validation Loss
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

        # Initialise the Trainer
        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stopping_callback],
            max_epochs=CONFIG['epochs'],
            gpus=-1,
            progress_bar_refresh_rate=30,
            precision=16,
            accelerator='dp'
        )

        data_module = JigsawDataModule(df_train, df_valid)
        model = JigsawModel(CONFIG['model_name'])

        trainer.fit(model, data_module)
