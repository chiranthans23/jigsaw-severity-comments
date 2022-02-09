from train_lstm import JigsawLSTMModel, CONFIG
from pre_process import encode_sentence
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import gc
import ast,emoji, string, re


from torch.utils.data import Dataset, DataLoader

# PyTorch Lightning
import pytorch_lightning as pl
MODEL_PATHS = [
    '../models/checkpoints/lstm/fold_0_lstm.ckpt',
    '../models/checkpoints/lstm/fold_1_lstm.ckpt',
    '../models/checkpoints/lstm/fold_2_lstm.ckpt',
    '../models/checkpoints/lstm/fold_3_lstm.ckpt',
    '../models/checkpoints/lstm/fold_4_lstm.ckpt'
]

class JigsawEncodedDataset(Dataset):
    def __init__(self, df):
        self.X = df["encoded"]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {"encoding":torch.from_numpy(self.X.loc[idx].astype(np.int32))}

    
    
def valid_fn(model, dataloader1, dataloader2, device):
    model.eval()
    model.freeze()
    model=model.to(device)
    dataset_size = 0
    running_loss = 0.0
    
    LT_PREDS = []
    MT_PREDS = []
    
    bar = tqdm(enumerate(dataloader1), total=len(dataloader1))
    for step, data in bar:
        enc = data['encoding']
        _, outputs = model(enc.to(device))
        MT_PREDS.append(outputs.view(-1).cpu().detach().numpy())

    bar = tqdm(enumerate(dataloader2), total=len(dataloader2))
    for step, data in bar:
        enc = data['encoding']
        _, outputs = model(enc.to(device))
        LT_PREDS.append(outputs.view(-1).cpu().detach().numpy()) 
    
    gc.collect()
    return np.concatenate(LT_PREDS),np.concatenate(MT_PREDS)


def inference(model_paths, dataloader1, dataloader2, device):
    final_preds1,final_preds2 = [],[]
    for i, path in enumerate(model_paths):
        model=JigsawLSTMModel.load_from_checkpoint(
        checkpoint_path=path,
        n_classes=CONFIG['num_classes'],
        vocab_size=CONFIG['vocab_size'],embedding_dim=CONFIG['embedding_dim'],hidden_dim=CONFIG['hidden_dim'],num_layers=CONFIG['num_layers']
        )
        
        print(f"Getting predictions for model {i+1}")
        lt_preds,mt_preds = valid_fn(model, dataloader1, dataloader2, device)
        final_preds1.append(lt_preds)
        final_preds2.append(mt_preds)
    final_preds1 = np.array(final_preds1)
    final_preds1 = np.mean(final_preds1, axis=0)
    final_preds2 = np.array(final_preds2)
    final_preds2 = np.mean(final_preds2, axis=0)
    print(f'val is : {(final_preds1 < final_preds2).mean()}')

if __name__=="__main__":    
    df1 = pd.read_csv("../input/jigsaw-toxic-severity-rating/validation_data_more_toxic.csv")
    df1['encoded']=df1['encoded'].apply(lambda x: np.fromstring(x, dtype=int, sep=' '))
    df2 = pd.read_csv("../input/jigsaw-toxic-severity-rating/validation_data_less_toxic.csv")
    df2['encoded']=df2['encoded'].apply(lambda x: np.fromstring(x, dtype=int, sep=' '))

    test_dataset1 = JigsawEncodedDataset(df1)
    test_loader1= DataLoader(test_dataset1, batch_size=CONFIG['valid_batch_size'],
                             num_workers=8, shuffle=False, pin_memory=True)

    test_dataset2 = JigsawEncodedDataset(df2)
    test_loader2= DataLoader(test_dataset2, batch_size=CONFIG['valid_batch_size'],
                             num_workers=8, shuffle=False, pin_memory=True)
    inference(MODEL_PATHS, test_loader1, test_loader2, CONFIG['device'])