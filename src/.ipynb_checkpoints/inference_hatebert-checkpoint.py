from train_hb import JigsawModel, JigsawDataset, CONFIG
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import gc


from torch.utils.data import Dataset, DataLoader

# PyTorch Lightning
import pytorch_lightning as pl
MODEL_PATHS = [
    '../models/checkpoints/roberta/fold_0_lstm.ckpt',
    '../models/checkpoints/roberta/fold_1_lstm.ckpt',
    '../models/checkpoints/roberta/fold_2_lstm.ckpt',
    '../models/checkpoints/roberta/fold_3_lstm.ckpt',
    '../models/checkpoints/roberta/fold_4_lstm.ckpt'
]

class JigsawDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.data = df
        self.max_len = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        less_toxic_text = self.data.iloc[index].less_toxic
        inputs_text = self.tokenizer.encode_plus(
                                less_toxic_text,
                                truncation=True,
                                max_length=self.max_len,
                                padding='max_length',
                                pad_to_max_length=True,
                                return_tensors='pt'
                    )

        lt_ids = inputs_text['input_ids']
        lt_mask = inputs_text['attention_mask']
        
        more_toxic_text = self.data.iloc[index].more_toxic
        inputs_text = self.tokenizer.encode_plus(
                                more_toxic_text,
                                truncation=True,
                                max_length=self.max_len,
                                padding='max_length',
                                pad_to_max_length=True,
                                return_tensors='pt'
                    )

        mt_ids = inputs_text['input_ids']
        mt_mask = inputs_text['attention_mask']
        
        return {
            'less_toxic_input_ids': lt_ids.flatten(),
            'less_toxic_attention_mask':lt_mask.flatten(),
            'more_toxic_input_ids': mt_ids.flatten(),
            'more_toxic_attention_mask':mt_mask.flatten()
        }

def valid_fn(model, dataloader, device):
    model.eval()
    model.freeze()
    model=model.to(device)
    dataset_size = 0
    running_loss = 0.0
    
    LT_PREDS = []
    MT_PREDS = []
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        lt_ids =data['less_toxic_input_ids']
        lt_mask =data['less_toxic_attention_mask']
        mt_ids = data['more_toxic_input_ids']
        mt_mask =data['more_toxic_attention_mask']
        _, outputs = model(lt_ids.to(device), lt_mask.to(device))
        preds=[np.exp(x) for x in outputs.cpu().flatten().numpy()]
        total=0
        cat_mtpl = {'obscene': 1, 'toxic': 1, 'threat': 1, 
                    'insult':1, 'severe_toxic': 2, 'identity_hate': 1}
        
        for label, prediction in zip(CONFIG['labels'], preds):
            if prediction>0.5:
                total+=cat_mtpl[label]*prediction
        
        LT_PREDS.append(total)
        
        _, outputs = model(mt_ids.to(device), mt_mask.to(device))
        preds=[np.exp(x) for x in outputs.cpu().flatten().numpy()]
        total=0

        for label, prediction in zip(CONFIG['labels'], preds):
            if prediction>0.5:
                total+=cat_mtpl[label]*prediction
        
        
        MT_PREDS.append(total)
    
    gc.collect()
    
    return LT_PREDS,MT_PREDS


def inference(model_paths, dataloader, device):
    final_preds1,final_preds2 = [],[]
    for i, path in enumerate(model_paths):
        model=JigsawModel.load_from_checkpoint(
        checkpoint_path=path,
        model_name=CONFIG['model_name'],
        n_classes=CONFIG['num_classes']
        )
        
        print(f"Getting predictions for model {i+1}")
        lt_preds,mt_preds = valid_fn(model, dataloader, device)
        final_preds1.append(lt_preds)
        final_preds2.append(mt_preds)
    
    final_preds1 = np.array(final_preds1)
    final_preds1 = np.mean(final_preds1, axis=0)
    final_preds2 = np.array(final_preds2)
    final_preds2 = np.mean(final_preds2, axis=0)
    
    print(f'val is : {(final_preds1 < final_preds2).mean()}')

if __name__=="__main__":    
    df = pd.read_csv("../input/jigsaw-toxic-severity-rating/validation_data.csv")
    df.head()

    test_dataset = JigsawDataset(df, CONFIG['tokenizer'], max_length=CONFIG['max_length'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['valid_batch_size'],
                             num_workers=8, shuffle=False, pin_memory=True)

    preds1 = inference(MODEL_PATHS, test_loader, CONFIG['device'])
#     df['score']=preds1
#     df['score'] = df['score'].apply(lambda x: (x + 1)/2)
#     df.to_csv('../output/hatebert-preds.csv')
    
    
