import random
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_data(df, infer=False):
    if infer:
        return df['img_path'].values
    return df['img_path'].values, df['artist'].values

def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")

def forhuggingface():
    df = pd.read_csv("/home/prml/Documents/ChanYoung/DACON/Artist-Classification/train.csv")
    
    for i in range(len(df)):
        path = df.iloc[i]['img_path'].split('/')[-1]
        df.loc[i, 'img_path'] = path
    df.to_csv("/home/prml/Documents/ChanYoung/DACON/Artist-Classification/forhug.csv")


# forhuggingface()