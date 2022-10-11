import albumentations as A
import pandas as pd
import torch
import os
from sklearn import preprocessing

from albumentations.pytorch.transforms import ToTensorV2
from dataset import CustomDataset
from torch.utils.data import DataLoader
import torch.nn as nn

from sklearn.model_selection import train_test_split

from utils import seed_everything, get_data
from models.BaseModel import BaseModel
from solver import Solver

CFG = {
    'IMG_SIZE':224,
    'MAX_EPOCH':1000,
    'EARLY_STOP':10,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':64,
    'MODEL_SAVE_PATH':"/home/prml/Documents/ChanYoung/DACON/Artist Classification/saved_model/eff0.pt",
    'CFG_PATH':"/home/prml/Documents/ChanYoung/DACON/Artist Classification/saved_model/eff0.csv",
    'SEED':41
}

seed_everything(CFG['SEED'])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

test_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])



df = pd.read_csv('./train.csv')

le = preprocessing.LabelEncoder()
df['artist'] = le.fit_transform(df['artist'].values)
for i in range(len(df)):
    a = os.path.join("/home/prml/Documents/ChanYoung/dataset/dacon_artist",df.iloc[i]['img_path'])
    df.loc[i,'img_path'] = a

train_df, val_df, _, _ = train_test_split(df, df['artist'].values, test_size=0.2, random_state=CFG['SEED'])
train_df = train_df.sort_values(by=['id'])
val_df = val_df.sort_values(by=['id'])

train_img_paths, train_labels = get_data(train_df)
val_img_paths, val_labels = get_data(val_df)

train_dataset = CustomDataset(train_img_paths, train_labels, train_transform)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = CustomDataset(val_img_paths, val_labels, test_transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

model = BaseModel(num_classes=len(le.classes_))
model = nn.DataParallel(model)
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
criterion = nn.CrossEntropyLoss().to(device)
scheduler = None


solver = Solver(model = model,
    optimizer=optimizer,
    criterion=criterion,
    train_loader= train_loader,
    val_loader=val_loader,
    test_loader=None,
    scheduler=scheduler,
    device=device,
    model_save_path=CFG['MODEL_SAVE_PATH'],
    csv_path=CFG['CFG_PATH'],
    max_epoch=CFG['MAX_EPOCH'],
    early_stop=CFG['EARLY_STOP'])

solver.train()