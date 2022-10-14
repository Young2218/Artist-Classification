import albumentations as A
import pandas as pd
import torch
import os
from sklearn import preprocessing
from vit_pytorch import ViT
from vit_pytorch.extractor import Extractor

from transformers import ViTFeatureExtractor, ViTForImageClassification
from albumentations.pytorch.transforms import ToTensorV2
from dataset import CustomDataset
from dataset_vit import VitDataset
from torch.utils.data import DataLoader
import torch.nn as nn

from sklearn.model_selection import train_test_split

from utils import seed_everything, get_data
from models.BaseModel import BaseModel, BaseModel2
from models.Coca import CoCa
from models.Vit import VitModel
from solver import Solver

CFG = {
    'IMG_SIZE':224,
    'MAX_EPOCH':1000,
    'EARLY_STOP':50,
    'LEARNING_RATE':1e-4,
    'BATCH_SIZE':64,
    'MODEL_SAVE_PATH_LOSS':"/home/prml/Documents/ChanYoung/DACON/Artist-Classification/saved_model/vit_loss.pt",
    'MODEL_SAVE_PATH_F1':"/home/prml/Documents/ChanYoung/DACON/Artist-Classification/saved_model/vit_f1.pt",
    'CFG_PATH':"/home/prml/Documents/ChanYoung/DACON/Artist-Classification/saved_model/vit.csv",
    'SEED':41
}

seed_everything(CFG['SEED'])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.HorizontalFlip(p=0.5),
                            A.RandomRotate90(p = 0.5),
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

id2label = {id:label for id, label in enumerate(df['artist'].unique())}
label2id = {label:id for id,label in id2label.items()}

train_df, val_df, _, _ = train_test_split(df, df['artist'].values, test_size=0.2, random_state=CFG['SEED'])
train_df = train_df.sort_values(by=['id'])
val_df = val_df.sort_values(by=['id'])

train_img_paths, train_labels = get_data(train_df)
val_img_paths, val_labels = get_data(val_df)

# train_dataset = CustomDataset(train_img_paths, train_labels, train_transform)
# train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

# val_dataset = CustomDataset(val_img_paths, val_labels, test_transform)
# val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

# ------------------------ MODEL SET ---------------------------------------------
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
train_dataset = VitDataset(train_img_paths, train_labels, feature_extractor,train_transform)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = VitDataset(val_img_paths, val_labels, feature_extractor,test_transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

model = VitModel(len(le.classes_), device)
print(model)
# ------------------------ MODEL SET ---------------------------------------------


model = nn.DataParallel(model)
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
criterion = nn.CrossEntropyLoss().to(device)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)



solver = Solver(model = model,
    optimizer=optimizer,
    criterion=criterion,
    train_loader= train_loader,
    val_loader=val_loader,
    test_loader=None,
    scheduler=scheduler,
    device=device,
    model_save_path_loss=CFG['MODEL_SAVE_PATH_LOSS'],
    model_save_path_f1=CFG['MODEL_SAVE_PATH_F1'],
    csv_path=CFG['CFG_PATH'],
    max_epoch=CFG['MAX_EPOCH'],
    early_stop=CFG['EARLY_STOP'])

solver.train()