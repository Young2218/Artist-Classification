from tabnanny import check
from tqdm.auto import tqdm
import numpy as np
import torch
import csv

from utils import competition_metric

class Solver():
    def __init__(self, model, optimizer, criterion,train_loader, val_loader, test_loader, scheduler, device, model_save_path_loss, model_save_path_f1, csv_path, max_epoch, early_stop):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.device = device
        self.max_epoch = max_epoch
        self.early_stop = early_stop
        self.criterion = criterion
        self.model_save_path_f1 = model_save_path_f1
        self.model_save_path_loss = model_save_path_loss
        self.csv_path = csv_path

    def train(self):
        self.model.to(self.device)
        best_loss = 10000
        best_f1 = 0
        check_early_stop = 0

        for epoch in range(1, self.max_epoch):
            train_loss = self.train_one_epoch()
            val_loss, val_f1 = self.val_one_epoch()

            check_early_stop += 1
            if best_loss > val_loss:
                best_loss = val_loss
                check_early_stop = 0
                torch.save({"state_dict": self.model.module.state_dict()},
                           self.model_save_path_loss)
            elif best_f1 < val_f1:
                best_f1 = val_f1
                check_early_stop = 0
                torch.save({"state_dict": self.model.module.state_dict()},
                           self.model_save_path_f1)

            if self.scheduler is not None:
                self.scheduler.step()    

            print(f'Epoch [{epoch}], Train Loss : [{train_loss:.5f}] Val Loss : [{val_loss:.5f}] Val F1 Score : [{val_f1:.5f}] ES: [{check_early_stop}]')
            
            train_loss = round(train_loss, 5)
            val_loss = round(val_loss, 5)
            self.write_csv_last_row([epoch, train_loss, val_loss, val_f1, check_early_stop])
            
            if check_early_stop > self.early_stop:
                print("EARLY STOP")
                break

    def train_one_epoch(self):
        train_loss_list = []
        self.model.train()
        for img, label in tqdm(iter(self.train_loader)):
            img, label = img.float().to(self.device), label.to(self.device)
            
            self.optimizer.zero_grad()

            model_pred = self.model(img)
            
            loss = self.criterion(model_pred, label)

            loss.backward()
            self.optimizer.step()

            train_loss_list.append(loss.item())

        train_loss = np.mean(train_loss_list)

        return train_loss

    def val_one_epoch(self):
        self.model.eval()
    
        model_preds = []
        true_labels = []
        
        val_loss = []
        
        with torch.no_grad():
            for img, label in tqdm(iter(self.val_loader)):
                img, label = img.float().to(self.device), label.to(self.device)
                
                model_pred = self.model(img)
                
                loss = self.criterion(model_pred, label)
                
                val_loss.append(loss.item())
                
                model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
                true_labels += label.detach().cpu().numpy().tolist()
            
        val_f1 = competition_metric(true_labels, model_preds)
        return np.mean(val_loss), val_f1

    def write_csv_last_row(self, row):
        f = open(self.csv_path, 'a', newline='')
        wr = csv.writer(f)
        wr.writerow(row)
        f.close()
    