import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
import argparse
from full_model import FullModel, WeightedCELoss
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from dataset import FNCDataset

def train(model, args):
      """
      training function for FullModel class.
      params:
            model: FullModel object
            epochs: training epochs
            dataloader: torch.utils.data.Dataloader object, init with FNCDataset object
      return:
            history: dict of loss, acc, val_loss, val_acc
      """
      train_dataset = FNCDataset(args.train_path, output_type='both')
      train_dataloader = DataLoader(train_dataset)
      val_dataset = FNCDataset(args.val_path, output_type="both")
      val_dataloader = DataLoader(val_dataset)
      
      epochs = args.epochs
      history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}
      
      #######################
      # replace with class weight
      weights = ...
      #######################
      
      model.to(device)
      model.train()
      
      opt = torch.optim.Adam(model.parameters(), lr=1e-3, decay=0)
      loss_func = WeightedCELoss(weights)
      for e in epochs:
            print(f'Epoch: {e}')
            # training
            losses = []
            accs = []
            for batch in tqdm(train_dataloader, total=len(train_dataloader)):
                  head_trans, body_trans, head_ext, body_ext, labels = batch
                  input = torch.cat([head_trans, body_trans, head_ext, body_ext], dim=0)
                  input.to(device)
                  output = model(batch)
                  loss = loss_func(output, labels)
                  loss.backward()
                  opt.step()
                  losses.append(loss)
                  
                  output = np.argmax(output, axis=1)
                  acc = accuracy_score(labels, output)
                  accs.append(acc)
            
            ep_loss = losses.detach().cpu().mean()
            ep_acc = acc.detach().cpu().mean()
            
            history['loss'].append(ep_loss)
            history['acc'].append(ep_acc)
            
            # validation
            val_losses = []
            val_accs = []
            for batch in val_dataloader:
                  head_trans, body_trans, head_ext, body_ext, labels = batch
                  input = torch.cat([head_trans, body_trans, head_ext, body_ext], dim=0)
                  input.to(device)
                  with torch.no_grad():
                        output = model(input)
                        val_loss = loss_func(output, labels)
                        val_losses.append(val_loss)
                        
                        output = np.argmax(output, axis=1)
                        val_acc = accuracy_score(labels, output)
                        val_accs.append(val_acc)
            
            ep_val_loss = val_losses.detach().cpu().mean()
            ep_val_acc = val_acc.detach().cpu().mean()
            
            history['val_loss'].append(ep_val_loss)
            history['val_acc'].append(ep_val_acc)
            
            print('loss: {ep_loss}, acc: {ep_acc}, val_loss: {ep_val_loss}, val_acc{ep_val_acc}')
                        
      return history
      
      
if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument('--device', type=int, default=0)
      parser.add_argument('--epochs', type=int, default=10)
      args = parser.parse_args()
      
      device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

      model = FullModel(input_shape, hidden_structure=[64, 32])
      
      history = train(model, args)
      