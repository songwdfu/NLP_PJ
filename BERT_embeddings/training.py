import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
import argparse
from full_model import FullModel, WeightedCELoss
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
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
      train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
      val_dataset = FNCDataset(args.val_path, output_type="both")
      val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
      
      epochs = args.epochs
      history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}
      
      #######################
      # replace with class weight
      weights = torch.tensor([0.017, 0.170, 0.070, 0.743]).to(device)
      #######################
      
      model.to(device)
      model.train()
      
      opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.001)
      loss_func = WeightedCELoss(weights)

      # early stopping
      best_loss = float('inf')
      best_weights = None
      patience = 5
      no_improvement = 0
      
      for e in range(1, epochs+1):
            print(f'Epoch: {e}', end='')
            # training
            losses = []
            accs = []
            for batch in tqdm(train_dataloader, total=len(train_dataloader)):
                  opt.zero_grad()
                  head_trans, body_trans, labels = batch  # head_ext, body_ext
                  input = torch.cat([head_trans, body_trans], dim=1).to(device)
                  labels = labels.to(device)
                  output = model(input)
                  loss = loss_func(output, labels)
                  loss.backward()
                  opt.step()
                  losses.append(loss.detach().cpu())
                  
                  output = torch.argmax(output, dim=1)
                  acc = accuracy_score(labels.detach().cpu(), output.detach().cpu())
                  accs.append(acc)
            
            ep_loss = np.mean(losses)
            ep_acc = np.mean(accs)
            
            history['loss'].append(ep_loss)
            history['acc'].append(ep_acc)
            
            # validation
            val_losses = []
            val_accs = []
            val_preds = []
            val_labels = []
            for batch in val_dataloader:
                  head_trans, body_trans, labels = batch
                  input = torch.cat([head_trans, body_trans], dim=1).to(device)
                  labels = labels.to(device)
                  with torch.no_grad():
                        output = model(input)
                        val_loss = loss_func(output, labels)
                        val_losses.append(val_loss.detach().cpu())
                        
                        output = torch.argmax(output, dim=1)
                        val_acc = accuracy_score(labels.detach().cpu(), output.detach().cpu())
                        val_accs.append(val_acc)

                        val_preds.extend(output.detach().cpu().tolist())
                        val_labels.extend(labels.detach().cpu().tolist())
            
            ep_val_loss = np.mean(val_losses)
            ep_val_acc = np.mean(val_acc)
            
            history['val_loss'].append(ep_val_loss)
            history['val_acc'].append(ep_val_acc)

            # early stopping
            if ep_val_loss < best_loss:
                  best_loss = ep_val_loss
                  best_weights = model.state_dict()
                  no_improvement = 0
            else:
                  no_improvement += 1
            
            if no_improvement >= patience:
                  print(f"Early stopping at {e} epochs: No improvement for {patience} epochs.")
                  print(f"Restoring weights from epoch {e-5}: ")
                  model.load_state_dict(best_weights)
                  print(f'loss: {ep_loss}, acc: {ep_acc}, val_loss: {ep_val_loss}, val_acc: {ep_val_acc}')
                  rep = classification_report(val_labels, val_preds)
                  print(rep)
                  torch.save(model.state_dict(), f'checkpoints/model_ep{e}.pth')
                  break

            print(f'loss: {ep_loss}, acc: {ep_acc}, val_loss: {ep_val_loss}, val_acc: {ep_val_acc}')
                        
            # save checkpoints
            if e % args.save_intvl == 0:
                  #################
                  # debug
                  rep = classification_report(val_labels, val_preds)
                  print(rep)
                  #################
                  torch.save(model.state_dict(), f'checkpoints/model_ep{e}.pth')

      return history
      
      
if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument('--device', type=int, default=0)
      parser.add_argument('--epochs', type=int, default=50)
      parser.add_argument('--train_path', type=str, default='/gpt_embeddings/fnc_train_mean.pkl')
      parser.add_argument('--val_path', type=str, default='/gpt_embeddings/fnc_val_mean.pkl')
      parser.add_argument('--save_intvl', type=int, default=10)
      args = parser.parse_args()
      
      device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

      input_shape = 768*2

      model = FullModel(input_shape, hidden_structure=[128, 128, 64, 32])
      
      history = train(model, args)
      
      pkl.dump(history, open(f'checkpoints/history_ep{args.epochs}.pkl','wb'))
      