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

def eval(model, args):
      """
      training function for FullModel class.
      params:
            model: FullModel object
            epochs: training epochs
            dataloader: torch.utils.data.Dataloader object, init with FNCDataset object
      return:
            history: dict of loss, acc, val_loss, val_acc
      """

      test_dataset = FNCDataset(args.test_path, output_type="both")
      test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
      
      # epochs = args.epochs
      # history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}
      
      #######################
      # replace with class weight
      weights = torch.tensor([0.017, 0.170, 0.070, 0.743]).to(device)
      #######################
      
      model.to(device)
      model.eval()
      
      # opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.001)
      loss_func = WeightedCELoss(weights)

      val_losses = []
      val_accs = []
      val_preds = []
      val_labels = []
      for batch in test_dataloader:
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
      
      print(f'test_loss: {ep_val_loss}, test_acc: {ep_val_acc}')
      rep = classification_report(val_labels, val_preds)
      # print(rep)

      return rep
      
      
if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument('--device', type=int, default=0)
      parser.add_argument('--test_path', type=str, default='/sbert_embeddings/fnc_comp_test_mean.pkl')
      parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_mpnet/total_model_complex_fnc_only_ep23.pth')
      args = parser.parse_args()
      
      device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

      input_shape = 768*2

      model = FullModel(input_shape, hidden_structure=[512, 256, 256, 128, 64])
      state_dict = torch.load(args.checkpoint_dir)
      model.load_state_dict(state_dict)
      
      rep = eval(model, args)
      
      print(rep)

      # pkl.dump(history, open(f'checkpoints/history_ep{args.epochs}.pkl','wb'))
      