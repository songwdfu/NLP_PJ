import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
import torch.nn.functional as F

class FullModel(nn.Module):
      """
      Full model using Transferred embeddings and external features
      """
      def __init__(self, input_shape, hidden_structure=[64, 32]):
            super().__init__()
            layer = []
            for i, hidden_size in enumerate(hidden_structure):
                  if i == 0:
                        layer.append(nn.Linear(input_shape, hidden_size))
                  else:
                        layer.append(nn.Linear(hidden_structure[i-1], hidden_size))
                  
                  layer.append(nn.ReLU())
            
            self.hidden_layers = nn.Sequential(*layer)
            self.fc_layer = nn.Linear(hidden_structure[-1], 4)
            self.softmax = nn.Softmax(-1)
            
            
      def forward(self, input):
            x = self.hidden_layers(input)
            x = self.fc_layer(x)
            output = self.softmax(x)
            return output
      
      def weighted_CE(self, input, target, weights):
            if len(weights)!=4:
                  raise ValueError('weight should be 4 dim')
            input = torch.clamp(input, min=1e-7, max=1-1e-7)
            

class WeightedCELoss(nn.Module):
      """
      class weighted categorical CE loss
      """
      def __init__(self, weights):
            super().__init__()
            self.weights = weights
      
      def forward(self, inputs, targets):
            loss = F.cross_entropy(inputs, targets, self.weights)
            return loss
            
                  
                  
                  
