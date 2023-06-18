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
      def __init__(self, input_shape, hidden_structure=[128, 128, 64, 32]):
            super().__init__()
            layer = []
            for i, hidden_size in enumerate(hidden_structure):
                  if i == 0:
                        layer.append(nn.Linear(input_shape, hidden_size))
                  else:
                        layer.append(nn.Linear(hidden_structure[i-1], hidden_size))
                  
                  layer.append(nn.BatchNorm1d(hidden_size))
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
            

class Transform(nn.Module):
      """
      MLP that transforms embedding into a 32-dim vec
      """
      def __init__(self, input_shape, hidden_structure=[128, 64, 32]):
            super().__init__()
            layer = []
            for i, hidden_size in enumerate(hidden_structure):
                  if i == 0:
                        layer.append(nn.Linear(input_shape, hidden_size))
                  else:
                        layer.append(nn.Linear(hidden_structure[i-1], hidden_size))
                  
                  # layer.append(nn.BatchNorm1d(hidden_size))
                  layer.append(nn.ReLU())
            
            self.hidden_layers = nn.Sequential(*layer)
            self.fc_layer = nn.Linear(hidden_structure[-1], 32)
            
            
      def forward(self, input):
            x = self.hidden_layers(input)
            output = self.fc_layer(x)
            return output
      

class LearnAlpha(nn.Module):
      def __init__(self, input_shape, hidden_structure=[]):
            super().__init__()
            
      def forward(self, input_head, input_body):
            # should be both 32 dim vec (output of TransformModel)
            alpha = 0.5
            return alpha


class TransformModel(nn.Module):
      """
      do MLP transform for head and body embeddings respectively and calc a cos_similarity as score
      """
      def __init__(self, input_shape=768):
            super().__init__()
            self.transform_head = Transform(input_shape)
            self.transform_body = Transform(input_shape)
            self.alpha = LearnAlpha(input_shape)
            self.softmax = nn.Softmax(-1)
            self.fc1 = nn.Linear(32 * 2 + 22, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 4)

            self.add_module("transform_head", self.transform_head)
            self.add_module("transform_body", self.transform_body)

      def forward(self, input_head, input_body, external):
            rep_head = self.transform_head(input_head)
            rep_body = self.transform_body(input_body)
            x = torch.cat([rep_head, rep_body, external], dim=1)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            output = self.softmax(x)
            return output
                  

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
