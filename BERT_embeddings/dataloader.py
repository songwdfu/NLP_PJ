import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class FNCDataloader(DataLoader):
      def __init__(self, dataset, batch_size, shuffle):
            super().__init__(dataset, batch_size, shuffle)
            

if __name__ == '__main__':
      import sys
      sys.path.insert(0, '.')
      from dataset import FNCDataset
      dataset = FNCDataset('/gpt_embeddings/fnc_train_mean.pkl', 'trans')
      dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
      for batch in dataloader:
            batch