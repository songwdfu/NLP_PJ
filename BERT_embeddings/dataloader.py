import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class FNCDataloader(DataLoader):
      def __init__(self):
            super().__init__()
            