import torch
import torch.nn
import pandas as pd
import pickle as pkl
from torch.utils.data import Dataset

class FNCDataset(Dataset):
      """
      FNCDataset class
      params:
            path: path to merged head-body pkl (stores a pd.DataFrame)
                  columns: head, body, head_transfer_embeddings, body_transfer_embeddings,
                        head_external_embeddings, body_external_embeddings
            output_type: 'ext', 'trans', or 'both'
                  return external features, transfer model embeddings, or both
      attributes:
            head_trans: gpt/bert headline embeddings
            body_trans: gpt/bert body embeddings
            head_ext: head external features
            body_ext: body external features
      return:
            a tuple of (head_trans, body_trans, head_ext, body_ext)
      """
      def __init__(self, path, output_type='both'):
            super().__init__()
            self.path = path
            self.output_type = output_type
            self.head_trans = []
            self.body_trans = []
            self.head_ext = []
            self.body_ext = []
            self.labels = []
            self._read()
      
      def _read(self):
            # df = pd.read_csv(self.path)
            df = pkl.load(open(self.path, 'rb'))
            if self.output_type == 'both':
                  self.head_trans = list(df.embeddings_head)
                  self.body_trans = list(df.embeddings_body)
                  # self.head_ext = list(df.head_external_embeddings)
                  # self.head_ext = list(df.body_external_embeddings)
                  self.labels = list(df.Stance)
            elif self.output_type == 'trans':
                  self.head_trans = list(df.embeddings_head)
                  self.body_trans = list(df.embeddings_body)
                  self.labels = list(df.Stance)
            else:
                  raise ValueError(r"Only 'both' and 'trans' are valid options")
      
      def __len__(self):
            return len(self.head_trans)
      
      def __getitem__(self, index):
            # only implemented trans
            return self.head_trans[index], self.body_trans[index], self.labels[index]
            
            # if self.output_type == 'both':
            #       return self.head_trans[index], self.body_trans[index], self.head_ext[index], self.body_ext[index]
            # if self.output_type == 'ext':
            #       return self.head_ext[index], self.body_ext[index]
            # if self.output_type == 'trans':
            #       return self.head_trans[index], self. body_trans[index]
      