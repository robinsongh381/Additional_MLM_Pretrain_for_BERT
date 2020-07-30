from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.utils.data import Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
import constant as model_config


class MLMDataset(Dataset): 
    def __init__(self,dtype):
        self.origin = torch.load('./data/{}_origin.pt'.format(dtype))
        self.mask = torch.load('./data/{}_mask.pt'.format(dtype))
        assert len(self.origin)==len(self.mask)
        self.length = len(self.origin)
        
    def __getitem__(self, idx):
        return self.mask[idx], self.origin[idx]
    
    def __len__(self):
        return self.length
    
    
def pad_collate(batch):

    (mask, origin) = zip(*batch)
    
    # masked sentence
    len_s1 = [len(x) for x in mask]
    max_token_len_s1 = max(len_s1)  
    input_mask = torch.tensor(len_s1) # valid length
    token_ids_mask = pad_sequences(mask, 
                              maxlen=max_token_len_s1, # model_config.maxlen 
                              value=model_config.pad_idx, 
                              padding='post',
                              dtype='long',
                              truncating='post')
    
    segment_ids_mask = [len(i)*[0] for i in token_ids_mask]
    
    # original sentence
    len_s2 = [len(y) for y in origin] 
    token_ids_origin = pad_sequences(origin, 
                              maxlen=max(len_s2) , # model_config.maxlen 
                              value=model_config.pad_idx, 
                              padding='post',
                              dtype='long',
                              truncating='post')

    return [token_ids_mask, input_mask, segment_ids_mask, token_ids_origin]