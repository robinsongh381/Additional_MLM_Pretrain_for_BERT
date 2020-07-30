from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.utils.data import Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
import constant as model_config


class MLMDataset(Dataset): 
    def __init__(self,dtype):
        self.masked_sent = torch.load('./data/{}_mask.pt'.format(dtype))
        self.label = torch.load('./data/{}_label.pt'.format(dtype))
        assert len(self.masked_sent)==len(self.label)
        self.length = len(self.label)
        
    def __getitem__(self, idx):
        return self.masked_sent[idx], self.label[idx]
    
    def __len__(self):
        return self.length
    
    
def pad_collate(batch):

    (masked_sent, label) = zip(*batch)
    
    # masked sentence
    len_s1 = [len(x) for x in masked_sent]
    max_token_len_s1 = max(len_s1)  
    input_mask = torch.tensor(len_s1) # valid length
    token_ids_mask = pad_sequences(masked_sent, 
                              maxlen=max_token_len_s1, # model_config.maxlen 
                              value=model_config.pad_idx, 
                              padding='post',
                              dtype='long',
                              truncating='post')
    
    segment_ids_mask = [len(i)*[0] for i in token_ids_mask]
    
    # original sentence
    # len_s2 = [len(y) for y in label] 
    token_ids_origin = pad_sequences(label, 
                              maxlen=max(len_s1) , # model_config.maxlen 
                              value=model_config.pad_idx, 
                              padding='post',
                              dtype='long',
                              truncating='post')

    return [token_ids_mask, input_mask, segment_ids_mask, token_ids_origin]