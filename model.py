import torch
import torch.nn as nn
        

class BertLM(nn.Module):
    def __init__(self, config, bert_model, distill='true'):
        super().__init__()
        
        self.distill = distill
        self.bert = bert_model 
        self.mask_lm = MaskedLanguageModel(config.hidden_size, config.vocab_size)

    def get_attention_mask(self, input_ids, valid_length):
        attention_mask = torch.zeros_like(input_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()
    
    def forward(self, input_ids, valid_length, token_type_ids=None):
        """
            input_ids      : (batch, maxlen)
            valid_length   : (batch)
            token_type_ids : (batch, maxlen)    
        """
        
        attention_mask = self.get_attention_mask(input_ids, valid_length)
        if self.distill=='true':
            outputs = self.bert(input_ids=input_ids.long(), attention_mask=attention_mask)
            all_encoder_layers, pooled_output = outputs[0], outputs[0][:,0,:]
        else:
            all_encoder_layers, pooled_output = self.bert(input_ids=input_ids.long(),
                                                      token_type_ids=token_type_ids,
                                                      attention_mask=attention_mask)
        return self.mask_lm(all_encoder_layers)
        
        
class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.linear(x)
        