import random
from tqdm import tqdm
import torch
from kobert_tokenizer import KoBertTokenizer
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
tr_data = list(open('./data/KorNLUDatasets/KorNLI/snli_1.0_train.ko.tsv'))[1:]
test_data = list(open('./data/KorNLUDatasets/KorNLI/xnli.test.ko.tsv'))[1:]
mode = ['tr', 'test']


def random_word(list_of_idx):
        output_label = []

        for i, index in enumerate(list_of_idx):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    replace_idx = tokenizer.mask_token_id

                # 10% randomly change token to random token
                elif prob < 0.9:
                    replace_idx = random.randrange(tokenizer.vocab_size)

                # 10% randomly change token to current token
                else:
                    replace_idx = index
            else:
                replace_idx = index
            output_label.append(replace_idx)


        return output_label


for m in mode:
    print('start processing {} file'.format(m))
    if m=='tr':
        data = tr_data
    else:
        data = test_data
    
    list_of_original_sent = []
    list_of_masked_sent = []
    
    for i in tqdm(data):
        sent1, sent2, label = i.split('\t')

        sent1_idx = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent1))
        sent1_rand = random_word(sent1_idx)
        # sent2_idx = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent2))
        # sent2_rand = random_word(sent2_idx)

        list_of_original_sent.append([tokenizer.cls_token_id]+sent1_idx+[tokenizer.sep_token_id])
        list_of_masked_sent.append([tokenizer.cls_token_id]+sent1_rand+[tokenizer.sep_token_id])

    assert len(list_of_original_sent)==len(list_of_masked_sent)
    
    torch.save(list_of_original_sent, './data/{}_origin.pt'.format(m))
    torch.save(list_of_masked_sent, './data/{}_mask.pt'.format(m))