import random
from tqdm import tqdm
import torch
from kobert_tokenizer import KoBertTokenizer
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
tr_data = list(open('./data/KorNLUDatasets/KorNLI/snli_1.0_train.ko.tsv'))[1:]
test_data = list(open('./data/KorNLUDatasets/KorNLI/xnli.test.ko.tsv'))[1:]
mode = ['tr', 'test']


def random_word(tokens):
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = tokenizer.mask_token_id

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.randrange(tokenizer.vocab_size)

            # 10% randomly change token to current token
            else:
                tokens[i] = token

            output_label.append(token)

        else:
            tokens[i] = token
            output_label.append(0)

    return tokens, output_label


print('Preprocess Start !')
mode = ['tr', 'test']
dataset = [tr_data, test_data]

for m in mode:
    print('start processing {} file'.format(m))
    if m=='tr':
        data = tr_data
    else:
        data = test_data
    
    list_of_masked_sent = []
    list_of_label_sent = []
    
    for i in tqdm(data):
        sent1, sent2, label = i.split('\t')

        sent1_idx = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent1))
        masked_sent, output_label = random_word(sent1_idx)

        # sent2_idx = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent2))
        # tokens, output_label = random_word(sent2_idx)

        list_of_masked_sent.append([tokenizer.cls_token_id]+masked_sent+[tokenizer.sep_token_id])
        list_of_label_sent.append([0]+output_label+[0])

    assert len(list_of_masked_sent)==len(list_of_label_sent)
    
    torch.save(list_of_masked_sent, './data/{}_mask.pt'.format(m))
    torch.save(list_of_label_sent, './data/{}_label.pt'.format(m))