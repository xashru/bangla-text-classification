from torch.utils.data import Dataset
import torch
from models.config import MODELS, TOKENS, TOKEN_IDX
import random


class BertDataset(Dataset):
    def __init__(self, _data, sequence_len, bert_model, num_class, balance):
        if balance:
            self.data = self.balance(_data, num_class)
        else:
            self.data = _data
        tokenizer = MODELS[bert_model][1]
        self.tokenizer = tokenizer.from_pretrained(bert_model)
        self.sequence_len = sequence_len
        token_style = MODELS[bert_model][3]
        self.start_token = TOKENS[token_style]['START_SEQ']
        self.end_token = TOKENS[token_style]['END_SEQ']
        self.pad_token = TOKENS[token_style]['PAD']
        self.pad_idx = TOKEN_IDX[token_style]['PAD']

    @staticmethod
    def balance(data, num_class):
        # get count
        count = {}
        for x in data:
            label = x[1]
            if label not in count:
                count[label] = 0
            count[label] += 1

        # minimum count
        min_count = 99999999
        for _, v in count.items():
            min_count = min(min_count, v)

        # filter
        random.shuffle(data)
        new_data = []
        count_rem = [min_count] * num_class
        for x in data:
            label = x[1]
            if count_rem[label] > 0:
                new_data.append(x)
            count_rem[label] -= 1

        return new_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index][0]
        label = self.data[index][1]
        tokens_text = self.tokenizer.tokenize(text)
        tokens = [self.start_token] + tokens_text + [self.end_token]
        if len(tokens) < self.sequence_len:
            tokens = tokens + [self.pad_token for _ in range(self.sequence_len - len(tokens))]
        else:
            tokens = tokens[:self.sequence_len - 1] + [self.end_token]

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)
        attn_mask = (tokens_ids_tensor != self.pad_idx).long()
        return tokens_ids_tensor, attn_mask, label
