# -*- coding: utf-8 -*-
"""
validate Bert-Whitening: https://kexue.fm/archives/8069
"""
import pandas as pd
import torch.nn.functional as func
import argparse
import copy
from abc import ABC
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from load_data import load_sentiment_data, get_dir_and_base_name
import numpy as np
from util import get_logger, get_args_str, get_tensorboard_writer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

parser = argparse.ArgumentParser()
parser.add_argument('--vec_dim', type=int, default=128)

parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--dataset_name', type=str, default='Instant_Video_5',
                    help='dataset name')
parser.add_argument('--dataset_path', type=str, default='/home/d1/shuaijie/data/Instant_Video_5/Instant_Video_5.json',
                    help='raw dataset file path')

parser.add_argument('--review_max_length', type=int, default=128)

args = parser.parse_args()


args.dataset_name = 'Digital_Music_5'
args.dataset_path = '/home/d1/shuaijie/data/Digital_Music_5/Digital_Music_5.json'
args.vec_dim = 64


args.model_short_name = 'BERT-Whitening'
args.pretrained_weight_shortcut = 'bert-base-uncased'


args.feat_save_path = f'../checkpoint/' \
                      f'{args.dataset_name}/' \
                      f'{args.model_short_name}/' \
                      f'{args.pretrained_weight_shortcut}_sentence_vectors_' \
                      f'dim_{args.vec_dim}.pkl'

bert_tokenizer = BertTokenizer.from_pretrained(args.pretrained_weight_shortcut,
                                               model_max_length=args.review_max_length)


class ReviewDataset(Dataset):

    def __init__(self, user, item, rating, review_text, tokenizer):

        self.user = np.array(user).astype(np.int64)
        self.item = np.array(item).astype(np.int64)
        self.r = np.array(rating).astype(np.float32)
        self.tokenizer = tokenizer
        self.docs = review_text

        self.__pre_tokenize()

    def __pre_tokenize(self):
        self.docs = [self.tokenizer.tokenize(x)
                     for x in tqdm(self.docs, desc='pre tokenize')]
        review_length = self.top_review_length(self.docs)
        self.docs = [x[:review_length] for x in self.docs]

    def __getitem__(self, idx):
        return self.user[idx], self.item[idx], self.r[idx], self.docs[idx]

    def __len__(self):
        return len(self.docs)

    @staticmethod
    def top_review_length(docs: list, top=0.8):
        sentence_length = [len(x) for x in docs]
        sentence_length.sort()
        length = sentence_length[int(len(sentence_length) * top)]
        length = 128 if length > 128 else length
        return length


def collate_fn(data):
    u, i, r, tokens = zip(*data)
    encoding = bert_tokenizer(tokens, return_tensors='pt', padding=True,
                              truncation=True, is_pretokenized=True)

    return torch.Tensor(u), torch.Tensor(i), torch.Tensor(r), \
        encoding['input_ids'], encoding['attention_mask']


def compute_kernel_bias(vecs, vec_dim):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    # vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    # return None, None
    # return W, -mu
    return W[:, :vec_dim], -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


@torch.no_grad()
def save_sentence_feat(params):
    train_data, valid_data, test_data, word2id, embeddings, _ = \
        load_sentiment_data(args.dataset_path)

    train_data = pd.concat([train_data, valid_data, test_data])

    train_dataset = ReviewDataset(
        train_data['user_id'].tolist(),
        train_data['item_id'].tolist(),
        train_data['rating'].tolist(),
        train_data['review_text'].tolist(),
        bert_tokenizer
    )
    data_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn)

    bert = BertModel.from_pretrained(params.pretrained_weight_shortcut).to(params.device)
    bert.config.output_hidden_states = True

    vecs = []
    for u, i, r, input_ids, mask in tqdm(data_loader):
        input_ids = input_ids.to(params.device)
        mask = mask.to(params.device)  # bs * seq_len

        outputs = bert(input_ids, mask)
        output1 = outputs[2][-2]  # bs * seq_len * 768
        output2 = outputs[2][-1]
        last2 = output1 + output2 / 2
        last2 = torch.sum(mask.unsqueeze(-1) * last2, dim=1) \
            / mask.sum(dim=1, keepdims=True)
        vecs.append(last2.cpu().numpy())
        # print(outputs.shape)
    vecs = np.vstack(vecs)
    kernel, bias = compute_kernel_bias(vecs, params.vec_dim)
    vecs = transform_and_normalize(vecs, kernel, bias)
    vecs = torch.from_numpy(vecs)

    ui = list(zip(train_data['user_id'].tolist(),
                  train_data['item_id'].tolist()))

    vecs = dict(zip(ui, vecs))
    torch.save(vecs, params.feat_save_path)
    print(f'Saved embeddings to {params.feat_save_path}')


if __name__ == '__main__':
    save_sentence_feat(args)
    # eval_feat(args)
