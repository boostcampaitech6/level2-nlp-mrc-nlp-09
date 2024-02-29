import os
import json
import pickle
import random
import time
import random
from contextlib import contextmanager
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from pprint import pprint

from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from datasets import DatasetDict, load_dataset, Dataset, concatenate_datasets, load_from_disk
from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
)

from dpr import DenseRetrieval, BertEncoder

datasets = load_from_disk("../data/train_dataset")
datasets['train'] = datasets['train'].remove_columns(['document_id','__index_level_0__'])
datasets['validation'] = datasets['validation'].remove_columns(['document_id','__index_level_0__'])
korquad = load_dataset('squad_kor_v1', features=datasets["train"].features)
datasets = DatasetDict({
    'train' : concatenate_datasets([datasets['train'], datasets['validation'], korquad['validation']]),
})

# 데이터셋과 모델은 아래와 같이 불러옵니다.
train_dataset = datasets['train']

args = TrainingArguments(
    output_dir="dense_retireval",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    num_train_epochs=8,
    weight_decay=0.01
)


p_encoder_path = '/data/ephemeral/huggingface/p_encoder_bert'
q_encoder_path = '/data/ephemeral/huggingface/q_encoder_bert'

# 혹시 위에서 사용한 encoder가 있다면 주석처리 후 진행해주세요 (CUDA ...)
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base", cache_dir='/data/ephemeral/huggingface')
p_encoder = BertEncoder.from_pretrained(p_encoder_path, cache_dir='/data/ephemeral/huggingface').to(args.device)
q_encoder = BertEncoder.from_pretrained(q_encoder_path, cache_dir='/data/ephemeral/huggingface').to(args.device)


# Retriever는 아래와 같이 사용할 수 있도록 코드를 짜봅시다.
retriever = DenseRetrieval(
    args=args,
    dataset=train_dataset,
    num_neg=2,
    tokenizer=tokenizer,
    p_encoder=p_encoder,
    q_encoder=q_encoder,
    do_train=True,
)

retriever.train()

retriever.get_dense_embeddings('../data/dense_v2.bin', corpus_path='../data/wiki_preprocessed_v2.json')

try:
    query = "미국의 대통령은 누구인가?"
    results = retriever.get_relevant_doc(query=query, k=5)

    print(f"[Search Query] {query}")

    indices = results[1]
    for i, idx in enumerate(indices):
        print(f"Top-{i + 1}th Passage (Index {idx})")
        print(retriever.contexts[idx])

finally:
    p_encoder.bert.save_pretrained('/data/ephemeral/huggingface/p_encoder_bert_v2')
    q_encoder.bert.save_pretrained('/data/ephemeral/huggingface/q_encoder_bert_v2')