import random
import time
from contextlib import contextmanager

import torch
from datasets import Dataset, load_from_disk

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    TrainingArguments,
)

from dpr import DenseRetrieval, BertEncoder
from bm25 import BM25
from sent_retrieval import STRetrieval


seed = 2024
deterministic = False

random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정
torch.manual_seed(seed) # torch random seed 고정
torch.cuda.manual_seed_all(seed)
if deterministic: # cudnn random seed 고정 - 고정 시 학습 속도가 느려질 수 있습니다. 
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class CustomRetriever:
    def __init__(self, p_encoder_path, q_encoder_path, weights=(0.8,0.2)):
        args = TrainingArguments(
            output_dir="dense_retireval",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=1,
        )

        self.weights = weights

        self.dpr_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base", cache_dir='/data/ephemeral/huggingface')
        self.p_encoder = BertEncoder.from_pretrained(p_encoder_path, cache_dir='/data/ephemeral/huggingface').to(args.device)
        self.q_encoder = BertEncoder.from_pretrained(q_encoder_path, cache_dir='/data/ephemeral/huggingface').to(args.device)

        self.dpr = DenseRetrieval(
            args=args,
            dataset=load_from_disk("../data/train_dataset")['train'],
            num_neg=2,
            tokenizer=self.dpr_tokenizer,
            p_encoder=self.p_encoder,
            q_encoder=self.q_encoder,
            do_train=False,
        )
        
        self.bm25_tokenizer = AutoTokenizer.from_pretrained('monologg/koelectra-base-v3-finetuned-korquad', cache_dir='/data/ephemeral/huggingface')
        self.bm25 = BM25(
            tokenize_fn=self.bm25_tokenizer.tokenize, data_path="../data", context_path="wiki_preprocessed_v2.json"
        )

        self.contexts = self.bm25.contexts

    def get_embeddings(self):
        self.dpr.get_dense_embeddings('../data/dense.bin', corpus_path='../data/wiki_preprocessed_v2.json')
        self.bm25.get_sparse_embedding()

    def retrieve(self, query_or_dataset, topk=1):
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query, k=1):
        bm25_doc_score, bm25_doc_indices = self.bm25.get_relevant_doc(query, k*2)
        dpr_doc_score, dpr_doc_indices = self.dpr.get_relevant_doc(query, k*2)
        pass

    def get_relevant_doc_bulk(self, queries, k=1):
        bm25_doc_scores, bm25_doc_indices = self.bm25.get_relevant_doc_bulk(queries, len(self.contexts))
        dpr_doc_scores, dpr_doc_indices = self.dpr.get_relevant_doc_bulk(queries, len(self.contexts))

        doc_scores = []
        doc_indices = []
        for i in range(len(queries)):
            doc_score = [0] * len(self.contexts)

            for idx in range(len(self.contexts)):
                doc_score[bm25_doc_indices[i][idx]] += bm25_doc_scores[i][idx] * self.weights[0]
                doc_score[dpr_doc_indices[i][idx]] += dpr_doc_scores[i][idx] * self.weights[1]
            
            doc_score = np.array(doc_score)
            sorted_doc_score_idx = np.argsort(doc_score)[::-1]
            doc_scores.append(doc_score[sorted_doc_score_idx].tolist()[:k])
            doc_indices.append(sorted_doc_score_idx.tolist()[:k])
        
        return doc_scores, doc_indices

        

