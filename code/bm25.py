import json
import os
import pickle
import time
import random
from contextlib import contextmanager

import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
from scipy.special import softmax

from rank_bm25 import BM25Okapi

seed = 2024
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class BM25:
    def __init__(self,tokenize_fn, data_path="../data/", context_path="wikipedia_documnets.json",):
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))  # set 은 매번 순서가 바뀌므로

        print(f"Lengths of unique contexts : {len(self.contexts)}")
        
        self.tokenize_fn = tokenize_fn

        self.bm25 = None

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

    def get_sparse_embedding(self):
        bm25_name = f'bm25_object.bin'
        bm25_path = os.path.join(self.data_path, bm25_name)

        if os.path.isfile(bm25_path):
            with open(bm25_path, "rb") as file:
                self.bm25 = pickle.load(file)
            print("Embedding pickle load.")

        else:
            with timer("Build bm25 passage"):
                self.bm25 = BM25Okapi(self.contexts, tokenizer=self.tokenize_fn)

            with open(bm25_path, "wb") as file:
                pickle.dump(self.bm25, file)
            
            print("bm25 pickle saved.")

    def get_relevant_doc(self, query, k=1):
        tokenized_query = self.tokenize_fn(query)
        result = self.bm25.get_scores(tokenized_query)
            
        result = softmax(result)
        sorted_result = np.argsort(result)[::-1]
        doc_score = result[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        
        return doc_score, doc_indices

    def get_relevant_doc_bulk(self, queries, k=1):
        tokenized_queris = [self.tokenize_fn(query) for query in queries]
        result_file = '../data/bm25_result.bin'
        
        if os.path.isfile(result_file):
            with open(result_file, "rb") as file:
                result = pickle.load(file)
            print("BM25 Result pickle load.")

        else:
            with timer("Build bm25 passage"):
                result = np.array(
                    [
                        self.bm25.get_scores(tokenized_query)
                        for tokenized_query in tqdm(tokenized_queris)
                    ]
                )
            with open(result_file, "wb") as file:
                pickle.dump(result, file)
            
            print("bm25 result pickle saved.")

        result = softmax(result, axis=1)

        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result_idx = np.argsort(result[i, :])[::-1] ## 역순으로 큰 것 부터 index 배열
            
            doc_scores.append(result[i, :][sorted_result_idx].tolist()[:k]) ## index에 맞게 점수 반환
            doc_indices.append(sorted_result_idx.tolist()[:k])
            
        return doc_scores, doc_indices