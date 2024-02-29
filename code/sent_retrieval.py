import json
import re
import random
import os
import pickle
import time
from contextlib import contextmanager

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from datasets import Dataset, load_dataset

import faiss
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import (
    BertPreTrainedModel, BertModel,
    AdamW, get_linear_schedule_with_warmup,
)

from sentence_transformers import SentenceTransformer, util

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


class STRetrieval:
    def __init__(self):
        self.model = SentenceTransformer('jhgan/ko-sroberta-multitask', cache_folder='/data/ephemeral/senttran')

        self.c_embeddings = None
        self.indexer = None

    def get_embeddings(self, corpus_path='../data/wiki_preprocessed_v2.json'):
        emd_path = '../data/str_embeddings.bin'
        with open(corpus_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.c_embeddings = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")

            self.c_embeddings = self.model.encode(self.contexts)

            with open(emd_path, "wb") as file:
                pickle.dump(self.c_embeddings, file)
            
            print("Embedding pickle saved.")
        
        self.contexts = np.array(self.contexts)

    def get_relevant_doc(self, query, k=1):
        pass

    def get_relevant_doc_bulk(self, queries, k=1):
        q_embs = self.model.encode(queries)
        scores = util.pytorch_cos_sim(q_embs, self.c_embeddings).numpy()

        doc_indices = np.argsort(scores, axis=1)[:,::-1][:,:k].tolist()
        doc_scores = np.sort(scores, axis=1)[:,::-1][:,:k].tolist()

        return doc_scores, doc_indices
    
    def retrieve(self, query_or_dataset, topk=1):
        assert self.c_embeddings is not None, "get_embedding() 메소드를 먼저 수행해줘야합니다."

        # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
        total = []
        with timer("query exhaustive search"):
            doc_scores, doc_indices = self.get_relevant_doc_bulk(
                query_or_dataset["question"], k=topk
            )
        for idx, example in enumerate(tqdm(query_or_dataset, desc="Dense retrieval: ")):
            tmp = {
                # Query와 해당 id를 반환합니다.
                "question": example["question"],
                "id": example["id"],
                # Retrieve한 Passage의 id, context를 반환합니다.
                "context": " ".join(
                    [self.contexts[pid] for pid in doc_indices[idx]]
                ),
            }
            if "context" in example.keys() and "answers" in example.keys():
                # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]
            total.append(tmp)

        cqas = pd.DataFrame(total)
        return cqas

    def build_faiss(self, num_clusters=64):
        indexer_path = f"../data/str_faiss_clusters{num_clusters}.index"
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def retrieve_faiss(self, query_or_dataset, topk=1):

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(self, query, k=1):
        query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(self, queries, k=1):
        query_vecs = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vecs) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()