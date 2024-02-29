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
    BertPreTrainedModel, BertModel, RobertaModel, RobertaPreTrainedModel,
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

def text_preprocess(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^A-Za-z0-9가-힣.?!,()~‘’“”"":%&《》〈〉''㈜·\-\'+\s一-龥サマーン]", "", text)  
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


# ST Negative Sampler (여기서 context 는 korquad v1 trainset)
class STNegativeSampling:
    def __init__(self, data_name='squad_kor_v1'):
        self.data_name = data_name
        self.dataset = load_dataset(self.data_name)
        self.contexts = [text_preprocess(item['context']) for item in self.dataset['train']]

        self.model = SentenceTransformer('jhgan/ko-sroberta-multitask', cache_folder='/data/ephemeral/senttran')
        self.doc_embeddings = None

        self.get_dense_embedding()

    def get_dense_embedding(self):
        emd_path = '../data/sentence_transformer_korquad.bin'
        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.doc_embeddings = pickle.load(file)
            print("ST Embedding pickle load.")
        else:
            print("Build passage(korquad) embedding")
            self.doc_embeddings = self.model.encode(self.contexts)

            with open(emd_path, "wb") as file:
                pickle.dump(self.doc_embeddings, file)
            
            print("ST Embedding pickle saved.")

    def get_negative_samples(self, queries, num_samples=1):
        q_embeddings = self.model.encode(queries)
        cosine_similarities = util.pytorch_cos_sim(q_embeddings, self.doc_embeddings).numpy()
        cosine_similarities = np.abs(cosine_similarities) # 0점에 가까울 수록 연관도 없음
        negative_samples_indices = np.argsort(cosine_similarities, axis=1)[:,:num_samples]
        negative_samples = np.array(self.contexts)[negative_samples_indices].tolist() # 2차원 배열
        return negative_samples



class RobertaEncoder(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaEncoder, self).__init__(config)
        self.bert = RobertaModel(config)
        self.init_weights()
      
    def forward(self,
            input_ids, 
            attention_mask=None,
            token_type_ids=None
        ): 
  
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]
        return pooled_output


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()
      
    def forward(self,
            input_ids, 
            attention_mask=None,
            token_type_ids=None
        ): 
  
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]
        return pooled_output



class DenseRetrieval:
    def __init__(self, args, dataset, num_neg, tokenizer, p_encoder, q_encoder, do_train=False):
        self.args = args
        self.dataset = dataset
        self.num_neg = num_neg
        self.do_train = do_train

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder.to(args.device)
        self.q_encoder = q_encoder.to(args.device)

        self.p_embedding = None
        self.indexer = None

    def prepare_negative(self, dataset=None, num_neg=2, tokenizer=None):
        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        # `num_neg` 는 2의 배수여야 한다 -> 1대1의 비율로 negative sampling을 하기 위함
        if self.num_neg % 2 == 1:
            self.num_neg += 1

        num_neg = self.num_neg
     
        corpus = list(set([text_preprocess(example) for example in dataset['context']])) # must have context key in dataset
        p_with_neg = []

        # Copurs Negative Sample Extract
        negative_samples = self.negative_sampler.get_negative_samples(queries=dataset['context'], num_samples=num_neg//2) # List Type
        corpus = np.array(corpus)

        # DONE
        # 1. bm25로 제일 유사하지 않은 context를 지만 답이 아닌 context를 negative sampling 으로 사용하기 (from korquad)
        # -> fix : bm25의 연산속도에 어려움 -> sentence transformer로 변경
        # 2. Train set 내에서 RandomSampling (Silver Passage (정식 용어 아님 주의))
        # 1번과 2번의 비울 = 1:1
        for i, c in enumerate(tqdm(dataset['context'], desc='Negative Sampling')):
            c = text_preprocess(c)
            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg//2)
                if not c in corpus[neg_idxs]:
                    p_with_neg.append(c) # golden passage
                    p_with_neg.extend(negative_samples[i]) # 1번방법
                    p_neg = corpus[neg_idxs]
                    p_with_neg.extend(p_neg) # 2번방법
                    break
        
        print('Query Tokenizing...')
        q_seqs = tokenizer(dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
        print('Passage Tokenizing...')
        p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors='pt')
        print('Tokenizing Done')

        max_len = p_seqs['input_ids'].size(-1)
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg+1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg+1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg+1, max_len)

        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size)

    def train(self, args=None):
        self.negative_sampler = STNegativeSampling()
        self.prepare_in_batch_negative(num_neg=self.num_neg)

        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        for _ in range(int(args.num_train_epochs)):
            for batch in tqdm(self.train_dataloader, desc=f'Batch epoch {_+1}/{args.num_train_epochs}'):
                
                batch_size = batch[3].size(0)

                self.p_encoder.train()
                self.q_encoder.train()
        
                targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                targets = targets.to(args.device)

                p_inputs = {
                    'input_ids': batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                    'attention_mask': batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                    'token_type_ids': batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
                }
        
                q_inputs = {
                    'input_ids': batch[3].to(args.device),
                    'attention_mask': batch[4].to(args.device),
                    'token_type_ids': batch[5].to(args.device)
                }
        
                p_outputs = self.p_encoder(**p_inputs)
                q_outputs = self.q_encoder(**q_inputs)

                p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
                q_outputs = q_outputs.view(batch_size, 1, -1)

                sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  #(batch_size, num_neg + 1)
                sim_scores = sim_scores.view(batch_size, -1)
                sim_scores = F.log_softmax(sim_scores, dim=1)

                loss = F.nll_loss(sim_scores, targets)

                loss.backward()
                optimizer.step()
                scheduler.step()

                self.p_encoder.zero_grad()
                self.q_encoder.zero_grad()

                global_step += 1

                torch.cuda.empty_cache()

                del p_inputs, q_inputs
            
            self.p_encoder.save_pretrained('/data/ephemeral/huggingface/p_encoder_v2')
            self.q_encoder.save_pretrained('/data/ephemeral/huggingface/q_encoder_v2')

    def prepare_in_batch_negative(self, dataset=None, tokenizer=None):
        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        print('Query Tokenizing...')
        q_seqs = tokenizer(dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
        print('Passage Tokenizing...')
        p_seqs = tokenizer([text_preprocess(example) for example in dataset['context']], padding="max_length", truncation=True, return_tensors='pt')
        print('Tokenizing Done')


        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size)

    def in_batch_train(self, args=None, valid_dataset=None):

        if args is None:
            args = self.args

        if valid_dataset is not None:
            self.prepare_validation(dataset=valid_dataset)

        self.prepare_in_batch_negative()

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
                {'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        ### 추가 부분 ###

        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        for _ in range(int(args.num_train_epochs)):
            for batch in tqdm(self.train_dataloader, desc=f'In Batch epoch {_+1}/{args.num_train_epochs}'):
                
                batch_size = batch[3].size(0)

                self.q_encoder.train()
                self.p_encoder.train()

                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)

                p_inputs = {'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2]
                            }

                q_inputs = {'input_ids': batch[3],
                            'attention_mask': batch[4],
                            'token_type_ids': batch[5]
                            }

                p_outputs = self.p_encoder(**p_inputs)  # (batch_size, emb_dim)
                q_outputs = self.q_encoder(**q_inputs)  # (batch_size, emb_dim)


                # Calculate similarity score & loss
                sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))  # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)

                # target: position of positive samples = diagonal element
                targets = torch.arange(0, batch_size).long()
                if torch.cuda.is_available():
                    targets = targets.to('cuda')

                sim_scores = F.log_softmax(sim_scores, dim=1)

                loss = F.nll_loss(sim_scores, targets)

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.q_encoder.zero_grad()
                self.p_encoder.zero_grad()
                global_step += 1

                torch.cuda.empty_cache()

            if valid_dataset is not None:
                val_score = self.evaluate()

    def prepare_validation(self, dataset=None, tokenizer=None):
        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        q_seqs = tokenizer(dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
        p_seqs = tokenizer([text_preprocess(example) for example in dataset['context']], padding="max_length", truncation=True, return_tensors='pt')

        valid_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )

        self.valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=self.args.per_device_eval_batch_size)

    def evaluate(self):
        self.q_encoder.eval()
        self.p_encoder.eval()
        with torch.no_grad():
            q_embs = []
            p_embs = []
            for batch in tqdm(self.valid_dataloader, desc='Validation'):

                batch = tuple(t.to(self.args.device) for t in batch)
                p_inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                }
                q_inputs = {
                    'input_ids': batch[3],
                    'attention_mask': batch[4],
                    'token_type_ids': batch[5]
                }
                p_emb = self.p_encoder(**p_inputs).to('cpu')
                q_emb = self.q_encoder(**q_inputs).to('cpu')
                p_embs.append(p_emb)
                q_embs.append(q_emb)

        p_embs = torch.stack(p_embs, dim=0).view(len(self.valid_dataloader.dataset), -1)
        q_embs = torch.stack(q_embs, dim=0).view(len(self.valid_dataloader.dataset), -1)

        result = torch.matmul(q_embs, torch.transpose(p_embs, 0, 1)).numpy()
        # result = torch.softmax(result, dim=1).numpy()

        # doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            # doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist())
        
        correct_1 = 0
        correct_5 = 0
        correct_10 = 0
        for answer, doc_i in enumerate(doc_indices):
            correct_1 += int(answer in doc_i[:1])
            correct_5 += int(answer in doc_i[:5])
            correct_10 += int(answer in doc_i[:10])
        
        accuaracy = correct_1/len(doc_indices), correct_5/len(doc_indices), correct_10/len(doc_indices)
        print(f'Top1, Top5, Top10 Accuracy : {accuaracy}')
        return accuaracy


    def get_dense_embeddings(self, emd_path, corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")

            valid_seqs = self.tokenizer(self.contexts, padding="max_length", truncation=True, return_tensors='pt')
            self.passage_dataset = TensorDataset(
                valid_seqs['input_ids'], valid_seqs['attention_mask'], valid_seqs['token_type_ids']
            )
            self.passage_dataloader = DataLoader(self.passage_dataset, batch_size=self.args.per_device_eval_batch_size)

            self.p_encoder.eval()
            with torch.no_grad():
                p_embs = []
                for batch in tqdm(self.passage_dataloader):

                    batch = tuple(t.to(self.args.device) for t in batch)
                    p_inputs = {
                        'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2]
                    }
                    p_emb = self.p_encoder(**p_inputs).to('cpu')
                    p_embs.append(p_emb)

            p_embs = torch.stack(p_embs, dim=0).view(len(self.passage_dataloader.dataset), -1)
            self.p_embedding = p_embs

            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            
            print("Embedding pickle saved.")
        
        self.contexts = np.array(self.contexts)

    def get_relevant_doc(self, query, k=1, args=None):
        if args is None:
            args = self.args

        q_encoder = self.q_encoder
        q_encoder.eval()

        q_seqs_val = self.tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to(args.device)
        q_emb = q_encoder(**q_seqs_val).to('cpu')  # (num_query=1, emb_dim)

        dot_prod_scores = torch.matmul(q_emb, torch.transpose(self.p_embedding, 0, 1))
        dot_prod_scores = torch.softmax(dot_prod_scores, dim=1)
        scores = torch.sort(dot_prod_scores, dim=1, descending=True).values.squeeze()
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

        scores = scores.detach().numpy().tolist()
        rank = rank.numpy().tolist()
        return scores[:k], rank[:k]

    def get_relevant_doc_bulk(self, queries, k=1, args=None):
        if args is None:
            args = self.args

        q_encoder = self.q_encoder
        q_encoder.eval()

        q_seqs_val = self.tokenizer(queries, padding="max_length", truncation=True, return_tensors='pt')
        test_dataset = TensorDataset(q_seqs_val['input_ids'], q_seqs_val['attention_mask'], q_seqs_val['token_type_ids'])
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.per_device_eval_batch_size)

        with torch.no_grad():
            q_embs = []
            for batch in tqdm(test_dataloader):
                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {
                        'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2]
                    }
                q_emb = q_encoder(**p_inputs).to('cpu')
                q_embs.append(q_emb)

            q_embs = torch.stack(q_embs, dim=0).view(len(test_dataset), -1)

        result = torch.matmul(q_embs, torch.transpose(self.p_embedding, 0, 1))
        result = torch.softmax(result, dim=1).numpy()

        doc_scores = []
        doc_indices = []
        for i in tqdm(range(result.shape[0])):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])

        return doc_scores, doc_indices
    
    def retrieve(self, query_or_dataset, topk=1):
        assert self.p_embedding is not None, "get_dense_embedding() 메소드를 먼저 수행해줘야합니다."

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
        indexer_path = f"../data/dpr_faiss_clusters{num_clusters}.index"
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