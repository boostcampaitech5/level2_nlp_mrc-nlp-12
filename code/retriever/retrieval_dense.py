import json
import os
import pickle
import sys
from typing import List, Optional, Tuple, Union
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import faiss
import pandas as pd
import torch
from datasets import load_from_disk
from datasets.arrow_dataset import Dataset
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed

from arguments import DataTrainingArguments, RetrieverArguments
from dpr.data_loaders import BaseDataset
from dpr.model import BertEncoder
from dpr.trainer import BiEncoderTrainer
from utils.utils import neat_logger, timer


class FaissRetrieval:
    def __init__(
        self,
        ctx_embeddings_path: str = None,
        indexer_path: str = None,
        num_clusters: Optional[int] = 48,
    ) -> None:
        """FaissRetrieval 객체를 초기화합니다.

        Args:
            ctx_embeddings_path (str): context embeddings의 경로. 기본값 None.
            indexer_path (str): indexer의 저장 경로. 기본값 None.
            num_clusters (Optional[int]): faiss index 구성 시 클러스터의 수. 기본값 48.
        """
        assert indexer_path is not None, 'Index 저장 경로를 지정하세요.'

        self.ctx_embeddings_path = ctx_embeddings_path
        self.num_clusters = num_clusters

        if not os.path.isfile(ctx_embeddings_path) or not os.path.isfile(indexer_path):
            self.build_faiss()
            self.save_index(indexer_path=indexer_path)

        self.indexer = self.load_index(indexer_path=indexer_path)

    def build_faiss(self) -> None:
        """Faiss 인덱서를 구축합니다.

        주어진 context embeddings를 기반으로 학습하고, embeddings를 인덱서에 추가합니다.
        """
        with open(self.ctx_embeddings_path, 'rb') as f:
            ctx_embds = pickle.load(f)
        emb_dim = ctx_embds.shape[-1]

        quantizer = faiss.IndexFlatL2(emb_dim)
        self.indexer = faiss.IndexIVFScalarQuantizer(
            quantizer, quantizer.d, self.num_clusters, faiss.METRIC_L2,
        )
        self.indexer.train(ctx_embds)
        self.indexer.add(ctx_embds)

    def load_index(
        self,
        indexer_path: str = None,
    ) -> faiss.Index:
        """저장된 Faiss 인덱서를 불러옵니다.
        
        Args:
            indexer_path (str): 저장된 인덱서의 경로. 기본값 None.
        """
        assert indexer_path is not None, 'Index 저장 경로를 지정하세요.'

        neat_logger('Loading Faiss indexer..')
        index = faiss.read_index(indexer_path)
        return index

    def save_index(
        self,
        indexer_path: str = None,
    ) -> None:
        """Faiss 인덱서를 지정된 경로에 저장합니다.

        Args:
            indexer_path (str): 인덱서를 저장할 경로. 기본값 None.
        """
        assert indexer_path is not None, 'Index 저장 경로를 지정하세요.'

        neat_logger('Saving Faiss indexer..')
        faiss.write_index(self.indexer, indexer_path)

    def get_relevant_doc(
        self,
        q_emb: torch.Tensor = None,
        top_k: Optional[int] = 1,
    ) -> Tuple[List, List]:
        """주어진 쿼리에 대해 가장 관련 있는 문서를 찾습니다.

        Args:
            q_emb (torch.Tensor): Dense Representation으로 표현된 쿼리 임베딩. 기본값 None.
            top_k (int): 반환할 상위 문서의 수. 기본값 1.
        """
        q_emb = q_emb.astype(np.float32)
        distances, indices = self.indexer.search(q_emb, top_k)
        return distances.tolist()[0], indices.tolist()[0]

    def get_relevant_docs_for_multiple_queries(
        self,
        q_embs: torch.Tensor = None,
        top_k: Optional[int] = 1,
    ) -> Tuple[List, List]:
        """주어진 여러 쿼리에 대해 가장 관련 있는 문서를 찾습니다.

        Args:
            q_embs (torch.Tensor): Dense Representation으로 표현된 여러 쿼리 임베딩. 기본값 None.
            top_k (int): 각 쿼리에 대해 반환할 상위 문서의 수. 기본값 1.
        """
        q_embs = np.array(
            [q_emb.astype(np.float32) for q_emb in q_embs]
        )
        distances, indices = self.indexer.search(q_embs, top_k)
        return distances.tolist(), indices.tolist()

    def retrieve(
        self,
        query_or_dataset: Union[str, Dataset],
        context_path: str,
        tokenizer: AutoTokenizer,
        q_encoder: BertEncoder,
        top_k: Optional[int] = 1,
        device: Optional[str] = 'cuda',
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """주어진 쿼리 또는 데이터셋에 대해 가장 관련 있는 문서를 찾습니다.

        Args:
            query_or_dataset (Union[str, Dataset]): 쿼리 문자열 또는 데이터셋.
            context_path (str): 위키 데이터의 경로.
            tokenizer (AutoTokenizer): 토크나이저.
            q_encoder (BertEncoder): 쿼리 인코더. 
            top_k (Optional[str]): 반환할 상위 문서의 수. 기본값 1.
            device (Optional[str]): 연산을 수행할 디바이스. 기본값 'cuda'.
        """
        assert self.indexer is not None, 'build_faiss 메서드를 먼저 실행하세요.'

        with open(context_path, 'r', encoding='utf-8') as f:
            wiki = json.load(f)
        contexts = list(
            dict.fromkeys([v['text'] for v in wiki.values()])
        )

        if isinstance(query_or_dataset, str):
            neat_logger('[Exhaustive search to query using dense passage retrieval (DPR)]\n{query_or_dataset}')
            input_query = tokenizer(
                query_or_dataset, padding='max_length', truncation=True, return_tensors='pt'
            )

            with torch.no_grad():
                output_query = q_encoder(**input_query).to('cpu').numpy()

            doc_scores, doc_indices = self.get_relevant_doc(output_query, top_k=top_k)
            for i in range(top_k):
                neat_logger(
                    f'Top-{i+1} passages with score {doc_scores[i]:4f}\n'
                    f'Doc index: {doc_indices[i]}\n{contexts[doc_indices[i]]}'
                )
            return (doc_scores, [contexts[doc_indices[i]] for i in range(top_k)])

        elif isinstance(query_or_dataset, TensorDataset) or isinstance(query_or_dataset, Dataset):
            input_queries = query_or_dataset['question']
            input_queries = tokenizer(
                input_queries, padding='max_length', truncation=True, return_tensors='pt'
            )

            with torch.no_grad():
                q_encoder.eval()
                output_queries = q_encoder(**input_queries).to('cpu').numpy()

            with timer('query exhaustive search using Faiss'):
                doc_scores, doc_indices = self.get_relevant_docs_for_multiple_queries(output_queries, top_k=top_k)

            total = []
            for idx, row in enumerate(tqdm(query_or_dataset, desc='Dense passage retrieval')):
                tmp = {
                    'question': row['question'],
                    'id': row['id'],
                    'context': ' '.join([contexts[pid] for pid in doc_indices[idx]])
                }

                if 'context' in row.keys() and 'anwers' in row.keys():
                    tmp['original_context'] = row['context']
                    tmp['answers'] = row['answers']

                # neat_logger(f'Given: {row}\n\n Inferred result: {tmp}')

                total.append(tmp)
            return pd.DataFrame(total)


def main():
    seed = 42
    set_seed(seed)

    parser = HfArgumentParser(
        (DataTrainingArguments, RetrieverArguments)
    )
    data_args, retriever_args = parser.parse_args_into_dataclasses()

    # PyTorch 버전과 XPU 사용 가부를 확인합니다.
    neat_logger(f'PyTorch version: [{torch.__version__}].')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    neat_logger(f'device: [{device}].')

    os.makedirs(retriever_args.dpr_encoder_save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(retriever_args.dpr_model_checkpoint)

    # Passage embeddings, Faiss 클러스터 인덱스 경로를 지정합니다.
    neat_logger('Defining passage embedding path..')
    indexer_path = f'../data/faiss_clusters_{retriever_args.dpr_num_faiss_clusters}.index'

    neat_logger('Loading dataset..')
    train_dataset = BaseDataset(
        dataset_path=data_args.train_dataset_path,
        tokenizer=tokenizer,
        in_batch_negatives=retriever_args.dpr_in_batch_negatives,
        num_neg_samples=retriever_args.dpr_num_neg_samples,
    )
    eval_dataset = load_from_disk(dataset_path=data_args.eval_dataset_path)

    neat_logger(f'Train dataset:\n{train_dataset}')
    neat_logger(f'Eval dataset:\n{eval_dataset}')

    # # 일부 학습 데이터셋의 passage ('context')를 로깅합니다.
    # for context in train_dataset['context'][:8]:
    #     neat_logger(context)

    neat_logger('Defining context(passage), question(query) encoders..')
    ctx_encoder = BertEncoder.from_pretrained(retriever_args.dpr_model_checkpoint)
    q_encoder = BertEncoder.from_pretrained(retriever_args.dpr_model_checkpoint)

    # 이미 학습된 ctx_encoder, q_encoder가 없으면 학습합니다.
    if (
        not os.path.isfile(retriever_args.dpr_ctx_encoder_path)
        or not os.path.isfile(retriever_args.dpr_q_encoder_path)
    ):
        neat_logger(
            'The context encoder and question encoder .pth files have not been found. '
            'Training will proceed for both the question encoder and the context encoder.'
        )
        neat_logger('Defining bi-encoder trainer..')
        training_args = TrainingArguments(
            output_dir='outputs_dpr',
            evaluation_strategy='epoch',
            per_device_train_batch_size=retriever_args.dpr_batch_size,
            per_device_eval_batch_size=retriever_args.dpr_batch_size,
            learning_rate=retriever_args.dpr_learning_rate,
            weight_decay=retriever_args.dpr_weight_decay,
            warmup_ratio=retriever_args.dpr_warmup_ratio,
            num_train_epochs=retriever_args.dpr_num_train_epochs,
            gradient_accumulation_steps=retriever_args.dpr_gradient_accumulation_steps,
            fp16=True,
        )

        neat_logger('Defining retriever..')
        trainer = BiEncoderTrainer(
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            in_batch_negatives=retriever_args.dpr_in_batch_negatives,
            num_neg_samples=retriever_args.dpr_num_neg_samples,
            ctx_encoder=ctx_encoder,
            q_encoder=q_encoder,
            ctx_encoder_path=retriever_args.dpr_ctx_encoder_path,
            q_encoder_path=retriever_args.dpr_q_encoder_path,
            context_path=data_args.context_path,
        )

        neat_logger('Training bi-encoder-based retriever..')
        trainer.train()

        # ctx_encoder & q_encoder를 저장합니다.
        neat_logger('Saving model weights..')
        trainer.save_model_weights()
    else:
        neat_logger('The context encoder and question encoder .pth files have been found.')

    # 지문 임베딩(passage embeddings)을 저장한 bin 파일이 없으면 새로이 만듭니다.
    if not os.path.isfile(indexer_path) or not os.path.isfile(retriever_args.dpr_ctx_embeddings_path):
        neat_logger('Indexer file or context embedding file have not been found.')
        neat_logger('Building ctx_embds setup..')
        ctx_encoder.load_state_dict(torch.load(retriever_args.dpr_ctx_encoder_path))

        # Wikipedia documents 파일 불러오기
        neat_logger('Loading wikipedia documents..')
        with open(data_args.context_path, 'r', encoding='utf-8') as f:
            wiki = json.load(f)
        search_corpus = list(dict.fromkeys([v['text'] for v in wiki.values()]))

        neat_logger(
            'Constructing wiki docs tokenizer, dataset, and dataloader..'
        )
        eval_ctx_seqs = tokenizer(
            search_corpus,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        eval_dataset = TensorDataset(
            eval_ctx_seqs['input_ids'],
            eval_ctx_seqs['attention_mask'],
            eval_ctx_seqs['token_type_ids'],
        )

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=retriever_args.dpr_batch_size,
        )

        ctx_embds = []
        with torch.no_grad():
            epoch_iterator = tqdm(
                eval_dataloader,
                desc='Building passage embeddings',
                position=0,
                leave=True,
            )
            ctx_encoder.eval()

            for batch in epoch_iterator:
                batch = tuple(b.cuda() for b in batch)

                ctx_inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                }

                outputs = ctx_encoder(**ctx_inputs).to('cpu').numpy()
                ctx_embds.extend(outputs)
        ctx_embds = np.array(ctx_embds)

        neat_logger('Saving passage embeddings..')
        with open(retriever_args.dpr_ctx_embeddings_path, 'wb') as f:
            pickle.dump(ctx_embds, f)
    else:
        neat_logger('Indexer file or context embedding file have been detected.')

    # Faiss index 파일을 만들고 저장합니다.
    neat_logger('Defininng Faiss retriever..')
    retriever = FaissRetrieval(
        ctx_embeddings_path=retriever_args.dpr_ctx_embeddings_path,
        indexer_path=indexer_path
    )

    q_encoder.load_state_dict(torch.load(retriever_args.dpr_q_encoder_path))

    df = retriever.retrieve(
        eval_dataset,
        context_path=data_args.context_path,
        tokenizer=tokenizer,
        q_encoder=q_encoder,
        top_k=data_args.top_k_retrieval,
    )
    neat_logger(f'DataFrame shape: {df.shape}')

    # 예제
    neat_logger('Loading wikipedia documents..')
    with open(data_args.context_path, 'r', encoding='utf-8') as f:
        wiki = json.load(f)
    search_corpus = list(dict.fromkeys([v['text'] for v in wiki.values()]))

    neat_logger('Examining a sample case..')
    # query = '금강산의 겨울 이름은?'
    question = '왕페이의 조부는?'
    neat_logger(f'Question: {question}')

    doc_tuple = retriever.retrieve(
        question,
        context_path=data_args.context_path,
        tokenizer=tokenizer,
        q_encoder=q_encoder,
        top_k=data_args.top_k_retrieval,
    )
    neat_logger(doc_tuple)
    neat_logger(f'Doc scores\n{doc_tuple[0]}')
    neat_logger(f'Docs\n{doc_tuple[1]}')
    neat_logger('Dense retrieval test has been ended.')


if __name__ == '__main__':
    main()
