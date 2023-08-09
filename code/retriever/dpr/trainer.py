import json
import os
import sys
from typing import List, Optional, Tuple
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from model import BertEncoder
from utils.utils import neat_logger


class BiEncoderTrainer:
    def __init__(
        self,
        args: TrainingArguments = None,
        train_dataset: torch.utils.data.Dataset = None,
        eval_dataset: torch.utils.data.Dataset = None,
        tokenizer: PreTrainedTokenizerBase = None,
        ctx_encoder: BertEncoder = None,
        q_encoder: BertEncoder = None,
        in_batch_negatives: bool = True,
        num_neg_samples: int = 3,
        ctx_encoder_path: str = 'models/ctx_encoder.pth',
        q_encoder_path: str = 'models/q_encoder.pth',
        context_path: str = '../data/wikipedia_documents.json',
    ) -> None:
        """Question 및 given passages 각각의 encoders를 train 하기 위한 클래스입니다."""
        self.args = args

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.tokenizer = tokenizer

        self.in_batch_negatives = in_batch_negatives
        self.num_neg_samples = num_neg_samples

        self.ctx_encoder = ctx_encoder
        self.q_encoder = q_encoder
        self.ctx_encoder_path = ctx_encoder_path
        self.q_encoder_path = q_encoder_path

        with open(context_path, 'r', encoding='utf-8') as f:
            wiki = json.load(f)

        self.search_corpus = list(dict.fromkeys([v['text'] for v in wiki.values()]))

        neat_logger('Start tokenizing wiki docs..')
        self.wiki_tokens = self.tokenizer(
            self.search_corpus, padding='max_length', truncation=True, return_tensors='pt'
        )
        neat_logger('Tokenizing wiki docs has been finished.')

    def _training_epoch(
        self,
        epoch_iterator: DataLoader,
        optimizer: Optimizer,
        scheduler: _LRScheduler
    ) -> float:
        batch_size = self.args.per_device_train_batch_size

        batch_loss = 0.0

        if self.in_batch_negatives:
            for training_step, batch in enumerate(epoch_iterator):
                self.ctx_encoder.train()
                self.q_encoder.train()

                if torch.cuda.is_available():
                    batch = tuple(b.cuda() for b in batch)
                    targets = torch.zeros(batch_size).long()
                    targets = targets.cuda()

                ctx_inputs = {
                    'input_ids': batch[0].view(batch_size * (self.num_neg_samples + 1), -1),
                    'attention_mask': batch[1].view(batch_size * (self.num_neg_samples + 1), -1),
                    'token_type_ids': batch[2].view(batch_size * (self.num_neg_samples + 1), -1)
                }
                q_inputs = {
                    'input_ids': batch[3],
                    'attention_mask': batch[4],
                    'token_type_ids': batch[5]
                }

                ctx_outputs = self.ctx_encoder(**ctx_inputs)  # (batch_size, emb_dim)
                q_outputs = self.q_encoder(**q_inputs)

                ctx_outputs = ctx_outputs.view(batch_size, (self.num_neg_samples + 1), -1)
                q_outputs = q_outputs.view(batch_size, 1, -1)

                similarity_scores = torch.bmm(q_outputs, torch.transpose(ctx_outputs, 1, 2)).squeeze()
                similarity_scores = similarity_scores.view(batch_size, -1)
                similarity_scores = F.log_softmax(similarity_scores, dim=1)

                loss = F.nll_loss(similarity_scores, targets)

                batch_loss += loss.item()

                loss.backward()

                if (training_step + 1) % self.args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    self.ctx_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                epoch_iterator.set_description(
                    f"Loss {loss:.04f} at step {training_step}"
                )

                del ctx_inputs, q_inputs

            torch.cuda.empty_cache()

            return batch_loss / len(epoch_iterator)

    def train(self) -> Tuple[BertEncoder, BertEncoder]:
        train_dataloader = self.get_train_dataloader()

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.ctx_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay
            },
            {
                'params': [p for n, p in self.ctx_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            },
            {
                'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay
            },
            {
                'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        training_total = len(train_dataloader) * self.args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=(training_total * self.args.warmup_ratio),
            num_training_steps=training_total
        )
        
        if torch.cuda.is_available():
            self.ctx_encoder.cuda()
            self.q_encoder.cuda()

        self.ctx_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(self.args.num_train_epochs)), desc='Epoch')
        best_score = 0

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc='Iter', leave=True)

            train_loss = self._training_epoch(epoch_iterator=epoch_iterator, optimizer=optimizer, scheduler=scheduler)
            neat_logger(f'Train loss: {train_loss:.4f}')

            top_1, top_5, top_10, top_30, top_50, top_100 = self.evaluate()

            if top_100 > best_score:
                self.save_model_weights()
                best_score = top_100

        return self.ctx_encoder, self.q_encoder

    def save_model_weights(self) -> None:
        torch.save(self.ctx_encoder.state_dict(), self.ctx_encoder_path)
        torch.save(self.q_encoder.state_dict(), self.q_encoder_path)
        # self.ctx_encoder.save_pretrained(self.ctx_encoder_path)
        # self.q_encoder.save_pretrained(self.q_encoder_path)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        """
        if self.train_dataset is None:
            raise ValueError('Trainer: training requires a train_dataset.')
        else:
            train_sampler = RandomSampler(self.train_dataset)

        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            drop_last=True,
        )

        return data_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                If provided, will override `self.eval_dataset`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        sampler = SequentialSampler(eval_dataset)

        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.args.per_device_eval_batch_size,
            drop_last=True,
        )

        return data_loader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test :class:`~torch.utils.data.DataLoader`.

        Args:
            test_dataset (obj:`Dataset`): The test dataset to use.
        """
        # We use the same batch_size as for eval.
        sampler = SequentialSampler(test_dataset)

        data_loader = DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=self.args.per_device_eval_batch_size,
            drop_last=True,
        )

        return data_loader

    def evaluate(self) -> List[float]:
        batch_size = self.args.per_device_eval_batch_size

        with torch.no_grad():
            self.ctx_encoder.eval()
            self.q_encoder.eval()

            question = self.eval_dataset['question']
            gold_passage = self.eval_dataset['context']

            q_seqs_eval = self.tokenizer(
                question, padding='max_length', truncation=True, return_tensors='pt'
            ).to('cuda')
            q_emb = self.q_encoder(**q_seqs_eval).to('cpu')  # (num_questions, emb_dim)

            wiki_iterator = TensorDataset(
                self.wiki_tokens['input_ids'],
                self.wiki_tokens['attention_mask'],
                self.wiki_tokens['token_type_ids']
            )
            wiki_dataloader = DataLoader(wiki_iterator, batch_size=batch_size)

            ctx_embs = []
            for context in tqdm(wiki_dataloader):
                if torch.cuda.is_available():
                    context = tuple(c.cuda() for c in context)

                ctx_inputs = {
                    'input_ids': context[0],
                    'attention_mask': context[1],
                    'token_type_ids': context[2]
                }

                ctx_emb = self.ctx_encoder(**ctx_inputs)
                ctx_embs.append(ctx_emb)

            ctx_embs = torch.cat(ctx_embs, dim=0).view(len(wiki_iterator), -1)
            ctx_embs = ctx_embs.to('cpu')  # (num_contexts, emb_dim)

            sim_scores = torch.matmul(q_emb, torch.transpose(ctx_embs, 0, 1))

            rank = torch.argsort(sim_scores, dim=1, descending=True).squeeze()

            def eval_score(k: int = 1) -> float:
                total = len(question)
                cnt = 0

                for i in range(total):
                    top_k = rank[i][:k]

                    pred_corpus = []
                    for top in top_k:
                        pred_corpus.append(self.search_corpus[top])

                    if gold_passage[i] in pred_corpus:
                        cnt += 1

                res = cnt / total
                neat_logger(f'Top-{k} score is {res:.4f}')

                return res

        neat_logger('********** Evaluation **********')
        return [
            eval_score(1),
            eval_score(5),
            eval_score(10),
            eval_score(30),
            eval_score(50),
            eval_score(100),
        ]
