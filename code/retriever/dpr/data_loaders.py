from typing import Optional

import numpy as np
from datasets import load_from_disk
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from utils.utils import neat_logger


class BaseDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        tokenizer: PreTrainedTokenizerBase,
        in_batch_negatives: Optional[bool] = True,
        num_neg_samples: int = 3,
    ):
        dataset = load_from_disk(dataset_path=dataset_path)

        if in_batch_negatives:
            neat_logger('Constructing in-batch negatives..')
            # corpus = np.array(list(set([example for example in dataset["context"]])))
            corpus = np.array(list(dict.fromkeys(dataset['context']).keys()))
            p_with_negs = []
            for base_passage in tqdm(dataset['context'], desc='Iteration for in-batch negatives'):
                while True:
                    idx_negs = np.random.randint(len(corpus), size=num_neg_samples)

                    if not base_passage in corpus[idx_negs]:
                        negative_passages = corpus[idx_negs]
                        p_with_negs.append(base_passage)
                        p_with_negs.extend(negative_passages)
                        break

        ctx_seqs = tokenizer(
            p_with_negs if in_batch_negatives else dataset['context'],
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        q_seqs = tokenizer(
            dataset['question'],
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        if in_batch_negatives:
            max_len = ctx_seqs['input_ids'].size(-1)  # ctx_seqs['input_ids'] ~ (bsz, num_negs+1, max_len)
            ctx_seqs['input_ids'] = ctx_seqs['input_ids'].view(-1, num_neg_samples + 1, max_len)
            ctx_seqs['attention_mask'] = ctx_seqs['attention_mask'].view(-1, num_neg_samples + 1, max_len)
            ctx_seqs['token_type_ids'] = ctx_seqs['token_type_ids'].view(-1, num_neg_samples + 1, max_len)

        self.dataset = TensorDataset(
            ctx_seqs['input_ids'], ctx_seqs['attention_mask'], ctx_seqs['token_type_ids'],
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
