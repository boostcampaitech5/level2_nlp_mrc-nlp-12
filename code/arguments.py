from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments related to the model, config, and tokenizer that we are going to fine-tune.
    """

    model_name_or_path: str = field(
        default='klue/roberta-large',
        metadata={
            'help': 'Path to pretrained model or model identifier from huggingface.co/models'
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Pretrained config name or path if not the same as model_name'
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Pretrained tokenizer name or path if not the same as model_name'
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments related to the data we are going to use for training and evaluating our model.
    """

    dataset_name: Optional[str] = field(
        default='../data/train_dataset',
        metadata={'help': 'The name of the dataset to use.'},
    )
    valid_dataset_name: Optional[str] = field(
        default='../data/train_dataset',
        metadata={'help': 'The name of the validation dataset to use.'},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={'help': 'Overwrite the cached training and evaluation sets'},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={'help': 'The number of processes to use for the preprocessing.'},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            'help': 'The maximum total input sequence length after tokenization. Sequences longer '
            'than this will be truncated, sequences shorter will be padded.'
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            'help': 'Whether to pad all samples to `max_seq_length`. '
            'If False, will pad the samples dynamically when batching to the maximum length in the batch (which can '
            'be faster on GPU but will be slower on TPU).'
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            'help': 'When splitting up a long document into chunks, how much stride to take between chunks.'
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            'help': 'The maximum length of an answer that can be generated. This is needed because the start '
            'and end predictions are not conditioned on one another.'
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={'help': 'Whether to run passage retrieval using sparse embedding.'},
    )
    num_clusters: int = field(
        default=64, metadata={'help': 'Define how many clusters to use for faiss.'}
    )
    top_k_retrieval: int = field(
        default=30,
        metadata={
            'help': 'Define how many top-k passages to retrieve based on similarity.'
        },
    )
    retrieval_method: str = field(
        default='bm25', metadata={'help': 'Choose retriever for passage retrieval'}
    )
    augment_data: Optional[str] = field(
        default=None,
        metadata={
            'help': 'New dataset name (ex. squad_kor_v1)'
        },
    )
    augment_valid_data: Optional[bool] = field(
        default=None,
        metadata={
            'help': 'Whether augment validation set or not'
        },
    )
    drop_context_duplicate: Optional[bool] = field(
        default=None,
        metadata={
            'help': 'Whether use unique context or not'
        },
    )
    retrieved_context: Optional[bool] = field(
        default=None,
        metadata={
            'help': 'Whether load retrieved context or not'
        },
    )
    sort_data: Optional[bool] = field(
        default=False,
        metadata={
            'help': 'Whether sort train dataset or not'
        },
    )
    context_path: str = field(
        default='../data/wikipedia_documents.json',
        metadata={
            'help': 'Path to the json file containing the context (Wikipedia documents) for retrieval.'
        },
    )
    train_dataset_path: Optional[str] = field(
        default='../data/train_dataset/train',
        metadata={
            'help': 'The name of the train dataset to use.'
        },
    )
    eval_dataset_path: Optional[str] = field(
        default='../data/train_dataset/validation',
        metadata={
            'help': 'The name of the eval dataset to use.'
        },
    )

@dataclass
class RetrieverArguments:
    """
    Arguments for dense passage retriever
    """
    dpr_num_train_epochs: int = field(
        default=20,
        metadata={
            'help': 'The number of training epochs for bi-encoders.'
        },
    )
    dpr_batch_size: int = field(
        default=4,
        metadata={
            'help': 'The batch size for training the bi-encoders.'
        },
    )
    dpr_num_neg_samples: int = field(
        default=7,
        metadata={
            'help': 'The number of neg samples to use during training.'
        },
    )
    dpr_num_faiss_clusters: int = field(
        default=48,
        metadata={
            'help': 'The number of FAISS clusters for quantization.'
        },
    )
    dpr_model_checkpoint: str = field(
        default='klue/bert-base',
        metadata={
            'help': 'The checkpoint or name of the pre-trained model to be loaded.'
        },
    )
    dpr_encoder_save_dir: str = field(
        default='models',
        metadata={
            'help': 'Directory where the trained DPR encoder models should be saved.'
        },
    )
    dpr_learning_rate: float = field(
        default=1e-5,
        metadata={
            'help': 'Learning rate for training the DPR models.'
        },
    )
    dpr_weight_decay: float = field(
        default=0.01,
        metadata={
            'help': 'Weight decay rate for training the DPR models.'
        },
    )
    dpr_ctx_encoder_path: str = field(
        default='models/ctx_encoder.pth',
        metadata={
            'help': 'Path to the pre-trained context encoder model.'
        },
    )
    dpr_q_encoder_path: str = field(
        default='models/q_encoder.pth',
        metadata={
            'help': 'Path to the pre-trained query encoder model.'
        },
    )
    dpr_ctx_embeddings_path: str = field(
        default='../data/passage_embeddings.bin',
        metadata={
            'help': 'Path to the pre-computed context embeddings.'
        },
    )
    dpr_gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            'help': 'Number of update steps to accumulate gradients before performing a backward/update pass.'
        }
    )
    dpr_in_batch_negatives: bool = field(
        default=True,
        metadata={
            'help': ''
        }
    )
    dpr_warmup_ratio: float = field(
        default=0.06,
        metadata={
            'help': ''
        }
    )


@dataclass
class WandbArguments:
    """
    Arguments for wandb setting
    """
    wandb_project_name: str = field(
        default='mrc_test',
    )
    wandb_entity: Optional[str] = field(
        default='salmons',
    )
    wandb_run_name: str = field(
        default='baseline+top_k_30'
    )
