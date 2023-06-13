#!/bin/bash
### install requirements for pstage3 baseline
# pip requirements
pip install torch==2.0.1
pip install datasets==2.12.0
pip install transformers==4.29.2
pip install --upgrade accelerate
pip install --upgrade evaluate
pip install tqdm
pip install pandas
pip install scikit-learn
pip install omegaconf
pip install wandb
pip install huggingface-hub

# faiss install (if you want to)
pip install faiss-gpu
pip install rank_bm25