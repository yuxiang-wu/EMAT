#!/bin/bash

EXP=$1

echo $EXP

python cat/retriever/embed.py \
    --model_name_or_path "checkpoints/${EXP}/latest_ckpt" \
  	--qas_to_embed "data/paq/PAQ_L1/PAQ_L1.filtered.jsonl" \
  	--output_path "checkpoints/${EXP}/embeddings/paq-l1.key.pt" \
  	--batch_size 1024 -v

python cat/retriever/build_index.py \
    --embeddings_dir "checkpoints/${EXP}/embeddings/paq-l1.key.pt" \
    --output_path "checkpoints/${EXP}/indices/paq-l1.key.hnsw.faiss" \
    --hnsw --SQ8 --store_n 32 --ef_construction 128 --ef_search 128 \
    --verbose