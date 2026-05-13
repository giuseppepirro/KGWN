# KGWN
Reference implementation Knowledge Graph Wave Networks (KGWN), with
inductive entity and relation reasoning, caching, and evaluation utilities.

## Repository Layout

- `src/inductive/`: core model, data pipeline, metrics, and caching.
- `src/train_inductive.py`: entity-induction training (link prediction).
- `src/train_inductive_relation.py`: relation-induction training (relation prediction).
- `src/precompute_inductive_subgraphs.py`: precompute subgraph caches (train/val/test).
- `src/generate_inductive_text_features.py`: entity text features.
- `src/generate_inductive_relation_text_features.py`: relation text features.
- `src/eval_inductive.py`: offline evaluation for saved checkpoints.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional (needed for BERT text features):
```bash
pip install transformers
```

If you want to avoid HuggingFace dependencies, use hash features:
`--entity_text_use_hash` and/or `--relation_text_use_hash`.

## Data Layout

Inductive datasets live under:
- `src/data/inductive/entities/<dataset>`
- `src/data/inductive/relations/<dataset>`

Entity-induction datasets (examples):
- `fb237_v1_ind`, `fb237_v2_ind`, `fb237_v3_ind`, `fb237_v4_ind`
- `wn18rr_v1_ind`, `wn18rr_v2_ind`, `wn18rr_v3_ind`, `wn18rr_v4_ind`

Relation-induction datasets (examples):
- `FB-25`, `FB-50`, `FB-75`, `FB-100`
- `NL-0`, `NL-25`, `NL-50`, `NL-75`, `NL-100`
- `WK-25`, `WK-50`, `WK-75`, `WK-100`
- `NELL-995-v1`

If a dataset name ends with `_ind`, the provided train/valid/test splits are used.
Otherwise, an entity-disjoint split is generated and cached under
`<dataset>/inductive/pTest...`.

## Dataset Downloads

This release does not include datasets. Download sources:
- FB15K-237: https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip
- WN18RR: https://github.com/TimDettmers/ConvE/raw/master/WN18RR.tar.gz
- Grail datasets: https://github.com/kkteru/grail/tree/master/data
- InGram datasets: https://github.com/bdi-lab/InGram/tree/main/data
These URLs are also listed in `src/datasets.py`. For other inductive splits,
use your own data and follow the folder layout above.

## Entity Induction (Link Prediction)

Train on all four versions by passing the base name:
```bash
python src/train_inductive.py \
  --induction_kind entity \
  --dataset fb237 \
  --device mps \
  --use_two_stream \
  --use_frequency_adaptation \
  --k_hops 2 \
  --batch_pos 128 \
  --neg_per_pos 2 \
  --eval_neg_per_pos -1
```

Notes:
- `--device mps` uses Apple Metal (MPS).
- `--use_two_stream` enables the lifted (direction-preserving) operator.
- Full ranking: `--eval_neg_per_pos -1`.

## Relation Induction (Relation Prediction)

Passing `FB` expands to `FB-25/50/75/100` automatically:
```bash
python src/train_inductive_relation.py \
  --dataset FB \
  --device mps \
  --use_two_stream \
  --use_frequency_adaptation \
  --k_hops 2 \
  --batch_size 64 \
  --rel_neg_per_pos 4 \
  --rel_margin 1.0
```

## Caching and Precomputation

Subgraphs and relation graphs can be cached on disk. By default:
- Subgraphs: `<split_dir>/subgraphs_k{K}/{train|test}`
- Relation graphs: `<split_dir>/relation_graphs/`

Precompute subgraphs (entity induction, all versions):
```bash
python src/precompute_inductive_subgraphs.py \
  --induction_kind entity \
  --dataset fb237 \
  --k_hops 2 \
  --use_augmented_graph \
  --neg_per_pos 2 \
  --neg_epochs 50 \
  --eval_neg_per_pos 99 \
  --eval_epochs 50 \
  --num_workers 8 \
  --precompute_relation_graphs
```

Precompute subgraphs (relation induction, all versions):
```bash
python src/precompute_inductive_subgraphs.py \
  --induction_kind relation \
  --dataset FB \
  --k_hops 2 \
  --use_augmented_graph \
  --neg_per_pos 2 \
  --neg_epochs 50 \
  --eval_neg_per_pos 99 \
  --eval_epochs 50 \
  --num_workers 8 \
  --precompute_relation_graphs
```

Negative sampling is deterministic per epoch (hash-based). To maximize cache hits,
precompute with the same `--neg_epochs` and `--eval_epochs` you plan to train with.

## Text Features

Entity and relation text features are generated automatically if missing.
You can also run them explicitly:
```bash
python src/generate_inductive_text_features.py --dataset fb237_v1_ind --induction_kind entity
python src/generate_inductive_relation_text_features.py --dataset FB-25 --induction_kind relation
```

## Outputs

- Entity induction checkpoints: `checkpoints/inductive/<dataset>_seedXX_<timestamp>/`
- Relation induction checkpoints: `src/checkpoints/inductive_relation/<dataset>_seedXX/`

Each run directory includes:
`best_model.pt`, `config.json`, `results.json`, and a `summary.txt` (entity induction).
