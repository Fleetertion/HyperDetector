# HyperDetector

HyperDetector is an APT detection framework extended with hypergraph neural networks and enhanced global perception modules.

This repository is the implementation for the project **"Advanced Persistent Threat Detection via Hypergraph Neural Networks with Enhanced Global Perception"**.

## Highlights

- Hypergraph-based message passing for higher-order dependency modeling.
- Optional Block Self-Attention (BSA) for enhanced global perception.
- Added support for `Clearscope` and `APT2021` in addition to `Wget`, `TRACE`, `THEIA`, and `CADETS`.
- Preprocessed data caches and trained checkpoints are already organized in this repository.

## Repository Structure

```text
.
|-- data/                 # datasets and cached graph files
|-- model/                # model definitions, training and evaluation modules
|-- result/               # checkpoints, cache, memory bank and evaluation artifacts
|-- utils/                # parsers, config, data loading
|-- train.py              # training entry
|-- test.py               # evaluation entry
`-- requirements.txt
```

## Environment

- Python 3.8
- PyTorch 1.12.1
- DGL 1.0.0
- scikit-learn 1.2.2

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Supported Datasets

HyperDetector currently supports the following datasets:

- `wget`: Unicorn Wget batch-level anomaly detection
- `trace`: DARPA TC E3 Trace
- `theia`: DARPA TC E3 THEIA
- `cadets`: DARPA TC E3 CADETS
- `clearscope`: Clearscope provenance dataset
- `apt2021`: SCVIC-APT-2021 / APT2021 flow-level dataset

The repository already contains processed caches for several datasets under `data/`. For large raw datasets, preprocessing scripts are kept in `utils/`.

## Data Preparation

### 1. Wget

- We used the same data processing method as the MAGIC method(https://github.com/FDUDSDE/MAGIC).
- Download `attack_baseline.tar.gz` and `benign.tar.gz` from the [Wget dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IA8UOS).
- Put raw `.log` files into `data/wget/raw/`.
- Run:

```bash
python utils/wget_parser.py
```

During training or evaluation, `utils/loaddata.py` will build and reuse `graphs.pkl`.

### 2. DARPA TC E3 (`trace`, `theia`, `cadets`)
- We used the same data processing method as the MAGIC method. Evaluation on the DARPA TC datasets using the ThreaTrace label.
- Download raw logs from the [DARPA Transparent Computing release](https://github.com/darpa-i2o/Transparent-Computing).
- Place files into the corresponding folders under `data/trace/`, `data/theia/`, and `data/cadets/`.
- Prepare ground-truth files such as `trace.txt`, `theia.txt`, and `cadets.txt`.
- Parse raw data with the corresponding parser.(https://github.com/threaTrace-detector/threaTrace/)

Note:

- Keep auxiliary log shards even if they are not directly used for train/test splits, because entity definitions may be referenced later in the event stream.
- Metadata and cached graphs are stored under each dataset folder.

### 3. Clearscope

- Put benign raw files into `data/clearscope/benign/`.
- Put attack raw files into `data/clearscope/attack/`.
- Run:

```bash
python utils/clearscope_parser.py
```

Optional quick parsing example:

```bash
python utils/clearscope_parser.py
```

### 4. APT2021

This repository already includes the HyperDetector-specific APT2021 pipeline in `utils/apt2021_pipeline.py`.


## Training

### Train from scratch

```bash
python train.py --dataset dataset
```

## Evaluation

Run evaluation with:

```bash
python test.py --dataset dataset
```


## Citation

If you use this repository in your research, please cite the corresponding paper/project:

```bibtex
@inproceedings{wu2026hyperdetector,
  title={HyperDetector: Advanced Persistent Threat Detection via Hypergraph Neural Networks with Enhanced Global Perception},
  author={Wu, Ziyue and Wang, Nan and Liu, Jiqiang and Dong, Hairong and Zhao, Xibin},
  booktitle={Proceedings of the ACM Web Conference 2026},
  pages={2673--2682},
  year={2026}
}
```