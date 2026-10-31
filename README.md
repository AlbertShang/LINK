# LINK: Lightweight and Inductive Node-to-Node Transition Learning for Traffic Kriging

LINK is a lightweight and inductive traffic kriging paradigm that formulates
traffic state reconstruction as a transferable node-to-node transition learning problem.

Instead of relying on implicit neighborhood aggregation, LINK explicitly
learns a shared propagation rule to model traffic-state transitions
between sensors. The learned rule can be transferred across different
source-destination pairs and propagation depths, enabling efficient
inductive kriging on unseen locations.

LINK instantiates this framework with a Spectral Propagation Network
(SPN), which captures dominant temporal patterns through spectral
propagation and recovers short-term variations with residual refinement.

Experiments are conducted on the TraffiKrig benchmark, including TraffiKrig-03,
TraffiKrig-04, TraffiKrig-07, and TraffiKrig-08.

## Repository Structure

``` text
LINK/
├── baselines/              # Baseline implementations
├── configs/                # Experiment configurations
├── data/                   # Data preprocessing scripts
├── experiments/            # Training and evaluation scripts
├── src/                    # LINK model implementation
├── outputs/                # Results and checkpoints
└── requirements.txt
```

## Environment

LINK is implemented with Python 3.10+ and PyTorch.

Install dependencies:

``` bash
pip install -r requirements.txt
```

## Dataset Preparation

The raw TraffiKrig data should be organized as:

``` text
archive/
├── node_order.npy
├── sensor_meta_feature.csv
├── adj_matrix.npy
└── year_2023/
    └── year_2023/
        ├── 2023_p01.npy
        ├── 2023_p02.npy
        ├── 2023_p03.npy
        └── 2023_p04.npy
```

Run preprocessing:

``` bash
python data/preprocess.py
```

The processed datasets will be generated under:

``` text
pure_dataset_topk/
├── pems03/
├── pems04/
├── pems07/
└── pems08/
```

## Training LINK

Run LINK on a specific dataset:

``` bash
python experiments/main.py --districts 3
```

Run all datasets:

``` bash
python experiments/main.py
```

## Baseline Experiments

Metadata-enhanced baselines:

``` bash
python baselines/STAGANN/examples/run_all_metadata.py     --districts 3 4 7 8     --top-k 3
```

Original baselines without metadata:

``` bash
python baselines/STAGANN/examples/run_all_no_metadata.py     --districts 3 4 7 8     --top-k 3
```

## Evaluation

Evaluate a trained LINK checkpoint:

``` bash
python experiments/main.py     --mode evaluate     --districts 3
```

Metrics:

-   MAE
-   RMSE
-   MAPE

## Cross-dataset Transfer Ablation

Freeze a PEMS03 checkpoint and evaluate it directly on PEMS04/07/08 without
training or fine-tuning on the target districts:

``` bash
python experiments/main.py --districts 3
python experiments/cross_dataset_transfer.py --include-source
```

To reuse an existing PEMS03 checkpoint, including an ablation variant:

``` bash
python experiments/cross_dataset_transfer.py --checkpoint outputs/pems03/best_model.pt --target-districts 4 7 8 --include-source
```

The experiment retains the source model configuration, weights, flow scale,
and Top-K. Each target uses its own preprocessed road network, metadata, and
saved node splits. Target `train` nodes provide observed traffic for kriging;
they do not update model parameters. Validation/test nodes remain held out
under the existing evaluation protocol. No target labels select checkpoints
or tune hyperparameters. All current datasets start on 2023-01-01; datasets
with another calendar origin must have calendar features aligned first.

Results are isolated under `outputs/ablation_cross_dataset/pems03/`: each
district gets `metrics.json`, and `summary.csv` / `summary.json` collect the
configured metrics (MAE, RMSE, MAPE by default) and valid counts.
`--include-source` adds a PEMS03 in-domain reference. `--save-completion`
also exports completed flows and validity masks. `--evaluation-chunk` controls
inference chunk size; `--max-time-steps` is for debugging only.

For the local-training comparison, evaluate independently trained target models:

``` bash
python experiments/main.py --mode evaluate --districts 4 7 8
```

Use the same model settings, Top-K, splits, metrics, and full time range.
Compare each target's local and transferred test errors to measure its transfer
penalty. The PEMS03 reference alone does not measure that penalty.

## Main Components

-   **Adaptive Subgraph Extraction**\
    Constructs target-specific propagation structures to avoid
    unnecessary aggregation.

-   **Shared Propagation Rule Learning**\
    Learns transferable node-to-node transition rules reused across
    propagation paths.

-   **Spectral Propagation Network (SPN)**\
    Propagates dominant temporal structures in the frequency domain and
    refines short-term variations through residual learning.


