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

Experiments are conducted on the TraffiKrig benchmark, including PEMS03,
PEMS04, PEMS07, and PEMS08.

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


