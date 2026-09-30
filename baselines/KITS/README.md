# KITS: Inductive Spatio-Temporal Kriging with Increment Training Strategy

This repository contains the code for our AAAI 2025 [paper](https://arxiv.org/abs/2311.02565) "*KITS: Inductive Spatio-Temporal Kriging with Increment Training Strategy*", where we design an increment training strategy for inductive spatio-temporal kriging.

> **Abstract**: *Sensors are commonly deployed to perceive the environment. However, due to the high cost, sensors are usually sparsely deployed. Kriging is the tailored task to infer the unobserved nodes (without sensors) using the observed nodes (with sensors). The essence of kriging task is transferability. Recently, several inductive spatio-temporal kriging methods have been proposed based on graph neural networks, being trained  based on a graph built on top of observed nodes via pretext tasks such as masking nodes out and reconstructing them. However, the graph in training is inevitably much sparser than the graph in inference that includes all the observed and unobserved nodes. The learned pattern cannot be well generalized for inference, denoted as <i>graph gap</i>. To address this issue, we first present a novel <i>Increment</i> training strategy: instead of masking nodes (and reconstructing them), we add virtual nodes into the training graph so as to mitigate the graph gap issue naturally. Nevertheless, the empty-shell virtual nodes without labels could have inferior features and lack supervision signals. To solve these issues, we pair each virtual node with its most similar observed node and fuse their features together; to enhance the supervision signal, we construct reliable pseudo labels for virtual nodes. As a result, the learned pattern of virtual nodes could be safely transferred to real unobserved nodes for reliable kriging. We name our new <u>K</u>riging model with <u>I</u>ncrement <u>T</u>raining <u>S</u>trategy as KITS. Extensive experiments demonstrate that KITS consistently outperforms existing methods by large margins, e.g., the improvement over MAE score could be as high as 18.33%.*

## Dependencies

- Python 3.8
- PyTorch 1.8.1
- PyTorch Lightning 1.4.0
- cuda 11.1
```
> conda env create -f env_{ubuntu,windows}.yaml
```

## Datasets

We utilize 8 datasets from different field in this paper:
- Traffic speed datasets:
    - METR-LA
    - PEMS-BAY
    - SEA-LOOP
- Traffic flow dataset:
    - PEMS07
- Air quality datasets (PM2.5):
    - AQI36
    - AQI
- Solar power datasets:
    - NREL-AL
    - NREL-MD

These datasets could be downloaded from this [datasets.zip](https://drive.google.com/file/d/1VQrSLNAr3qr2LAsEK1-_CbBbu6vr0G63/view?usp=sharing), and compressed to the current path.

## Usage

### Shared pure PEMS Top-K=3 benchmark

From the repository root, use the adapter below to run PEMS03, PEMS04, PEMS07,
and PEMS08 sequentially. Each district runs both `no_metadata` and `metadata`
by default:

```bash
python baselines/KITS/run_pure.py --districts 3 4 7 8 --top-k 3 --device cuda
```

It reads `pure_dataset_topk/pemsXX/district_X_flow.npy`,
`district_X_adj_inout_topk3.npy`, `adj_index.pkl`, and
`district_X_node_metadata.npz` directly. The node type and
direction one-hot features are copied from that artifact, while the freeway
one-hot vocabulary is fitted on training nodes only; unseen freeway IDs use
an unknown bucket. A small projection adds these features to KITS's initial
node representation. Virtual training nodes receive zero metadata. The
`no_metadata` variant uses the same model parameters and dimensions, but zeros
all real-node metadata, as in the other baselines' metadata ablations. The
projection has no bias, so zero metadata adds exactly zero to the original
KITS node representation. KITS is trained
only on `train_nodes` with its original virtual-node increment/cycle objective.
Validation uses `train_nodes + valid_nodes`, and testing uses
`train_nodes + test_nodes`; only the held-out nodes contribute to the scores.
The shared 24-step windows and 70% temporal training split match the other
PEMS baselines. MAE, RMSE, and percent MAPE are reported in the original flow
units, excluding zero targets, and the best validation-MAE checkpoint is used
for testing. Both variants have separate checkpoints and metrics JSON under
`baselines/KITS/results/pemsXX_topk3/{no_metadata,metadata}/`; a combined CSV
is written to `baselines/KITS/results/topk3_pure_results.csv`.

To run only one variant, add `--modes metadata` or `--modes no_metadata`.
Training stops after 10 consecutive epochs without validation-MAE improvement
by default (`--patience 10`), then tests the best checkpoint. To evaluate an
existing checkpoint without retraining, use `--test-only`, for example:

```bash
python baselines/KITS/run_pure.py --districts 3 --modes no_metadata --top-k 3 --test-only
```

For a checkpoint saved by the earlier no-metadata runner, pass its path with
`--checkpoint`. The test-only path also accepts that runner's model without
metadata-projection weights.

```bash
python baselines/KITS/run_pure.py --districts 3 --modes no_metadata --top-k 3 \
  --test-only --checkpoint baselines/KITS/results/pems03_topk3/best.pt
```

The default batch size is 4 to limit memory use on PEMS07. Adjust
`--batch-size` and `--eval-batch-size` if needed. Use `--dry-run` to list input
paths without starting training.

### Original KITS datasets

- Run the following commands for training and testing.

- **Training**:
    - E.g., train KITS with missing ratio of 0.5:
      ```
      python train.py --config config/kits/{la_point, bay_point, sea_loop_point, pems07_point, aqi36, aqi, nrel_al_point, nrel_md_point}.yaml --miss-rate 0.5 --lr 0.0002 --patience 50
      ```

- **Testing**
    - The pretrained KITS (with random seed 1, with missing ratio 0.5) could be downloaded from [final_model.zip](https://drive.google.com/file/d/1uj74MTy6zukWrnwQ_3zoMP67hurtnerP/view?usp=sharing).
    - E.g., test KITS with missing ratio of 0.5:
      ```
      python train.py --config config/kits/{la_point, bay_point, sea_loop_point, pems07_point, aqi36, aqi, nrel_al_point, nrel_md_point}.yaml --pretrained-model final_model/{la_point, bay_point, sea_loop_point, pems07_point, aqi36, aqi, nrel_al_point, nrel_md_point}/final/seed1_rm0.5/best.ckpt
      ```

## References

This repo is mainly built based on [grin](https://github.com/Graph-Machine-Learning-Group/grin). Thanks for their great work!
