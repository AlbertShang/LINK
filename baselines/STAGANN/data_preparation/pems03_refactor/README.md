# Pure PEMS → STAGANN conversion

`generate_training_data.py` converts the selected pure PEMS district data into
the artifact layout read by STAGANN. It preserves the pure dataset's configured
directed in/out Top-K weighted adjacency and its saved train/valid/test node split.

From the workspace root, generate PEMS03 with:

```powershell
python baselines/STAGANN/data_preparation/pems03_refactor/generate_training_data.py
```

This writes `baselines/STAGANN/datasets/pems03_metadata_topk5_71_24/`. The metadata
dataset is intentionally separate and never replaces the older
`pems03_refactor_71_24/` artifacts. The default
format matches the original STAGANN PEMS03 preparation: 24-step windows, 70%
of time windows for training, the remaining windows shared by validation/test,
standard normalization, and flow/TOD/DOW channels.

It builds `metadata.pkl` from the new canonical
`district_<id>_node_metadata.npz` artifact.  The node vector contains exactly
the static metadata retained by `src`: Type (5-way one-hot), Direction (9-way
one-hot), and a train-node-only Freeway vocabulary plus an unknown bucket.
For the current PEMS03 split this is 22 dimensions (`5 + 9 + 7 + 1`), so a
sample has 25 channels: `[flow, metadata(22), TOD, DOW]`. Lat/Lng and station
ID are not model inputs.

The pure PEMS03 flow contains missing readings.  Before normalization, the
converter fills each node's gaps by temporal linear interpolation (using the
nearest valid value for leading/trailing gaps).  The raw pure files remain
unchanged.  Use `--missing-value-strategy error` to stop instead of imputing.

To adapt another pure district after its input directory has been created,
change only `--district`:

```powershell
python baselines/STAGANN/data_preparation/pems03_refactor/generate_training_data.py --district 4
python baselines/STAGANN/data_preparation/pems03_refactor/generate_training_data.py --district 7
python baselines/STAGANN/data_preparation/pems03_refactor/generate_training_data.py --district 8
```

Each district needs `district_<id>_flow.npy`,
`district_<id>_adj_inout_topk<K>.npy`, `preprocess_config.json`,
`district_<id>_node_metadata.npz`, and `adj_index.pkl` in
`pure_dataset/pems0<id>/` (the additionally nested extracted layout
`pure_dataset/pure_dataset/pems0<id>/` is detected as well). Existing output
artifacts are protected; pass `--overwrite` only when intentionally regenerating them.

After installing STAGANN's Python dependencies, run every metadata variant on
PEMS04/07/08 from the workspace root:

```powershell
python baselines/STAGANN/examples/run_all_metadata.py --districts 4 7 8 --gpus 0
```

Use `--districts 3 4 7 8` for the complete four-dataset experiment. Omitting
`--districts` remains backward compatible and runs PEMS03 only.

The command runs DualSTN, GCN, GIN, GRIN, IAGCN, IGNNK, INCREASE, ordinary
kriging, SATCN, and STAGANN in isolated sequential subprocesses. It creates
each converted dataset from `pure_dataset/pems0X/` when missing and
automatically regenerates stale artifacts. Checkpoints and result CSVs are
isolated per dataset under
`checkpoints/pems0X_metadata_topk5_71_24/metadata/`, so neither datasets nor
legacy experiment outputs overwrite one another.

Useful checks and subsets:

```powershell
python baselines/STAGANN/examples/run_all_metadata.py --dry-run
python baselines/STAGANN/examples/run_all_metadata.py --districts 4 7 8 --models GCN GIN STAGANN --gpus 0
python baselines/STAGANN/examples/run_all_metadata.py --districts 4 --overwrite-data --gpus 0
```

The original model implementations and `*_PEMS0X.py` configurations remain
unchanged. Every metadata experiment has a separate `*_PEMS0X_METADATA.py`
configuration. `MetadataSTKrigingDataset` broadcasts the aligned static node
vector over each 24-step window and appends it immediately after traffic flow,
producing `[flow, metadata..., TOD, DOW]`. Each model uses its own explicit
metadata architecture (`MetadataGCN`, `MetadataGIN`, etc.), which learns a
metadata-conditioned traffic-flow representation before invoking its unchanged
legacy backbone. All original optimizer, scheduler, epoch, masking,
batch-size, worker, gradient-clipping, and early-stopping settings are retained.
For direct comparison with `src`, metadata runs report MAPE as a percentage
(`14.9` means `14.9%`); legacy STAGANN logs report the unmultiplied ratio.

To start only the PEMS07 STAGANN metadata configuration from `baselines/STAGANN`:

```powershell
python examples/run.py -c examples/STAGANN/STAGANN_PEMS07_METADATA.py --gpus 0
```
