import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from argparse import ArgumentParser
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch

try:
    # Keep compatibility with the original package layout.
    from test.baselines.STAGANN.stkriging import launch_training
except ModuleNotFoundError:
    # Support running this standalone refactor directly from baselines/STAGANN.
    from stkriging import launch_training

#torch.set_num_threads(1) # aviod high cpu avg usage


#
def parse_args():
    parser = ArgumentParser(description="Run time series forecasting model in BasicTS framework!")

    parser.add_argument("-c", "--cfg", default="examples/STAGANN/STAGANN_PEMS03.py", help="training config")

    parser.add_argument("--gpus", default="0", help="visible gpus")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    launch_training(args.cfg, args.gpus)
