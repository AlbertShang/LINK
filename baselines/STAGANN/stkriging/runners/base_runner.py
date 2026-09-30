import os
import json
import time
from pathlib import Path
from typing import Dict, Optional, Union, Tuple

import setproctitle
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from easytorch import Runner
from easytorch.utils import master_only, TimePredictor, get_local_rank, is_master
from easytorch.core.data_loader import build_data_loader
from easytorch.core.checkpoint import save_ckpt
from easytorch.utils.data_prefetcher import DevicePrefetcher


class BaseRunner(Runner):
    """
        An expanded easytorch runner for benchmarking time series models.
            - Support test loader and test process.
            - Support early stopping via TRAIN.EARLY_STOPPING config.
            - Log total training time at the end of training.
    """

    def __init__(self, cfg: dict):
        """Init

        Args:
            cfg (dict): all in one configurations
        """

        super().__init__(cfg)

        # validate every `val_interval` epoch
        self.val_interval = cfg["VAL"].get("INTERVAL", 1) if hasattr(cfg, "VAL") else None
        # test every `test_interval` epoch
        self.test_interval = cfg["TEST"].get("INTERVAL", 1) if hasattr(cfg, "TEST") else None

        # declare data loader
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None

        # early stopping state (configured in init_training)
        self.early_stopping_enabled = False
        self.early_stop_patience = 10
        self.early_stop_min_delta = 0.0
        self.early_stop_counter = 0
        self.early_stop = False
        self.early_stop_metric = "val_MAE"
        self.early_stop_greater_best = False

        # fit higher easy-torch version
        if not hasattr(self,"to_running_device"):
            from easytorch.device import to_device
            self.to_running_device = to_device

        # set proctitle
        proctitle_name = "{0}({1})".format(cfg["MODEL"].get(
            "NAME", " "), cfg.get("DATASET_NAME", " "))
        setproctitle.setproctitle("{0}@BasicTS".format(proctitle_name))

    @staticmethod
    def define_model(cfg: Dict) -> nn.Module:
        return cfg["MODEL"]["ARCH"](**cfg.MODEL.PARAM)

    def init_training(self, cfg: dict):
        """Initialize training and support test dataloader.

        Args:
            cfg (dict): config
        """

        super().init_training(cfg)
        # init test
        if hasattr(cfg, "TEST"):
            self.init_test(cfg)

        # ---- early stopping config ----
        # Supports either:
        #   CFG.TRAIN.EARLY_STOPPING = True / False  (with optional PATIENCE / MIN_DELTA)
        # or a nested dict:
        #   CFG.TRAIN.EARLY_STOPPING = {ENABLED, PATIENCE, MIN_DELTA, METRIC, GREATER_BEST}
        es_cfg = cfg["TRAIN"].get("EARLY_STOPPING", None)
        self.early_stop = False
        self.early_stop_counter = 0

        if es_cfg is None or es_cfg is False:
            self.early_stopping_enabled = False
        elif es_cfg is True:
            self.early_stopping_enabled = True
            self.early_stop_patience = int(cfg["TRAIN"].get("PATIENCE", 10))
            self.early_stop_min_delta = float(cfg["TRAIN"].get("MIN_DELTA", 0.0))
            self.early_stop_metric = cfg["TRAIN"].get("EARLY_STOP_METRIC", "val_MAE")
            self.early_stop_greater_best = bool(cfg["TRAIN"].get("EARLY_STOP_GREATER_BEST", False))
        else:
            # nested EasyDict / dict
            self.early_stopping_enabled = bool(es_cfg.get("ENABLED", True))
            self.early_stop_patience = int(es_cfg.get("PATIENCE", 10))
            self.early_stop_min_delta = float(es_cfg.get("MIN_DELTA", 0.0))
            self.early_stop_metric = es_cfg.get("METRIC", "val_MAE")
            self.early_stop_greater_best = bool(es_cfg.get("GREATER_BEST", False))

        if self.early_stopping_enabled:
            self.logger.info(
                "Early stopping enabled: metric={}, patience={}, min_delta={}, greater_best={}".format(
                    self.early_stop_metric,
                    self.early_stop_patience,
                    self.early_stop_min_delta,
                    self.early_stop_greater_best,
                )
            )

    @master_only
    def init_test(self, cfg: dict):
        """Initialize test.

        Args:
            cfg (dict): config
        """

        self.test_interval = cfg["TEST"].get("INTERVAL", 1)
        self.test_data_loader = self.build_test_data_loader(cfg)
        self.register_epoch_meter("test_time", "test", "{:.2f} (s)", plt=False)

    def build_test_data_loader(self, cfg: dict) -> DataLoader:
        """Build val dataset and dataloader.
        Build dataset by calling ```self.build_train_dataset```,
        build dataloader by calling ```build_data_loader```.

        Args:
            cfg (dict): config

        Returns:
            val data loader (DataLoader)
        """

        dataset = self.build_test_dataset(cfg)
        return build_data_loader(dataset, cfg["TEST"]["DATA"])

    @staticmethod
    def build_test_dataset(cfg: dict):
        """It can be implemented to a build dataset for test.

        Args:
            cfg (dict): config

        Returns:
            val dataset (Dataset)
        """

        raise NotImplementedError()

    def train(self, cfg: dict):
        """Train model with early stopping and total training time logging.

        Args:
            cfg (dict): config
        """

        self.init_training(cfg)
        benchmark_path = cfg["TRAIN"].get("BENCHMARK_RESULT_PATH")
        benchmark_device = (
            next(self.model.parameters()).device if benchmark_path else None
        )

        # train time predictor (for ETA of remaining epochs)
        train_time_predictor = TimePredictor(self.start_epoch, self.num_epochs)
        # wall-clock total training time
        total_train_start_time = time.time()

        # training loop
        last_epoch = self.start_epoch
        for epoch_index in range(self.start_epoch, self.num_epochs):
            epoch = epoch_index + 1
            last_epoch = epoch
            self.on_epoch_start(epoch)
            if benchmark_device is not None and benchmark_device.type == "cuda":
                torch.cuda.synchronize(benchmark_device)
                torch.cuda.reset_peak_memory_stats(benchmark_device)
            benchmark_start = time.perf_counter() if benchmark_path else None
            epoch_start_time = time.time()
            # start training
            self.model.train()

            # tqdm process bar
            if cfg.get("TRAIN.DATA.DEVICE_PREFETCH", False):
                data_loader = DevicePrefetcher(self.train_data_loader)
            else:
                data_loader = self.train_data_loader
            data_loader = (
                tqdm(data_loader)
                if get_local_rank() == 0 and not benchmark_path else data_loader
            )

            # data loop
            optimizer_steps = 0
            for iter_index, data in enumerate(data_loader):
                loss = self.train_iters(epoch, iter_index, data)
                if loss is not None:
                    self.backward(loss)
                    optimizer_steps += 1
            # update lr_scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            if benchmark_path:
                if benchmark_device.type == "cuda":
                    torch.cuda.synchronize(benchmark_device)
                seconds = time.perf_counter() - benchmark_start
                training_peak_allocated = (
                    torch.cuda.max_memory_allocated(benchmark_device)
                    if benchmark_device.type == "cuda" else None
                )
                training_peak_reserved = (
                    torch.cuda.max_memory_reserved(benchmark_device)
                    if benchmark_device.type == "cuda" else None
                )
                self.model.eval()
                with torch.inference_mode():
                    inference = self.test(benchmark=True)
                if is_master():
                    if not isinstance(inference, dict):
                        raise RuntimeError(
                            "Inference benchmark did not return timing metadata"
                        )
                    inference["batch_size"] = cfg["TEST"]["DATA"]["BATCH_SIZE"]
                    parameter_count = sum(
                        parameter.numel() for parameter in self.model.parameters()
                    )
                    parameter_bytes = sum(
                        parameter.numel() * parameter.element_size()
                        for parameter in self.model.parameters()
                    )
                    buffer_bytes = sum(
                        buffer.numel() * buffer.element_size()
                        for buffer in self.model.buffers()
                    )
                    result = {
                        "seconds": seconds, "steps": optimizer_steps,
                        "batch_size": cfg["TRAIN"]["DATA"]["BATCH_SIZE"],
                        "parameter_count": parameter_count,
                        "parameter_bytes": parameter_bytes,
                        "buffer_bytes": buffer_bytes,
                        "peak_gpu_allocated_bytes": training_peak_allocated,
                        "peak_gpu_reserved_bytes": training_peak_reserved,
                        "device": str(benchmark_device),
                        "gpu_name": (
                            torch.cuda.get_device_name(benchmark_device)
                            if benchmark_device.type == "cuda" else ""
                        ),
                        "torch_version": torch.__version__,
                        "inference": inference,
                    }
                    destination = Path(benchmark_path)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                    self.logger.info("Training epoch benchmark: {}".format(result))
                return

            epoch_end_time = time.time()
            # epoch time
            self.update_epoch_meter("train_time", epoch_end_time - epoch_start_time)
            self.on_epoch_end(epoch)

            expected_end_time = train_time_predictor.get_expected_end_time(epoch)

            # estimate training finish time
            if epoch < self.num_epochs and not self.early_stop:
                self.logger.info("The estimated training finish time is {}".format(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(expected_end_time))))

            # early stopping
            if self.early_stop:
                self.logger.info(
                    "Early stopping triggered at epoch {} "
                    "(no improvement for {} validation checks, patience={}).".format(
                        epoch, self.early_stop_counter, self.early_stop_patience
                    )
                )
                break

        # Capture validation predictions at the actual final epoch (whether the
        # loop completed normally or stopped early), rather than a fixed config epoch.
        # Use a manual validation pass instead of self.validate() to avoid
        # re-registering meters that already exist (ValueError: Meter val_time already existed).
        if self.val_data_loader is not None and hasattr(self, "_capture_validation_plot"):
            self._plot_epoch = last_epoch
            self._val_plot_data = []
            self._capture_validation_plot = True
            self.model.eval()
            with torch.no_grad():
                for iter_index, data in enumerate(self.val_data_loader):
                    self.val_iters(iter_index, data)
            self.on_validating_end(train_epoch=None)

        # log total training time
        total_train_time = time.time() - total_train_start_time
        hours = int(total_train_time // 3600)
        minutes = int((total_train_time % 3600) // 60)
        seconds = total_train_time % 60
        self.logger.info(
            "Total training time: {:.2f}s ({:d}h {:02d}m {:05.2f}s)".format(
                total_train_time, hours, minutes, seconds
            )
        )

        # log training finish time
        self.logger.info("The training finished at {}".format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        ))

        self.on_training_end()

    # support test process
    def on_epoch_end(self, epoch: int):
        """Callback at the end of an epoch.

        Args:
            epoch (int): current epoch.
        """

        # print train meters
        self.print_epoch_meters("train")
        # tensorboard plt meters
        self.plt_epoch_meters("train", epoch)
        # validate
        if self.val_data_loader is not None and epoch % self.val_interval == 0:
            self.validate(train_epoch=epoch)
        # test
        if self.test_data_loader is not None and epoch % self.test_interval == 0:
            self.test_process(train_epoch=epoch)
        # save model
        self.save_model(epoch)
        # reset meters
        self.reset_epoch_meters()

    @master_only
    def save_best_model(self, epoch: int, metric_name: str, greater_best: bool = True):
        """Save the best model while training, and update early-stopping state.

        Args:
            epoch (int): current epoch.
            metric_name (str): metric name used to measure the model, must be registered in `epoch_meter`.
            greater_best (bool, optional): `True` means greater value is best (e.g. acc);
                `False` means lower value is best (e.g. loss/MAE). Defaults to True.

        Returns:
            bool: True if the metric improved (and best model was saved), else False.
        """

        metric = self.meter_pool.get_avg(metric_name)
        best_metric = self.best_metrics.get(metric_name)

        # decide improvement with optional min_delta (only used when early stopping is on)
        min_delta = self.early_stop_min_delta if self.early_stopping_enabled else 0.0
        if best_metric is None:
            improved = True
        elif greater_best:
            improved = metric > best_metric + min_delta
        else:
            improved = metric < best_metric - min_delta

        if improved:
            self.best_metrics[metric_name] = metric
            model = self.model.module if isinstance(self.model, DDP) else self.model
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": self.optim.state_dict(),
                "best_metrics": self.best_metrics,
            }
            ckpt_path = os.path.join(
                self.ckpt_save_dir,
                "{}_best_{}.pt".format(self.model_name, metric_name.replace("/", "_")),
            )
            save_ckpt(ckpt_dict, ckpt_path, self.logger)

            if self.early_stopping_enabled and metric_name == self.early_stop_metric:
                self.early_stop_counter = 0
                self.logger.info(
                    "EarlyStopping: {} improved to {:.4f}. Counter reset to 0/{}.".format(
                        metric_name, metric, self.early_stop_patience
                    )
                )
            return True

        # no improvement
        if self.early_stopping_enabled and metric_name == self.early_stop_metric:
            self.early_stop_counter += 1
            self.logger.info(
                "EarlyStopping: no improvement in {} ({:.4f}). Counter: {}/{}.".format(
                    metric_name, metric, self.early_stop_counter, self.early_stop_patience
                )
            )
            if self.early_stop_counter >= self.early_stop_patience:
                self.early_stop = True
        return False

    @torch.no_grad()
    @master_only
    def test_process(self, cfg: dict = None, train_epoch: int = None):
        """The whole test process.

        Args:
            cfg (dict, optional): config
            train_epoch (int, optional): current epoch if in training process.
        """

        # init test if not in training process
        if train_epoch is None:
            self.init_test(cfg)

        self.on_test_start()

        test_start_time = time.time()
        self.model.eval()

        # test
        self.test()

        test_end_time = time.time()
        self.update_epoch_meter("test_time", test_end_time - test_start_time)
        # print test meters
        self.print_epoch_meters("test")
        if train_epoch is not None:
            # tensorboard plt meters
            self.plt_epoch_meters("test", train_epoch // self.test_interval)

        self.on_test_end()

    @master_only
    def on_test_start(self):
        """Callback at the start of testing.
        """

        pass

    @master_only
    def on_test_end(self):
        """Callback at the end of testing.
        """

        pass

    def test(self, train_epoch: int = None):
        """It can be implemented to define testing details.

        Args:
            train_epoch (int, optional): current epoch if in training process.
        """

        raise NotImplementedError()
