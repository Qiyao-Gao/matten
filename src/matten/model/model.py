"""
Base Lightning model for regression and classification.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.cli import instantiate_class
from torch import Tensor

from matten.data.data import DataPoint
from matten.model.task import Task, TaskType
from matten.model.utils import TimeMeter
from matten.utils import ToCartesian


class BaseModel(pl.LightningModule):
    """
    Base matten model for regression and classification tasks.

    This class accepts any type of data as batch. Subclass determines how the batch
    should be dealt with.
    """

    def __init__(
        self,
        tasks: Union[Task, List[Task], Dict[str, Task]] = None,
        backbone_hparams: Dict[str, Any] = None,
        dataset_hparams: Dict[str, Any] = None,
        optimizer_hparams: Dict[str, Any] = None,
        lr_scheduler_hparams: Dict[str, Any] = None,
        trainer_hparams: Dict[str, Any] = None,
        data_hparams: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.optimizer_hparams = optimizer_hparams
        self.lr_scheduler_hparams = lr_scheduler_hparams

        # backbone model
        self.backbone, extra_layers_dict = self.init_backbone(
            backbone_hparams, dataset_hparams
        )
        if extra_layers_dict is not None:
            self.extra_layers_dict = nn.ModuleDict(extra_layers_dict)

        # tasks
        self.tasks = self.init_tasks(tasks)

        # losses
        self.loss_fns = {name: task.init_loss() for name, task in self.tasks.items()}

        # metrics: {mode: {task_name: MetricCollection}}
        self.metrics = nn.ModuleDict()
        for mode in ["train", "val", "test"]:
            mode_key = "metric_" + mode
            self.metrics[mode_key] = nn.ModuleDict()
            for name, task in self.tasks.items():
                mc = task.init_metric_as_collection()
                self.metrics[mode_key][name] = mc

        # timer
        self.timer = TimeMeter()

        # callback monitor key
        self.monitor_key = "val/score"

    def init_backbone(
        self,
        backbone_hparams: Dict[str, Any],
        dataset_hparams: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        raise NotImplementedError

    def init_tasks(
        self, tasks: Union[Task, List[Task], Dict[str, Task]]
    ) -> Dict[str, Task]:
        if isinstance(tasks, dict):
            for name, t in tasks.items():
                assert name == t.name, f"Task name not consistent; got {name} and {t.name}"
        elif isinstance(tasks, Task):
            tasks = {tasks.name: tasks}
        elif isinstance(tasks, list):
            tasks = {t.name: t for t in tasks}
        else:
            raise ValueError(f"Unsupported tasks type {type(tasks)}")
        return tasks

    def forward(
        self,
        batch,
        mode: Optional[str] = None,
        task_name: str = "elastic_tensor_full",
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        graphs, labels = self.preprocess_batch(batch)

        if mode is None or mode.lower() == "none":
            preds = self.decode(graphs, **kwargs)
            preds = self.transform_prediction(preds, task_name=task_name)
            labels = self.transform_target(labels, task_name=task_name)
        elif mode == "backbone":
            preds = self.backbone(graphs, **kwargs)
        else:
            supported = (None, "backbone")
            raise ValueError(f"Expect mode to be one of {supported}; got {mode}")

        return preds, labels

    def transform_prediction(self, prediction: Dict[str, Tensor], task_name: str = None) -> Dict[str, Tensor]:
        return prediction

    def transform_target(self, target: Dict[str, Tensor], task_name: str = None) -> Dict[str, Tensor]:
        return target

    def preprocess_batch(self, batch) -> Tuple[Any, Dict[str, Tensor]]:
        raise NotImplementedError

    def decode(self, model_input, *args, **kwargs) -> Dict[str, Tensor]:
        raise NotImplementedError

    def compute_loss(
        self,
        preds: Dict[str, Tensor],
        labels: Dict[str, Tensor],
        weight: Tensor = None,
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        individual_losses = {}
        total_loss = 0.0

        for task_name, task in self.tasks.items():
            p = preds[task_name]
            l = labels[task_name]
            p = task.transform_pred_loss(p)
            l = task.transform_target_loss(l)

            if weight is not None:
                p = p * weight
                l = l * weight

            if task.task_type == TaskType.CLASSIFICATION and task.is_binary():
                p = p.reshape(-1)
                l = l.reshape(-1).to(torch.get_default_dtype())

            loss_fn = self.loss_fns[task_name]
            loss = loss_fn(p, l)
            individual_losses[task_name] = loss
            total_loss = total_loss + task.loss_weight * loss

        return individual_losses, total_loss

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch, "train")
        self.update_metrics(preds, labels, "train")
        return {"loss": loss}

    def on_training_epoch_end(self):
        self.compute_metrics("train")

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch, "val")
        self.update_metrics(preds, labels, "val")
        return {"loss": loss}

    def on_validation_epoch_end(self):
        _, score = self.compute_metrics("val")

        if score is not None:
            self.log(self.monitor_key, score, on_step=False, on_epoch=True, prog_bar=True)

        delta_t, cumulative_t = self.timer.update()
        self.log("epoch time", delta_t, on_step=False, on_epoch=True, prog_bar=True)
        self.log("cumulative time", cumulative_t, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch, "test")
        self.update_metrics(preds, labels, "test")
        return {"loss": loss}

    def on_test_epoch_end(self):
        self.compute_metrics("test")

    def shared_step(self, batch, mode: str):
        batch_size = batch.num_graphs

        graphs, labels = self.preprocess_batch(batch)
        preds = self.decode(graphs)

        if "atom_selector" in labels:
            selector = labels["atom_selector"]
            preds = {k: v[selector] for k, v in preds.items()}

        target_weight = graphs.get("target_weight", None)
        individual_loss, total_loss = self.compute_loss(preds, labels, weight=target_weight)

        self.log_dict(
            {f"{mode}/loss/{task_name}": loss for task_name, loss in individual_loss.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            f"{mode}/total_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        return total_loss, preds, labels

    def update_metrics(self, preds: Dict, labels: Dict, mode: str):
        mode_key = "metric_" + mode

        for task_name, metric in self.metrics[mode_key].items():
            task = self.tasks[task_name]

            p = task.transform_pred_metric(preds[task_name])
            l = task.transform_target_metric(labels[task_name])

            if task.task_type == TaskType.CLASSIFICATION:
                if task.is_binary():
                    p = torch.sigmoid(p.reshape(-1))
                else:
                    p = torch.argmax(p, dim=1)

            metric(p, l)

    def compute_metrics(
        self, mode, log: bool = True
    ) -> Tuple[Dict[str, Tensor], Union[Tensor, None]]:
        mode_key = "metric_" + mode

        total_score = None
        individual_score = {}

        for task_name, metric_coll in self.metrics[mode_key].items():
            score = metric_coll.compute()
            individual_score[task_name] = score

            if log:
                for metric_name, metric_value in score.items():
                    self.log(
                        f"{mode_key}/{metric_name}/{task_name}",
                        metric_value,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                    )

            task = self.tasks[task_name]
            metric_agg_dict = task.metric_aggregation()
            if metric_agg_dict:
                total_score = 0 if total_score is None else total_score
                for metric_name, weight in metric_agg_dict.items():
                    total_score = total_score + score[metric_name] * weight

            metric_coll.reset()

        return individual_score, total_score

    def configure_optimizers(self):
        model_params = (filter(lambda p: p.requires_grad, self.parameters()),)
        optimizer = instantiate_class(model_params, self.optimizer_hparams)

        scheduler = self._config_lr_scheduler(optimizer)

        if scheduler is None:
            return optimizer
        else:
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": self.monitor_key}

    def _config_lr_scheduler(self, optimizer):
        class_path = self.lr_scheduler_hparams.get("class_path")
        if class_path is None or class_path == "none":
            scheduler = None
        else:
            scheduler = instantiate_class(optimizer, self.lr_scheduler_hparams)
        return scheduler


class ModelForPyGData(BaseModel):
    """
    A lightning model working with data provided as PyG batched data.
    """

    # --------------------------
    # 1) Cartesian test-only metrics helpers (FULL Cartesian MAE/MSE)
    # --------------------------
    @staticmethod
    def _tensor_rank_from_formula(formula: str) -> int:
        lhs = formula.split("=")[0].replace("-", "").strip()
        return len(lhs)

    @staticmethod
    def _squeeze_irreps(x: Tensor) -> Tensor:
        if x.ndim == 3 and x.shape[1] == 1:
            return x[:, 0, :]
        return x

    @staticmethod
    @torch.no_grad()
    def _cartesian_full_mae_mse_from_irreps(
        pred_irreps: Tensor,
        target_irreps: Tensor,
        formula: str,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return (pred_vec, targ_vec, mae, mse) for Cartesian space metrics."""
        pred_irreps = ModelForPyGData._squeeze_irreps(pred_irreps)
        target_irreps = ModelForPyGData._squeeze_irreps(target_irreps)

        to_cart = ToCartesian(formula)

        pred_cart = to_cart(pred_irreps)
        targ_cart = to_cart(target_irreps)

        rank = ModelForPyGData._tensor_rank_from_formula(formula)

        pred_cart = pred_cart.reshape(pred_cart.shape[0], *([3] * rank))
        targ_cart = targ_cart.reshape(targ_cart.shape[0], *([3] * rank))

        pred_vec = pred_cart.reshape(pred_cart.shape[0], -1)
        targ_vec = targ_cart.reshape(targ_cart.shape[0], -1)

        diff = pred_vec - targ_vec
        mae = diff.abs().mean()
        mse = (diff ** 2).mean()
        return pred_vec, targ_vec, mae, mse

    # --------------------------
    # 2) Robust formula getter (CRITICAL FIX)
    # --------------------------
    def _get_tensor_formula(self, task_name: str, task) -> str:
        """
        按优先级取 formula，确保和数据/模型一致：
        1) task 自己带的（如果有）
        2) hparams 里的 data_hparams / data 里的 tensor_target_formula
        3) hparams 里的 backbone_hparams / model 的 output_formula
        找不到就直接报错，避免静默用错公式
        """
        # 1) task-level (如果你的 Task 类里存了 formula)
        for attr in ("tensor_target_formula", "target_formula", "formula", "output_formula"):
            v = getattr(task, attr, None)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # 2) data-level
        for key in ("data_hparams", "data", "dataset_hparams"):
            dh = self.hparams.get(key, None)
            if isinstance(dh, dict):
                v = dh.get("tensor_target_formula", None)
                if isinstance(v, str) and v.strip():
                    return v.strip()

        # 3) model/backbone-level
        for key in ("backbone_hparams", "model_hparams", "model"):
            mh = self.hparams.get(key, None)
            if isinstance(mh, dict):
                v = mh.get("output_formula", None)
                if isinstance(v, str) and v.strip():
                    return v.strip()

        # 如果你希望按任务名区分，也可以在这里做 task_name->formula 映射

        raise RuntimeError(
            f"[Cartesian metric] Cannot find tensor formula for task={task_name}. "
            f"Please set data.tensor_target_formula AND model.output_formula in your yaml."
        )

    # --------------------------
    # 3) Your existing batch processing
    # --------------------------
    def preprocess_batch(self, batch: DataPoint) -> Tuple[DataPoint, Dict[str, Tensor]]:
        graphs = batch
        graphs = graphs.to(self.device)

        labels = {name: graphs.y[name] for name in self.tasks}
        if "atom_selector" in graphs.y:
            labels["atom_selector"] = graphs.y["atom_selector"]

        graphs = graphs.tensor_property_to_dict()
        return graphs, labels

    def decode(self, model_input: DataPoint, *args, **kwargs) -> Dict[str, Tensor]:
        return self.backbone(model_input)

    # --------------------------
    # 4) Override test_step: keep irreps metrics + add FULL Cartesian MAE/MSE (test-only)
    # --------------------------
    def on_test_start(self):
        """Reset accumulation for MAE/MAD computation."""
        super().on_test_start()
        self._test_cartesian_preds = {}
        self._test_cartesian_targets = {}

    def on_test_epoch_end(self):
        """Compute and log MAE/MAD (MAE divided by MAD) for wandb."""
        super().on_test_epoch_end()

        accum = getattr(self, "_test_cartesian_preds", None)
        if not accum:
            return

        for task_name, pred_list in self._test_cartesian_preds.items():
            if not pred_list or not self._test_cartesian_targets.get(task_name):
                continue
            pred_vec = torch.cat(pred_list, dim=0)
            targ_vec = torch.cat(self._test_cartesian_targets[task_name], dim=0)

            mae = (pred_vec - targ_vec).abs().mean().item()
            mad = (targ_vec - targ_vec.mean()).abs().mean().item()
            mae_over_mad = mae / mad if mad > 1e-10 else float("nan")

            self.log(
                f"metric_test/MAE_over_MAD/{task_name}",
                mae_over_mad,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                f"test/MAE_over_MAD/{task_name}",
                mae_over_mad,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        self._test_cartesian_preds = {}
        self._test_cartesian_targets = {}

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch, "test")
        self.update_metrics(preds, labels, "test")

        batch_size = batch.num_graphs

        for task_name, task in self.tasks.items():
            if task.task_type != TaskType.REGRESSION:
                continue

            p_ir = task.transform_pred_metric(preds[task_name])
            l_ir = task.transform_target_metric(labels[task_name])

            # ✅ 关键：取到正确的 ijk=ikj / ij=ji / ijkl=... 等
            formula = self._get_tensor_formula(task_name, task)

            pred_vec, targ_vec, mae_cart, mse_cart = ModelForPyGData._cartesian_full_mae_mse_from_irreps(
                pred_irreps=p_ir,
                target_irreps=l_ir,
                formula=formula,
            )

            # Accumulate for MAE/MAD computation in on_test_epoch_end
            if task_name not in self._test_cartesian_preds:
                self._test_cartesian_preds[task_name] = []
                self._test_cartesian_targets[task_name] = []
            self._test_cartesian_preds[task_name].append(pred_vec.detach().cpu())
            self._test_cartesian_targets[task_name].append(targ_vec.detach().cpu())

            self.log(
                f"metric_test/MAE_Cartesian/{task_name}",
                mae_cart,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=False,
            )
            self.log(
                f"metric_test/MSE_Cartesian/{task_name}",
                mse_cart,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=False,
            )
            # Mirror Cartesian metrics under `test/*` so they are easier to spot
            # in W&B default test panels.
            self.log(
                f"test/MAE_Cartesian/{task_name}",
                mae_cart,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=False,
            )
            self.log(
                f"test/MSE_Cartesian/{task_name}",
                mse_cart,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=False,
            )

        return {"loss": loss}
