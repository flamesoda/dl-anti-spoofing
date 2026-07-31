import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=self.use_amp
        ):
            outputs = self.model(**batch)
            batch.update(outputs)
            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        if self.is_train:
            self.scaler.scale(batch["loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            self._clip_grad_norm()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.lr_scheduler is not None and self.cfg_trainer.get(
                "scheduler_step_per_batch", False
            ):
                self.lr_scheduler.step()

        batch_size = batch["labels"].shape[0]
        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item(), n=batch_size)

        for met in metric_funcs:
            if getattr(met, "is_epoch_metric", False):
                met.update(**batch)
            else:
                metrics.update(met.name, met(**batch), n=batch_size)
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_amp = bool(
            self.cfg_trainer.get("use_amp", False)
            and str(self.device).startswith("cuda")
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    @staticmethod
    def _reset_epoch_metrics(metric_functions):
        for metric in metric_functions:
            if getattr(metric, "is_epoch_metric", False):
                metric.reset()

    @staticmethod
    def _finalize_epoch_metrics(metric_functions, tracker):
        for metric in metric_functions:
            if getattr(metric, "is_epoch_metric", False):
                tracker.update(metric.name, metric.compute())
