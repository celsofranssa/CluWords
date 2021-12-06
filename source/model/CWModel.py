import json

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torch import nn, Tensor
from torchmetrics import MetricCollection, F1



class CWModel(pl.LightningModule):

    def __init__(self, hparams):
        super(CWModel, self).__init__()
        self.save_hyperparameters(hparams)

        self.encoder = instantiate(hparams.encoder)

        self.pooling = instantiate(hparams.pooling)

        self.cls_head = torch.nn.Sequential(
            torch.nn.Dropout(hparams.dropout),
            torch.nn.Linear(hparams.hidden_size, hparams.num_classes),
            torch.nn.LogSoftmax(dim=-1)
        )

        self.train_metrics = self._get_metrics(prefix="train_")
        self.val_metrics = self._get_metrics(prefix="val_")
        self.test_metrics = self._get_metrics(prefix="test_")

        self.loss = nn.NLLLoss()

    def _get_metrics(self, prefix):
        return MetricCollection(
            metrics={
                "Mic-F1": F1(num_classes=self.hparams.num_classes, average="micro"),
                "Wei-F1": F1(num_classes=self.hparams.num_classes, average="weighted")
            },
            prefix=prefix)

    def forward(self, text, attention_mask):
        return self.encoder(text, attention_mask)


    def _cls(self, text, attention_mask):
        encoder_outputs  = self(text, attention_mask)
        pooled =  self.pooling(
            attention_mask, encoder_outputs
        )
        return self.cls_head(pooled)


    def training_step(self, batch, batch_idx):
        text, true_cls = batch["text"], batch["cls"]
        attention_mask = torch.gt(text, 0).int()
        pred_cls = self._cls(text, attention_mask)

        train_loss = self.loss(pred_cls, true_cls)

        # log training loss
        self.log('train_loss', train_loss)

        self.log_dict(self.train_metrics(pred_cls, true_cls), prog_bar=True)

        return train_loss

    def training_epoch_end(self, outs):
        self.train_metrics.compute()

    def validation_step(self, batch, batch_idx):
        text, true_cls = batch["text"], batch["cls"]
        attention_mask = torch.gt(text, 0).int()
        pred_cls = self._cls(text, attention_mask)

        val_loss = self.loss(pred_cls, true_cls)

        # log val loss
        self.log('val_loss', val_loss)

        # log val metrics
        self.log_dict(self.val_metrics(pred_cls, true_cls), prog_bar=True)

    def validation_epoch_end(self, outs):
        self.val_metrics.compute()

    def test_step(self, batch, batch_idx):
        idx, text, true_cls = batch["idx"], batch["text"], batch["cls"]
        attention_mask = torch.gt(text, 0).int()
        pred_cls = self._cls(text, attention_mask)

        # log test metrics
        self.log_dict(self.test_metrics(pred_cls, true_cls), prog_bar=True)

    def test_epoch_end(self, outs):
        test_result = self.test_metrics.compute()
        self._checkpoint_test_result(
            test_result,
            self.hparams.stat.dir+self.hparams.stat.name)

    def _checkpoint_test_result(self, test_result, test_result_path):
        test_result = {k: v.tolist() for k, v in test_result.items()}
        with open(test_result_path, "w") as test_results_file:
            test_results_file.write(json.dumps(test_result))

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        idx, text, true_cls = batch["idx"], batch["text"], batch["cls"]
        attention_mask = torch.gt(text, 0).int()

        return {
                "idx": idx,
                "text": text,
                "rpr": self(text, attention_mask).last_hidden_state,
                "pred_cls": self._cls(text, attention_mask),
                "true_cls": true_cls,
            }

    def configure_optimizers(self):
        # optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.hparams.weight_decay,
            amsgrad=True)

        # scheduler
        step_size_up = round(0.03 * self.num_training_steps)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            mode='triangular2',
            base_lr=self.hparams.base_lr,
            max_lr=self.hparams.max_lr,
            step_size_up=step_size_up,
            cycle_momentum=False)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and number of epochs."""
        steps_per_epochs = len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
        max_epochs = self.trainer.max_epochs
        return steps_per_epochs * max_epochs
