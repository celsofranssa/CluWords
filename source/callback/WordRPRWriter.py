from pathlib import Path
from typing import Any, List

import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor


class WordRPRWriter(BasePredictionWriter):

    def __init__(self, params):
        super(WordRPRWriter, self).__init__(params.write_interval)
        self.params=params
        self.predictions = []

    def write_on_batch_end(
            self, trainer, pl_module, prediction: Any, batch_indices: List[int], batch: Any,
            batch_idx: int, dataloader_idx: int
    ):

        predictions = []

        for idx, text, rpr, pred_cls, true_cls in zip(
                prediction["idx"].tolist(),
                prediction["text"].tolist(),
                prediction["rpr"].tolist(),
                prediction["pred_cls"].tolist(),
                prediction["true_cls"].tolist()):

            pad_index = text.index(self.params.PAD)
            predictions.append({
                "idx": idx,
                "text": text[:pad_index],
                "rpr": rpr[:pad_index],
                "pred_cls": pred_cls,
                "true_cls": true_cls
            })

        self._checkpoint(predictions, dataloader_idx, batch_idx)



    def _checkpoint(self, predictions, dataloader_idx, batch_idx):
        dir = f"{self.params.dir}fold_{self.params.fold}/"
        Path(dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            predictions,
            f"{dir}{dataloader_idx}_{batch_idx}.rpr"
        )

