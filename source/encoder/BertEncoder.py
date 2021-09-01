import torch
from pytorch_lightning import LightningModule
from transformers import BertModel


class BertEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, architecture, output_attentions):
        super(BertEncoder, self).__init__()

        self.encoder = BertModel.from_pretrained(
            architecture,
            output_attentions=output_attentions
        )
        #self.pooling = pooling

    def forward(self, input_ids, attention_mask):

        encoder_outputs = self.encoder(input_ids, attention_mask)

        return attention_mask, encoder_outputs.last_hidden_state

        # return self.pooling(
        #     attention_mask,
        #     encoder_outputs
        # )
