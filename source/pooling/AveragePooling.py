import torch
from pytorch_lightning import LightningModule


class AveragePooling(LightningModule):
    """
    Performs average pooling on the last hidden-states transformer output.
    """

    def __init__(self):
        super(AveragePooling, self).__init__()

    def forward(self, attention_mask, encoder_outputs):
        """

        """
        hidden_states = encoder_outputs.last_hidden_state
        attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden_states = torch.sum(hidden_states * attention_mask, 1)
        sum_attention_mask = torch.sum(attention_mask, 1)
        return torch.div(
            sum_hidden_states,
            torch.clamp(sum_attention_mask, min=1e-9)
        )
