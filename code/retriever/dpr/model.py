from typing import Optional

import torch
from transformers import (
    BertModel,
    BertPreTrainedModel,
    PretrainedConfig,
)


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config: PretrainedConfig = None):
        super(BertEncoder, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
    ):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]  # cls token shape ~ (batch_size, hidden_size)
        return pooled_output
