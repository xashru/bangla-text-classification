from transformers.modeling_roberta import *
from .config import MODELS
import torch


class MyRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return x


class MyRoBerta(RobertaPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.classifier = MyRobertaClassificationHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        output = self.classifier(sequence_output)

        return output


class TextRoBERTa(nn.Module):
    def __init__(self, pretrained_model, num_class, fine_tune):
        super(TextRoBERTa, self).__init__()
        self.bert = MyRoBerta.from_pretrained(pretrained_model)
        # Freeze bert layers
        if not fine_tune:
            for p in self.bert.parameters():
                p.requires_grad = False

        bert_dim = MODELS[pretrained_model][2]
        self.classifier = nn.Linear(bert_dim, num_class)

    def forward(self, x, attn_masks):
        outputs = self.bert(x, attention_mask=attn_masks)
        logits = self.classifier(outputs)
        return logits
