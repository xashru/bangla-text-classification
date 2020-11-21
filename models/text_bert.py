from transformers.modeling_bert import *
from .config import MODELS


class MyBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        return pooled_output


class TextBERT(nn.Module):
    def __init__(self, pretrained_model, num_class, fine_tune):
        super(TextBERT, self).__init__()
        self.bert = MyBert.from_pretrained(pretrained_model)
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
