import collections
import torch
from torch import nn
from transformers import RobertaPreTrainedModel, RobertaModel, BertPreTrainedModel, BertModel

Outputs = collections.namedtuple("RobertaOutputs", ["special_loss", "special_logits", "bio_loss", "bio_logits"])

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaForBioDst(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        for i in range(config.num_special_slots):
            self.add_module("special_classifier_{}".format(i), ClassificationHead(config, num_labels=config.num_special_classes))

        self.bio_classifier = ClassificationHead(config, num_labels=config.num_bio_classes)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
        token_type_ids: torch.LongTensor = None,
        special_labels: torch.FloatTensor = None,
        bio_labels: torch.LongTensor = None,
    ):
        
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]

        special_logits = []

        for i in range(self.config.num_special_slots):
            special_logits.append(getattr(self, "special_classifier_{}".format(i))(pooled_output))

        special_logits = torch.stack(special_logits).permute(1, 0, 2).contiguous()
        bio_logits = self.bio_classifier(sequence_output)

        special_loss = None
        if special_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            special_loss = loss_fct(special_logits.view(-1, self.config.num_special_classes), special_labels.view(-1))
        
        bio_loss = None
        if bio_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            bio_loss = loss_fct(bio_logits.view(-1, self.config.num_bio_classes), bio_labels.view(-1))
            
        return {'special_loss':special_loss, 'special_logits':special_logits, 'bio_loss':bio_loss, 'bio_logits':bio_logits}


class BertForBioDst(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)

        for i in range(config.num_special_slots):
            self.add_module("special_classifier_{}".format(i), ClassificationHead(config, num_labels=config.num_special_classes))

        self.bio_classifier = ClassificationHead(config, num_labels=config.num_bio_classes)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
        token_type_ids: torch.LongTensor = None,
        special_labels: torch.FloatTensor = None,
        bio_labels: torch.LongTensor = None,
    ):
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,)

        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]

        special_logits = []

        for i in range(self.config.num_special_slots):
            special_logits.append(getattr(self, "special_classifier_{}".format(i))(pooled_output))

        special_logits = torch.stack(special_logits).permute(1, 0, 2).contiguous()
        bio_logits = self.bio_classifier(sequence_output)

        special_loss = None
        if special_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            special_loss = loss_fct(special_logits.view(-1, self.config.num_special_classes), special_labels.view(-1))
        
        bio_loss = None
        if bio_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            bio_loss = loss_fct(bio_logits.view(-1, self.config.num_bio_classes), bio_labels.view(-1))
            
        return {'special_loss':special_loss, 'special_logits':special_logits, 'bio_loss':bio_loss, 'bio_logits':bio_logits}