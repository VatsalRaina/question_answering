import torch
import torch.nn as nn

from dataclasses import dataclass

from transformers import ElectraTokenizer
from transformers import ElectraForQuestionAnswering as HFElectraForQuestionAnswering
# from transformers.modeling_electra import ElectraClassificationHead
from transformers.file_utils import ModelOutput

from typing import Optional, Tuple

__all__ = [
    'qa_electra_large',
    'qa_electra_large_combo',
]


@dataclass
class QuestionAnsweringModelOutputCombo(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        answerable_probs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 1)`):
            probability score of being answerable.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    answerable_probs: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ElectraClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.gelu = nn.GELU()

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ElectraForQuestionAnswering(HFElectraForQuestionAnswering):
    def __init__(self, config):
        super(ElectraForQuestionAnswering, self).__init__(config = config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        return super(ElectraForQuestionAnswering, self).forward(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            start_positions = start_positions,
            end_positions = end_positions,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict
        )


class ElectraForQuestionAnsweringCombo(HFElectraForQuestionAnswering):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringCombo, self).__init__(config = config)

        # Ensure that these models are initialised
        assert callable(self.electra)
        assert callable(self.qa_outputs)

        # Add separate answerability module
        self.answerability = ElectraClassificationHead(config)
        self.answerability_loss = nn.BCELoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        answerable_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = discriminator_hidden_states[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # This is a separate module for answerability
        answerable_probs = torch.sigmoid(self.answerability(sequence_output))

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # Sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index, reduction = 'none')
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            import pdb; pdb.set_trace()
            # Now mask out losses corresponding to unanswerable examples
            if self.user_args.answer_train_separation:
                assert answerable_labels is not None

                total_loss = torch.masked_select(total_loss, answerable_labels)
                total_loss = total_loss.sum()/sequence_output.size(0)

        combo_loss = None
        if answerable_labels is not None:
            combo_loss = self.answerability_loss(answerable_probs, answerable_labels)

        # Combine losses
        loss = None
        if (total_loss is not None) and (combo_loss is not None):
            loss = total_loss + self.user_args.answer_alpha * combo_loss

        if not return_dict:
            output = (
                start_logits,
                end_logits,
                answerable_probs,
            ) + discriminator_hidden_states[1:]
            return ((loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutputCombo(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            answerable_probs = answerable_probs,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


def qa_electra_large(args, tokenizer_only = False, **kwargs):

    # Model identifier
    electra_large = "google/electra-large-discriminator"

    # Load tokenizer
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)
    if tokenizer_only: return tokenizer

    # Load pre-trained model
    model = ElectraForQuestionAnswering.from_pretrained(electra_large)

    return model, tokenizer


def qa_electra_large_combo(args, tokenizer_only = False, **kwargs):
    # Model identifier
    electra_large = "google/electra-large-discriminator"

    # Load tokenizer
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)
    if tokenizer_only: return tokenizer

    # Load pre-trained model
    model = ElectraForQuestionAnsweringCombo.from_pretrained(electra_large)
    model.user_args = args

    return model, tokenizer

