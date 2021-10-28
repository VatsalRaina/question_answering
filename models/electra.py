import torch
import torch.nn as nn

from dataclasses import dataclass

from transformers import ElectraTokenizer
from transformers import ElectraForQuestionAnswering as HFElectraForQuestionAnswering
from transformers.file_utils import ModelOutput

from typing import Optional, Tuple

__all__ = [
    'qa_electra_large',
    'qa_electra_large_modified',
    'qa_electra_large_combo',
    'qa_electra_large_combo_modified',
]


def get_default_device(use_cuda = True):
    """
    Returns cuda/cpu device
    """
    return torch.device('cuda') if (use_cuda and torch.cuda.is_available()) else torch.device('cpu')


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

    def __init__(self, config, num_labels = None):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.out_proj = nn.Linear(
            config.hidden_size,
            config.num_labels if num_labels is None else num_labels
        )
        self.gelu = nn.GELU()

    def forward(self, features, **kwargs):
        # Map only the first feature corresponding to [CLS]
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

### ELECTRA MODELS ###
class ElectraForQuestionAnswering(HFElectraForQuestionAnswering):
    def __init__(self, config):
        super(ElectraForQuestionAnswering, self).__init__(config = config)

    @staticmethod
    def get_context_mask(ids: torch.Tensor, device = None):
        """
        Returns a bool mask corresponding to context locations.
        This is assuming the input has two SEP tokens with id 102.
        Input should have shape (batch, seqlen (default 512))
        """

        # This returns the batch number and index for each batch
        _, indices = (ids == 102).nonzero(as_tuple = True)

        # Ensure that indices can be cast into a shape (batch, 2)
        indices = indices.view(ids.size(0), 2)

        # Create mask for context
        mask = torch.arange(ids.size(-1), device = device).expand(*ids.size())
        mask = torch.logical_and(indices[:, 0, None] < mask, mask < indices[:, 1, None])
        return mask

    @staticmethod
    def forward_loss(start_positions, end_positions, start_logits, end_logits, reduction = 'mean', context_mask = None):
        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)

        # Sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        if context_mask is not None:
            # This masking operation ensures that logits not corresponding to elements of interest
            # have extremely small logits not affecting the probability mask of remaining elements
            start_logits = start_logits + context_mask.log()
            end_logits = end_logits + context_mask.log()

        # Define loss function and get the loss for both start and end
        loss_fct = nn.CrossEntropyLoss(ignore_index = ignored_index, reduction = reduction)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return total_loss

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
            reduction='mean',
            use_context_mask=False,
            **kwargs,
    ):
        """
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

        total_loss = None
        if start_positions is not None and end_positions is not None:

            # Get the context mask when logits outside context should not be used
            context_mask = None if use_context_mask is False else \
                self.get_context_mask(input_ids, device = get_default_device())

            total_loss = self.forward_loss(
                start_positions = start_positions,
                end_positions = end_positions,
                start_logits = start_logits,
                end_logits = end_logits,
                context_mask = context_mask.float(),
                reduction = reduction
            )

        if not return_dict:
            output = (start_logits, end_logits, ) + discriminator_hidden_states[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutputCombo(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


class ElectraForQuestionAnsweringModified(ElectraForQuestionAnswering):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringModified, self).__init__(config = config)

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
        return super(ElectraForQuestionAnsweringModified, self).forward(
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
            return_dict = return_dict,
            use_context_mask = True,
        )


class ElectraForQuestionAnsweringCombo(ElectraForQuestionAnswering):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringCombo, self).__init__(config = config)

        # Ensure that these models are initialised
        assert callable(self.electra)
        assert callable(self.qa_outputs)

        # Add separate answerability module
        self.answerability = ElectraClassificationHead(config, num_labels = 1)
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
        use_context_mask=False,
        **kwargs,
    ):
        """
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

            # Get the context mask when logits outside context should not be used
            context_mask = None if use_context_mask is False else \
                self.get_context_mask(input_ids, device=get_default_device())

            total_loss = self.forward_loss(
                start_positions=start_positions,
                end_positions=end_positions,
                start_logits=start_logits,
                end_logits=end_logits,
                context_mask=context_mask.float(),
                reduction='none' if self.user_args.answer_train_separation else 'mean'
            )

            # Now mask out losses corresponding to unanswerable examples
            if self.user_args.answer_train_separation:
                assert answerable_labels is not None

                total_loss = torch.masked_select(total_loss, answerable_labels.to(torch.bool))
                total_loss = total_loss.sum()/sequence_output.size(0)

        combo_loss = None
        if answerable_labels is not None:
            combo_loss = self.answerability_loss(
                answerable_probs.squeeze(-1),
                answerable_labels.to(torch.float)
            )

        # Combine losses
        loss = None
        if (total_loss is not None) and (combo_loss is not None):
            loss = total_loss + self.user_args.answer_alpha * combo_loss

        if not return_dict:
            output = (start_logits, end_logits, answerable_probs, ) + discriminator_hidden_states[1:]
            return ((loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutputCombo(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            answerable_probs = answerable_probs,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


class ElectraForQuestionAnsweringComboModified(ElectraForQuestionAnsweringCombo):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringComboModified, self).__init__(config = config)

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
        return super(ElectraForQuestionAnsweringComboModified, self).forward(
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
            return_dict = return_dict,
            use_context_mask = True,
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


def qa_electra_large_modified(args, tokenizer_only = False, **kwargs):

    # Model identifier
    electra_large = "google/electra-large-discriminator"

    # Load tokenizer
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)
    if tokenizer_only: return tokenizer

    # Load pre-trained model
    model = ElectraForQuestionAnsweringModified.from_pretrained(electra_large)

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


def qa_electra_large_combo_modified(args, tokenizer_only = False, **kwargs):
    # Model identifier
    electra_large = "google/electra-large-discriminator"

    # Load tokenizer
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)
    if tokenizer_only: return tokenizer

    # Load pre-trained model
    model = ElectraForQuestionAnsweringComboModified.from_pretrained(electra_large)
    model.user_args = args

    return model, tokenizer
