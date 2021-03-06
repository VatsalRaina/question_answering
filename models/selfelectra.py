import torch

from transformers import ElectraTokenizer

from loss import DirichletEstimationLoss
from modules import multiplicativegaussianlayer

from loss import dirichlet_kl_divergence
from .electra import (
    QuestionAnsweringModelOutputCombo,
    ElectraForQuestionAnsweringModified,
    ElectraForQuestionAnsweringComboModified
)

__all__ = [
    'qa_electra_large_self',
    'qa_electra_large_combo_self',
]


def get_default_device(use_cuda = True):
    """
    Returns cuda/cpu device
    """
    return torch.device('cuda') if (use_cuda and torch.cuda.is_available()) else torch.device('cpu')


class ElectraForQuestionAnsweringSelf(ElectraForQuestionAnsweringModified):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringSelf, self).__init__(config = config)

    def forward_logits(self, sequence_output):
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        return start_logits, end_logits

    def forward_student_loss(self, start_logits, end_logits, noisy_start_logits, noisy_end_logits, context_mask = None):
        loss_fct = DirichletEstimationLoss()
        start_loss = loss_fct(self.user_args, logits = start_logits, noisy_logits = noisy_start_logits, context_mask = context_mask)
        end_loss = loss_fct(self.user_args, logits = end_logits, noisy_logits = noisy_end_logits, context_mask = context_mask)

        student_loss = 0.50 * (start_loss + end_loss)
        return student_loss

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

        # This has a size of (batch, seqlen, hidden)
        sequence_output = discriminator_hidden_states[0]

        # Get start and end logits
        start_logits, end_logits = self.forward_logits(sequence_output)

        total_loss = None
        if start_positions is not None and end_positions is not None:

            # Get the noisy version of sequence_output which has size (batch, num, seqlen, hidden)
            noisy_sequence_outputs = sequence_output.unsqueeze(1).repeat(1, self.user_args.num_passes, 1, 1)

            # Get noisy predictions and train masked cross entropy on those
            noisy_start_logits, noisy_end_logits = self.forward_logits(noisy_sequence_outputs)

            # Get the context mask when logits outside context should not be used
            context_mask = self.get_context_mask(input_ids, device = get_default_device())

            total_loss = self.forward_loss(
                start_positions = start_positions,
                end_positions = end_positions,
                start_logits = noisy_start_logits.mean(1),
                end_logits = noisy_end_logits.mean(1),
                context_mask = context_mask.float(),
                reduction = 'mean'
            )

            # Now get the student losses based on fitting a target dirichlet
            student_loss = self.forward_student_loss(
                start_logits = start_logits,
                end_logits = end_logits,
                noisy_start_logits = noisy_start_logits,
                noisy_end_logits = noisy_end_logits,
                context_mask = context_mask.float()
            )

            # Update total loss
            total_loss += student_loss * self.user_args.self_ratio

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


class ElectraForQuestionAnsweringComboSelf(ElectraForQuestionAnsweringComboModified):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringComboSelf, self).__init__(config = config)

    def forward_logits(self, sequence_output):
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        return start_logits, end_logits

    def forward_student_loss(self, start_logits, end_logits, noisy_start_logits, noisy_end_logits, reduction = 'mean', context_mask = None):
        loss_fct = DirichletEstimationLoss()
        start_loss = loss_fct(
            self.user_args,
            logits = start_logits,
            noisy_logits = noisy_start_logits,
            context_mask = context_mask,
            reduce = True if reduction == 'mean' else False
        )
        end_loss = loss_fct(
            self.user_args,
            logits = end_logits,
            noisy_logits = noisy_end_logits,
            context_mask = context_mask,
            reduce = True if reduction == 'mean' else False
        )

        student_loss = 0.50 * (start_loss + end_loss)
        return student_loss

    @staticmethod
    def forward_student_loss_flat(start_logits, end_logits, reduction = 'mean', context_mask = None):
        start_loss = dirichlet_kl_divergence(
            log_alphas=start_logits,
            log_alphas_target=torch.zeros_like(start_logits),
            temperature_scale_num=1.0,
            mask=context_mask,
            reduce=True if reduction == 'mean' else False
        )
        end_loss = dirichlet_kl_divergence(
            log_alphas=end_logits,
            log_alphas_target=torch.zeros_like(end_logits),
            temperature_scale_num=1.0,
            mask=context_mask,
            reduce=True if reduction == 'mean' else False
        )
        flat_student_loss = (start_loss + end_loss) / 2
        return flat_student_loss

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

        # This has a size of (batch, seqlen, hidden)
        sequence_output = discriminator_hidden_states[0]

        # Get start and end logits
        start_logits, end_logits = self.forward_logits(sequence_output)

        # This is a separate module for answerability
        answerable_probs = torch.sigmoid(self.answerability(sequence_output))

        total_loss = None
        if start_positions is not None and end_positions is not None:

            # Get the noisy version of sequence_output which has size (batch, num, seqlen, hidden)
            noisy_sequence_outputs = sequence_output.unsqueeze(1).repeat(1, self.user_args.num_passes, 1, 1)

            # Get noisy predictions and train masked cross entropy on those
            noisy_start_logits, noisy_end_logits = self.forward_logits(noisy_sequence_outputs)

            # Get the context mask when logits outside context should not be used
            context_mask = self.get_context_mask(input_ids, device=get_default_device())

            total_loss = self.forward_loss(
                start_positions = start_positions,
                end_positions = end_positions,
                start_logits = noisy_start_logits.mean(1),
                end_logits = noisy_end_logits.mean(1),
                context_mask = context_mask.float(),
                reduction = 'none' if self.user_args.answer_train_separation else 'mean'
            )

            # Now get the student losses based on fitting a target dirichlet
            student_loss = self.forward_student_loss(
                start_logits = start_logits,
                end_logits = end_logits,
                noisy_start_logits = noisy_start_logits,
                noisy_end_logits = noisy_end_logits,
                context_mask = context_mask.float(),
                reduction='none' if self.user_args.answer_train_separation else 'mean'
            )

            # Update total loss
            total_loss += student_loss * self.user_args.self_ratio

            # Now mask out losses corresponding to unanswerable examples
            if self.user_args.answer_train_separation and not self.user_args.flatten_train_separation:
                assert answerable_labels is not None

                total_loss = torch.masked_select(total_loss, answerable_labels.to(torch.bool))
                total_loss = total_loss.sum()/sequence_output.size(0)

            elif self.user_args.answer_train_separation and self.user_args.flatten_train_separation:
                assert answerable_labels is not None

                unanswerable_loss = self.forward_loss_flat(
                    start_logits = start_logits,
                    end_logits = end_logits,
                    reduction = 'none',
                    context_mask = context_mask,
                )

                # unanswerable_student_loss = self.forward_student_loss_flat(
                #     start_logits = start_logits,
                #     end_logits = end_logits,
                #     reduction='none',
                #     context_mask = context_mask,
                # )

                # Update total loss
                # unanswerable_loss += unanswerable_student_loss * self.user_args.self_ratio

                # Choose the elements corresponding to answerable examples, maximising correct probability
                choice_mask = answerable_labels.to(torch.bool)
                answerable_loss = torch.masked_select(total_loss, choice_mask)

                # Choose the elements corresponding to unanswerable examples, seeking a flat distribution
                unanswerable_loss = torch.masked_select(unanswerable_loss, ~choice_mask)

                total_loss = (answerable_loss.sum() + unanswerable_loss.sum()) / sequence_output.size(0)

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


def qa_electra_large_self(args, tokenizer_only = False, **kwargs):
    # Model identifier
    electra_large = "google/electra-large-discriminator"

    # Load tokenizer
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)
    if tokenizer_only: return tokenizer

    # Load pre-trained model
    model = ElectraForQuestionAnsweringSelf.from_pretrained(electra_large)
    model.user_args = args

    # Set model noise layer
    model.srt = multiplicativegaussianlayer(
        noise_a = args.noise_a,
        noise_b = args.noise_b,
    )

    print("===> Using multiplicative noise with params ({:.2f}, {:.2f})".format(args.noise_a, args.noise_b))

    return model, tokenizer


def qa_electra_large_combo_self(args, tokenizer_only = False, **kwargs):
    # Model identifier
    electra_large = "google/electra-large-discriminator"

    # Load tokenizer
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)
    if tokenizer_only: return tokenizer

    # Load pre-trained model
    model = ElectraForQuestionAnsweringComboSelf.from_pretrained(electra_large)
    model.user_args = args

    # Set model noise layer
    model.srt = multiplicativegaussianlayer(
        noise_a = args.noise_a,
        noise_b = args.noise_b,
    )

    print("===> Using multiplicative noise with params ({:.2f}, {:.2f})".format(args.noise_a, args.noise_b))

    return model, tokenizer