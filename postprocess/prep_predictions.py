#! /usr/bin/env python

"""
Prepare predictions for official SQuAD evaluation
"""
# TODO There is a fair amount of code repetition here which can probably be refactored for readability
# TODO: Split the code up in this file into smaller more readable functions and add more comments

import os
import sys
import json
import collections

dirname, filename = os.path.split(os.path.abspath(__file__))
sys.path.append(dirname+'/..')

import numpy as np
import scipy as sp
import copy as c

from datasets import load_dataset
from transformers import ElectraTokenizer, ElectraConfig
from utils import get_args_prep
from uncertainty.logits import load_class as load_uncertainty_class

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

# Get all arguments for postprocessing
args = get_args_prep().parse_args()


def normalise(arr):
    arr = np.array(arr)
    return (arr - arr.mean())/arr.std()


def strip_accents(text):
    text = text.replace("ö", "o").replace("ü", "u").replace("á", "a").replace("é", "e").replace("í", "i")
    text = text.replace("ó", "o").replace("ú", "u").replace("ñ", "n").replace("ç", "c").replace("â", "a")
    text = text.replace("ê", "e").replace("î", "i").replace("ô", "o").replace("û", "u").replace("à", "a")
    text = text.replace("è", "e").replace("ì", "i").replace("ò", "o").replace("ù", "u").replace("ë", "e")
    text = text.replace("ï", "i").replace("ä", "a").replace("ć", "c").replace("ń", "n").replace("ś", "s")
    text = text.replace("ź", "z").replace("ł", "l").replace("ż", "z").replace("ą", "a").replace("ę", "e")
    text = text.replace("š", "s").replace("ř", "r").replace("ů", "u").replace("č", "c").replace("ě", "e")
    text = text.replace("ž", "z").replace("ý", "y").replace("ā", "a").replace("ē", "e").replace("ī", "i")
    text = text.replace("ō", "o").replace("ū", "u").replace("õ", "o").replace("\u1ea1", "a").replace("\u1eb1", "a")
    text = text.replace("\u1ec7", "e").replace("å", "a")
    return text


def clean_answer(answer, context):
    if answer != "":
        noSpaceToWithSpace = []
        for pos, char in enumerate(context):
            if char != ' ':
                noSpaceToWithSpace.append(pos)
        passage_noSpace = strip_accents(context.lower().replace(" ", ""))
        answer_noSpace = strip_accents(answer.replace(" ", ""))
        if answer_noSpace in passage_noSpace:
            start_char_noSpace = passage_noSpace.find(answer_noSpace)
            end_char_noSpace = start_char_noSpace + len(answer_noSpace) - 1
            start_char = noSpaceToWithSpace[start_char_noSpace]
            end_char = noSpaceToWithSpace[end_char_noSpace]
            answer = context[start_char:end_char + 1]
    return answer


def get_best_indices(start_indices, end_indices, start_logits, end_logits):

    # Keep track of all preliminary predictions
    PredictionInfo = collections.namedtuple("PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit"])

    prediction_info = [PredictionInfo(
        start_index = 0, end_index = 0,
        start_logit = start_logits[0],
        end_logit = end_logits[0]
    )]

    for start_index in start_indices:
        for end_index in end_indices:
            # Throw out invalid predictions
            if end_index < start_index: continue

            # Append valid predictions
            prediction_info.append(PredictionInfo(
                start_index=start_index, end_index=end_index,
                start_logit=start_logits[start_index],
                end_logit=end_logits[end_index]
            ))

    # Sort preliminary predictions by their score
    prediction_info = sorted(prediction_info, key = lambda x: (x.start_logit + x.end_logit), reverse = True)

    # Return the prediction with the jointly highest score
    return prediction_info[0]


def ood_detection(domain_labels, measures, mode, rev):

    # Convert to high bit float
    measures = np.asarray(measures, dtype=np.float128)

    if rev: measures *= -1.0

    if mode == 'PR':
        precision, recall, thresholds = precision_recall_curve(domain_labels, measures)
        aupr = auc(recall, precision)

        # Find the best threshold for F1 score
        best_result = find_best_f1(precision, recall, thresholds)

        return aupr, best_result

    elif mode == 'ROC':
        # fpr, tpr, thresholds = roc_curve(domain_labels, measures)
        auroc = roc_auc_score(domain_labels, measures)
        return auroc


def find_best_f1(precision, recall, threshold):

    # Calculate F1 score
    f_score = (2 * precision * recall) / (precision + recall)

    # Find all nan positions
    nan_pos = np.squeeze(np.argwhere(np.isnan(f_score)))

    # Remove them from the f1 score
    f_score_masked = np.delete(f_score, nan_pos, None)

    # Find the best score
    best = np.amax(f_score_masked)

    # Find the position of best score
    pos = np.where(f_score == best)

    return precision[pos], recall[pos], threshold[pos], best


def main(args):

    # Store command for future reference
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')

    with open('CMDs/postprocess.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Number of models
    n = len(args.load_dirs)

    # Save all predictions within this dictionary
    logit_predictions = {'start': [], 'end': []}

    # Iterate over all directory paths
    for path in args.load_dirs:
        filename = "" if args.dataset == 'squad' else "_" + args.dataset

        # Load start logits
        file = os.path.join(path, "pred_start_logits{}.npy".format(filename))
        logit_predictions['start'].append(np.load(file))

        # Load end logits
        file = os.path.join(path, "pred_end_logits{}.npy".format(filename))
        logit_predictions['end'].append(np.load(file))

    unanswerability_probs = []
    if args.use_unans_probs == 1:
        for path in args.load_dirs:
            file = os.path.join(path, "pred_unans_probs{}.npy".format("_" + args.dataset))
            unanswerability_probs.append(np.load(file))

        # Ensemble answerability predictions
        unanswerability_probs = np.mean(unanswerability_probs, axis=0)

    # Convert into numpy arrays
    # The logits will have dimension (num models, dataset size, maxlen)
    for key, item in logit_predictions.items():
        logit_predictions[key] = np.array(item)

    # Loading dev data
    dev_data = load_dataset(args.dataset, split='validation')

    # TODO: This should be specified based on what model arch we are using, see test script
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-large-discriminator', do_lower_case=True)

    # Find predictions as word spans
    span_predictions = {}

    # Get all uncertainties
    unc_predictions = {}

    # Store unanswerability labels for the detection task
    unans_labels = []

    # Use the start logit for the [CLS] token
    unans_preds = []

    # Store qids to ensure ordering
    qids = []

    for i, ex in enumerate(dev_data):

        # Get the unanswerability label
        unans_labels.append(1 if len(ex["answers"]["text"]) == 0 else 0)

        # Store qid for uncertainties later on
        qids.append(ex["id"])

        # Get the question, context and id
        question, context, qid = ex["question"], ex["context"], ex["id"]

        # The input to the transformer model is based on concatenating question and context
        # this is serparated by a special SEP token
        concatenation = question + " [SEP] " + context

        # Tokenizer converts input to be compatible with chosen model
        # Note we truncate inputs exceesding max length
        input_encodings_dict = tokenizer(concatenation, truncation=True, max_length=args.max_len, padding="max_length")

        # Extract necessary pieces from tokenizer output, sequence of token ids
        input_ids = input_encodings_dict['input_ids']

        # Find occurrence of SEP tokens to isolate the context
        first_sep_idx = input_ids.index(102) + 1
        last_sep_idx  = input_ids[::-1].index(102) + 1

        # Process each example separately
        # Get the logits for all models in the ensemble (num models, seqlen)
        all_start_logits = logit_predictions['start'][:, i, first_sep_idx:-last_sep_idx]
        all_end_logits = logit_predictions['end'][:, i, first_sep_idx:-last_sep_idx]
        all_start_logits = sp.special.log_softmax(all_start_logits, axis = -1)
        all_end_logits = sp.special.log_softmax(all_end_logits, axis = -1)

        # Ensemble start and end logits
        start_logits = sp.special.logsumexp(all_start_logits, axis=0) - np.log(n)
        end_logits = sp.special.logsumexp(all_end_logits, axis=0) - np.log(n)

        # Get the unanswerablility cls predictions
        unans_preds.append(start_logits[0])

        # Tokenize contest and remove special tokens
        context_ids = tokenizer.encode(context)[1:-1]

        # Sort our start and end logits from largest to smallest, keeping track of the index
        start_idx_and_logit = sorted(enumerate(start_logits), key=lambda x: x[1], reverse=True)
        end_idx_and_logit = sorted(enumerate(end_logits), key=lambda x: x[1], reverse=True)

        # Select top 10 indexes for computational efficiency
        start_indices = [idx for idx, _ in start_idx_and_logit[:10]]
        end_indices = [idx for idx, _ in end_idx_and_logit[:10]]

        # Get best predictions
        best = get_best_indices(start_indices, end_indices, start_logits, end_logits)

        # Extract answer from context
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(
                context_ids[best.start_index:best.end_index + 1]
            )
        )

        # Check for unanswerability
        if args.dataset == "squad_v2":

            # Initialise estimator and get the uncertainties
            estimator = load_uncertainty_class(args)
            uncertainties = estimator(args, all_start_logits, all_end_logits)

            # Set uncertainties for later processing
            for unc_name in uncertainties:
                if unc_name not in unc_predictions:
                    unc_predictions[unc_name] = {}
                unc_predictions[unc_name][qid] = uncertainties[unc_name]

        # The answer after detokenizing often doesn't even end up being an extract from the context
        # due to spacing around various characters e.g. punctuation
        # Hence, it is necessary to match the answer to an extract from the context
        span_predictions[qid] = clean_answer(answer, context)

    if args.dataset != "squad_v2":
        with open(os.path.join(args.save_dir, 'predictions.json'), 'w') as fp:
            json.dump(span_predictions, fp)
        return

    # Store predictions without any uncertainty thresholding
    with open(os.path.join(args.save_dir, 'unc_none_squad_v2_predictions.json'), 'w') as fp:
        json.dump(span_predictions, fp)

    # Now process the uncertainties and set certain answer
    for unc_name, uncs in unc_predictions.items():

        # Copy the span predictions
        unc_span_predictions = c.deepcopy(span_predictions)

        # According to threshold fraction convert
        threshold = np.array(list(uncs.values()))
        threshold = np.quantile(threshold, 1 - args.threshold_frac)

        # Now any uncertainty exceeding this threshold will have its answer set to nan
        for qid, answer in unc_span_predictions.items():

            # If the uncertainty exceeds the threshold then set the answer to ""
            unc_span_predictions[qid] = "" if uncs[qid] > threshold else answer

        with open(os.path.join(args.save_dir, unc_name + '_squad_v2_predictions.json'), 'w') as fp:
            json.dump(unc_span_predictions, fp)

    # Get detection performance for each uncertainty
    domain_labels = np.array(unans_labels)

    for unc_name, uncs in unc_predictions.items():

        # Get the uncertainty measures
        measures = np.array([unc_predictions[unc_name][qid] for qid in qids])

        # Get the aupr and best threshold performance
        aupr, [pr, re, th, f1] = ood_detection(domain_labels, measures, mode='PR', rev = False)

        # Get the auroc
        auroc = ood_detection(domain_labels, measures, mode='ROC', rev = False)

        print("\n\nDetection of Unanswerability")
        if unc_name == 'unc_log_conf_expected':
            print(measures.mean())
        if unc_name == 'unc_entropy_expected':
            print(measures.mean())
        print(unc_name)
        print("Precision:", pr)
        print("Recall:   ", re)
        print("Threshold:", th)
        print("F1:       ", f1)
        print("AUROC:    ", auroc)
        print("AUPR:     ", aupr)

    if args.use_unans_probs == 1:
        aupr, [pr, re, th, f1] = ood_detection(domain_labels, unanswerability_probs, mode='PR', rev = False)
        auroc = ood_detection(domain_labels, unanswerability_probs, mode='ROC', rev = False)
        print("\n\nDetection of Unanswerability")
        print("Unanswerability probabilities")
        print("Precision:", pr)
        print("Recall:   ", re)
        print("Threshold:", th)
        print("F1:       ", f1)
        print("AUROC:    ", auroc)
        print("AUPR:     ", aupr)

        # Copy the span predictions
        unans_span_predictions = c.deepcopy(span_predictions)

        # According to threshold fraction convert
        threshold = np.array(list(unanswerability_probs))
        threshold = np.quantile(threshold, 1 - args.threshold_frac)

        # Now any uncertainty exceeding this threshold will have its answer set to nan
        for count, (qid, answer) in enumerate(unans_span_predictions.items()):

            # If the uncertainty exceeds the threshold then set the answer to ""
            unans_span_predictions[qid] = "" if unanswerability_probs[count] > threshold else answer

        with open(os.path.join(args.save_dir, 'unans_squad_v2_predictions.json'), 'w') as fp:
            json.dump(unans_span_predictions, fp)

    if args.use_unans_probs_implicit == 1:
        unans_preds = np.asarray(unans_preds)
        aupr, [pr, re, th, f1] = ood_detection(domain_labels, unans_preds, mode='PR', rev = False)
        auroc = ood_detection(domain_labels, unans_preds, mode='ROC', rev = False)
        print("\n\nDetection of Unanswerability")
        print("Unanswerability probabilities implicit")
        print("Precision:", pr)
        print("Recall:   ", re)
        print("Threshold:", th)
        print("F1:       ", f1)
        print("AUROC:    ", auroc)
        print("AUPR:     ", aupr)

        # Copy the span predictions
        unans_span_predictions = c.deepcopy(span_predictions)

        # According to threshold fraction convert
        threshold = np.array(list(unans_preds))
        threshold = np.quantile(threshold, 1 - args.threshold_frac)

        # Now any uncertainty exceeding this threshold will have its answer set to nan
        for count, (qid, answer) in enumerate(unans_span_predictions.items()):

            # If the uncertainty exceeds the threshold then set the answer to ""
            unans_span_predictions[qid] = "" if unans_preds[count] > threshold else answer

        with open(os.path.join(args.save_dir, 'unans_implicit_squad_v2_predictions.json'), 'w') as fp:
            json.dump(unans_span_predictions, fp)

    """
    if args.use_joint_thresholding and (args.use_unans_probs == 1 or args.use_unans_probs_implicit == 1):

        print("\n\nProcessing joing thresholding with TH = {:.4f} and JTH = {:.4f}".format(
            args.threshold_frac, args.joint_threshold_frac
        ))

        # Copy the span predictions
        primary_unans_span_predictions = c.deepcopy(span_predictions)

        # Get the main unanswerability scores
        scores = unanswerability_probs if args.use_unans_probs == 1 else unans_preds

        # Now threshold with the main metric
        threshold = np.array(list(scores))
        threshold = np.quantile(threshold, 1 - (args.threshold_frac - args.joint_threshold_frac))

        # Now any uncertainty exceeding this threshold will have its answer set to nan
        for count in range(len(primary_unans_span_predictions.keys())):

            # Get the qid
            qid = qids[count]

            # Get the answer
            answer = primary_unans_span_predictions[qid]

            # If the uncertainty exceeds the threshold then set the answer to ""
            primary_unans_span_predictions[qid] = "" if scores[count] > threshold else answer

        for second_unc_name, second_uncs in unc_predictions.items():

            # Copy the span predictions
            secondary_unans_span_predictions = c.deepcopy(primary_unans_span_predictions)

            # Get secondary metric
            secondary_scores = c.deepcopy(second_uncs)

            # Now any uncertainty exceeding this threshold will have its answer set to nan
            for count in range(len(secondary_unans_span_predictions.keys())):

                # Get the qid
                qid = qids[count]

                # Set thresholded items to negtive inf
                if scores[count] > threshold: secondary_scores[qid] = -float('inf')

            # Now threshold with the main metric
            secondary_threshold = np.array(list(secondary_scores.values()))
            secondary_threshold = np.quantile(secondary_threshold, 1 - args.joint_threshold_frac)

            for qid, answer in secondary_unans_span_predictions.items():

                # If the uncertainty exceeds the threshold then set the answer to ""
                secondary_unans_span_predictions[qid] = "" if secondary_scores[qid] > secondary_threshold else answer

            mask = np.array(list(secondary_unans_span_predictions.values())) == ""
            print("Joint", second_unc_name)
            print("Joint Fraction unanswered: {}/{} = {:.2f}".format(mask.sum(), len(mask), mask.sum()/len(mask)))

            with open(os.path.join(args.save_dir, second_unc_name + '_joint_squad_v2_predictions.json'), 'w') as fp:
                json.dump(secondary_unans_span_predictions, fp)
    """

    if args.use_joint_thresholding and (args.use_unans_probs == 1 or args.use_unans_probs_implicit == 1):

        print("\n\nProcessing joing thresholding with TH = {:.4f} and JTH = {:.4f}".format(
            args.threshold_frac, args.joint_threshold_frac
        ))

        # Copy the span predictions
        primary_unans_span_predictions = c.deepcopy(span_predictions)

        # Get the main unanswerability scores
        scores = unanswerability_probs if args.use_unans_probs == 1 else unans_preds
        scores = normalise(scores)

        for second_unc_name, second_uncs in unc_predictions.items():

            # Copy the span predictions
            secondary_unans_span_predictions = c.deepcopy(primary_unans_span_predictions)

            # Get secondary metric
            secondary_scores = c.deepcopy(second_uncs)

            # Normalise secondary scores
            secondary_scores = np.array([secondary_scores[qid] for qid in qids])
            secondary_scores = normalise(secondary_scores)

            # Combine normalised scores
            scores = scores + args.joint_threshold_frac * secondary_scores

            # Now threshold the joint metric
            secondary_threshold = np.quantile(secondary_scores, 1 - args.threshold_frac)

            # Now any uncertainty exceeding this threshold will have its answer set to nan
            for count, (qid, answer) in enumerate(secondary_unans_span_predictions.items()):
                # If the uncertainty exceeds the threshold then set the answer to ""
                secondary_unans_span_predictions[qid] = "" if secondary_threshold[count] > threshold else answer

            mask = np.array(list(secondary_unans_span_predictions.values())) == ""
            print("Joint", second_unc_name)
            print("Joint Fraction unanswered: {}/{} = {:.2f}".format(mask.sum(), len(mask), mask.sum()/len(mask)))

            with open(os.path.join(args.save_dir, second_unc_name + '_joint_squad_v2_predictions.json'), 'w') as fp:
                json.dump(secondary_unans_span_predictions, fp)


if __name__ == '__main__':
    main(args)