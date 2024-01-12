import collections
from tqdm import tqdm
import json
import torch
import os

def compute_f1(list_ref, list_hyp):
    """Compute F1 score from reference (grouth truth) list and hypothesis list.
    Args:
        list_ref: List of true elements.
        list_hyp: List of postive (retrieved) elements.
    Returns:
        A F1Scores object containing F1, precision, and recall scores.
    """

    F1Scores = collections.namedtuple("F1Scores", ["f1", "precision", "recall", "tp", "fp", "fn"])

    ref = collections.Counter(list_ref)
    hyp = collections.Counter(list_hyp)
    true = sum(ref.values())
    positive = sum(hyp.values())
    true_positive = sum((ref & hyp).values())
    precision = float(true_positive) / positive if positive else 1.0
    recall = float(true_positive) / true if true else 1.0

    false_positive = positive - true_positive
    false_negative = true - true_positive

    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:  # The F1-score is defined to be 0 if both precision and recall are 0.
        f1 = 0.0

    return F1Scores(f1=f1, precision=precision, recall=recall, tp=true_positive, fp=false_positive, fn=false_negative)

def train(model, optimizer, train_loader, device):
    """
    Train 1 epoch.
    Args:
        model: Model to be training. (torch.nn.Module)
        optimizer: Optimizer to use. (torch.optim)
        train_loader: DataLoader for training. (torch.utils.data.DataLoader)
    Returns:
        model: Trained Model. (torch.nn.Module)
        special_loss: Average of Losses Shown while Training about Special Slots. (Float)
        bio_loss: Average of Losses Shown while Training about Bio Tags. (Float)
    """

    model.train()

    tqdm_train_loader = tqdm(train_loader)
    total_special_loss = []
    total_bio_loss = []

    for batch_idx, batch in enumerate(tqdm_train_loader):

        batch = {key: value.to(device) for key, value in batch.items()}

        outputs = model(**batch)

        loss = outputs['special_loss'].mean() + outputs['bio_loss'].mean()
        total_special_loss.append(outputs['special_loss'].mean().item())
        total_bio_loss.append(outputs['bio_loss'].mean().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tqdm_train_loader.set_description("special loss: {} | bio loss: {}".format(
            round(sum(total_special_loss) / len(total_special_loss), 6),
            round(sum(total_bio_loss) / len(total_bio_loss), 6)
        ))

    return {
        "model": model,
        "special_loss": round(sum(total_special_loss) / len(total_special_loss), 6),
        "bio_loss": round(sum(total_bio_loss) / len(total_bio_loss), 6)
    }

def evaluate(model, eval_set, eval_loader, device):
    """
    Evaluate the Evaluate Dataloader.
    Args:
        model: Model to be Training. (torch.nn.Module)
        eval_set: Evaluation Dataset. (torch.utils.data.Dataset)
        eval_loader: DataLoader for Evaluating. (torch.utils.data.DataLoader)
    Returns:
        result: Evaluation Results. (list(dict))
            sentence: Utterances. (str)
            targets: The Original Dialog State. (list(str))
            predictions: The Prediction of Dialog State. (list(str))
    """

    model.eval()

    tqdm_eval_loader = tqdm(eval_loader)
    result = []

    for batch_idx, batch in enumerate(tqdm_eval_loader):

        batch = {key: value.to(device) for key, value in batch.items()}

        outputs = model(**batch)

        special_preds = outputs["special_logits"].argmax(-1)
        bio_preds = outputs["bio_logits"].argmax(-1)

        batch_size = batch["input_ids"].shape[0]
        for sample_idx in range(batch_size):

            pad_idx = torch.where(batch["input_ids"][sample_idx] == eval_set.tokenizer.sep_token_id)[0][0].item()

            special_pred = special_preds[sample_idx]
            bio_pred = bio_preds[sample_idx][:pad_idx]
            
            preds = []
            labels = []

            for i, p in enumerate(special_pred != 0):
                if p:
                    preds.append(eval_set.special_slots[i] + "-" + eval_set.special_classes[special_pred[i]])

            tmp_slot = None
            tmp_value = None
            for i, p in enumerate(bio_pred != 0):
                if p:
                    if eval_set.bio_classes[bio_pred[i]].startswith("B-"):
                        if tmp_value is not None:
                            preds.append(tmp_slot + "-" + eval_set.tokenizer.decode(tmp_value))
                        tmp_slot = eval_set.bio_classes[bio_pred[i]][2:]
                        tmp_value = [batch["input_ids"][sample_idx][i]]

                    elif eval_set.bio_classes[bio_pred[i]].startswith("I-") and tmp_value is not None:
                        tmp_value.append(batch["input_ids"][sample_idx][i])

                else:
                    if tmp_slot is not None and tmp_value is not None:
                        preds.append(tmp_slot + "-" + eval_set.tokenizer.decode(tmp_value))
                        tmp_slot = None
                        tmp_value = None
    
            if tmp_slot is not None and tmp_value is not None:
                preds.append(tmp_slot + "-" + eval_set.tokenizer.decode(tmp_value))

            preds = list(set(preds))
            preds = [k.replace('##', '') for k in preds]
            preds = [pred for pred in preds if eval_set.tokenizer.unk_token not in pred]

            preds.sort()

            special_label = batch["special_labels"][sample_idx]
            bio_label = batch["bio_labels"][sample_idx][:pad_idx]

            for i, l in enumerate(special_label != 0):
                if l:
                    labels.append(eval_set.special_slots[i] + "-" + eval_set.special_classes[special_label[i]])

            tmp_slot = None
            tmp_value = None
            for i, l in enumerate(bio_label != 0):
                if l and bio_label[i] != -100:
                    if eval_set.bio_classes[bio_label[i]].startswith("B-"):
                        if tmp_value is not None:
                            labels.append(tmp_slot + "-" + eval_set.tokenizer.decode(tmp_value))

                        tmp_slot = eval_set.bio_classes[bio_label[i]][2:]
                        tmp_value = [batch["input_ids"][sample_idx][i]]
                    elif eval_set.bio_classes[bio_label[i]].startswith("I-") and eval_set.bio_classes[bio_label[i]][2:] == tmp_slot:
                        tmp_value.append(batch["input_ids"][sample_idx][i])
                else:
                    if tmp_slot is not None and tmp_value is not None:
                        labels.append(tmp_slot + "-" + eval_set.tokenizer.decode(tmp_value))
                        tmp_slot = None
                        tmp_value = None

            if tmp_slot is not None and tmp_value is not None:
                labels.append(tmp_slot + "-" + eval_set.tokenizer.decode(tmp_value))

            labels = list(set(labels))
            labels = [k.replace('##', '') for k in labels]
            labels = [label for label in labels if eval_set.tokenizer.unk_token not in label]

            labels.sort()

            result.append({
                "sentence": eval_set.samples[batch_idx * batch_size + sample_idx]["utterance"],
                "targets": labels,
                "predictions": preds,
            })

    return result


def get_scores(result, dataset):
    """
    Scoring for Evaluation Results.
    Args:
        result: Evalution Result. (list(dict))
        dataset: Evaluation Dataset. (torch.utils.data.Dataset)
    Returns:
        dialogs: A Dialogues of the Results for each Utterances. (list(dict))
        total_f1: Average of F1 Scores for each Utterances. (float)
    """

    total_f1 = []
    total_precision = []
    total_recall = []
    total_tp = []
    total_fp = []
    total_fn = []
    outputs = dict()

    for i, sample in enumerate(result):

        origin_sample = dataset.samples[i]

        if origin_sample["dialog_id"] not in outputs:
            outputs[origin_sample["dialog_id"]] = list()
            targets = sample["targets"].copy()
            predictions = sample["predictions"].copy()
        else:
            targets = (sample["targets"] + outputs[origin_sample["dialog_id"]][-1]["targets"]).copy()
            predictions = (sample["predictions"] + outputs[origin_sample["dialog_id"]][-1]["predictions"]).copy()

        dialog = {
            "utterance_id": origin_sample["utterance_id"],
            "utterance": origin_sample["utterance"],
            "targets": sorted(list(set(targets))),
            "predictions": sorted(list(set(predictions))),
        }

        if dialog["targets"] or dialog["predictions"]:
            f1 = compute_f1(dialog["targets"], dialog["predictions"]).f1
            precision = compute_f1(dialog["targets"], dialog["predictions"]).precision
            recall = compute_f1(dialog["targets"], dialog["predictions"]).recall
            tp = compute_f1(dialog["targets"], dialog["predictions"]).tp
            fp = compute_f1(dialog["targets"], dialog["predictions"]).fp
            fn = compute_f1(dialog["targets"], dialog["predictions"]).fn
            total_f1.append(f1)
            total_precision.append(precision)
            total_recall.append(recall)
            total_tp.append(tp)
            total_fp.append(fp)
            total_fn.append(fn)
        else:
            f1 = 0.
            precision = 0.
            recall = 0.

        outputs[origin_sample["dialog_id"]].append(dialog)

    return_precision = sum(total_tp) / (sum(total_tp) + sum(total_fp)) if sum(total_tp) + sum(total_fp) else 0.
    return_recall = sum(total_tp) / (sum(total_tp) + sum(total_fn)) if sum(total_tp) + sum(total_fp) else 0.
    return_f1 = 2 * (return_precision * return_recall) / (return_precision + return_recall) if return_precision + return_recall else 0.

    return {
        "dialogs": outputs,
        "sample_f1": sum(total_f1) / len(total_f1),
        "total_f1": return_f1,
        "total_precision": return_precision,
        "total_recall": return_recall,
        "total_tp": sum(total_tp),
        "total_fp": sum(total_fp),
        "total_fn": sum(total_fn)
    }
