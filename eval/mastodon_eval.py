from sklearn.metrics import classification_report
import numpy as np
import torch

def evalDA(haty, goldy):
    """
    Evaluate Dialogue Act (DA) classification performance using Precision, Recall, and F1-score.

    Args:
    - haty (list): Predicted Dialogue Act labels.
    - goldy (list): Ground truth Dialogue Act labels.

    Returns:
    - prec (float): Weighted average precision for all classes.
    - reca (float): Weighted average recall for all classes.
    - f1 (float): Weighted average F1-score.
    """
    nclasses = 15  # The number of Dialogue Act classes, change this based on your dataset
    nok = [0.] * nclasses  # Correctly predicted instances for each class
    nrec = [0.] * nclasses  # Total predicted instances for each class
    ntot = [0.] * nclasses  # Total actual instances for each class

    for i in range(len(haty)):
        recy = haty[i]
        gldy = goldy[i]
        ntot[gldy] += 1
        nrec[recy] += 1
        if recy == gldy:
            nok[gldy] += 1

    nsamps = sum(ntot)
    preval = [float(ntot[i]) / float(nsamps) for i in range(nclasses)]

    prec = 0.
    reca = 0.
    raweval = "DAraweval "

    for j in range(nclasses):
        tp = nok[j]
        pr, re = 0., 0.
        if nrec[j] > 0:
            pr = float(tp) / float(nrec[j])  # Precision for class j
        if ntot[j] > 0:
            re = float(tp) / float(ntot[j])  # Recall for class j
        raweval += f"Class {j}: P={pr:.4f}, R={re:.4f} | "
        prec += pr * preval[j]
        reca += re * preval[j]

    print(raweval)
    if prec + reca == 0.:
        f1 = 0.
    else:
        f1 = 2. * prec * reca / (prec + reca)  # F1 score
    return prec, reca, f1


def evalSE(haty, goldy):
    """
    Evaluate Sentiment Analysis (SE) performance using Precision, Recall, and F1-score.

    Args:
    - haty (list): Predicted sentiment labels (0: Neutral, 1: Positive, 2: Negative).
    - goldy (list): Ground truth sentiment labels.

    Returns:
    - avg_prec (float): Average precision across positive and negative sentiment.
    - avg_reca (float): Average recall across positive and negative sentiment.
    - avg_f1 (float): Average F1-score across positive and negative sentiment.
    """
    nclasses = 3  # Sentiment classes: Neutral (0), Positive (1), Negative (2)
    nok = [0.] * nclasses
    nrec = [0.] * nclasses
    ntot = [0.] * nclasses

    for i in range(len(haty)):
        recy = haty[i]
        gldy = goldy[i]
        ntot[gldy] += 1
        nrec[recy] += 1
        if recy == gldy:
            nok[gldy] += 1

    raweval = "SEraweval "
    f1pos, f1neg = 0., 0.
    pr_pos, re_pos = 0., 0.
    pr_neg, re_neg = 0., 0.

    # Evaluate Positive sentiment (1)
    for j in (1,):
        tp = nok[j]
        pr, re = 0., 0.
        if nrec[j] > 0:
            pr = float(tp) / float(nrec[j])  # Precision for positive sentiment
        if ntot[j] > 0:
            re = float(tp) / float(ntot[j])  # Recall for positive sentiment
        pr_pos, re_pos = pr, re
        raweval += f"Positive: P={pr:.4f}, R={re:.4f} | "
        if pr + re > 0.:
            f1pos = 2. * pr * re / (pr + re)

    # Evaluate Negative sentiment (2)
    for j in (2,):
        tp = nok[j]
        pr, re = 0., 0.
        if nrec[j] > 0:
            pr = float(tp) / float(nrec[j])  # Precision for negative sentiment
        if ntot[j] > 0:
            re = float(tp) / float(ntot[j])  # Recall for negative sentiment
        pr_neg, re_neg = pr, re
        raweval += f"Negative: P={pr:.4f}, R={re:.4f} | "
        if pr + re > 0.:
            f1neg = 2. * pr * re / (pr + re)

    print(raweval)

    # Average Precision, Recall, and F1 score across Positive and Negative sentiments
    avg_prec = (pr_pos + pr_neg) / 2.
    avg_reca = (re_pos + re_neg) / 2.
    avg_f1 = (f1pos + f1neg) / 2.
    return avg_prec, avg_reca, avg_f1


def evaluate(model, dev_loader, device):
    """
    Evaluate the model on the validation dataset.

    Args:
    - model: The trained model.
    - dev_loader: The DataLoader object for the validation dataset.
    - device: The device to run the model on (GPU or CPU).

    Returns:
    - A dictionary with precision, recall, and F1 scores for both DA and SE tasks.
    """
    model.eval()
    all_sentiment_preds = []
    all_act_preds = []
    all_sentiment_labels = []
    all_act_labels = []
    all_speaker_preds = []
    all_speaker_labels = []

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_labels = batch['sentiment'].to(device)
            act_labels = batch['act'].to(device)
            speakers = batch['speakers']
            utterance_positions = batch['utterance_positions']
            current_utterance_position_index = batch['current_utterance_position_index']
            importance_scores = batch['importance_scores']
            importance_mask = batch['importance_mask']

            outputs = model(
                input_ids,
                attention_mask,
                sentiment_labels=None,
                act_labels=None,
                speakers=speakers,
                utterance_positions=utterance_positions,
                current_utterance_position_index=current_utterance_position_index,
                importance_scores=importance_scores,
                importance_mask=importance_mask
            )

            loss, sentiment_logits, act_logits, speaker_logits_list, same_speaker_labels_list, importance_logits_list, importance_labels_list = outputs

            # Sentiment predictions
            sentiment_preds = torch.argmax(sentiment_logits, dim=1).cpu().numpy()
            act_preds = torch.argmax(act_logits, dim=1).cpu().numpy()

            all_sentiment_preds.extend(sentiment_preds)
            all_act_preds.extend(act_preds)
            all_sentiment_labels.extend(sentiment_labels.cpu().numpy())
            all_act_labels.extend(act_labels.cpu().numpy())

            # Speaker predictions
            if len(speaker_logits_list) > 0:
                speaker_logits_cat = torch.cat(speaker_logits_list, dim=0)
                speaker_preds = (torch.sigmoid(speaker_logits_cat) > 0.5).int().cpu().numpy()
                speaker_labels = torch.cat(same_speaker_labels_list, dim=0).cpu().numpy()

                all_speaker_preds.extend(speaker_preds)
                all_speaker_labels.extend(speaker_labels)

        # Evaluate Dialogue Act (DA) classification
        prec_da, reca_da, f1_da = evalDA(all_act_preds, all_act_labels)

        # Evaluate Sentiment (SE) classification
        prec_se, reca_se, f1_se = evalSE(all_sentiment_preds, all_sentiment_labels)

        # Calculate speaker accuracy
        if all_speaker_labels:
            speaker_accuracy = np.mean(np.array(all_speaker_preds) == np.array(all_speaker_labels))
        else:
            speaker_accuracy = 0.0

        # Return evaluation metrics
        return {
            'act': {'precision': prec_da, 'recall': reca_da, 'f1': f1_da},
            'sentiment': {'precision': prec_se, 'recall': reca_se, 'f1': f1_se},
            'speaker': {'accuracy': speaker_accuracy}
        }
