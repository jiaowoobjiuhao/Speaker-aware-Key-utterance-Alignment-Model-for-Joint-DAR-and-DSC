import torch
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

def evaluate(model, data_loader, device):
    model.eval()
    all_sentiment_preds = []
    all_act_preds = []
    all_sentiment_labels = []
    all_act_labels = []
    all_speaker_preds = []
    all_speaker_labels = []
    total_importance_loss = 0.0
    total_importance_steps = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_labels = batch['sentiment'].to(device)
            act_labels = batch['act'].to(device)
            speakers = batch['speakers']
            utterance_positions = batch['utterance_positions']
            current_utterance_position_index = batch['current_utterance_position_index']
            importance_scores = batch['importance_scores']
            importance_mask = batch['importance_mask']

            _, sentiment_logits, act_logits, speaker_logits_list, same_speaker_labels_list, importance_logits_list, importance_labels_list = model(
                input_ids,
                attention_mask,
                speakers=speakers,
                utterance_positions=utterance_positions,
                current_utterance_position_index=current_utterance_position_index,
                importance_scores=importance_scores,
                importance_mask=importance_mask
            )

            # Sentiment and act predictions
            sentiment_preds = torch.argmax(sentiment_logits, dim=1).cpu().numpy()
            act_preds = torch.argmax(act_logits, dim=1).cpu().numpy()

            all_sentiment_preds.extend(sentiment_preds)
            all_act_preds.extend(act_preds)
            all_sentiment_labels.extend(sentiment_labels.cpu().numpy())
            all_act_labels.extend(act_labels.cpu().numpy())

            # Speaker predictions
            if len(speaker_logits_list) > 0:
                speaker_logits_cat = torch.cat(speaker_logits_list, dim=0)
                speaker_logits_cat = torch.cat(speaker_logits_list, dim=0)
                speaker_labels_cat = torch.cat(same_speaker_labels_list, dim=0)
                speaker_preds = (torch.sigmoid(speaker_logits_cat) > 0.5).int().cpu().numpy()
                speaker_labels = speaker_labels_cat.cpu().numpy()

                all_speaker_preds.extend(speaker_preds)
                all_speaker_labels.extend(speaker_labels)

            # Importance predictions and loss
            for importance_logits, importance_scores_i in zip(importance_logits_list, importance_labels_list):
                if importance_logits.numel() > 0:
                    importance_loss_fct = torch.nn.MSELoss(reduction='mean')
                    importance_loss = importance_loss_fct(importance_logits.cpu(), importance_scores_i.cpu())
                    total_importance_loss += importance_loss.item()
                    total_importance_steps += 1

    # Generate classification reports
    sentiment_report = classification_report(all_sentiment_labels, all_sentiment_preds,
                                             target_names=sentiment_encoder.classes_, output_dict=True)
    act_report = classification_report(all_act_labels, all_act_preds,
                                       target_names=act_encoder.classes_, output_dict=True)

    # Speaker prediction accuracy
    if all_speaker_labels:
        speaker_accuracy = accuracy_score(all_speaker_labels, all_speaker_preds)
    else:
        speaker_accuracy = 0.0

    # Average importance loss
    if total_importance_steps > 0:
        avg_importance_loss = total_importance_loss / total_importance_steps
    else:
        avg_importance_loss = 0.0

    # Return metrics
    return {
        'act': {
            'precision': act_report['macro avg']['precision'],
            'recall': act_report['macro avg']['recall'],
            'f1': act_report['macro avg']['f1-score']
        },
        'sentiment': {
            'precision': sentiment_report['macro avg']['precision'],
            'recall': sentiment_report['macro avg']['recall'],
            'f1': sentiment_report['macro avg']['f1-score']
        },
        'speaker': {
            'accuracy': speaker_accuracy
        },
        'importance_loss': avg_importance_loss
    }