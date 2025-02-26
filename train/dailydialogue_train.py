import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2TokenizerFast, DebertaV2Model, AdamW
from transformers.models.bert.modeling_bert import BertLayer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import time
from tqdm import tqdm
import numpy as np
import random
import torch.backends.cudnn as cudnn

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    cudnn.deterministic = True
    cudnn.benchmark = False

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Modified read_erc_with_history function
def read_erc_with_history(input_path, tokenizer, max_history=4):
    data_list = []
    speakers_list = []  # Collect all speakers

    with open(input_path, encoding="utf-8") as file_read:
        data = json.load(file_read)

        for dialogues in data:
            if len(dialogues) == 0:
                continue

            history = []
            speakers_so_far = []
            num_utterances = len(dialogues)
            for i in range(num_utterances):
                current_speaker = f"Speaker{i % 2 + 1}"
                current_utterance = dialogues[i]['utterance']
                sentiment = dialogues[i]['sentiment']
                act = dialogues[i]['act']

                # Collect speakers
                speakers_so_far.append(current_speaker)
                speakers_list.append(current_speaker)

                # Collect recent max_history utterances
                relevant_history = history[-max_history:]  # List format
                relevant_speakers = speakers_so_far[-max_history:]

                # Prepare input text
                input_text = "The following is a conversation history. Predict the Sentiment Label and the Dialogue Act Label for the last utterance based on the preceding context. "

                if relevant_history:
                    history_text = ' [SEP] '.join(relevant_history)
                    input_text += history_text + ' [SEP] '

                # Append current utterance
                input_text += f"{current_speaker}: {current_utterance}"

                # Update history
                history.append(f"{current_speaker}: {current_utterance}")

                # Current utterance index in the history (always the last one)
                current_utterance_index = len(relevant_history)

                data_list.append({
                    "input_text": input_text,
                    "sentiment": sentiment,
                    "act": act,
                    "speakers": relevant_speakers.copy(),
                    "current_utterance_index": current_utterance_index,
                    "history": relevant_history,
                    "current_utterance": f"{current_speaker}: {current_utterance}"
                })

    # Encode speakers
    speaker_encoder = LabelEncoder()
    speaker_encoder.fit(speakers_list)

    # Convert 'speakers' in 'data_list' to integer codes
    for entry in data_list:
        entry['speakers'] = speaker_encoder.transform(entry['speakers']).tolist()

    return data_list, speaker_encoder

# Define Dataset
class ConversationDataset(Dataset):
    def __init__(self, data_list, sentiments, acts, importance_labels_list, tokenizer, max_length):
        self.texts = [item['input_text'] for item in data_list]
        self.sentiments = sentiments
        self.acts = acts
        self.speakers = [item['speakers'] for item in data_list]  # List of speaker codes
        self.current_utterance_indices = [item['current_utterance_index'] for item in data_list]
        self.importance_labels_list = importance_labels_list
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Store utterance positions
        self.utterance_positions_list = []

        for text in self.texts:
            # Encode text and get positions of each utterance
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                return_token_type_ids=False,
                return_attention_mask=True,
                return_offsets_mapping=True,
                padding='max_length',
                truncation=True
            )

            input_ids = encoding['input_ids']

            # Find positions of each utterance using [SEP] tokens
            sep_token_id = self.tokenizer.sep_token_id
            utterance_positions = [i for i, token_id in enumerate(input_ids)
                                   if token_id == sep_token_id]

            # Include [CLS] position
            utterance_positions = [0] + [pos + 1 for pos in utterance_positions]

            # Adjust for truncation
            utterance_positions = [pos for pos in utterance_positions if pos < self.max_length]

            # Store positions
            self.utterance_positions_list.append(utterance_positions)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        sentiment = self.sentiments[idx]
        act = self.acts[idx]
        speakers = self.speakers[idx]  # List of speaker codes
        current_utterance_index = self.current_utterance_indices[idx]
        importance_scores = self.importance_labels_list[idx]['importance_scores']  # Original length
        importance_mask = self.importance_labels_list[idx]['importance_mask']  # Original length

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        # Get utterance positions
        utterance_positions = torch.tensor(
            self.utterance_positions_list[idx], dtype=torch.long)

        # Adjust speakers list to match number of utterances
        num_utterances = min(utterance_positions.size(0), len(speakers))
        utterance_positions = utterance_positions[:num_utterances]
        speakers = speakers[:num_utterances]

        # Convert speakers to tensor
        speakers = torch.tensor(speakers, dtype=torch.long)

        # Current utterance position index
        if num_utterances > 0:
            current_utterance_position_index = min(
                current_utterance_index, num_utterances - 1)
        else:
            current_utterance_position_index = 0

        # Adjust importance_scores and importance_mask length
        importance_scores = torch.tensor(importance_scores, dtype=torch.float)
        importance_mask = torch.tensor(importance_mask, dtype=torch.float)

        num_history_sentences = num_utterances - 1  # Exclude current utterance

        if importance_scores.size(0) > num_history_sentences:
            # Truncate
            importance_scores = importance_scores[-num_history_sentences:]
            importance_mask = importance_mask[-num_history_sentences:]
        elif importance_scores.size(0) < num_history_sentences:
            # Pad
            padding_size = num_history_sentences - importance_scores.size(0)
            scores_padding = torch.zeros(padding_size, dtype=torch.float)
            mask_padding = torch.zeros(padding_size, dtype=torch.float)
            importance_scores = torch.cat([scores_padding, importance_scores], dim=0)
            importance_mask = torch.cat([mask_padding, importance_mask], dim=0)
        else:
            # Keep as is
            pass

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'sentiment': torch.tensor(sentiment, dtype=torch.long),
            'act': torch.tensor(act, dtype=torch.long),
            'speakers': speakers,  # Tensor
            'utterance_positions': utterance_positions,
            'current_utterance_position_index':
                torch.tensor(current_utterance_position_index, dtype=torch.long),
            'importance_scores': importance_scores,  # Tensor, length=num_history_sentences
            'importance_mask': importance_mask       # Tensor, length=num_history_sentences
        }

# Define custom collate_fn
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    sentiments = torch.stack([item['sentiment'] for item in batch])
    acts = torch.stack([item['act'] for item in batch])
    current_utterance_position_index = torch.stack(
        [item['current_utterance_position_index'] for item in batch])

    # Keep variable-length sequences
    utterance_positions = [item['utterance_positions'] for item in batch]
    speakers = [item['speakers'] for item in batch]
    importance_scores = [item['importance_scores'] for item in batch]
    importance_mask = [item['importance_mask'] for item in batch]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'sentiment': sentiments,
        'act': acts,
        'speakers': speakers,
        'utterance_positions': utterance_positions,
        'current_utterance_position_index': current_utterance_position_index,
        'importance_scores': importance_scores,  # List, length=batch_size
        'importance_mask': importance_mask       # List, length=batch_size
    }

# Define the model with modifications
class MultiTaskDeberta(torch.nn.Module):
    def __init__(self, model_name, num_sentiment_labels, num_act_labels):
        super(MultiTaskDeberta, self).__init__()
        self.deberta = DebertaV2Model.from_pretrained(
            model_name, output_hidden_states=True
        )

        self.num_sentiment_labels = num_sentiment_labels
        self.num_act_labels = num_act_labels

        # Define classifiers
        self.sentiment_classifier = torch.nn.Linear(
            self.deberta.config.hidden_size, num_sentiment_labels
        )
        self.act_classifier = torch.nn.Linear(
            self.deberta.config.hidden_size, num_act_labels
        )

        # Define detectors for speaker relation and importance
        self.speaker_relation_detector = torch.nn.Linear(
            self.deberta.config.hidden_size * 4, 1
        )
        self.importance_relation_detector = torch.nn.Linear(
            self.deberta.config.hidden_size * 4, 1
        )

        # Define MHA layers for speaker and importance modules
        self.speaker_mha_layers = 2
        self.importance_mha_layers = 2
        self.config = self.deberta.config

        # Define speaker MHA layers
        for i in range(self.speaker_mha_layers):
            layer = BertLayer(self.config)
            self.add_module(f"Speaker_MHA_{i}", layer)

        # Define importance MHA layers
        for i in range(self.importance_mha_layers):
            layer = BertLayer(self.config)
            self.add_module(f"Importance_MHA_{i}", layer)

        # Define fusion functions
        self.fusion_fct_speaker = torch.nn.Sequential(
            torch.nn.Linear(self.config.hidden_size * 4, self.config.hidden_size),
            torch.nn.Tanh()
        )
        self.fusion_fct_importance = torch.nn.Sequential(
            torch.nn.Linear(self.config.hidden_size * 4, self.config.hidden_size),
            torch.nn.Tanh()
        )

    def forward(self, input_ids, attention_mask, sentiment_labels=None,
                act_labels=None, speakers=None, utterance_positions=None,
                current_utterance_position_index=None,
                importance_scores=None, importance_mask=None):
        # Encode input and get hidden states
        outputs = self.deberta(input_ids=input_ids,
                               attention_mask=attention_mask,
                               output_hidden_states=True)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        all_hidden_states = outputs.hidden_states  # Tuple of layers

        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        hidden_size = hidden_states.size(-1)

        # Extract speaker_hidden_states and importance_hidden_states from appropriate layers
        speaker_hidden_states = all_hidden_states[-(self.speaker_mha_layers + 1)].detach()  # Tensor of shape (batch_size, seq_len, hidden_size)
        importance_hidden_states = all_hidden_states[-(self.importance_mha_layers + 1)]  # Similar shape

        # Initialize lists to collect results
        fused_hidden_states_list = []
        speaker_logits_list = []
        same_speaker_labels_list = []
        importance_loss_list = []
        importance_logits_list = []
        importance_labels_list = []

        # Iterate over each sample in the batch
        for i in range(batch_size):
            # Get data for the current sample
            input_ids_i = input_ids[i]
            attention_mask_i = attention_mask[i]
            hidden_states_i = hidden_states[i]  # (seq_len, hidden_size)
            speaker_hidden_states_i = speaker_hidden_states[i]  # (seq_len, hidden_size)
            importance_hidden_states_i = importance_hidden_states[i]  # (seq_len, hidden_size)
            speakers_i = speakers[i].to(device)
            utterance_positions_i = utterance_positions[i].to(device)
            current_idx = current_utterance_position_index[i].item()

            # Expand attention mask
            attention_mask_i_expanded = attention_mask_i.unsqueeze(0).unsqueeze(0)
            attention_mask_i_expanded = attention_mask_i_expanded.to(dtype=next(self.parameters()).dtype)
            attention_mask_i_expanded = (1.0 - attention_mask_i_expanded) * -10000.0  # (1, 1, seq_len)

            # Speaker MHA module
            speaker_hidden_states_i = speaker_hidden_states_i.unsqueeze(0)  # (1, seq_len, hidden_size)
            for j in range(self.speaker_mha_layers):
                mha_layer = getattr(self, f"Speaker_MHA_{j}")
                layer_outputs = mha_layer(hidden_states=speaker_hidden_states_i,
                                          attention_mask=attention_mask_i_expanded)
                speaker_hidden_states_i = layer_outputs[0]

            speaker_hidden_states_i = speaker_hidden_states_i.squeeze(0)  # (seq_len, hidden_size)

            # Speaker relation detection
            num_utterances = utterance_positions_i.size(0)
            if num_utterances > 1:
                positions = utterance_positions_i  # (num_utterances,)
                utterance_embs = speaker_hidden_states_i[positions, :]  # (num_utterances, hidden_size)
                current_utterance_emb = utterance_embs[current_idx]  # (hidden_size,)

                # Exclude current utterance embedding
                other_indices = list(range(num_utterances))
                other_indices.remove(current_idx)
                other_utterance_embs = utterance_embs[other_indices, :]  # (num_other, hidden_size)

                # Compute speaker relation features
                current_emb_expanded = current_utterance_emb.unsqueeze(0).expand(other_utterance_embs.size(0), -1)
                speaker_features = torch.cat([
                    other_utterance_embs,
                    current_emb_expanded,
                    other_utterance_embs * current_emb_expanded,
                    other_utterance_embs - current_emb_expanded
                ], dim=-1)  # (num_other, hidden_size * 4)

                speaker_logit = self.speaker_relation_detector(speaker_features).squeeze(-1)  # (num_other,)
                speaker_logits_list.append(speaker_logit)

                # Compute same speaker labels
                masked_speaker = speakers_i[current_idx]
                other_speakers = speakers_i[other_indices]
                same_speaker_labels = (other_speakers == masked_speaker).float()
                same_speaker_labels_list.append(same_speaker_labels)
            else:
                speaker_logit = torch.tensor([], device=device)
                same_speaker_labels = torch.tensor([], device=device)

            # Fuse hidden states and speaker module output
            fused_hidden_states_speaker_i = self.fusion_fct_speaker(
                torch.cat([
                    hidden_states_i,
                    speaker_hidden_states_i,
                    hidden_states_i * speaker_hidden_states_i,
                    hidden_states_i - speaker_hidden_states_i
                ], dim=-1)
            )  # (seq_len, hidden_size)

            # Importance MHA module
            importance_hidden_states_i = importance_hidden_states_i.unsqueeze(0)  # (1, seq_len, hidden_size)
            for j in range(self.importance_mha_layers):
                mha_layer = getattr(self, f"Importance_MHA_{j}")
                layer_outputs = mha_layer(hidden_states=importance_hidden_states_i,
                                          attention_mask=attention_mask_i_expanded)
                importance_hidden_states_i = layer_outputs[0]

            importance_hidden_states_i = importance_hidden_states_i.squeeze(0)  # (seq_len, hidden_size)

            # Importance relation detection
            if num_utterances > 1:
                positions = utterance_positions_i  # (num_utterances,)
                utterance_embs = importance_hidden_states_i[positions, :]  # (num_utterances, hidden_size)
                current_utterance_emb = utterance_embs[current_idx]  # (hidden_size,)

                # Exclude current utterance embedding
                history_indices = list(range(current_idx))  # History utterance indices
                if len(history_indices) > 0:
                    history_utterance_embs = utterance_embs[history_indices, :]  # (num_history, hidden_size)

                    current_emb_expanded = current_utterance_emb.unsqueeze(0).expand(history_utterance_embs.size(0), -1)
                    importance_features = torch.cat([
                        history_utterance_embs,
                        current_emb_expanded,
                        history_utterance_embs * current_emb_expanded,
                        history_utterance_embs - current_emb_expanded
                    ], dim=-1)  # (num_history, hidden_size * 4)

                    importance_logits = self.importance_relation_detector(importance_features).squeeze(-1)  # (num_history,)
                    importance_logits_list.append(importance_logits)

                    # Compute importance loss
                    importance_scores_i = importance_scores[i].to(device)
                    importance_mask_i = importance_mask[i].to(device)
                    importance_scores_i = importance_scores_i[-len(history_indices):]
                    importance_mask_i = importance_mask_i[-len(history_indices):]

                    importance_loss_fct = torch.nn.MSELoss(reduction='none')
                    importance_loss_i = importance_loss_fct(importance_logits, importance_scores_i)
                    masked_importance_loss = (importance_loss_i * importance_mask_i).sum() / importance_mask_i.sum().clamp(min=1e-8)
                    importance_loss_list.append(masked_importance_loss)
                    importance_labels_list.append(importance_scores_i)
                else:
                    importance_loss_list.append(torch.tensor(0.0, device=device))
                    importance_logits_list.append(torch.tensor([], device=device))
                    importance_labels_list.append(torch.tensor([], device=device))
            else:
                importance_loss_list.append(torch.tensor(0.0, device=device))
                importance_logits_list.append(torch.tensor([], device=device))
                importance_labels_list.append(torch.tensor([], device=device))

            # Fuse hidden states and importance module output
            fused_hidden_states_i = self.fusion_fct_importance(
                torch.cat([
                    fused_hidden_states_speaker_i,
                    importance_hidden_states_i,
                    fused_hidden_states_speaker_i * importance_hidden_states_i,
                    fused_hidden_states_speaker_i - importance_hidden_states_i
                ], dim=-1)
            )  # (seq_len, hidden_size)

            # Add fused hidden states to the list
            fused_hidden_states_list.append(fused_hidden_states_i)

        # Convert list to tensor
        fused_hidden_states = torch.stack(fused_hidden_states_list, dim=0)  # (batch_size, seq_len, hidden_size)

        # Get positions of current utterances for classification
        current_utterance_positions = []
        for i in range(batch_size):
            if utterance_positions[i].numel() == 0:
                current_utterance_positions.append(torch.tensor(0, device=input_ids.device))
            else:
                idx = current_utterance_position_index[i].item()
                if idx >= utterance_positions[i].size(0):
                    idx = utterance_positions[i].size(0) - 1
                current_pos = utterance_positions[i][idx]
                current_utterance_positions.append(current_pos)
        current_utterance_positions = torch.stack(current_utterance_positions).to(input_ids.device)

        # Clamp positions to prevent out-of-bounds indexing
        current_utterance_positions = torch.clamp(current_utterance_positions, max=fused_hidden_states.size(1) - 1)

        pooled_output = fused_hidden_states[
            torch.arange(batch_size, device=input_ids.device), current_utterance_positions, :
        ]

        sentiment_logits = self.sentiment_classifier(pooled_output)
        act_logits = self.act_classifier(pooled_output)

        # Compute loss
        total_loss = None
        if sentiment_labels is not None and act_labels is not None:
            # Sentiment and act prediction use CrossEntropyLoss
            loss_fct = torch.nn.CrossEntropyLoss()
            sentiment_loss = loss_fct(sentiment_logits, sentiment_labels)
            act_loss = loss_fct(act_logits, act_labels)

            # Speaker prediction loss
            if len(speaker_logits_list) > 0:
                speaker_logits_cat = torch.cat(speaker_logits_list, dim=0)
                speaker_labels_cat = torch.cat(same_speaker_labels_list, dim=0).to(device)
                bce_loss_fct = torch.nn.BCEWithLogitsLoss()
                speaker_loss = bce_loss_fct(speaker_logits_cat, speaker_labels_cat)
            else:
                speaker_loss = torch.tensor(0.0, device=device)

            # Importance prediction loss
            if len(importance_loss_list) > 0:
                importance_loss = torch.stack(importance_loss_list).mean()
            else:
                importance_loss = torch.tensor(0.0, device=device)

            # Aggregate all losses
            total_loss = sentiment_loss + act_loss + 0.8 * speaker_loss + 0.5 * importance_loss

        return total_loss, sentiment_logits, act_logits, speaker_logits_list, same_speaker_labels_list, importance_logits_list, importance_labels_list

# Evaluation function (unchanged)
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

if __name__ == "__main__":
    set_seed(42)
    # Data paths
    train_path = '/public/home/acy7tl31mi/demo2-classification/dataset/dailydialogue/train.json'
    dev_path = '/public/home/acy7tl31mi/demo2-classification/dataset/dailydialogue/dev.json'

    # Load tokenizer
    tokenizer = DebertaV2TokenizerFast.from_pretrained(
        "/public/home/acy7tl31mi/demo2-classification/classify-deberta/trainmodel-deberta-v3-base")
    model_name = "/public/home/acy7tl31mi/demo2-classification/classify-deberta/trainmodel-deberta-v3-base"

    # Read data with history limit
    train_data, speaker_encoder = read_erc_with_history(train_path, tokenizer, max_history=4)
    dev_data, _ = read_erc_with_history(dev_path, tokenizer, max_history=4)  # Reuse same encoder

    # Load importance labels
    with open('/public/home/acy7tl31mi/demo2-classification/classify-deberta/dailydialogue-train_importance_labels-debertav2xl.json', 'r', encoding='utf-8') as f:
        train_importance_labels = json.load(f)

    with open('/public/home/acy7tl31mi/demo2-classification/classify-deberta/dailydialogue-dev_importance_labels-debertav2xl.json', 'r', encoding='utf-8') as f:
        dev_importance_labels = json.load(f)

    # Label encoding
    sentiment_encoder = LabelEncoder()
    act_encoder = LabelEncoder()

    train_sentiments = [item['sentiment'] for item in train_data]
    train_acts = [item['act'] for item in train_data]

    dev_sentiments = [item['sentiment'] for item in dev_data]
    dev_acts = [item['act'] for item in dev_data]

    train_sentiments_encoded = sentiment_encoder.fit_transform(train_sentiments)
    train_acts_encoded = act_encoder.fit_transform(train_acts)

    dev_sentiments_encoded = sentiment_encoder.transform(dev_sentiments)
    dev_acts_encoded = act_encoder.transform(dev_acts)

    np.save('sentiment_classes.npy', sentiment_encoder.classes_)
    np.save('act_classes.npy', act_encoder.classes_)
    np.save('speaker_classes.npy', speaker_encoder.classes_)

    # Create datasets and loaders
    BATCH_SIZE = 8
    MAX_LENGTH = 256

    train_dataset = ConversationDataset(
        data_list=train_data,
        sentiments=train_sentiments_encoded,
        acts=train_acts_encoded,
        importance_labels_list=train_importance_labels,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    dev_dataset = ConversationDataset(
        data_list=dev_data,
        sentiments=dev_sentiments_encoded,
        acts=dev_acts_encoded,
        importance_labels_list=dev_importance_labels,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn)

    # Instantiate model
    num_sentiment_labels = len(sentiment_encoder.classes_)
    num_act_labels = len(act_encoder.classes_)
    model = MultiTaskDeberta(model_name, num_sentiment_labels, num_act_labels)
    model.to(device)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    # Train model
    EPOCHS = 30

    best_avg_f1 = 0.0
    best_act_f1 = 0.0
    best_sentiment_f1 = 0.0

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0
        total_batches = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}"):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_labels = batch['sentiment'].to(device)
            act_labels = batch['act'].to(device)
            speakers = batch['speakers']
            utterance_positions = batch['utterance_positions']
            current_utterance_position_index = batch['current_utterance_position_index']
            importance_scores = batch['importance_scores']
            importance_mask = batch['importance_mask']

            loss, _, _, _, _, _, _ = model(
                input_ids,
                attention_mask,
                sentiment_labels=sentiment_labels,
                act_labels=act_labels,
                speakers=speakers,
                utterance_positions=utterance_positions,
                current_utterance_position_index=current_utterance_position_index,
                importance_scores=importance_scores,
                importance_mask=importance_mask
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / total_batches
        epoch_time = time.time() - start_time

        print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}, '
              f'Time: {epoch_time:.2f}s')

        metrics = evaluate(model, dev_loader, device)

        print(f'Epoch {epoch + 1}/{EPOCHS} Evaluation:')
        print(f'Sentiment Classification - Precision: '
              f'{metrics["sentiment"]["precision"]:.4f}, Recall: '
              f'{metrics["sentiment"]["recall"]:.4f}, F1-score: '
              f'{metrics["sentiment"]["f1"]:.4f}')
        print(f'Dialogue Act Classification - Precision: '
              f'{metrics["act"]["precision"]:.4f}, Recall: '
              f'{metrics["act"]["recall"]:.4f}, F1-score: '
              f'{metrics["act"]["f1"]:.4f}')
        print(f'Speaker Prediction - Accuracy: '
              f'{metrics["speaker"]["accuracy"]:.4f}')
        print(f'Importance Prediction - Average MSE Loss: '
              f'{metrics["importance_loss"]:.4f}')

        sentiment_f1 = metrics["sentiment"]["f1"]
        act_f1 = metrics["act"]["f1"]
        avg_f1 = (sentiment_f1 + act_f1) / 2

        # Save model if any of the F1 scores improved
        save_model = False
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            save_model = True
            print(f"Average F1 improved to {best_avg_f1:.4f}")
        if sentiment_f1 > best_sentiment_f1:
            best_sentiment_f1 = sentiment_f1
            save_model = True
            print(f"Sentiment F1 improved to {best_sentiment_f1:.4f}")
        if act_f1 > best_act_f1:
            best_act_f1 = act_f1
            save_model = True
            print(f"Act F1 improved to {best_act_f1:.4f}")

        if save_model:
            torch.save(model.state_dict(),
                       '/public/home/acy7tl31mi/demo2-classification/classify-deberta/bestmodel/dailydialogue-best_model_deberta-base-v3-train-four-history-dialogue-train_importance_labels-addspeakerpredict.pth')
            print(f"Best model saved with act F1: {best_act_f1:.4f}, "
                  f"sentiment F1: {best_sentiment_f1:.4f}")

    print(f"Training completed. Best act F1: {best_act_f1:.4f}, "
          f"Best sentiment F1: {best_sentiment_f1:.4f}")
