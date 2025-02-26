import torch
from transformers import DebertaV2TokenizerFast, DebertaV2Model, AdamW
from transformers.models.bert.modeling_bert import BertLayer

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
