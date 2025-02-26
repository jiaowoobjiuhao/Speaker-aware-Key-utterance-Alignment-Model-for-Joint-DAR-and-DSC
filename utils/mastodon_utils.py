import torch
import random
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from transformers import DebertaV2TokenizerFast
import torch.backends.cudnn as cudnn  # For controlling deterministic behavior in CUDA


def set_seed(seed=42):
    """
    Set the random seed for reproducibility across all libraries.

    Args:
    - seed (int): The seed value to be set for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic operations in CUDA
    cudnn.deterministic = True
    cudnn.benchmark = False


def load_tokenizer(model_name):
    """
    Load the DeBERTa tokenizer using Hugging Face's `DebertaV2TokenizerFast`.

    Args:
    - model_name (str): Path or identifier for the pre-trained DeBERTa model.

    Returns:
    - tokenizer: Tokenizer object used to tokenize input text.
    """
    return DebertaV2TokenizerFast.from_pretrained(model_name)


def encode_labels(data_list, label_encoder, label_key='sentiment'):
    """
    Encode labels using a pre-fitted LabelEncoder.

    Args:
    - data_list (list of dict): List of data entries containing labels.
    - label_encoder (LabelEncoder): A pre-fitted encoder.
    - label_key (str): The key used to access the label in the data (e.g., 'sentiment').

    Returns:
    - Encoded labels as a numpy array.
    """
    return label_encoder.transform([item[label_key] for item in data_list])


def create_label_encoder(data_list, label_key='sentiment'):
    """
    Create and fit a LabelEncoder to the labels in the data.

    Args:
    - data_list (list of dict): List of data entries containing labels.
    - label_key (str): The key used to access the label in the data (e.g., 'sentiment').

    Returns:
    - label_encoder (LabelEncoder): The fitted LabelEncoder object.
    """
    label_encoder = LabelEncoder()
    labels = [item[label_key] for item in data_list]
    label_encoder.fit(labels)
    return label_encoder


def read_erc(input_path, tokenizer):
    """
    Read and preprocess the ERC (Emotion Recognition) dataset by adding conversation history
    and tokenizing the texts.

    Args:
    - input_path (str): The path to the dataset file (in JSON format).
    - tokenizer (Tokenizer): A tokenizer object used to tokenize the text.

    Returns:
    - data_list (list of dict): A list of dictionaries containing preprocessed data.
    """
    data_list = []
    with open(input_path, encoding="utf-8") as file_read:
        data = json.load(file_read)

        for dialogues in data:
            if len(dialogues) == 0:
                continue

            history = []
            speakers_so_far = []
            for i, utterance_data in enumerate(dialogues):
                current_speaker = utterance_data['speaker']
                utterance = utterance_data['utterance']
                sentiment = utterance_data['sentiment']
                act = utterance_data['act']

                # Update conversation history with the current utterance and speaker
                if len(history) >= 6:  # Limit to a maximum of 6 previous utterances
                    history.pop(0)
                    speakers_so_far.pop(0)
                history.append(f"{current_speaker}: {utterance}")
                speakers_so_far.append(current_speaker)

                # Construct the input text with up to 6 previous turns
                input_text = f" {tokenizer.sep_token} ".join(history[-6:])

                # Adjust the current utterance index based on the size of the history
                current_utterance_index = min(i, 5)  # Max index is 5 due to history length limit

                data_list.append({
                    "input_text": input_text,
                    "sentiment": sentiment,
                    "act": act,
                    "speakers": speakers_so_far.copy(),
                    "current_utterance_index": current_utterance_index  # Index of the current utterance
                })

    return data_list


class ConversationDataset(torch.utils.data.Dataset):
    """
    Custom dataset to load conversation data and process the input for the model.

    Args:
    - texts (list): List of tokenized conversation inputs (texts).
    - sentiments (list): List of sentiment labels (encoded).
    - acts (list): List of dialogue act labels (encoded).
    - speakers (list): List of speaker information (encoded).
    - current_utterance_indices (list): List of indices for the current utterance in the history.
    - importance_labels_list (list): List of importance labels for each utterance.
    - tokenizer (Tokenizer): Tokenizer used to encode text.
    - max_length (int): The maximum length for padding/truncation of text.
    """

    def __init__(self, texts, sentiments, acts, speakers,
                 current_utterance_indices, importance_labels_list, tokenizer, max_length):
        self.texts = texts
        self.sentiments = sentiments
        self.acts = acts
        self.speakers = speakers
        self.current_utterance_indices = current_utterance_indices
        self.importance_labels_list = importance_labels_list
        self.tokenizer = tokenizer
        self.max_length = max_length

        # List to store positions of utterances
        self.utterance_positions_list = []

        for text in self.texts:
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
            sep_token_id = self.tokenizer.sep_token_id
            utterance_positions = [i for i, token_id in enumerate(input_ids)
                                   if token_id == sep_token_id]

            # Add [CLS] token position (index 0)
            utterance_positions = [0] + [pos + 1 for pos in utterance_positions]
            self.utterance_positions_list.append(utterance_positions)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        sentiment = self.sentiments[idx]
        act = self.acts[idx]
        speakers = self.speakers[idx]
        current_utterance_index = self.current_utterance_indices[idx]
        importance_labels = self.importance_labels_list[idx]
        importance_scores = importance_labels['importance_scores']
        importance_mask = importance_labels['importance_mask']

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

        utterance_positions = torch.tensor(self.utterance_positions_list[idx], dtype=torch.long)
        num_utterances = min(len(utterance_positions), len(speakers))
        utterance_positions = utterance_positions[:num_utterances]
        speakers = speakers[:num_utterances]

        speakers = torch.tensor(speakers, dtype=torch.long)

        if num_utterances > 0:
            current_utterance_position_index = min(current_utterance_index, num_utterances - 1)
        else:
            current_utterance_position_index = 0

        importance_scores = torch.tensor(importance_scores, dtype=torch.float)
        importance_mask = torch.tensor(importance_mask, dtype=torch.float)

        importance_length = importance_scores.size(0)
        if importance_length > num_utterances:
            importance_scores = importance_scores[:num_utterances]
            importance_mask = importance_mask[:num_utterances]
        elif importance_length < num_utterances:
            padding_size = num_utterances - importance_length
            scores_padding = torch.zeros(padding_size, dtype=torch.float)
            mask_padding = torch.zeros(padding_size, dtype=torch.float)
            importance_scores = torch.cat([importance_scores, scores_padding], dim=0)
            importance_mask = torch.cat([importance_mask, mask_padding], dim=0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'sentiment': torch.tensor(sentiment, dtype=torch.long),
            'act': torch.tensor(act, dtype=torch.long),
            'speakers': speakers,
            'utterance_positions': utterance_positions,
            'current_utterance_position_index': torch.tensor(current_utterance_position_index, dtype=torch.long),
            'importance_scores': importance_scores,
            'importance_mask': importance_mask
        }


def collate_fn(batch):
    """
    Custom collate function to handle padding and batching of sequences.

    Args:
    - batch (list): List of samples to be collated.

    Returns:
    - A dictionary containing batched input data and labels.
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    sentiments = torch.stack([item['sentiment'] for item in batch])
    acts = torch.stack([item['act'] for item in batch])
    current_utterance_position_index = torch.stack([item['current_utterance_position_index'] for item in batch])

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
        'importance_scores': importance_scores,
        'importance_mask': importance_mask
    }
