import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
from sklearn.preprocessing import LabelEncoder
from transformers import DebertaV2TokenizerFast


def set_seed(seed=42):
    """
    Set the random seed to ensure reproducibility of the experiments.

    Parameters:
    - seed: The seed value to be set for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for results to be reproducible
    cudnn.deterministic = True
    cudnn.benchmark = False


def encode_labels(data_list, label_encoder, label_key='sentiment'):
    """
    Encode the labels using a pre-fitted LabelEncoder.

    Parameters:
    - data_list: List of data entries (usually dictionaries containing labels).
    - label_encoder: A pre-fitted LabelEncoder instance used to encode labels.
    - label_key: The key in the data dictionary that contains the label to encode (default is 'sentiment').

    Returns:
    - Encoded labels as a numpy array.
    """
    return label_encoder.transform([item[label_key] for item in data_list])


def load_tokenizer(model_name):
    """
    Load the DeBERTa tokenizer from the Hugging Face library.

    Parameters:
    - model_name: The name or path of the pre-trained model.

    Returns:
    - tokenizer: The tokenizer object for the DeBERTa model.
    """
    return DebertaV2TokenizerFast.from_pretrained(model_name)


def create_label_encoder(data_list, label_key='sentiment'):
    """
    Create and fit a LabelEncoder for the specified label in the data.

    Parameters:
    - data_list: List of data entries (usually dictionaries containing labels).
    - label_key: The key in the data dictionary that contains the label to encode (default is 'sentiment').

    Returns:
    - label_encoder: A fitted LabelEncoder instance.
    """
    label_encoder = LabelEncoder()
    labels = [item[label_key] for item in data_list]
    label_encoder.fit(labels)
    return label_encoder


def collate_fn(batch):
    """
    Custom batch collation function to process variable-length sequences and labels.

    This function is used by the DataLoader to aggregate data into tensors.

    Parameters:
    - batch: List of data samples in the current batch, typically dictionaries.

    Returns:
    - A dictionary containing tensors for input data and labels, ready for the model.
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    sentiments = torch.stack([item['sentiment'] for item in batch])
    acts = torch.stack([item['act'] for item in batch])
    current_utterance_position_index = torch.stack([item['current_utterance_position_index'] for item in batch])

    # Handle variable-length sequences
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
        'importance_mask': importance_mask  # List, length=batch_size
    }


def save_model(model, path):
    """
    Save the model's state_dict to the specified file path.

    Parameters:
    - model: The trained model object.
    - path: The path where the model should be saved.

    """
    torch.save(model.state_dict(), path)
    print(f"Model has been saved to {path}")


def load_model(model, path):
    """
    Load a model's state_dict from the specified file path.

    Parameters:
    - model: The model instance to which the state_dict will be loaded.
    - path: The path to the saved model's state_dict.

    Returns:
    - model: The model with the loaded state_dict.
    """
    model.load_state_dict(torch.load(path))
    print(f"Model has been loaded from {path}")
    return model
