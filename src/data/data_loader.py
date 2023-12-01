import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DataLoader Configuration
DATA_LOADER_CONFIGS = {
    'batch_size': 32,
    'max_length': 512,
    'val_split': 0.2,
    'shuffle': True,
    'random_state': 42
}


class QADataset(Dataset):
    """Custom Dataset for handling QA data."""

    def __init__(self, tokenizer, data_path, config):
        logger.info(f"Loading data from {data_path}")
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = config.get('max_length', 512)

    def __len__(self):
        return len(self.data)

    def tokenize_text(self, text):
        return self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        ).input_ids.squeeze(0)

    def __getitem__(self, idx):
        question = self.data.iloc[idx]['question']
        answer = self.data.iloc[idx]['answer']

        question_tokens = self.tokenize_text(question)
        answer_tokens = self.tokenize_text(answer)

        return question_tokens, answer_tokens


def get_data_loaders(tokenizer, data_path, config=None):
    """Create data loaders from the dataset."""

    if config is None:
        config = DATA_LOADER_CONFIGS

    logger.info("Creating Data Loaders...")
    dataset = QADataset(tokenizer, data_path, config)
    train_data, val_data = train_test_split(
        dataset,
        test_size=config['val_split'],
        shuffle=config['shuffle'],
        random_state=config['random_state']
    )

    train_loader = DataLoader(
        train_data,
        batch_size=config['batch_size'],
        shuffle=config['shuffle']
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config['batch_size'],
        shuffle=False
    )

    return train_loader, val_loader
