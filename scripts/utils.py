import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

class TextSummarizationDataset(Dataset):
    def __init__(self, texts, summaries, tokenizer, max_length):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = self.tokenizer(
            summary,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze()
        }

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(data_path):
    import pandas as pd
    df = pd.read_csv(data_path)
    return df['text'].tolist(), df['summary'].tolist()

def create_data_loader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)