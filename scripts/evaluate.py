import torch
import argparse
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from utils import get_device, load_data, TextSummarizationDataset, create_data_loader
import os

def main(data_path):
    # Parameters
    model_dir = './models'
    max_length = 512
    batch_size = 8

    # Load Model and Tokenizer
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    # Device
    device = get_device()
    model.to(device)

    # Load Dataset
    texts, summaries = load_data(data_path)
    val_dataset = TextSummarizationDataset(texts, summaries, tokenizer, max_length)
    val_loader = create_data_loader(val_dataset, batch_size)

    # Evaluation Function
    def evaluate(model, data_loader, device):
        model.eval()
        total_preds = []
        total_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                total_preds.extend(preds.cpu().numpy())
                total_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(total_labels, total_preds)
        f1 = f1_score(total_labels, total_preds, average='weighted')

        return accuracy, f1

    # Evaluate
    accuracy, f1 = evaluate(model, val_loader, device)
    print(f'Accuracy: {accuracy}')
    print(f'F1 Score: {f1}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing the data')
    args = parser.parse_args()
    main(args.data_path)