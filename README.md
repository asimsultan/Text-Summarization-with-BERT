# Text Summarization with BERT

Welcome to the Text Summarization with BERT project! This project focuses on summarizing long texts into concise summaries using BERT.

## Introduction

Text summarization is a common task in natural language processing (NLP) where the goal is to produce a shorter version of a text that preserves its main points. In this project, we leverage the power of BERT to perform text summarization.

## Dataset

For this project, we will use the [CNN/Daily Mail dataset](https://huggingface.co/datasets/cnn_dailymail) available via Hugging Face's datasets library. The dataset consists of news articles (long texts) and their corresponding summaries.

## Project Overview

### Prerequisites

- Python 3.6 or higher
- PyTorch
- Hugging Face Transformers
- Datasets

### Installation

To set up the project, follow these steps:

```bash
# Clone this repository and navigate to the project directory:
git clone https://github.com/your-username/bert_summarization.git
cd bert_summarization
```

### Install the required packages:
```bash
pip install -r requirements.txt
```
 Ensure your data includes texts and their corresponding summaries. Place these files in the data/ directory.
 The data should be in a CSV file with at least two columns: text and summary.

To fine-tune the BERT model for text summarization, run the following command:
```bash
python scripts/train.py --data_path data/sample_texts.csv
```
 To evaluate the performance of the fine-tuned model, run:
```bash
python scripts/evaluate.py --data_path data/sample_texts.csv
```