import torch
from torch.utils.data import Dataset
from transformers import BertModel
import torch.nn as nn

# Custom Dataset class for ParaCrawl data
class ParaCrawlDataset(Dataset):
    def __init__(self, source_texts, target_texts, tokenizer, max_len=128):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]

        # Tokenize the source and target sentences with truncation enabled
        inputs = self.tokenizer.encode_plus(
            source_text, 
            target_text,
            max_length=self.max_len,
            truncation=True,  # Truncate sequences if they exceed the max length
            padding='max_length',  # Pad shorter sequences
            return_tensors="pt"  # Return PyTorch tensors
        )

        # Placeholder for a quality score (to be replaced with actual scores)
        quality_score = torch.tensor(0.5, dtype=torch.float)

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'score': quality_score
        }

class QEModel(nn.Module):
    def __init__(self, bert_model_name):
        super(QEModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)  # Single output for quality score

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token output (pooled output)
        pooled_output = outputs.pooler_output
        score = self.regressor(pooled_output)
        return score