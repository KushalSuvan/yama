from tokenizers import Tokenizer

import torch
from torch.utils.data import Dataset

import nltk
from nltk.tokenize import sent_tokenize


# scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# def greedy_oracle(article, summary):
#     sentences = sent_tokenize(article)
#     remaining = list(range(len(sentences)))

#     selected, current_summary = [], ""

#     while remaining:
#         best_idx, best_score = -1, 0.0


class ExtractiveSummarizationDataset(Dataset):

    def __init__(self, ds, tokenizer: Tokenizer):
        self.ds = ds
        self.tokenizer = tokenizer
        nltk.download('punkt_tab')
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        pair = self.ds[idx]

        judgement = pair['judgement']
        summary = pair['summary']

        # print(self.ds[idx])

        # print(f'judgement: {judgement}')
        # print(f'summary: {summary}')

        judgement_sentences = sent_tokenize(judgement)
        summary_sentences = sent_tokenize(summary)

        
        summary_set = set(summary_sentences)
        target = [1 if s.strip() in summary_set else 0 for s in judgement_sentences]
        target = torch.tensor(target, dtype=torch.float)


        # print(f'jsents: {judgement_sentences}')
        # print(f'ssents: {summary_sentences}')

        judgement_tokens = self.tokenizer(judgement_sentences, truncation=True, padding="max_length", return_tensors="pt", is_split_into_words=False)
        summary_tokens = self.tokenizer(summary_sentences, truncation=True, padding="max_length", return_tensors="pt", is_split_into_words=False)

        return {
            "jtokens": judgement_tokens, 
            "stokens": summary_tokens,
            "judgement_sentences": judgement_sentences,
            "summary_sentences": summary_sentences,
            "target": target
        }


        