from torch.utils.data import Dataset

import nltk
from nltk.tokenize import sent_tokenize

from rouge_score import rouge_scorer

from transformers import AutoTokenizer

# scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# def greedy_oracle(article, summary):
#     sentences = sent_tokenize(article)
#     remaining = list(range(len(sentences)))

#     selected, current_summary = [], ""

#     while remaining:
#         best_idx, best_score = -1, 0.0


class ExtractiveSummarizationDataset(Dataset):

    def __init__(self, ds):
        self.ds = ds
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
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

        # print(f'jsents: {judgement_sentences}')
        # print(f'ssents: {summary_sentences}')

        judgement_tokens = self.tokenizer(judgement_sentences, truncation=True, padding=True, return_tensors="pt", is_split_into_words=False)
        summary_tokens = self.tokenizer(summary_sentences, truncation=True, padding=True, return_tensors="pt", is_split_into_words=False)



        return judgement_tokens, summary_tokens


        