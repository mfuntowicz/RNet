import re
import string
from collections import namedtuple
from itertools import islice
from typing import Dict, List, Union, Tuple
from ujson import load

import torch
from torch.utils.data import Dataset
from torchtext.data import Dataset as TextDataset, Example, Field, NestedField, RawField, LabelField


__author__ = 'Morgan Funtowicz'
__email__ = 'morgan.funtowicz@naverlabs.com'

SquadEntry = namedtuple('SquadEntry', ['id', 'content', 'question', 'context', 'answer', 'answer_start', 'answer_end'])


# Functions taken from TF implementation for fair comparison
def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def detokenize(seq, vocab):
    return ' '.join([vocab[i] for i in seq])


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def read_squad(squad: Dict, tokenizer, limit=-1) -> List[SquadEntry]:
    if limit > 0:
        return list(islice(iterate_squad(squad['data'], tokenizer), limit))
    else:
        return list(iterate_squad(squad['data'], tokenizer))


def iterate_squad(squad: Dict, tokenizer) -> SquadEntry:
    # Iterate over each article
    for article in squad:
        # Iterate over each paragraph
        for paragraph in article['paragraphs']:
            text = paragraph['context'].strip().replace("''", '" ').replace("``", '" ')
            text_tokens = tokenizer(text)
            spans = convert_idx(text, text_tokens)

            # Iterate over each (question, answer) for each paragraph
            for qa in paragraph['qas']:
                qid = qa['id']
                question = qa['question'].strip().replace("''", '" ').replace("``", '" ')

                # Iterate over all the answers
                answers_text, answers_span = [], []

                # TODO: Support for SQuAD 2
                for answer in qa['answers']:
                    answer_text = answer['text']
                    answer_start = answer['answer_start']
                    answer_end = answer_start + len(answer_text)

                    answers_text += [answer_text]
                    answer_span = []

                    for idx, span in enumerate(spans):
                        if not (answer_end <= span[0] or answer_start >= span[1]):
                            answer_span.append(idx)

                    # Extract real start / end position
                    answers_span += [[answer_span[0], answer_span[-1]]]

                answers_span = answers_span[0]
                yield SquadEntry(qid, text, question, text, answers_text, answers_span[0], answers_span[-1])


class SquadDataset(Dataset):
    def __init__(self, path: str, tokenizer: callable, limit: int = -1):
        super().__init__()
        with open(path, 'r') as squad_f:
            self._entries = read_squad(load(squad_f), tokenizer, limit)

    def __getitem__(self, index) -> SquadEntry:
        return self._entries[index]

    def __len__(self) -> int:
        return len(self._entries)

    def to_torchtext(self, tokenizer, fields=None, return_fields=False) -> Union[TextDataset, Tuple[TextDataset, List]]:

        # Embedding to build
        if fields is None:
            text_f = Field(tokenize=tokenizer, dtype=torch.int64, batch_first=True, include_lengths=True)

            char_f = NestedField(
                Field(tokenize=list, batch_first=True, dtype=torch.int64),
                tokenize=tokenizer, include_lengths=True
            )

            fields = [
                ('id', RawField()),
                (('context_w', 'context_c'), (text_f, char_f)),
                (('question_w', 'question_c'), (text_f, char_f)),
                ('context', RawField()),
                ('answer', RawField()),
                ('answer_start', LabelField(use_vocab=False)),
                ('answer_end', LabelField(use_vocab=False))
            ]

        # Mapping dataset
        dataset = TextDataset(list(map(lambda x: Example.fromlist(x, fields), self._entries)), fields)

        if return_fields:
            return dataset, fields
        else:
            return dataset


class SquadDebugDataset(SquadDataset):
    def __init__(self, path: str, tokenizer: callable):
        super().__init__(path, tokenizer, 128)

