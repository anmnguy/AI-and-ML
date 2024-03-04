'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import os
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import word_tokenize
import torch
import string
from code.base_class.dataset import dataset

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name_train = None
    dataset_source_file_name_test = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load_csv(self):
        data_path = os.path.join(self.dataset_source_folder_path, self.dataset_source_file_name)
        df = pd.read_csv(data_path)
        jokes = df['Joke'].tolist()
        return jokes

    def tokenizer(self, text):
        tokens = word_tokenize(text.lower())
        table = str.maketrans('', '', string.punctuation)
        tokens = [w.translate(table) for w in tokens if w.isalpha()]
        return tokens

    def build_vocab_and_get_sequences(self, jokes):
        tokenizer = get_tokenizer('basic_english')

        def yield_tokens(data_iter):
            for joke in data_iter:
                yield tokenizer(joke)

        vocab = build_vocab_from_iterator(yield_tokens(jokes), specials=["<unk>", "<pad>", "<start>", "<end>"])
        vocab.set_default_index(vocab["<unk>"])

        # Tokenizing jokes and converting to index sequences
        sequences = [[vocab["<start>"]] + [vocab[token] for token in tokenizer(joke)] + [vocab["<end>"]] for joke in
                     jokes]
        return sequences, vocab

    def load(self):
        print('loading data...')
        jokes = self.load_csv()
        sequences, vocab = self.build_vocab_and_get_sequences(jokes)

        # Padding sequences for uniform length
        sequence_tensor = pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True,
                                       padding_value=vocab["<pad>"])

        return sequence_tensor, vocab