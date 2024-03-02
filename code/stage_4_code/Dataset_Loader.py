'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
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

    def clean(self, text):
        # split into words
        tokens = word_tokenize(text)
        # convert to lower case
        tokens = [w.lower() for w in tokens]
        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        return words

    def pad_sequences(self, sequences, max_length):
        padded_sequences = []
        for seq in sequences:
            if len(seq) < max_length:
                # pad the sequence with zeros to match the max length
                padded_seq = seq + [0] * (max_length - len(seq))
            else:
                # truncate the sequence if it exceeds the max length
                padded_seq = seq[:max_length]
            padded_sequences.append(padded_seq)
        return padded_sequences

    
    def load(self):
        print('loading data...')
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        train_dir = self.dataset_source_folder_path + self.dataset_source_file_name_train
        test_dir = self.dataset_source_folder_path + self.dataset_source_file_name_test
        vocab = defaultdict(lambda: len(vocab))

        for label in ['pos','neg']:
            label_dir = os.path.join(train_dir, label)
            for filename in os.listdir(label_dir):
                with open(os.path.join(label_dir, filename), 'r', encoding='utf-8') as file:
                    review = file.read()
                    clean_review = self.clean(review)
                    X_train.append([vocab[word] for word in clean_review])
                    y_train.append(1 if label == 'pos' else 0)

        for label in ['pos','neg']:
            label_dir = os.path.join(test_dir, label)
            for filename in os.listdir(label_dir):
                with open(os.path.join(label_dir, filename), 'r', encoding='utf-8') as file:
                    review = file.read()
                    clean_review = self.clean(review)
                    X_test.append([vocab[word] for word in clean_review])
                    y_test.append(1 if label == 'pos' else 0)

        vocab = dict(vocab)

        max_length = 150
        X_train = self.pad_sequences(X_train, max_length)
        X_test= self.pad_sequences(X_test, max_length)

        X_train = torch.LongTensor(X_train)
        y_train = torch.Tensor(y_train)
        X_test = torch.LongTensor(X_test)
        y_test = torch.Tensor(y_test)

        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}, vocab