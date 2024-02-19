'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pickle
from code.base_class.dataset import dataset

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name_train = None
    dataset_source_file_name_test = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        f = open(self.dataset_source_folder_path + self.dataset_source_file_name_train, 'rb')
        data = pickle.load(f)
        f.close()

        for instance in data['train']:
            image_matrix = instance['image']
            image_label = instance['label']
            X_train.append(image_matrix)
            y_train.append(image_label)

        for instance in data['test']:
            image_matrix = instance['image']
            image_label = instance['label']
            X_test.append(image_matrix)
            y_test.append(image_label)

        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}