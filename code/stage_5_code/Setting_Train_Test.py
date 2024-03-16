'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from torch.utils.data import DataLoader
import numpy as np


class Setting_Train_Test(setting):

    def load_run_save_evaluate(self):

        # load dataset
        loaded_data = self.dataset.load()
        graph = loaded_data['graph']
        train_test_val = loaded_data['train_test_val']

        X_train, y_train = graph['X'][train_test_val['idx_train']], graph['y'][train_test_val['idx_train']]

        # run MethodModule
        self.method.data = loaded_data
        self.method.input_dim = np.array(X_train).shape[1]
        self.method.output_dim = np.array(y_train).max() + 1
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return learned_result
