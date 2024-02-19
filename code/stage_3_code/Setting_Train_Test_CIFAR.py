'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
import torch
import numpy as np

class Setting_Train_Test_CIFAR(setting):

    def load_run_save_evaluate(self):

        # load dataset
        loaded_data = self.dataset.load()

        X_train, y_train = loaded_data['X_train'], loaded_data['y_train']
        X_test, y_test = loaded_data['X_test'], loaded_data['y_test']

        BATCH_SIZE = 32

        # convert data to tensor
        torch_X_train = torch.from_numpy(np.asarray(X_train)).type(torch.LongTensor)
        torch_y_train = torch.from_numpy(np.asarray(y_train)).type(torch.LongTensor)  # data type is long

        torch_X_test = torch.from_numpy(np.asarray(X_test)).type(torch.LongTensor)
        torch_y_test = torch.from_numpy(np.asarray(y_test)).type(torch.LongTensor)  # data type is long

        torch_X_train = torch_X_train.view(-1, 3, 32, 32).float()
        torch_X_test = torch_X_test.view(-1, 3, 32, 32).float()

        # pytorch train and test sets
        train = torch.utils.data.TensorDataset(torch_X_train, torch_y_train)
        test = torch.utils.data.TensorDataset(torch_X_test, torch_y_test)

        # data loader
        train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        self.method.train_loader = train_loader
        self.method.test_loader = test_loader
        learned_result, evaluations = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        return evaluations
