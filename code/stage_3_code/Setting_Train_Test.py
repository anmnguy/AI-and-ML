'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
import torch
import numpy as np

class Setting_Train_Test(setting):

    def load_run_save_evaluate(self):

        # load dataset
        loaded_data = self.dataset.load()

        X_train, y_train = loaded_data['X_train'], loaded_data['y_train']
        X_test, y_test = loaded_data['X_test'], loaded_data['y_test']

        # normalize dataset
        X_train_np = np.array(X_train)
        X_test_np = np.array(X_test)
        X_train_normalized = X_train_np / 255.0
        X_test_normalized = X_test_np / 255.0

        X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
        X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.int64)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        self.method.train_loader = train_loader
        self.method.test_loader = test_loader
        learned_result, accuracy = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        return accuracy
