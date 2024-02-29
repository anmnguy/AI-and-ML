'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from torch.utils.data import DataLoader, TensorDataset


class Setting_Train_Test(setting):

    def load_run_save_evaluate(self):

        # load dataset
        loaded_data, vocab = self.dataset.load()
        X_train, y_train = loaded_data['X_train'], loaded_data['y_train']
        X_test, y_test = loaded_data['X_test'], loaded_data['y_test']

        train = TensorDataset(X_train, y_train)
        test = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train, batch_size=32, shuffle=True)
        test_loader = DataLoader(test, batch_size=32, shuffle=True)

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        self.method.train_loader = train_loader
        self.method.test_loader = test_loader
        self.method.vocab = vocab
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        return learned_result