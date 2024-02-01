'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting


class Setting_Train_Test(setting):

    def load_run_save_evaluate(self):

        # load dataset
        loaded_data = self.dataset.load()

        X_train, y_train = loaded_data['X_train'], loaded_data['y_train']
        X_test, y_test = loaded_data['X_test'], loaded_data['y_test']

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None

