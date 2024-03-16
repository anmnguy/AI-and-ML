from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.stage_4_code.Method_RNN_Generation import Method_RNN
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Setting_Train_Test import Setting_Train_Test
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report

# ---- Recurrent Neural Network script for generation ----
if 1:

    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('RNN', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/'
    data_obj.dataset_source_file_name_train = 'text_generation/train'
    data_obj.dataset_source_file_name_test = 'text_generation/test'

    method_obj = Method_RNN('recurrent neural network', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/RNN_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test('train test no split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    result = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: {}'.format(accuracy_score(result['true_y'], result['pred_y'])))
    print('Classification Report: ')
    print(classification_report(result['true_y'], result['pred_y'], target_names=['pos','neg']))
    print('************ Finish ************')
    # ------------------------------------------------------
