from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from code.stage_3_code.Face_Net import Face_Net
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Setting_Train_Test import Setting_Train_Test
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import warnings

# ---- Digit Recognition Convolutional Neural Network Script ----
if 1:
    warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow.*")
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('MNIST', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name_train = 'MNIST'

    method_obj = Method_CNN_MNIST('convolutional neural network', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test('train test no split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    evaluations = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CNN Accuracy: ', evaluations['accuracy'])
    print('CNN Precision: ', evaluations['precision'])
    print('CNN Recall: ', evaluations['recall'])
    print('CNN F1: ', evaluations['f1'])
    print('************ Finish ************')
    # ------------------------------------------------------

# ---- Facial Recognition Convolutional Neural Network Script ----
if 0:
    warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow.*")
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('ORL', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name_train = 'ORL'

    method_obj = Face_Net('convolutional neural network', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/ORL_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test('train test no split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    accuracy = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CNN Accuracy: ', accuracy)
    print('************ Finish ************')
    # ------------------------------------------------------

# ---- Colored Objects Identification Convolutional Neural Network Script ----
if 0:
    warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow.*")
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('CIFAR', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name_train = 'CIFAR'

    method_obj = Method_CNN('convolutional neural network', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test('train test no split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    accuracy = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CNN Accuracy: ', accuracy)
    print('************ Finish ************')
    # ------------------------------------------------------
