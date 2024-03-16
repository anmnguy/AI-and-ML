from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from code.stage_5_code.Method_GCN import Method_GCN
from code.stage_5_code.Result_Saver import Result_Saver
from code.stage_5_code.Setting_Train_Test import Setting_Train_Test
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report

# ---- Graph Neural Network script ----
if 1:

    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('GNN', '')
    data_obj.dataset_source_folder_path = '../../data/stage_5_data/cora/'
    data_obj.dataset_name = 'cora'

    method_obj = Method_GCN('graph neural network', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_5_result/GNN_'
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
    print(classification_report(result['true_y'], result['pred_y'], digits=4))
    print('************ Finish ************')
    # ------------------------------------------------------
