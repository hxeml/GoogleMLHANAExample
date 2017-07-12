import os
script_dir = os.path.dirname(__file__)
rel_path = "params.config"
abs_file_path = os.path.join(script_dir, rel_path)


HOSTNAME='HOSTNAME'
PORT='PORT'
USER='USER'
PASSWORD='PASSWORD'
TENSOR_SCHEMA='TENSOR_SCHEMA'
TENSOR_TRAINING_DATA_TABLE='TENSOR_TRAINING_DATA_TABLE'
TENSOR_TEST_DATA_TABLE='TENSOR_TEST_DATA_TABLE'
TENSOR_RESULT_TABLE='TENSOR_RESULT_TABLE'


ACCURACY='accuracy'
ACCURACY_BASELINE_LABEL_MEAN='accuracy/baseline_label_mean'
ACCURACY_THRESHOLD_MEAN='accuracy/threshold_0.500000_mean'
AUC='auc'
GLOBAL_STEP='global_step'
LABEL_ACTUAL_MEAN='labels/actual_label_mean'
LABEL_PREDICTION_MEAN='labels/prediction_mean'
LOSS='loss'
PRECISION='precision/positive_threshold_0.500000_mean'
RECALL='recall/positive_threshold_0.500000_mean'

METRICS=[ACCURACY, ACCURACY_BASELINE_LABEL_MEAN, ACCURACY_THRESHOLD_MEAN, AUC, GLOBAL_STEP, LABEL_ACTUAL_MEAN, LABEL_PREDICTION_MEAN, LOSS, PRECISION, RECALL]



def getParamsFromFile():
    with open(abs_file_path) as f:
       lines = list(f)
    params = {}
    for l in lines:
        l = l.rstrip("\r\n")
        entry = l.split('=')
        params[str(entry[0])] = str(entry[1])
    return params
