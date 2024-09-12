import torch


# SET THE NETWORKS 

from dlModels.CIFAR10 import resnet_cifar10,mobilenetv2_cifar10, densenet_cifar10, vgg_cifar10, googlenet_cifar10
from dlModels.GTSRB import resnet_GTSRB, vgg_GTSRB, densenet_GTSRB
from dlModels.CIFAR100 import densenet_cifar100, resnet_cifar100, googlenet_cifar100

'''
FAULT_MODEL available: 'stuck-at_params', 'byzantine_neuron'


NETWORK available: 'ResNet18', 'ResNet20', 'ResNet32', 'ResNet44', 
                          'DenseNet121', 'DenseNet161','MobileNetV2', 
                          'GoogLeNet', 'Vgg11_bn', 'Vgg13_bn'


DATASET available: 'CIFAR10', 'CIFAR100', 'GTSRB'



'''
# enable the fault list generation
FAULT_LIST_GENERATION = True

# enable the fault injection
FAULTS_INJECTION = False

# 0 : masked, 1: non.critic, 2: critic
FI_ANALYSIS = False
FI_ANALYSIS_SUMMARY = True

# network and dataset to use
DATASET_NAME = 'CIFAR10'
NETWORK_NAME = 'ResNet20'

# if you want to check  only the accuracy of the clean model
ONLY_CLEAN_INFERENCE = True
# ------------------------------------ FAULT LIST SETTINGS ------------------------------------



# FAULT LIST
SEED = 38

# FAULT LIST PARAMETERS
error_margin = 0.01
probability = 0.7
confidence_constant = 2.58
bit = 32

modules_to_fault = (torch.nn.Conv2d, torch.nn.Linear)

FAULT_LIST_PATH = f'output/fault_list/{DATASET_NAME}/{NETWORK_NAME}/'
FAULT_LIST_NAME = f'{NETWORK_NAME}_{SEED}_fault_list.csv'

# ------------------------------------ FAULT INJECTION SETTINGS ------------------------------------

#fault to inject in the model from the faul list
FAULTS_TO_INJECT = 20

# use the GPU is available
USE_CUDA_0 = True
USE_CUDA_1 = False

# forbif the logging of the results ----- MI SEMBRA NON FUNZIONI
NO_LOG_RESULTS = False

# test set batch size
BATCH_SIZE = 64

# fault model to use (check the top of the file for the available models)
FAULT_MODEL = 'stuck-at_params'

# dataset to use (check the top of the file for the available datasets)
DATASET = DATASET_NAME

# network to use (check the top of the file for the available networks)
NETWORK = NETWORK_NAME

# threshold under which an error is undetected
THRESHOLD = 0.0

# ------------------------------------ FAULT ANALYSIS SETTINGS ------------------------------------# 

# Batch from which to start and end the analysis to segment the analysis due to memory problems (in the case you have troubles with the memory)
RAM_LIMIT = False
BATCH_START = 0
BATCH_END = 13

        
# ------------------------------------ SAVE SETTINGS ------------------------------------

# SAVE CLEAN OFM
SAVE_CLEAN_OFM = True

# SAVE FAULTY OFM
SAVE_FAULTY_OFM = True

# SAVE FAULTY OUTPUT
SAVE_FAULTY_OUTPUT = True

# OFM TO SAVE
if SAVE_FAULTY_OFM:   
    INPUT_FMAPS_TO_SAVE = 'layer1.0.conv1'
else:
    INPUT_FMAPS_TO_SAVE = None

# ------------------------------------ PATHS ------------------------------------

# CLEAN FOLDER PATHS
CLEAN_FM_FOLDER = f'output/clean_feature_maps/{DATASET}/{NETWORK}/batch_{BATCH_SIZE}'
CLEAN_OUTPUT_FOLDER = f'output/clean_output/{DATASET}/{NETWORK}/batch_{BATCH_SIZE}'

# FAULTY FOLDER PATHS
FAULTY_FM_FOLDER = f'output/faulty_feature_maps/{DATASET}/{NETWORK}/batch_{BATCH_SIZE}/{FAULT_MODEL}'
FAULTY_OUTPUT_FOLDER = f'output/faulty_output/{DATASET}/{NETWORK}/batch_{BATCH_SIZE}'

# MODULES TO SAVE OFM AND IFM
MODULE_CLASSES = (torch.nn.Conv2d)
MODULE_CLASSES_FAULT_LIST = (torch.nn.Conv2d)

# DATASET PATHS
DATASET_PATH = f'Datasets/'

# MODELS PATHS
MODELS_PATH = f'dlModels/'

# PRETRAINED MODEL PATHS
MODEL_TH_PATH = f'dlModels/{DATASET}/pretrained/{NETWORK}_{DATASET}.th'
MODEL_PT_PATH = f'dlModels/{DATASET}/pretrained/{NETWORK}_{DATASET}.pt'
MODEL_PTH_PATH = f'dlModels/{DATASET}/pretrained/{NETWORK}_{DATASET}.pth'

# FAULT ANALYSIS PATHS
FI_ANALYSIS_PATH = f'results/{DATASET}/{NETWORK}/batch_{BATCH_SIZE}/'
FI_SUM_ANALYSIS_PATH = f'./results_summary/{DATASET}/{NETWORK}/batch_{BATCH_SIZE}/{NETWORK}_summary.csv'













