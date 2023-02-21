import os
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

def read_file_linebyline(filename, encoding=None):
    with open(filename) as rf:
        data = [line.strip() for line in rf]
    return data


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_instance(module, config, *args, **kwargs):
    return getattr(module, config['type'])(*args, **kwargs, **config['args'])


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def SNString2Float(array):
    result = []
    for i in array:
        if isfloat(i):
            result.append(i)
        else:
            n = float(i[:i.find("e")])
            p = float(i[i.find("e")+2:])
            result.append(n * pow(10,p))
    return np.array(result)


def splitData(features, labels):
    X_train,X_test, y_train, y_test = train_test_split(features,labels,test_size=0.3, random_state=0, shuffle=True)
    return X_train, X_test, y_train, y_test		


def metric(y_true, y_pred):
	result = []
	for i in range(len(y_true[0])):
		result.append(classification_report(y_true[:,i], y_pred[:,i]))
	return result


def split_by_interval(matrix,interval):
    result = []
    length = matrix.size()[0]
    for i in range(0,length,interval):
        if (i+interval) <= length:
            result.append(matrix[i:i+interval][:])
        else:
            result.append(matrix[i:length][:])
    return result

