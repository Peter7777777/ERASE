#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import platform
import datetime
import wrapt
import configparser
from getpass import getuser
from socket import gethostname
from sklearn.datasets import load_svmlight_file
from joblib import Memory


def isWindowsSystem():
    return 'Windows' in platform.system()


def isLinuxSystem():
    return 'Linux' in platform.system()


def tfGpuSet(gpuNo="0", printInfo="0"):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuNo
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = printInfo


def checkOS():
    if isWindowsSystem():
        return 'Windows'
    elif isLinuxSystem():
        return 'Linux'




def DatasetIsTCGA(dataName):
    if dataName in TCGASets:
        return True
    else:
        return False


def GetVarName(var):
    return list(dict(var=var).keys())[0]


def GetFunName():
    import traceback
    return traceback.extract_stack()[-2][2]


def __initMemCache():
    if isWindowsSystem():
        return Memory("D:/WorkSpaces/PyCharmWorkSpace/MemoryCache")
    elif isLinuxSystem():
        return Memory("MemoryCache")


mem = __initMemCache()


def DefaultDataPath(dataName):
    LabDatasPath = f'datasets'
    if dataName in ('citeseer', 'cora', 'pubmed'):
        defaultDataPath = LabDatasPath + f'/{dataName}'
    else:
        defaultDataPath = LabDatasPath + f'/{dataName}.csv'
    return defaultDataPath


def Start_time():
    return datetime.datetime.now()


start_time = Start_time()


def Now_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def lookData(dataPath, type="rb"):
    with open(dataPath, type) as f:
        for fLine in f:
            print(fLine)


def LoadDataset(dataName, onehot=False, TrainTest=False):
    if dataName in ('citeseer', 'cora', 'pubmed'):
        if TrainTest:
            X_train, y_train, X_test, y_test = loadGraphData(dataName, onehot=onehot, TrainTest=TrainTest)
            return X_train, y_train, X_test, y_test
        X, y = loadGraphData(dataName, onehot=onehot, TrainTest=TrainTest)
    else:
        X, y = loadCSVData(dataName)
    return X, y


@mem.cache
def loadNPZData(dataName, dataPath=None, onehot=False):
    dataPath = dataPath if dataPath else DefaultDataPath(dataName)
    data = np.load(dataPath)
    X, y = data['X'], data['y']
    if onehot:
        from sklearn.preprocessing import LabelBinarizer
        y = LabelBinarizer().fit_transform(y)
        return X, y
    return X, y


@mem.cache
def loadGraphData(dataName, dataPath=None, onehot=False, TrainTest=False):
    from gutils import load_data
    adj, features, labels, train_mask, val_mask, test_mask = load_data(dataName, dataPath)
    if not onehot:
        labels = np.argmax(labels, axis=1)
    if TrainTest:
        X_train, y_train = features[train_mask], labels[train_mask]
        X_test, y_test = features[test_mask], labels[test_mask]
        return X_train, y_train, X_test, y_test
    else:
        return features, labels


@mem.cache
def loadCSVData(dataName, dataPath=None, dataFolder='CSV_Datasets', nfiles='', no='', isXy=True):
    if no == '':
        dataPath = dataPath if dataPath else DefaultDataPath(dataName)
        data = pd.read_csv(dataPath, header=None, engine='python')
        if isXy:
            X, y = data.iloc[:, 0:-1], data.iloc[:, -1:]
        else:
            return data
    else:
        assert isXy is True
        if isWindowsSystem():
            Xpath = f'D:/LabDatas/{dataFolder}/{dataName}/{nfiles}/{dataName}_NO_{no}.csv'
            ypath = f'D:/LabDatas/{dataFolder}/{dataName}/{nfiles}/{dataName}_labels.csv'
        elif isLinuxSystem():
            Xpath = f'/home/LabDatas/{dataFolder}/{dataName}/{nfiles}/{dataName}_NO_{no}.csv'
            ypath = f'/home/LabDatas/{dataFolder}/{dataName}/{nfiles}/{dataName}_labels.csv'
        X = pd.read_csv(Xpath, header=None, engine='python')
        y = pd.read_csv(ypath, header=None, engine='python')
    X = X.values
    y = y.squeeze().values
    return X, y


def instances_features_num(X):
    instances_num, features_num = X.shape
    return instances_num, features_num


def Split_X_Features(X, indices, startCol=0, colLength=100):
    if isinstance(X, pd.DataFrame):
        X = X[indices[startCol:startCol + colLength]]
        X.columns = range(colLength)
    else:
        X = X[:, indices[startCol: startCol + colLength]]
    return X