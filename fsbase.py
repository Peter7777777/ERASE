#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = "Alex"

import os
import sys
import argparse
import uuid
import multiprocessing
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import glob_variables as gl
from util import checkOS


class LabConfig:
    def __init__(self, random_state, dataName, test_size, classifier):
        self.os = checkOS()
        self.lab_id = uuid.uuid1()
        self.random_state = random_state
        self.dataName = dataName
        self.test_size = test_size
        self.classifier = classifier

    def testNameToType(self):
        if self.test_size == 0.3:
            return "70%-30%"


class DatasetUtil(LabConfig):
    def __init__(self, random_state, dataName, test_size, classifier):
        super(DatasetUtil, self).__init__(random_state, dataName, test_size, classifier)

    @staticmethod
    def GetLoadDataset(dataName, onehot=False, TrainTest=False):
        from util import LoadDataset
        return LoadDataset(dataName, onehot, TrainTest)

    @staticmethod
    def TransDataset(X, y, train_index, test_index):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        return X_train, X_test, y_train, y_test

    @staticmethod
    def InstancesFeaturesNum(X):
        instances_num, features_num = X.shape
        return instances_num, features_num

    @staticmethod
    def CalAcc(classifier, X_train, X_test, y_train, y_test, **kwargs):
        if classifier[-2:] == "NN":
            for k in range(1, 100):
                if classifier == f"sklearn_{k}NN" or classifier == f"{k}NN":
                    kwargs.update({"n_neighbors": k})
                    clf = KNeighborsClassifier(**kwargs)
                    break
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        acc = accuracy_score(predict, y_test)
        return acc

    @staticmethod
    def CalFsAcc(classifier, fset, X_train, X_test, y_train, y_test):
        if not fset:
            return 0.0
        acc = gl.cache.get(frozenset(fset))
        if acc is None:
            if isinstance(X_test, pd.DataFrame):
                acc = DatasetUtil.CalAcc(classifier, X_train[list(fset)], X_test[list(fset)], y_train, y_test)
            else:
                acc = DatasetUtil.CalAcc(classifier, X_train[:, list(fset)], X_test[:, list(fset)], y_train, y_test)
            gl.cache.add(frozenset(fset), acc)
        return acc

    @staticmethod
    def CalFsDR(fset, feature_length):
        dr = 1.0 - (1.0 * len(fset) / feature_length)
        return dr


class FeatureSelection(DatasetUtil):
    def __init__(self, methodName, random_state, dataName, test_size, classifier):
        super(FeatureSelection, self).__init__(random_state, dataName, test_size, classifier)
        self.methodName = methodName
        self.solve = self.__MethodNameToFun()

    def __MethodNameToFun(self):
        MethodDict = {"ENUMFS": self.GetENUMFS}
        return MethodDict.get(self.methodName)

    def OutResult(self, result):
        print("Parallel execution {} success on {} dataset using {} to {} test and random seed is {}".format(
            self.methodName, self.dataName, self.classifier, self.test_size, self.random_state))
        print("Result is {}\n".format(result))

    def ParallelError(self):
        print("Parallel execution {} error on {} dataset using {} to {} test and random seed is {}\n".format(
            self.methodName, self.dataName, self.classifier, self.test_size, self.random_state))

    def Run(self, *args):
        args, result = list(args) if args else list(), list()
        pool = gl.get_value("multiprocessing")
        X, y = self.GetLoadDataset(self.dataName)
        if self.test_size > 1:
            skf = StratifiedKFold(n_splits=self.test_size, random_state=self.random_state)
            for train_index, test_index in skf.split(X, y):
                X_train, X_test, y_train, y_test = self.TransDataset(X, y, train_index, test_index)
                task_args = args + [X_train, X_test, y_train, y_test]
                result.append(pool.apply_async(func=self.solve, args=task_args, callback=self.OutResult,
                                               error_callback=self.ParallelError))
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                                random_state=self.random_state)
            task_args = args + [X_train, X_test, y_train, y_test]
            result.append(pool.apply_async(func=self.solve, args=task_args, callback=self.OutResult,
                                           error_callback=self.ParallelError))

def GetConsoleArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu_use", type=int, default=10 if os.cpu_count() >= 12 else os.cpu_count() - 1,
                        help="a number of cpu_use")
    parser.add_argument("--test_size", default=(0.3,))
    parser.add_argument("--classifier", nargs='*', default=("1NN",),
                        help="a list of train_name=(1NN, 3NN, 5NN, SVM, DT)")
    parser.add_argument("--write_to_file", type=bool, default=False, help="a bool of write_to_file")
    return parser.parse_args()