#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""a module of Ls_Forest"""

__author__ = "Alex"

import os
import sys
# dirPath = os.path.dirname(os.path.abspath(sys.argv[0]))
# projectPath = os.path.dirname(dirPath)
# sys.path.append(projectPath)
# import warnings
# warnings.filterwarnings("ignore")
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
import DataUtils.glob_variables as gl
from DataUtils.util import checkOS


class LabConfig:
    def __init__(self, random_state, dataName, test_size, classifier):
        self.os = checkOS()
        self.lab_id = uuid.uuid1()
        self.random_state = random_state
        self.dataName = dataName
        self.test_size = test_size
        self.classifier = classifier

    def testNameToType(self):
        """
        根据验证类型返回验证方法
        :return: TestType
        """
        if self.test_size == 0.3:
            return "70%-30%"
        elif self.test_size == 0.4:
            return "60%-40%"
        elif self.test_size == 2:
            return "2-fold"
        elif self.test_size == 10:
            return "10-fold"


class DatasetUtil(LabConfig):
    def __init__(self, random_state, dataName, test_size, classifier):
        super(DatasetUtil, self).__init__(random_state, dataName, test_size, classifier)

    @staticmethod
    def GetLoadDataset(dataName, onehot=False, TrainTest=False):
        from DataUtils.util import LoadDataset
        return LoadDataset(dataName, onehot, TrainTest)

    @staticmethod
    def TransDataset(X, y, train_index, test_index):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        return X_train, X_test, y_train, y_test

    @staticmethod
    def InstancesFeaturesNum(X):
        """
        更据X的数据类型，返回对应的实例数和特征数
        :param X: 数据
        :return: 实例数，特征数
        """
        # if isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame) or sp.sparse.isspmatrix(X):
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
                elif classifier == f"faiss_{k}NN":
                    from fast_knn import FaissKNeighborsClassifier
                    clf = FaissKNeighborsClassifier(k)
                    break
                elif classifier == f"torch_{k}NN":
                    from fast_knn import TorchKNeighborsClassifier
                    clf = TorchKNeighborsClassifier(k)
                    break
        elif classifier == "SVM":
            clf = SVC(**kwargs)
        elif classifier == "DT":
            clf = DecisionTreeClassifier(**kwargs)
        elif classifier == "RF":
            kwargs.update({"n_estimators": 500, "criterion": "gini", "n_jobs": -1})
            clf = RandomForestClassifier(**kwargs)
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        acc = accuracy_score(predict, y_test)  # 计算分类准确率
        return acc

    @staticmethod
    def CalFsAcc(classifier, fset, X_train, X_test, y_train, y_test):
        """
        计算特征子集分类准确率
        :return:
        """
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
        MethodDict = {"ENUMFS": self.GetENUMFS, "PFSFOA": self.GetPFSFOA, "LsForest": self.GetLsForest,
                      "SFSRF": self.GetSFSRF, "SBSRF": self.GetSBSRF, "SFSXGB": self.GetSFSXGB,
                      "SBSXGB": self.GetSBSXGB}
        return MethodDict.get(self.methodName)

    '''
    e.g.
    def GetENUMFS(self, X_train, X_test, y_train, y_test):
        from FS.SearchStrategy import SearchStrategy
        from FS.ENUMFS import ENUMFS
        searchStrategy = SearchStrategy("ENUMFS")
        return ENUMFS(searchStrategy, X_train, X_test, y_train, y_test, self)
    '''
    def OutResult(self, result):
        if gl.get_value("write_to_db"):
            import datetime
            from DataUtils.util import checkOS
            from DataUtils.write2db import WriteResult, ResultSQL
            sql = ResultSQL(self.methodName)
            if self.methodName == "ENUMFS":
                ca, dr, featureStr, runTime, originalCA = result
                DataName, Classifier, TestSize, OriginalCA, CA, DR, RandSeed, RunTime, OutTime, FeatureStr, OS = self.dataName, self.classifier, self.testNameToType(), originalCA * 100, ca * 100, dr * 100, self.random_state, runTime, datetime.datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S'), ",".join('%s' % n for n in featureStr), checkOS()
                WriteResult(sql, (
                DataName, Classifier, TestSize, OriginalCA, CA, DR, RandSeed, RunTime, OutTime, FeatureStr, OS))
            else:
                ca, dr, featureStr, runTime = result
                DataName, Classifier, TestSize, CA, DR, RandSeed, RunTime, OutTime, FeatureStr, OS, Args, Labid = self.dataName, self.classifier, self.testNameToType(), ca * 100, dr * 100, self.random_state, runTime, datetime.datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S'), ",".join('%s' % n for n in featureStr), checkOS(), self.GetArgs(), self.lab_id
                WriteResult(sql, (
                DataName, Classifier, TestSize, CA, DR, RandSeed, RunTime, OutTime, FeatureStr, OS, Args, Labid))
        else:
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
                # 暂时只有这一种写法拼并行参数不会导致循环内部改变的参数错误，其它方法均会存在X_train, X_test, y_train, y_test有时候相同的情况
                task_args = args + [X_train, X_test, y_train, y_test]
                # 目前 kwds= 这个 字典参数， 只能接受不在循环内部改变值的参数，否则子进程中的值会错乱
                result.append(pool.apply_async(func=self.solve, args=task_args, callback=self.OutResult,
                                               error_callback=self.ParallelError))
                # self.solve(*task_args)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                                random_state=self.random_state)
            task_args = args + [X_train, X_test, y_train, y_test]
            result.append(pool.apply_async(func=self.solve, args=task_args, callback=self.OutResult,
                                           error_callback=self.ParallelError))
            # self.solve(*task_args)


def GetConsoleArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu_use", type=int, default=10 if os.cpu_count() >= 12 else os.cpu_count() - 1,
                        help="a number of cpu_use")  # 并行所需要使用的CPU核数
    parser.add_argument("--test_size", default=(0.3,), help="a number of test_size=(0.3,2,10)")  # test_size=(0.3,2,10)
    parser.add_argument("--classifier", nargs='*', default=("1NN",),
                        help="a list of train_name=(1NN, 3NN, 5NN, SVM, DT)")  # 执行训练的算法 train_name=["1NN", "SVM", "3NN", "5NN", "DT"]
    parser.add_argument("--write_to_db", type=bool, default=True, help="a bool of write_to_db")
    parser.add_argument("--write_to_file", type=bool, default=False, help="a bool of write_to_file")
    return parser.parse_args()