#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = "Alex"

import os
import gc
import warnings
import socket
import multiprocessing
import json
import copy
import numpy as np
import scipy as sp
import xgboost as xgb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import log_loss, accuracy_score
from fsbase import DatasetUtil
from toolkit import GetStatistics
from tqdm import tqdm
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")


class SearchStrategies(DatasetUtil):
    def __init__(self, classifier, X_train, X_test, y_train, y_test):
        self.classifier = classifier
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.feature_length = self.InstancesFeaturesNum(X_test)[1]

    @staticmethod
    def FindRelations(f, featureRelation):
        fRelations = []
        for relation in featureRelation:
            f1, f2 = map(int, relation.split('|'))
            if f1 == f and f2 != f and f2 not in fRelations:
                fRelations.append(f2)
            elif f2 == f and f1 != f and f1 not in fRelations:
                fRelations.append(f1)
        return fRelations

    def AddRelationFeatures(self, feature, featureRelation, optimalFSet, optimalFitness):
        fRelations = self.FindRelations(feature, featureRelation)  # 找到和当前feature相关的特征集合
        for rf in fRelations:
            if rf not in optimalFSet:
                optimalFSet.add(rf)
                tempFitness = self.CalFitness(optimalFSet)
                if tempFitness < optimalFitness:
                    optimalFitness = tempFitness
                else:
                    optimalFSet.remove(rf)
        return optimalFSet, optimalFitness

    def RemoveFeatures(self, removeRank, optimalFSet, optimalFitness):
        for mf in removeRank:
            if mf in optimalFSet:
                optimalFSet.remove(mf)
                tempFitness = self.CalFitness(optimalFSet)
                if tempFitness < optimalFitness:
                    optimalFitness = tempFitness
                else:
                    optimalFSet.add(mf)
        return optimalFSet, optimalFitness

    def SFSR(self, addRank, featureRelation, removeRank):
        CAList, DRList, FitnessList = [], [], []
        optimalFSet, optimalFitness = set(), 1.0
        for f in addRank:
            if f not in optimalFSet:
                optimalFSet.add(f)
                tempFitness = self.CalFitness(optimalFSet)
                if tempFitness < optimalFitness:
                    optimalFitness = tempFitness
                    optimalFSet, optimalFitness = self.AddRelationFeatures(f, featureRelation, optimalFSet,
                                                                           optimalFitness)
                    optimalFSet, optimalFitness = self.RemoveFeatures(removeRank, optimalFSet, optimalFitness)
                else:
                    optimalFSet.remove(f)
            else:
                optimalFSet, optimalFitness = self.AddRelationFeatures(f, featureRelation, optimalFSet, optimalFitness)
                optimalFSet, optimalFitness = self.RemoveFeatures(removeRank, optimalFSet, optimalFitness)
            CA = round(self.CalFsAcc(self.classifier, optimalFSet, self.X_train, self.X_test, self.y_train, self.y_test) * 100, 4)
            DR = round(self.CalFsDR(optimalFSet, self.feature_length) * 100, 4)
            Fitness = round(optimalFitness * 100, 4)
            CAList.append(CA)
            DRList.append(DR)
            FitnessList.append(Fitness)
        return CAList[-1], DRList[-1], FitnessList[-1], CAList[0:100], DRList[0:100], FitnessList[0:100], list(optimalFSet)

    def CalFitness(self, fset, alpha=0.99, beta=0.01):
        acc = self.CalFsAcc(self.classifier, fset, self.X_train, self.X_test, self.y_train, self.y_test)
        dr = self.CalFsDR(fset, self.feature_length)
        fv = alpha * (1.0 - acc) + beta * (1.0 - dr)
        return fv


class XGBLFS(xgb.XGBClassifier):
    def __init__(self, X_train, X_test, y_train, y_test, base_classifier="5NN", max_depth=5, learning_rate=0.1,
                 n_estimators=100, verbosity=1, silent=None, objective="binary:logistic", booster='gbtree', n_jobs=10,
                 nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
                 colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5,
                 random_state=0, seed=None, missing=None, **kwargs):
        super(xgb.XGBClassifier, self).__init__(
            max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
            verbosity=verbosity, silent=silent, objective=objective, booster=booster,
            n_jobs=n_jobs, nthread=nthread, gamma=gamma,
            min_child_weight=min_child_weight, max_delta_step=max_delta_step,
            subsample=subsample, colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel, colsample_bynode=colsample_bynode,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight,
            base_score=base_score, random_state=random_state, seed=seed, missing=missing,
            **kwargs)

        self.base_classifier = base_classifier
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test


    @staticmethod
    def ConvertX(X, use_label=None, use_times=1):
        if sp.sparse.issparse(X):
            X = X.todense()
        if use_label is not None:
            y = use_label
            while use_times > 1:
                y = np.concatenate((y, use_label), axis=1)
                use_times -= 1
            feature_dense_y = tf.cast(np.concatenate((y, X), axis=1), dtype=tf.float32)
        else:
            feature_dense_y = tf.cast(X, dtype=tf.float32)
        return feature_dense_y

    @staticmethod
    def LapChange(X, W, K=10, feature_dense_y=None, use_label=None, use_times=1):

        def tf_normalize_adj(inputs):
            params_shape = tf.shape(inputs)[0]
            inputs = tf.cast(inputs, dtype=tf.float32) + tf.cast(tf.eye(params_shape), dtype=tf.float32)
            row_sum = tf.reduce_sum(inputs, axis=1)
            d_inv_sqrt = tf.pow(row_sum, -0.5)
            tf.clip_by_value(d_inv_sqrt, clip_value_min=0.01, clip_value_max=1000)
            outputs = tf.transpose(d_inv_sqrt * inputs) * d_inv_sqrt
            return outputs

        if sp.sparse.issparse(X):
            X = X.todense()
        feature_dense_y = XGBLFS.ConvertX(X, use_label=use_label, use_times=use_times) if feature_dense_y is None else feature_dense_y
        feature_dense_y = feature_dense_y * W
        sim_matrix = feature_dense_y @ tf.transpose(feature_dense_y)
        top_values, top_indices = tf.nn.top_k(sim_matrix, k=K)
        kthvalue = tf.math.reduce_min(top_values, axis=-1)
        adj = tf.cast((sim_matrix > kthvalue), tf.float32) * sim_matrix
        processed_adj = tf_normalize_adj(adj)
        lap_mul_input = processed_adj @ X
        return lap_mul_input.numpy()

    @staticmethod
    def GetFeatureImportances(xgb_booster, importance_type):
        importances = xgb.Booster.get_score(xgb_booster, importance_type=importance_type)
        indices = sorted(importances.items(), key=lambda item: item[1], reverse=True)
        print(indices)
        indices = [int(i[0]) for i in indices]
        return indices

    @staticmethod
    def GetFeatureInteractions(interactions_dict, SortBy='averagegain', Desc=True, Depth='Depth1'):
        SortFun = {
            'gain': lambda x: x.Gain,
            'fscore': lambda x: x.FScore,
            'fscoreweighted': lambda x: x.FScoreWeighted,
            'fscoreweightedaverage': lambda x: x.FScoreWeightedAverage,
            'averagegain': lambda x: x.AverageGain,
            'expectedgain': lambda x: x.ExpectedGain
        }[SortBy.lower()]
        interactions_sort = sorted(interactions_dict[Depth], key=SortFun, reverse=Desc)
        indices = [f.Name for f in interactions_sort]
        return indices

    @staticmethod
    def GetFeatureInteractions(interactions_dict, SortBy='averagegain', Desc=True, Depth='Depth1'):
        SortFun = {
            'gain': lambda x: x.Gain,
            'fscore': lambda x: x.FScore,
            'fscoreweighted': lambda x: x.FScoreWeighted,
            'fscoreweightedaverage': lambda x: x.FScoreWeightedAverage,
            'averagegain': lambda x: x.AverageGain,
            'expectedgain': lambda x: x.ExpectedGain
        }[SortBy.lower()]
        interactions_sort = sorted(interactions_dict[Depth], key=SortFun, reverse=Desc)
        indices = [f.Name for f in interactions_sort]
        return indices

    def Run(self):
        clf = self.fit(self.X_train, self.y_train)
        booster = clf.get_booster()
        booster.feature_names = list(
            map(lambda feature_name: feature_name[1:] if feature_name[0] == 'f' else feature_name,
                booster.feature_names))
        ensemble_trees = xgb.Booster.get_dump(booster, with_stats=True)
        interactions = GetStatistics(booster, ensemble_trees, MaxTrees=self.n_estimators, MaxInteractionDepth=self.max_depth, SortBy='averageGain')
        avgGainRank = map(int, self.GetFeatureInteractions(interactions, SortBy='averageGain', Desc=True, Depth='Depth0'))
        avgGainRelation = self.GetFeatureInteractions(interactions, SortBy='averageGain', Desc=True, Depth='Depth1')
        reversFScoreRank = map(int, self.GetFeatureInteractions(interactions, SortBy='FScore', Desc=False, Depth='Depth0'))  # 暂时用FScore, 先不用cover
        searchStrategy = SearchStrategies(self.base_classifier, self.X_train, self.X_test, self.y_train, self.y_test)
        return searchStrategy.SFSR(avgGainRank, avgGainRelation, reversFScoreRank)


def OptimizerWByFOA(X, y, test_size, random_state, tree_numbers=15, lifeTime=5, loop=10, areaLimit=30, transferRate=0.05, K=10, use_label=None, use_times=1):
    class FOA:
        def __init__(self, W, lap_mul_input, lap_mul_input_y, test_size, random_state):
            self.W = W
            self.age = 0
            max_depth = 5
            n_estimators = 100
            clf = xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators)
            X_train, X_test, y_train, y_test = train_test_split(lap_mul_input, lap_mul_input_y, test_size=test_size, random_state=random_state)
            y_train_onehot = self.encode.fit_transform(y_train)  # y_train 转 onehot
            clf = clf.fit(X_train, y_train)
            y_predict_prob = clf.predict_proba(X_train)
            self.XgbLoss = log_loss(y_train_onehot, y_predict_prob)
            booster = clf.get_booster()
            booster.feature_names = list(map(lambda feature_name: feature_name[1:] if feature_name[0] == 'f' else feature_name, booster.feature_names))
            ensemble_trees = xgb.Booster.get_dump(booster, with_stats=True)
            interactions = GetStatistics(booster, ensemble_trees, MaxTrees=n_estimators, MaxInteractionDepth=max_depth, SortBy='averageGain')
            self.avgGainRank = list(map(int, XGBLFS.GetFeatureInteractions(interactions, SortBy='averageGain', Desc=True, Depth='Depth0')))

        @classmethod
        def InitClassVariable(cls, X, y, test_size, random_state, lifeTime=10, areaLimit=60, transferRate=0.05, K=10, use_label=None, use_times=1):
            cls.X, cls.y = X, y
            cls.encode = LabelBinarizer()
            cls.test_size, cls.random_state = test_size, random_state
            cls.lifeTime, cls.areaLimit, cls.transferRate = lifeTime, areaLimit, transferRate
            cls.K = K
            cls.use_label, cls.use_times = use_label, use_times
            cls.trainableWs = []
            cls.candidateWs = []

        @classmethod
        def InitForest(cls, tree_numbers=30):
            cls.feature_dense_y = XGBLFS.ConvertX(cls.X, use_label=cls.use_label, use_times=cls.use_times)
            cls.variableNumbers = cls.feature_dense_y.shape[1]
            cls.GSC = int(np.sqrt(cls.variableNumbers))
            for _ in range(tree_numbers):
                W = tf.random.uniform([1, cls.variableNumbers], 0.707, 1.414)
                lap_mul_input = XGBLFS.LapChange(cls.X, W, K=cls.K, feature_dense_y=cls.feature_dense_y)
                cls.trainableWs.append(cls(W.numpy(), lap_mul_input, cls.y, cls.test_size, cls.random_state))

        @classmethod
        def LocalSeeding(cls):
            GenerateWs = []
            for tree in tqdm(cls.trainableWs):
                if tree.age == 0:
                    for v in tree.avgGainRank:
                        deltaX = np.random.uniform(-0.1, 0.1)
                        new_W = copy.copy(tree.W)
                        new_W[0][v] = min(max((tree.W[0][v] + deltaX), 0.707), 1.414)
                        lap_mul_input = XGBLFS.LapChange(cls.X, tf.convert_to_tensor(new_W), K=cls.K, feature_dense_y=cls.feature_dense_y)
                        GenerateWs.append(cls(new_W, lap_mul_input, cls.y, cls.test_size, cls.random_state))
                tree.age += 1
                if tree.age > cls.lifeTime:
                    cls.candidateWs.append(tree)
            cls.trainableWs = list(filter(lambda old_tree: old_tree.age <= cls.lifeTime, cls.trainableWs))
            cls.trainableWs.extend(GenerateWs)
            cls.trainableWs = sorted(cls.trainableWs, key=lambda tree: tree.XgbLoss)
            cls.candidateWs.extend(cls.trainableWs[cls.areaLimit:])
            cls.trainableWs = cls.trainableWs[:cls.areaLimit]

        @classmethod
        def GlobalSeeding(cls):
            candidateTreeIndices = np.random.choice(len(cls.candidateWs), int(cls.transferRate * len(cls.candidateWs)), replace=False)
            for index in candidateTreeIndices:
                tree = cls.candidateWs[index]
                GSC_variables = np.random.choice(cls.variableNumbers, cls.GSC, replace=False)
                for v in GSC_variables:
                    deltaX = np.random.uniform(-0.1, 0.1)
                    new_W = copy.copy(tree.W)
                    new_W[0][v] = min(max((tree.W[0][v] + deltaX), 0.707), 1.414)
                lap_mul_input = XGBLFS.LapChange(cls.X, tf.convert_to_tensor(new_W), K=cls.K, feature_dense_y=cls.feature_dense_y)
                cls.trainableWs.append(cls(new_W, lap_mul_input, cls.y, cls.test_size, cls.random_state))

        @classmethod
        def Update(cls):
            cls.trainableWs = sorted(cls.trainableWs, key=lambda tree: tree.XgbLoss)
            cls.trainableWs[0].age = 0

    FOA.InitClassVariable(X, y, test_size, random_state, lifeTime=lifeTime, areaLimit=areaLimit, transferRate=transferRate, K=K, use_label=use_label, use_times=use_times)
    FOA.InitForest(tree_numbers=tree_numbers)
    loop = 3
    for _ in tqdm(range(loop)):
        FOA.LocalSeeding()
        FOA.GlobalSeeding()
        FOA.Update()
        gc.collect()
    bestW = FOA.trainableWs[0].W
    return bestW


def main(random_state, dataName, test_size, classifier, k):
    fsutil = DatasetUtil(random_state, dataName, test_size, classifier)
    try:
        if test_size < 1:
            X, y = fsutil.GetLoadDataset(dataName, onehot=False)
            bestW = OptimizerWByFOA(X, y, test_size, random_state, tree_numbers=15, lifeTime=5, loop=10, areaLimit=30, transferRate=0.05, K=k)
            gc.collect()
            X = XGBLFS.LapChange(X, bestW, K=k)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            print("start")
            featureSelection = XGBLFS(X_train, X_test, y_train, y_test, classifier)
            CA, DR, fitness, CAList, DRList, FitnessList, FeatureStr = featureSelection.Run()
            print(CA)
    except:
        pass
