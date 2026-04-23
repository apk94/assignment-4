import pandas as pd
import csv
from requests import get
import json
from datetime import datetime, timedelta, date
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, cosine
from scipy.stats import pearsonr
import csv
import re
import pandas as pd
import argparse
import collections
import json
import glob
import math
import os
import requests
import string
import sys
import time
import xml
import random

class Recommender(object):
    def __init__(self, training_set, test_set):
        if isinstance(training_set, str):
            self.training_set = pd.read_csv(training_set)
        else:
            self.training_set = training_set.copy()
        if isinstance(test_set, str):
            self.test_set = pd.read_csv(test_set)
        else:
            self.test_set = test_set.copy()

    def train_user_euclidean(self, data_set, userId):
        weights = {}
        other_users = [c for c in data_set.columns if c != 'movieId' and c != userId]
        for other in other_users:
            mask = data_set[userId].notna() & data_set[other].notna()
            u, v = data_set.loc[mask, userId], data_set.loc[mask, other]
            if len(u) == 0:
                weights[other] = 0.0
            else:
                dist = np.sqrt(np.sum((u.values - v.values) ** 2))
                weights[other] = 1.0 / (1.0 + dist)
        return weights

    def train_user_manhattan(self, data_set, userId):
        weights = {}
        other_users = [c for c in data_set.columns if c != 'movieId' and c != userId]
        for other in other_users:
            mask = data_set[userId].notna() & data_set[other].notna()
            u, v = data_set.loc[mask, userId], data_set.loc[mask, other]
            if len(u) == 0:
                weights[other] = 0.0
            else:
                dist = cityblock(u, v)
                weights[other] = 1.0 / (1.0 + dist)
        return weights

    def train_user_cosine(self, data_set, userId):
        weights = {}
        other_users = [c for c in data_set.columns if c != 'movieId' and c != userId]
        for other in other_users:
            mask = data_set[userId].notna() & data_set[other].notna()
            u, v = data_set.loc[mask, userId], data_set.loc[mask, other]
            if len(u) == 0:
                weights[other] = 0.0
            else:
                sim = 1.0 - cosine(u, v)
                weights[other] = float(sim) if not np.isnan(sim) else 0.0
        return weights

    def train_user_pearson(self, data_set, userId):
        weights = {}
        other_users = [c for c in data_set.columns if c != 'movieId' and c != userId]
        for other in other_users:
            mask = data_set[userId].notna() & data_set[other].notna()
            u, v = data_set.loc[mask, userId], data_set.loc[mask, other]
            if len(u) < 2 or np.std(u) == 0 or np.std(v) == 0:
                weights[other] = 0.0
            else:
                corr, _ = pearsonr(u, v)
                weights[other] = float(corr) if not np.isnan(corr) else 0.0
        return weights

    def train_user(self, data_set, distance_function, userId):
        if distance_function == 'euclidean':
            return self.train_user_euclidean(data_set, userId)
        elif distance_function == 'manhattan':
            return self.train_user_manhattan(data_set, userId)
        elif distance_function == 'cosine':
            return self.train_user_cosine(data_set, userId)
        elif distance_function == 'pearson':
            return self.train_user_pearson(data_set, userId)
        else:
            return None

    def get_user_existing_ratings(self, data_set, userId):
        user_col = data_set[userId]
        rated_mask = user_col.notna()
        movie_ids = data_set.loc[rated_mask, 'movieId'].values
        ratings   = user_col[rated_mask].values
        return [(int(mid), float(r)) for mid, r in zip(movie_ids, ratings)]

    def predict_user_existing_ratings_top_k(self, data_set, sim_weights, userId, k):
        sorted_neighbours = sorted(sim_weights.items(), key=lambda x: x[1], reverse=True)
        existing = self.get_user_existing_ratings(data_set, userId)
        predictions = []
        for movieId, _ in existing:
            test_mask  = data_set['movieId'] == movieId
            train_mask = self.training_set['movieId'] == movieId
            numerator   = 0.0
            denominator = 0.0
            used = 0
            for neighbour, weight in sorted_neighbours:
                if used >= k:
                    break
                neighbour_rating = None
                if neighbour in data_set.columns and test_mask.any():
                    val = data_set.loc[test_mask, neighbour].values[0]
                    if not pd.isna(val):
                        neighbour_rating = val
                if neighbour_rating is None and neighbour in self.training_set.columns and train_mask.any():
                    val = self.training_set.loc[train_mask, neighbour].values[0]
                    if not pd.isna(val):
                        neighbour_rating = val
                if neighbour_rating is None:
                    continue
                numerator   += weight * neighbour_rating
                denominator += abs(weight)
                used += 1
            if denominator == 0 or used == 0:
                continue
            pred_rating = numerator / denominator
            predictions.append((int(movieId), float(pred_rating)))
        return predictions

    def evaluate(self, existing_ratings, predicted_ratings):
        actual = {m: r for m, r in existing_ratings}
        pred = {m: r for m, r in predicted_ratings}
        common = set(actual.keys()) & set(pred.keys())
        ratio = len(common) / len(actual) if actual else 0.0
        if not common:
            return {'rmse': 0.0, 'ratio': float(ratio)}
        mse = sum([(actual[m] - pred[m])**2 for m in common]) / len(common)
        return {'rmse': float(math.sqrt(mse)), 'ratio': float(ratio)}

    def single_calculation(self, distance_function, userId, k_values):
        user_existing_ratings = self.get_user_existing_ratings(self.test_set, userId)
        print("User has {} existing and {} missing movie ratings".format(len(user_existing_ratings), len(self.test_set) - len(user_existing_ratings)), file=sys.stderr)
        print('Building weights')
        sim_weights = self.train_user(self.training_set[self.test_set.columns.values.tolist()], distance_function, userId)
        result = []
        for k in k_values:
            print('Calculating top-k user prediction with k={}'.format(k))
            top_k_existing_ratings_prediction = self.predict_user_existing_ratings_top_k(self.test_set, sim_weights, userId, k)
            result.append((k, self.evaluate(user_existing_ratings, top_k_existing_ratings_prediction)))
        return result

    def aggregate_calculation(self, distance_functions, userId, k_values):
        print()
        result_per_k = {}
        for func in distance_functions:
            print("Calculating for {} distance metric".format(func))
            for calc in self.single_calculation(func, userId, k_values):
                if calc[0] not in result_per_k:
                    result_per_k[calc[0]] = {}
                result_per_k[calc[0]]['{}_rmse'.format(func)] = calc[1]['rmse']
                result_per_k[calc[0]]['{}_ratio'.format(func)] = calc[1]['ratio']
            print()
        result = []
        for k in k_values:
            row = {'k':k}
            row.update(result_per_k[k])
            result.append(row)
        columns = ['k']
        for func in distance_functions:
            columns.append('{}_rmse'.format(func))
            columns.append('{}_ratio'.format(func))
        result = pd.DataFrame(result, columns=columns)
        return result

if __name__ == "__main__":
    recommender = Recommender("data/train.csv", "data/small_test.csv")
    print("Training set has {} users and {} movies".format(len(recommender.training_set.columns[1:]), len(recommender.training_set)))
    print("Testing set has {} users and {} movies".format(len(recommender.test_set.columns[1:]), len(recommender.test_set)))
    result = recommender.aggregate_calculation(['euclidean', 'cosine', 'pearson', 'manhattan'], "0331949b45", [1, 2, 3, 4])
    print(result)
