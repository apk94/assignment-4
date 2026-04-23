import pandas as pd
import numpy as np
import math
from scipy.spatial.distance import euclidean, cityblock, cosine
from scipy.stats import pearsonr

class Recommender(object):
    def __init__(self, training_set, test_set):
        self.training_set = pd.read_csv(training_set) if isinstance(training_set, str) else training_set.copy()
        self.test_set = pd.read_csv(test_set) if isinstance(test_set, str) else test_set.copy()
    
    def train_user_euclidean(self, data_set, userId):
        weights = {}
        u_ratings = data_set[userId]
        for other in [c for c in data_set.columns if c not in ['movieId', userId]]:
            mask = u_ratings.notna() & data_set[other].notna()
            if mask.any():
                dist = euclidean(u_ratings[mask], data_set.loc[mask, other])
                weights[other] = 1.0 / (1.0 + dist)
            else:
                weights[other] = 0.0
        return weights
        
    def train_user_manhattan(self, data_set, userId):
        weights = {}
        u_ratings = data_set[userId]
        for other in [c for c in data_set.columns if c not in ['movieId', userId]]:
            mask = u_ratings.notna() & data_set[other].notna()
            if mask.any():
                dist = cityblock(u_ratings[mask], data_set.loc[mask, other])
                weights[other] = 1.0 / (1.0 + dist)
            else:
                weights[other] = 0.0
        return weights

    def train_user_cosine(self, data_set, userId):
        weights = {}
        u_ratings = data_set[userId]
        for other in [c for c in data_set.columns if c not in ['movieId', userId]]:
            mask = u_ratings.notna() & data_set[other].notna()
            if mask.any():
                sim = 1.0 - cosine(u_ratings[mask], data_set.loc[mask, other])
                weights[other] = float(sim) if not np.isnan(sim) else 0.0
            else:
                weights[other] = 0.0
        return weights
   
    def train_user_pearson(self, data_set, userId):
        weights = {}
        u_ratings = data_set[userId]
        for other in [c for c in data_set.columns if c not in ['movieId', userId]]:
            mask = u_ratings.notna() & data_set[other].notna()
            u_vec, v_vec = u_ratings[mask], data_set.loc[mask, other]
            if len(u_vec) < 2 or np.std(u_vec) < 1e-9 or np.std(v_vec) < 1e-9:
                weights[other] = 0.0
            else:
                try:
                    corr, _ = pearsonr(u_vec, v_vec)
                    weights[other] = float(corr) if not np.isnan(corr) else 0.0
                except:
                    weights[other] = 0.0
        return weights

    def train_user(self, data_set, distance_function, userId):
        methods = {'euclidean': self.train_user_euclidean, 'manhattan': self.train_user_manhattan,
                   'cosine': self.train_user_cosine, 'pearson': self.train_user_pearson}
        return methods[distance_function](data_set, userId)

    def get_user_existing_ratings(self, data_set, userId):
        user_col = data_set[userId]
        mask = user_col.notna()
        return list(zip(data_set.loc[mask, 'movieId'].astype(int), user_col[mask].astype(float)))

    def predict_user_existing_ratings_top_k(self, data_set, sim_weights, userId, k):
        existing = self.get_user_existing_ratings(data_set, userId)
        predictions = []

        for movieId, _ in existing:
            # 1. Get the movie ratings row from training set
            movie_rows = self.training_set[self.training_set['movieId'] == movieId]
            if movie_rows.empty:
                continue
            
            movie_row = movie_rows.iloc[0]
            
            # 2. Find ALL neighbors who rated this movie
            candidates = []
            for neighbor, weight in sim_weights.items():
                if neighbor in self.training_set.columns:
                    rating = movie_row[neighbor]
                    if not pd.isna(rating):
                        candidates.append((neighbor, weight, rating))
            
            if not candidates:
                continue

            # 3. Sort by weight (desc) then name (asc) and take top K
            candidates.sort(key=lambda x: (-x[1], str(x[0])))
            top_k_neighbors = candidates[:k]
            
            num = sum(w * r for _, w, r in top_k_neighbors)
            den = sum(abs(w) for _, w, _ in top_k_neighbors)
            
            if den != 0:
                predictions.append((int(movieId), float(num / den)))
        
        return predictions
            
    def evaluate(self, existing_ratings, predicted_ratings):
        actual = {m: r for m, r in existing_ratings}
        pred = {m: r for m, r in predicted_ratings if r is not None}
        
        common = set(actual.keys()) & set(pred.keys())
        ratio = len(pred) / len(actual) if actual else 0.0
        
        if not common:
            return {'rmse': 0.0, 'ratio': round(ratio, 4)}
            
        sq_diff_sum = sum((float(actual[m]) - float(pred[m]))**2 for m in common)
        rmse = math.sqrt(sq_diff_sum / len(common))
        
        return {'rmse': round(rmse, 4), 'ratio': round(ratio, 4)}
    
    def single_calculation(self, distance_function, userId, k_values):
        user_existing = self.get_user_existing_ratings(self.test_set, userId)
        sim_weights = self.train_user(self.training_set, distance_function, userId)

        result = []
        for k in k_values:
            preds = self.predict_user_existing_ratings_top_k(self.test_set, sim_weights, userId, k)
            result.append((k, self.evaluate(user_existing, preds)))
        return result

    def aggregate_calculation(self, distance_functions, userId, k_values):
        res_dict = {k: {} for k in k_values}
        for func in distance_functions:
            for k, eval_res in self.single_calculation(func, userId, k_values):
                res_dict[k][f'{func}_rmse'] = eval_res['rmse']
                res_dict[k][f'{func}_ratio'] = eval_res['ratio']
        
        final_rows = []
        for k in k_values:
            row = {'k': k}
            row.update(res_dict[k])
            final_rows.append(row)
        
        cols = ['k']
        for func in distance_functions:
            cols.extend([f'{func}_rmse', f'{func}_ratio'])
            
        return pd.DataFrame(final_rows, columns=cols)
