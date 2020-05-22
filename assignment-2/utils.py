import gc

from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import ndcg_score, make_scorer
import numpy as np
import xgboost as xgb
from xgboost import DMatrix, XGBRanker
from xgboost.sklearn import XGBClassifier
from xgboostextension.scorer import RankingScorer
from xgboostextension.scorer.metrics import ndcg
import pandas as pd
# from xgboostextension import XGBRanker


class HackXGBRanker(XGBRanker):
    def fit(self, *args, **xargs):
        # print(args)
        # print(xargs)
        args = list(args)
        X = args[0]
        groups = X[:, 0]
        X = X[:, 1:]
        args[0] = X
        xargs['group'] = pd.Series(groups).value_counts(
            sort=False).sort_index()
        return super().fit(*args, **xargs)

    def predict(self, *args, **xargs):
        args = list(args)
        args[0] = args[0][:, 1:]
        res = super().predict(*args, **xargs)
        return res

    def predict_proba(self, *args, **xargs):
        return self.predict(*args, **xargs)


# def run(X, y, groups):

#     train_dmatrix = DMatrix(x_train, y_train)
#     valid_dmatrix = DMatrix(x_valid, y_valid)
#     test_dmatrix = DMatrix(x_test)

#     train_dmatrix.set_group(group_train)
#     valid_dmatrix.set_group(group_valid)
#     test_dmatrix.set_group(group_test)
#     params = {'objective': 'rank:pairwise', 'eta': 0.5, 'gamma': 1.0,
#                 'min_child_weight': 0.5, 'max_depth': 8,'eval_metric':'ndcg@5','nthread':16}

# def train_and_eval(train_dmatrix, valid_dmatrix, params):
#     xgb_model = xgb.train(params, train_dmatrix, num_boost_round=1000,
#                             evals=[(valid_dmatrix, 'validation')])
#     xgb_model
def groups2group(groups):
    return pd.Series(groups).value_counts(sort=False).sort_index()


def run(X, y, groups):
    gc.collect()
    # grid_search = GridSearchCV(
    #     ranker,
    #     {
    #         'n_estimators': [5,40],
    #         'max_depth': [3]
    #     },
    #     cv=GroupKFold(2),
    #     # scoring=RankingScorer(ndcg(5)),
    #     scoring=make_scorer(ndcg_score, greater_is_better=True, needs_proba=True, k=5),
    #     verbose=100,
    #     n_jobs=1
    # )
    params = {'objective': 'rank:ndcg', 'n_estimators': 40, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.5,
              'early_stopping_rounds': 10, 'seed': 5}
    gkf = GroupKFold(2)
    for train_inds, val_inds in gkf.split(X, y, groups):
        X_train, X_val = X[train_inds], X[val_inds]
        y_train, y_val = y[train_inds], y[val_inds]
        groups_train, groups_val = groups[train_inds], groups[val_inds]
        group_train = groups2group(groups_train)
        group_val = groups2group(groups_val)
        gc.collect()
        ranker = XGBRanker(*params)
        ranker.fit(X_train, y_train, group_train, verbose=True, eval_set=[
                   (X_train, y_train), (X_val, y_val)], eval_group=[group_train, group_val], eval_metric='ndcg@5')
        

    # grid_search.fit(X, y, groups=groups)
    # print("Cross validation results:")
    # print(grid_search.cv_results_)


if __name__ == "__main__":
    X = np.array([0., 0., 0., 0.])
    X = X.reshape([4, 1])
    y = np.array([1, 0, 2, 1])
    groups = np.array([0, 0, 2, 2], dtype='int32')
    X = np.concatenate((X, groups[:, None]), 1)
    # group = np.array([2,2], dtype='int32')

    gkf = GroupKFold(10)
    print(len(X))
    subsampled_inds = next(iter(gkf.split(X, y, groups)))[-1]
    print(len(subsampled_inds))
    X = X[subsampled_inds]
    y = y[subsampled_inds]
    groups = groups[subsampled_inds]
    run(X, y, groups)
