import os, sys, inspect

# Colorful error messages
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Neutral', call_pdb=False)

import numpy as np
import math

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import models

def get_r2(estimator, X, y):
       
    y_pred = estimator.predict(X)
    r2 = r2_score(y, y_pred)

    return r2

def fit_estimator_on_subset_of_features(best_estimators, output, X, y):
    new_estimator_base = clone(best_estimators[output][2])
    new_estimator = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='median'), StandardScaler(), new_estimator_base)
    new_estimator.fit(X, y)
    return new_estimator

def get_top_n_features(feature_subsets, output, n):
    features_up_top_n = feature_subsets[output][n]
    return features_up_top_n

def get_cv_scores_on_feature_subsets(feature_subsets, datasets, best_estimators):
    cv_scores_on_feature_subsets = {}
    
    for i, output in enumerate(feature_subsets):
        if output in datasets.keys():
            print("Getting CV scores on feature subsets for " + output + " (" + str(i+1) + "/" + str(len(feature_subsets)) + ")")
            cv_scores_on_feature_subsets[output] = []
            for nb_features in feature_subsets[output].keys():
                X_train, y_train = datasets[output]["X_train"], datasets[output]["y_train"]
                top_n_features = get_top_n_features(feature_subsets, output, nb_features)
                new_estimator = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='median'), StandardScaler(), clone(best_estimators[output][2]))
                cv_scores = cross_val_score(new_estimator, X_train[top_n_features], y_train, cv = KFold(n_splits=8), scoring='r2')
                cv_scores_on_feature_subsets[output].append(cv_scores.mean())
    return cv_scores_on_feature_subsets

def get_performances_on_feature_subsets_per_output(output, feature_subsets, estimators_on_feature_subsets, datasets, use_test_set):

    if use_test_set == 1:
        X_test, y_test = datasets[output]["X_test"], datasets[output]["y_test"]
    else:
        X_test, y_test = datasets[output]["X_val"], datasets[output]["y_val"]

    metrics_on_subsets = {}
    
    for nb_features in feature_subsets[output].keys():
        # Create new pipeline with the params of the best estimator (need to re-train the imputer on less features)
        print("Getting metrics on feature subsets for " + output + " with " + str(nb_features) + " features")
        top_n_features = get_top_n_features(feature_subsets, output, nb_features)
        new_estimator = estimators_on_feature_subsets[output][nb_features]
        
        metrics = get_r2(new_estimator, X_test[top_n_features], y_test)
        metrics_on_subsets[nb_features] = metrics
    
    return metrics_on_subsets

def get_performances_on_feature_subsets(feature_subsets, datasets, best_estimators, estimators_on_feature_subsets, use_test_set):
    cv_scores_on_feature_subsets = get_cv_scores_on_feature_subsets(feature_subsets, datasets, best_estimators)

    performances_on_subsets = {}
    
    for i, output in enumerate(feature_subsets):
        if output in datasets.keys():
            print("Getting performances on feature subsets for " + output + " (" + str(i+1) + "/" + str(len(feature_subsets)) + ")")
            performances_on_subsets[output] = get_performances_on_feature_subsets_per_output(output, feature_subsets, estimators_on_feature_subsets, datasets, use_test_set)

    return performances_on_subsets, cv_scores_on_feature_subsets