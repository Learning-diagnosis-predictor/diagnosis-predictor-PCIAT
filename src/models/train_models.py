import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning" # Seems to be the only way to suppress multi-thread sklearn warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.linear_model import ElasticNet

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import RandomizedSearchCV

import sys, inspect

from joblib import load, dump

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util, data, models, util

DEBUG_MODE = True

def build_output_dir_name(params_from_create_datasets):
    # Part with the datetime
    datetime_part = util.get_string_with_current_datetime()

    # Part with the params
    params_part = util.build_param_string_for_dir_name(params_from_create_datasets) + "___" +\
                  util.build_param_string_for_dir_name({"debug_mode": DEBUG_MODE})
    
    return datetime_part + "___" + params_part

def set_up_directories():

    # Create directory in the parent directory of the project (separate repo) for output data, models, and reports
    data_dir = "../diagnosis_predictor_PCIAT_data/"
    util.create_dir_if_not_exists(data_dir)

    # Input dirs
    input_data_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "data/create_datasets/")

    # Create directory inside the output directory with the run timestamp and params:
    #    - [params from create_datasets.py]
    #    - use other diags as input
    #    - debug mode
    params_from_create_datasets = models.get_params_from_current_data_dir_name(input_data_dir)
    current_output_dir_name = build_output_dir_name(params_from_create_datasets)

    models_dir = data_dir + "models/" + "train_models/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(models_dir)

    reports_dir = data_dir + "reports/" + "train_models/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(reports_dir) 

    return {"input_data_dir": input_data_dir, "models_dir": models_dir, "reports_dir": reports_dir}

def set_up_load_directories():
    # When loading existing models, can't take the newest directory, we just created it, it will be empty. 
    #   Need to take the newest non-empty directory.

    data_dir = "../diagnosis_predictor_PCIAT_data/"
    
    load_data_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "data/create_datasets/")
    load_models_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "models/train_models/")
    load_reports_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "reports/train_models/")
    
    return {"load_data_dir": load_data_dir, "load_models_dir": load_models_dir, "load_reports_dir": load_reports_dir}
    
def get_base_models_and_param_grids():
    
    # Define base models
    rf = RandomForestRegressor(n_estimators=200 if DEBUG_MODE else 400)
    svr = svm.SVR()
    en = ElasticNet()
    
    # Impute missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    
    # Standardize data
    scaler = StandardScaler()

    # Make pipelines
    rf_pipe = make_pipeline(imputer, scaler, rf)
    svr_pipe = make_pipeline(imputer, scaler, svr)
    en_pipe = make_pipeline(imputer, scaler, en)

    # Define parameter grids to search for each pipe
    from scipy.stats import loguniform, uniform
    rf_param_grid = {
        'randomforestregressor__max_depth' : np.random.randint(5, 150, 30),
        'randomforestregressor__min_samples_split': np.random.randint(2, 50, 30),
        'randomforestregressor__n_estimators': np.random.randint(50, 400, 10),
        'randomforestregressor__min_samples_leaf': np.random.randint(1, 20, 30),
        'randomforestregressor__max_features': ['auto', 'sqrt', 'log2', 0.25, 0.5, 0.75, 1.0]
    }
    svr_param_grid = {
        'svr__C': loguniform(1e-03, 1e+02),
        'svr__gamma': loguniform(1e-03, 1e+02),
        'svr__degree': uniform(2, 5),
        'svr__epsilon': loguniform(1e-03,1),
        'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
    en_param_grid = {
        'elasticnet__alpha': loguniform(1e-5, 100),
        'elasticnet__l1_ratio': uniform(0, 1)
    }
    
    base_models_and_param_grids = [
        (rf_pipe, rf_param_grid),
        (svr_pipe, svr_param_grid),
        (en_pipe, en_param_grid)
    ]
    if DEBUG_MODE:
        base_models_and_param_grids = [base_models_and_param_grids[-1]] # Only do LR in debug mode
        #base_models_and_param_grids = base_models_and_param_grids

    return base_models_and_param_grids

def get_best_estimator(base_model, grid, X_train, y_train):
    cv = KFold(n_splits=3 if DEBUG_MODE else 8)
    rs = RandomizedSearchCV(estimator=base_model, param_distributions=grid, cv=cv, scoring="neg_mean_absolute_error", n_iter=50 if DEBUG_MODE else 200, n_jobs = -1, verbose=1) #neg_mean_absolute_error
    
    print("Fitting", base_model, "...")
    print(y_train.dtype)
    rs.fit(X_train, y_train) 
    
    best_estimator = rs.best_estimator_
    best_score = rs.best_score_
    sd_of_score_of_best_estimator = rs.cv_results_['std_test_score'][rs.best_index_]

    if DEBUG_MODE and (util.get_base_model_name_from_pipeline(best_estimator) == "elasticnet" or 
                        util.get_base_model_name_from_pipeline(best_estimator) == "linearregression"):
        # In debug mode print top features from LR
        models.print_top_features_from_lr(best_estimator, X_train, 10)

    return (best_estimator, best_score, sd_of_score_of_best_estimator)

def find_best_estimator_for_output_and_its_score(X_train, y_train, performance_margin):
    base_models_and_param_grids = get_base_models_and_param_grids()
    best_estimators_and_scores = []
    
    for (base_model, grid) in base_models_and_param_grids:
        best_estimator_for_model, best_score_for_model, sd_of_score_of_best_estimator_for_model = get_best_estimator(base_model, grid, X_train, y_train)
        model_type = list(base_model.named_steps.keys())[-1]
        best_estimators_and_scores.append([model_type, best_estimator_for_model, best_score_for_model, sd_of_score_of_best_estimator_for_model])
    
    best_estimators_and_scores = pd.DataFrame(best_estimators_and_scores, columns = ["Model type", "Best estimator", "Best score", "SD of best score"])
    print(best_estimators_and_scores)
    best_estimator = best_estimators_and_scores.sort_values("Best score", ascending=False)["Best estimator"].iloc[0]
    best_score = best_estimators_and_scores[best_estimators_and_scores["Best estimator"] == best_estimator]["Best score"].iloc[0]
    sd_of_score_of_best_estimator = best_estimators_and_scores[best_estimators_and_scores["Best estimator"] == best_estimator]["SD of best score"].iloc[0]
    
    # If elasticnet is not much worse than the best model, prefer elasticnet (much faster than rest)
    best_base_model = best_estimators_and_scores[best_estimators_and_scores["Best estimator"] == best_estimator]["Model type"].iloc[0]
    if best_base_model != "elasticnet":
        lr_score = best_estimators_and_scores[best_estimators_and_scores["Model type"] == "elasticnet"]["Best score"].iloc[0]
        print("lr_score: ", lr_score, "; best_score: ", best_score)
        if best_score - lr_score <= performance_margin:
            best_estimator = best_estimators_and_scores[best_estimators_and_scores["Model type"] == "elasticnet"]["Best estimator"].iloc[0]
            best_score = best_estimators_and_scores[best_estimators_and_scores["Best estimator"] == best_estimator]["Best score"].iloc[0]
            sd_of_score_of_best_estimator = best_estimators_and_scores[best_estimators_and_scores["Best estimator"] == best_estimator]["SD of best score"].iloc[0]
        
    print("best estimator:")
    print(best_estimator)
    
    return best_estimator, best_score, sd_of_score_of_best_estimator

# Find best estimator
def find_best_estimators_and_scores(datasets, output_cols, performance_margin):
    best_estimators = {}
    scores_of_best_estimators = {}
    sds_of_scores_of_best_estimators = {}
    for i, output in enumerate(output_cols):
        print(output, f'{i+1}/{len(output_cols)}')

        X_train = datasets[output]["X_train_train"]
        y_train = datasets[output]["y_train_train"]
        
        best_estimator_for_output, best_score_for_output, sd_of_score_of_best_estimator_for_output = find_best_estimator_for_output_and_its_score(X_train, y_train, performance_margin)
        best_estimators[output] = best_estimator_for_output
        sds_of_scores_of_best_estimators[output] = sd_of_score_of_best_estimator_for_output
        scores_of_best_estimators[output] = best_score_for_output

        scores_of_best_estimators[output] = -best_score_for_output # Convert to positive score

        if DEBUG_MODE and util.get_base_model_name_from_pipeline(best_estimators[output]) == "elasticnet":
            # In debug mode print top features from LR
            models.print_top_features_from_lr(best_estimators[output], X_train, 10)
            
    return best_estimators, scores_of_best_estimators, sds_of_scores_of_best_estimators

def build_df_of_best_estimators_and_their_score_sds(best_estimators, scores_of_best_estimators, sds_of_scores_of_best_estimators):

    best_estimators_and_score_sds = []
    for output in best_estimators.keys():
        best_estimator = best_estimators[output]
        score_of_best_estimator = scores_of_best_estimators[output]
        sd_of_score_of_best_estimator = sds_of_scores_of_best_estimators[output]
        model_type = util.get_base_model_name_from_pipeline(best_estimator)
        best_estimators_and_score_sds.append([output, model_type, best_estimator, score_of_best_estimator, sd_of_score_of_best_estimator])
    best_estimators_and_score_sds = pd.DataFrame(best_estimators_and_score_sds, columns = ["Output", "Model type", "Best estimator", "Best score", "SD of best score"])
    best_estimators_and_score_sds["Score - SD"] = best_estimators_and_score_sds['Best score'] - best_estimators_and_score_sds['SD of best score'] 
    return best_estimators_and_score_sds

def dump_estimators_and_performances(dirs, best_estimators, scores_of_best_estimators, sds_of_scores_of_best_estimators):
    print(dirs["models_dir"])
    dump(best_estimators, dirs["models_dir"]+'best-estimators.joblib', compress=1)
    dump(scores_of_best_estimators, dirs["reports_dir"]+'scores-of-best-estimators.joblib', compress=1)
    dump(sds_of_scores_of_best_estimators, dirs["reports_dir"]+'sds-of-scores-of-best-estimators.joblib', compress=1)

def save_coefficients_of_lr_models(best_estimators, datasets, output_cols, output_dir):
    for output in output_cols:
        best_estimator = best_estimators[output]
        if util.get_base_model_name_from_pipeline(best_estimator) == "elasticnet":
            X_train = datasets[output]["X_train_train"]
            models.save_coefficients_from_lr(output, best_estimator, X_train, output_dir)

def main(performance_margin = 0.02, models_from_file = 1):
    models_from_file = int(models_from_file)
    performance_margin = float(performance_margin) # Margin of error for ROC AUC (for prefering linear models over other models)

    dirs = set_up_directories()
    load_dirs = set_up_load_directories()

    datasets = load(load_dirs["load_data_dir"]+'datasets.joblib')
    output_cols = list(datasets.keys())
    print("Train set shape: ", datasets[output_cols[0]]["X_train_train"].shape)

    if DEBUG_MODE:
        #output_cols = output_cols[0:2]
        pass

    if models_from_file == 1:
        
        best_estimators = load(load_dirs["load_models_dir"]+'best-estimators.joblib')
        scores_of_best_estimators = load(load_dirs["load_reports_dir"]+'scores-of-best-estimators.joblib')
        sds_of_scores_of_best_estimators = load(load_dirs["load_reports_dir"]+'sds-of-scores-of-best-estimators.joblib')

        dump_estimators_and_performances(dirs, best_estimators, scores_of_best_estimators, sds_of_scores_of_best_estimators)
    else: 
        # Find best models for each output
        best_estimators, scores_of_best_estimators, sds_of_scores_of_best_estimators = find_best_estimators_and_scores(datasets, output_cols, performance_margin)
        
        # Save best estimators 
        dump_estimators_and_performances(dirs, best_estimators, scores_of_best_estimators, sds_of_scores_of_best_estimators)
       
    # Build and save dataframe of best estimators and their scores
    df_of_best_estimators_and_their_score_sds = build_df_of_best_estimators_and_their_score_sds(best_estimators, scores_of_best_estimators, sds_of_scores_of_best_estimators)
    df_of_best_estimators_and_their_score_sds.to_csv(dirs["reports_dir"] + "df_of_best_estimators_and_their_scores.csv", float_format='%.3f')
    print(df_of_best_estimators_and_their_score_sds)

    # Save feature coefficients for logistic regression models
    save_coefficients_of_lr_models(best_estimators, datasets, output_cols, dirs["reports_dir"])

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])