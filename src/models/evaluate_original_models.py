import os, inspect
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning" # Seems to be the only way to suppress multi-thread sklearn warnings

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error

import sys

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util, models, data

def build_output_dir_name(params_from_train_models, params_from_evaluate_original_models):
    # Part with the datetime
    datetime_part = util.get_string_with_current_datetime()

    return datetime_part + "___" + util.build_param_string_for_dir_name(params_from_train_models) + "___" +\
        util.build_param_string_for_dir_name(params_from_evaluate_original_models)

def set_up_directories(use_test_set):

    data_dir = "../diagnosis_predictor_PCIAT_data/"

    # Input dirs
    input_data_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "data/create_datasets/")
    models_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "models/train_models/")
    input_reports_dir = models.get_newest_non_empty_dir_in_dir(data_dir+ "reports/train_models/")

    # Output dirs

    # Create directory inside the output directory with the run timestamp and params:
    #    - [params from train_models.py]
    #    - use test set
    params_from_train_models = models.get_params_from_current_data_dir_name(input_data_dir)
    params_from_current_file = {"use_test_set": use_test_set}
    current_output_dir_name = build_output_dir_name(params_from_train_models, params_from_current_file)

    output_reports_dir = data_dir + "reports/evaluate_original_models/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(output_reports_dir)

    return {"input_data_dir": input_data_dir, "models_dir": models_dir, "input_reports_dir": input_reports_dir, "output_reports_dir": output_reports_dir}

def add_output_col_values_range(results, datasets):
    for output in datasets:
        full_dataset_y = datasets[output]["y_train"].append(datasets[output]["y_test"]) # Reconstruct full dataset from train and test
        results.loc[results["Output"] == output, "Min value"] = min(full_dataset_y)
        results.loc[results["Output"] == output, "Max value"] = max(full_dataset_y)
    return results

def get_mae(X, y, estimator):
    y_pred = estimator.predict(X)
    mae = mean_absolute_error(y, y_pred)
    return mae

def get_maes_on_test_set(best_estimators, datasets, use_test_set, output_cols):
    maes = {}
    for output in output_cols:
        print(output)
        print(util.get_base_model_name_from_pipeline(best_estimators[output]))
        estimator = best_estimators[output]
        
        if use_test_set == 1:
            X, y = datasets[output]["X_test"], datasets[output]["y_test"]
        else:
            X, y = datasets[output]["X_val"], datasets[output]["y_val"]

        mae = get_mae(X, y, estimator)
        maes[output] = mae

    # Example of scores: {'Output1': 0.5, 'Output2': 0.6, 'Output3': 0.7}
    # Convert to a dataframe
    results = pd.DataFrame.from_dict(maes, columns=["MAE"], orient="index").sort_values("MAE", ascending=False).reset_index().rename(columns={'index': 'Output'})
    print(results)
    results = add_output_col_values_range(results, datasets)

    return results

def get_maes_cv_from_grid_search(reports_dir, output_cols):
    auc_cv_from_grid_search = pd.read_csv(reports_dir + "df_of_best_estimators_and_their_scores.csv")
    auc_cv_from_grid_search = auc_cv_from_grid_search[auc_cv_from_grid_search["Output"].isin(output_cols)][["Output", "Best score", "SD of best score", "Score - SD"]]
    auc_cv_from_grid_search.columns = ["Output", "MAE Mean CV", "MAE SD CV", "MAE Mean CV - SD"]
    return auc_cv_from_grid_search

def get_maes(best_estimators, datasets, use_test_set, output_cols, input_reports_dir):
    maes_cv_from_grid_search = get_maes_cv_from_grid_search(input_reports_dir, output_cols)
    maes_on_test_set = get_maes_on_test_set(best_estimators, datasets, use_test_set=use_test_set, output_cols=output_cols)
    maes = maes_cv_from_grid_search.merge(maes_on_test_set, on="Output").sort_values(by="MAE Mean CV - SD", ascending=False)
    return maes

def main(use_test_set=1):
    use_test_set = int(use_test_set)

    dirs = set_up_directories(use_test_set)

    from joblib import load
    best_estimators = load(dirs["models_dir"]+'best-estimators.joblib')
    
    output_cols = best_estimators.keys()

    datasets = load(dirs["input_data_dir"]+'datasets.joblib')

    # Print performances of models on validation set
    maes = get_maes(best_estimators, datasets, use_test_set=use_test_set, output_cols=output_cols, input_reports_dir=dirs["input_reports_dir"])

    if use_test_set == 1:
        maes.to_csv(dirs["output_reports_dir"]+"performance_table_all_features.csv", index=False)    

if __name__ == "__main__":
    main(sys.argv[1])