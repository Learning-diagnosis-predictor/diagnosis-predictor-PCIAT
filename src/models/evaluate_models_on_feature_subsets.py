import os, sys, inspect
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning" # Seems to be the only way to suppress multi-thread sklearn warnings

# Colorful error messages
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Neutral', call_pdb=False)

import pandas as pd

from joblib import dump, load

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import models, util

DEBUG_MODE = False

def build_output_dir_name(params_from_previous_script):
    # Part with the datetimeS
    datetime_part = util.get_string_with_current_datetime()

    return datetime_part + "___" + util.build_param_string_for_dir_name(params_from_previous_script)

def set_up_directories():

    data_dir = "../diagnosis_predictor_PCIAT_data/"

    # Input dirs
    input_data_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "data/create_datasets/")
    print("Reading data from: " + input_data_dir)
    input_models_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "models/train_models/")
    print("Reading models from: " + input_models_dir)
    input_reports_dir = models.get_newest_non_empty_dir_in_dir(data_dir+ "reports/identify_feature_subsets/")
    print("Reading reports from: " + input_reports_dir)
    
    # Output dirs
    params_from_previous_script = models.get_params_from_current_data_dir_name(input_data_dir)
    current_output_dir_name = build_output_dir_name(params_from_previous_script)
    
    output_models_dir = data_dir + "models/" + "evaluate_models_on_feature_subsets/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(output_models_dir)

    output_reports_dir = data_dir + "reports/" + "evaluate_models_on_feature_subsets/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(output_reports_dir)

    return {"input_data_dir": input_data_dir,  "input_models_dir": input_models_dir, "output_models_dir": output_models_dir, "input_reports_dir": input_reports_dir, "output_reports_dir": output_reports_dir}

def set_up_load_directories():
    data_dir = "../diagnosis_predictor_PCIAT_data/"
    load_reports_dir = models.get_newest_non_empty_dir_in_dir(data_dir+ "reports/evaluate_models_on_feature_subsets/")
    load_models_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "models/" + "evaluate_models_on_feature_subsets/")
    return {"load_reports_dir": load_reports_dir, "load_models_dir": load_models_dir}

def make_and_write_cv_r2_table(r2_on_subsets, dir):
    r2_on_subsets = pd.DataFrame.from_dict(r2_on_subsets)
    r2_on_subsets.index = range(1, len(r2_on_subsets)+1)
    r2_on_subsets = r2_on_subsets.rename(columns={"index": "Output"})

    r2_on_subsets.to_csv(dir+'cv-r2-on-subsets.csv')

    return r2_on_subsets

def make_performance_table(performances_on_feature_subsets):
    # Create a table for r2 for each number of features
    # Build list of lists where each list is a row in the table
    r2_table = []
    
    for output in performances_on_feature_subsets:
        # Each row in the table will have the output and then each columns will be the performance for each number of features
        output_row_r2 = [output] +  list(performances_on_feature_subsets[output].values())
        r2_table.append(output_row_r2)
    
    r2_df = pd.DataFrame(r2_table, columns=["Output"] + list(performances_on_feature_subsets[output].keys()))
    
    # Sort outputnoses by performance on max number of features (sort values by last column)
    r2_df = r2_df.sort_values(by=r2_df.columns[-1], ascending=False)
    
    # Transpose so that each column is a output and each row is a number of features
    r2_df = r2_df.transpose()
    
    # Rename columns and index
    r2_df.columns = r2_df.iloc[0]
    r2_df = r2_df.drop(r2_df.index[0])
    r2_df.index.name = "Number of features"

    # Reverse order order of rows so max number of features is at the top
    r2_df = r2_df.iloc[::-1]

    return r2_df

def get_and_write_optimal_nbs_features(r2_table, dir):
    optimal_nbs_features = {}

    for diag in auc_table.columns:
        # Get max score at number of features in the longest subcsale among those that perform best for each diag (from HBN-scripts repo)
        max_score = auc_table[diag].iloc[0:27].max() 
    for output in r2_table.columns:
        max_score = r2_table[output].max()
        optimal_score = max_score - 0.01
        # Get index of the first row with a score >= optimal_score
        optimal_nbs_features[output] = r2_table[output][r2_table[output] >= optimal_score].index[0]

    print(optimal_nbs_features)
    util.write_dict_to_file(optimal_nbs_features, dir, "optimal-nb-features.txt")

    return optimal_nbs_features

def make_and_write_test_set_performance_table(performances_on_feature_subsets, dir, optimal_nbs_features):
    r2_test_set_table = make_performance_table(performances_on_feature_subsets)
    r2_test_set_table.to_csv(dir+'r2-on-subsets-test-set.csv')

def main(models_from_file = 1):
    models_from_file = int(models_from_file)

    dirs = set_up_directories()

    feature_subsets = load(dirs["input_reports_dir"]+'feature-subsets.joblib')
    datasets = load(dirs["input_data_dir"]+'datasets.joblib')
    best_estimators = load(dirs["input_models_dir"]+'best-estimators.joblib')

    if DEBUG_MODE == True:
        # In debug mode, only use first output
        datasets = {list(datasets.keys())[0]: datasets[list(datasets.keys())[0]]}
        feature_subsets = {list(feature_subsets.keys())[0]: feature_subsets[list(feature_subsets.keys())[0]]}
        best_estimators = {list(best_estimators.keys())[0]: best_estimators[list(best_estimators.keys())[0]]}

    if models_from_file == 1:
        load_dirs = set_up_load_directories()

        performances_on_feature_subsets = load(load_dirs["load_reports_dir"]+'performances-on-feature-subsets.joblib')    
        cv_scores_on_feature_subsets = load(load_dirs["load_reports_dir"]+'cv-scores-on-feature-subsets.joblib')

        # Save reports to newly created directories
        dump(performances_on_feature_subsets, dirs["output_reports_dir"]+'performances-on-feature-subsets.joblib')
        dump(cv_scores_on_feature_subsets, dirs["output_reports_dir"]+'cv-scores-on-feature-subsets.joblib')
    else:
        performances_on_feature_subsets, cv_scores_on_feature_subsets = models.get_performances_on_feature_subsets(feature_subsets, datasets, best_estimators, use_test_set = 1)
        dump(performances_on_feature_subsets, dirs["output_reports_dir"]+'performances-on-feature-subsets.joblib')
        dump(cv_scores_on_feature_subsets, dirs["output_reports_dir"]+'cv-scores-on-feature-subsets.joblib')
    
    cv_r2_table = make_and_write_cv_r2_table(cv_scores_on_feature_subsets, dirs["output_reports_dir"])
    optimal_nbs_features = get_and_write_optimal_nbs_features(cv_r2_table, dirs["output_reports_dir"])
    make_and_write_test_set_performance_table(performances_on_feature_subsets, dirs["output_reports_dir"], optimal_nbs_features)

if __name__ == "__main__":
    main(sys.argv[1])