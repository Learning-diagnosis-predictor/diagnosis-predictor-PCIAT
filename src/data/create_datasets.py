import pandas as pd

from sklearn.model_selection import train_test_split

from joblib import dump
import sys, os, inspect

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util, data, features

def build_output_dir_name(first_assessment_to_drop, use_other_outputs_as_input, only_free_assessments):
    # Part with the datetime
    datetime_part = util.get_string_with_current_datetime()

    # Part with the params
    params = {"first_assessment_to_drop": first_assessment_to_drop, "use_other_outputs_as_input": use_other_outputs_as_input, 
              "only_free_assessments": only_free_assessments}
    params_part = util.build_param_string_for_dir_name(params)
    
    return datetime_part + "___" + params_part

def set_up_directories(first_assessment_to_drop, use_other_diags_as_input, only_free_assessments):

    # Create directory in the parent directory of the project (separate repo) for output data, models, and reports
    data_dir = "../diagnosis_predictor_PCIAT_data/"
    util.create_dir_if_not_exists(data_dir)

    # Create directory inside the output directory with the run timestamp and first_assessment_to_drop param
    current_output_dir_name = build_output_dir_name(first_assessment_to_drop, use_other_diags_as_input, only_free_assessments)

    data_statistics_dir = data_dir + "reports/create_datasets/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(data_statistics_dir)
    util.create_dir_if_not_exists(data_statistics_dir+"figures/")

    data_output_dir = data_dir + "data/create_datasets/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(data_output_dir)

    return {"data_statistics_dir": data_statistics_dir, "data_output_dir": data_output_dir}

def get_cols_from_output_assessments(all_cols, output_cols):
    cols_from_output_assessments = []
    for col in output_cols:
        assessment_name = col.split(",")[0]
        cols_from_output_assessments += [x for x in all_cols if assessment_name in x]

    return cols_from_output_assessments

def customize_input_cols_per_output(input_cols, output):
    
    if output == "WISC,WISC_PSI": # PSI used for FIQ calculation
        input_cols = [x for x in input_cols if x != "Diag.Borderline Intellectual Functioning" and x!= "Diag.Intellectual Disability-Mild"]
                      
    return input_cols

def get_input_cols_per_output(full_dataset, output, output_cols, use_other_outputs_as_input):

    cols_from_output_assessments = get_cols_from_output_assessments(full_dataset.columns, output_cols)
    
    if use_other_outputs_as_input == 1:
        input_cols = [x for x in full_dataset.columns if 
                            not x in cols_from_output_assessments
                            and not x == output]
    else:
        input_cols = [x for x in full_dataset.columns if 
                            not x in cols_from_output_assessments
                            and not x == output
                            and not x.startswith("Diag.")]
    
    input_cols = customize_input_cols_per_output(input_cols, output)
    print("Input assessemnts used: ", list(set([x.split(",")[0] for x in input_cols])))
    
    return input_cols

def bin_continuous_var(col, n_bins):
    return pd.qcut(col, q=n_bins, labels=False, duplicates='drop')

def split_datasets_per_output(full_dataset, output_cols, split_percentage, use_other_diags_as_input):
    datasets = {}
    for output in output_cols:
        
        output_col = output

        # Drop columns from input that we don't want there
        input_cols = get_input_cols_per_output(full_dataset, output, output_cols, use_other_diags_as_input)

        # Split train, validation, and test sets, stratify by the binned target variable
        # Create a new categorical variable by discretizing the continuous target variable (except recent grades, which is already only 5 values)
        full_dataset[f'{output_col}_binned'] = bin_continuous_var(full_dataset[output_col], 10) if output_col != "PreInt_EduHx,recent_grades" else full_dataset[output_col]
        X_train, X_test, y_train, y_test = train_test_split(full_dataset[input_cols], full_dataset[output_col], test_size=split_percentage, stratify=full_dataset[f'{output_col}_binned'], random_state=1)

        y_train_binned = bin_continuous_var(y_train, 10) if output_col != "PreInt_EduHx,recent_grades" else y_train
        X_train_train, X_val, y_train_train, y_val = train_test_split(X_train[input_cols], y_train, test_size=split_percentage, stratify=y_train_binned, random_state=1)
            
        X_train = X_train[input_cols]
        X_test = X_test[input_cols]
        X_train_train = X_train_train[input_cols]
        X_val = X_val[input_cols]
        
        datasets[output] = { "X_train": X_train,
                        "X_test": X_test,
                        "y_train": y_train,
                        "y_test": y_test,
                        "X_train_train": X_train_train,
                        "X_val": X_val,
                        "y_train_train": y_train_train,
                        "y_val": y_val,
                    }
    return datasets

def get_1sd_examples_in_ds(full_dataset, output_cols):
    one_sd_examples_in_ds = {}
    # Check number of examples in -1sd and +1sd for bidirectional outputs and +1sd for unidirectional outputs

    for output_col in output_cols:

        mean = full_dataset[output_col].mean()
        std = full_dataset[output_col].std()

        one_sd = mean + std
        indices_plus_one_sd = full_dataset[full_dataset[output_col] >= one_sd].index
        num_plus_one_sd = len(indices_plus_one_sd)

        num_minus_one_sd = 0
        if "WISC" in output_col:
            minus_one_sd = mean - std
            indices_minus_one_sd = full_dataset[full_dataset[output_col] <= minus_one_sd].index
            num_minus_one_sd = len(indices_minus_one_sd)

        print(output_col, " - Mean: ", mean, "Std: ", std, "1sd: ", one_sd, "Num -1sd: ", num_minus_one_sd, "Num +1sd: ", num_plus_one_sd)

        one_sd_examples_in_ds[output_col] = num_minus_one_sd + num_plus_one_sd
    
    return one_sd_examples_in_ds
    
def find_outputs_w_enough_1sd_examples_in_val_set(one_sd_examples_in_ds, output_cols, split_percentage, min_1sd_examples_val_set):
    outputs_w_enough_1sd_examples_in_val_set = []
    for output in output_cols:
        one_sd_examples_val_set = one_sd_examples_in_ds[output] * (1-split_percentage) * split_percentage 
        if one_sd_examples_val_set >= min_1sd_examples_val_set:
            outputs_w_enough_1sd_examples_in_val_set.append(output)
    return outputs_w_enough_1sd_examples_in_val_set

def save_dataset_stats(datasets, output_cols, full_dataset, dir):
    stats = {}
    stats["n_rows_full_ds"] = full_dataset.shape[0]
    stats["n_rows_train_ds"] = datasets[output_cols[0]]["X_train_train"].shape[0]
    stats["n_rows_val_ds"] = datasets[output_cols[0]]["X_val"].shape[0]
    stats["n_rows_test_ds"] = datasets[output_cols[0]]["X_test"].shape[0]
    stats["n_input_cols"] = datasets[output_cols[0]]["X_train_train"].shape[1] - len(output_cols)
    # To df
    stats_df = pd.DataFrame.from_dict(stats, orient="index")
    stats_df.columns = ["Value"]
    stats_df.to_csv(dir + "dataset_stats.csv")

def main(only_assessment_distribution, first_assessment_to_drop, use_other_diags_as_input, only_free_assessments):
    only_assessment_distribution = int(only_assessment_distribution)
    use_other_diags_as_input = int(use_other_diags_as_input)
    only_free_assessments = int(only_free_assessments)

    dirs = set_up_directories(first_assessment_to_drop, use_other_diags_as_input, only_free_assessments)

    output_cols = ["PCIAT,PCIAT_Total", 
                   "IAT,IAT_Total", 
                   "PreInt_EduHx,recent_grades", 
                   "WHODAS_P,WHODAS_P_Total", 
                   "WHODAS_SR,WHODAS_SR_Score", 
                   "CIS_P,CIS_P_Score", 
                   "CIS_SR,CIS_SR_Total", 
                   "WISC,WISC_PSI"]

    data.make_full_dataset(only_assessment_distribution, first_assessment_to_drop, only_free_assessments, dirs, output_cols)

    if only_assessment_distribution == 0:
        full_dataset = pd.read_csv(dirs["data_output_dir"] + "item_lvl.csv")
        #full_dataset = features.make_new_output_cols(full_dataset) # Not making any new features for now

        # Print dataset shape
        print("Full dataset shape: Number of rows: ", full_dataset.shape[0], "Number of columns: ", full_dataset.shape[1])

        # Get list of column names with "Diag." prefix, where number of 
        # positive examples is > threshold
        min_1sd_ex_in_val_set = 20
        split_percentage = 0.2
        one_sd_examples_in_ds = get_1sd_examples_in_ds(full_dataset, output_cols)
        
        output_cols = find_outputs_w_enough_1sd_examples_in_val_set(one_sd_examples_in_ds, output_cols, split_percentage, min_1sd_ex_in_val_set)

        # Create datasets for each output (different input and output columns)
        datasets = split_datasets_per_output(full_dataset, output_cols, split_percentage, use_other_diags_as_input)

        save_dataset_stats(datasets, output_cols, full_dataset, dirs["data_statistics_dir"])
            
        dump(datasets, dirs["data_output_dir"]+'datasets.joblib', compress=1)

        # Save number of positive examples for each output to csv (convert dict to df)
        one_sd_examples_col_name = f"Examples outside 1sd out of {full_dataset.shape[0]}"
        pd.DataFrame(one_sd_examples_in_ds.items(), columns=["Output", one_sd_examples_col_name]).sort_values(one_sd_examples_col_name, ascending=False).to_csv(dirs["data_statistics_dir"]+"number-of-1sd-examples.csv")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])