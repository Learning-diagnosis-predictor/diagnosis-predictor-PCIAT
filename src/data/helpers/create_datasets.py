from sklearn.model_selection import train_test_split 
import pandas as pd

def get_cols_from_output_assessments(all_cols, output_cols):
    cols_from_output_assessments = []
    for col in output_cols:
        assessment_name = col.split(",")[0]
        cols_from_output_assessments += [x for x in all_cols if assessment_name in x]

    return cols_from_output_assessments

def make_new_output_cols(full_dataset, output_col): ####
    # IAt and PCIAT scoring
    # 0-30 Normal
    # 31-49 Mild
    # 50-79 Moderate
    # 80-100 Severe
    if output_col in ["PCIAT,PCIAT_Total", "IAT,IAT_Total"]:
        
        full_dataset["OUTPUT."+output_col] = full_dataset[output_col].apply(lambda x: 0 if x <= 30 else 1 if x <= 49 else 2 if x <= 79 else 3)
        print(full_dataset["OUTPUT."+output_col].value_counts())
        print(full_dataset[[output_col, "OUTPUT."+output_col]])
    elif output_col in ["CIS_P,CIS_P_Score", "CIS_SR,CIS_SR_Total"]:
        # Mild: 11-14, Moderate: 15-19, Severe: 20+
        full_dataset["OUTPUT."+output_col] = full_dataset[output_col].apply(lambda x: 0 if x <= 14 else 1 if x <= 19 else 2)
    else:
        full_dataset["OUTPUT."+output_col] = full_dataset[output_col]
    return full_dataset

def get_input_and_output_cols_for_output(full_dataset, output, output_cols):

    cols_from_output_assessments = get_cols_from_output_assessments(full_dataset.columns, output_cols)
    
    input_cols = [x for x in full_dataset.columns if 
                    not x in cols_from_output_assessments
                    and not x.startswith("Diag.")
                    and not x == "Diag.No Diagnosis Given"]
    
    output_col = output
    
    return input_cols, output_col

def bin_continuous_var(col, n_bins):
    return pd.qcut(col, q=n_bins, labels=False, duplicates='drop')

def create_datasets(full_dataset, output_cols, split_percentage):
    datasets = {}
    for output in output_cols:

        full_dataset = make_new_output_cols(full_dataset, output) ####
        
        input_cols, output_col = get_input_and_output_cols_for_output(full_dataset, output, output_cols)
        
        # Split train, validation, and test sets
        # Create a new categorical variable by discretizing the continuous target variable (except recent grades, which is already only 5 values)
        # full_dataset[f'{output_col}_binned'] = bin_continuous_var(full_dataset[output_col], 10) if output_col != "PreInt_EduHx,recent_grades" else full_dataset[output_col]
        # X_train, X_test, y_train, y_test = train_test_split(full_dataset[input_cols], full_dataset[output_col], test_size=split_percentage, stratify=full_dataset[f'{output_col}_binned'], random_state=1)

        # y_train_binned = bin_continuous_var(y_train, 10) if output_col != "PreInt_EduHx,recent_grades" else y_train
        # X_train_train, X_val, y_train_train, y_val = train_test_split(X_train[input_cols], y_train, test_size=split_percentage, stratify=y_train_binned, random_state=1)
    
        #####
        output_col = "OUTPUT."+output_col
        X_train, X_test, y_train, y_test = train_test_split(full_dataset[input_cols], full_dataset[output_col], test_size=split_percentage, stratify=full_dataset[output_col], random_state=1)
        X_train_train, X_val, y_train_train, y_val = train_test_split(X_train[input_cols], y_train, test_size=split_percentage, stratify=y_train, random_state=1)

        print(output_col, ": ")
        print(f"Train set: {X_train_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Train set: {y_train_train.shape}")
        print(f"Validation set: {y_val.shape}")
        print(f"Test set: {y_test.shape}")
        print(y_train_train.value_counts())
        print(y_val.value_counts())
        print(y_test.value_counts())
    
        #####
        datasets[output] = { "X_train": X_train,
                        "X_test": X_test,
                        "y_train": y_train,
                        "y_test": y_test,
                        "X_train_train": X_train_train,
                        "X_val": X_val,
                        "y_train_train": y_train_train,
                        "y_val": y_val}
    return datasets