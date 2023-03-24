from sklearn.model_selection import train_test_split 
import pandas as pd

def get_cols_from_output_assessments(all_cols, output_cols):
    cols_from_output_assessments = []
    for col in output_cols:
        assessment_name = col.split(",")[0]
        cols_from_output_assessments += [x for x in all_cols if assessment_name in x]

    return cols_from_output_assessments

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
        
        input_cols, output_col = get_input_and_output_cols_for_output(full_dataset, output, output_cols)
        
        # Split train, validation, and test sets
        # Create a new categorical variable by discretizing the continuous target variable (except recent grades, which is already only 5 values)
        full_dataset[f'{output_col}_binned'] = bin_continuous_var(full_dataset[output_col], 10) if output_col != "PreInt_EduHx,recent_grades" else full_dataset[output_col]
        X_train, X_test, y_train, y_test = train_test_split(full_dataset[input_cols], full_dataset[output_col], test_size=split_percentage, stratify=full_dataset[f'{output_col}_binned'], random_state=1)

        y_train_binned = bin_continuous_var(y_train, 10) if output_col != "PreInt_EduHx,recent_grades" else y_train
        X_train_train, X_val, y_train_train, y_val = train_test_split(X_train[input_cols], y_train, test_size=split_percentage, stratify=y_train_binned, random_state=1)
    
        datasets[output] = { "X_train": X_train,
                        "X_test": X_test,
                        "y_train": y_train,
                        "y_test": y_test,
                        "X_train_train": X_train_train,
                        "X_val": X_val,
                        "y_train_train": y_train_train,
                        "y_val": y_val}
    return datasets