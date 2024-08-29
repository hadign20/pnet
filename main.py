import os
import pandas as pd
import numpy as np
from src.feature_selection.correlation import remove_collinear_features
from src.feature_selection.feature_selection import *
from src.model.model_building import evaluate_models, save_classification_results
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from src.visualization.plotting import *
import joblib

from src.model.model_building import compute_metrics, compute_confidence_interval


#=========================================
# set paths
#=========================================
data_path = r'D:\projects\PNET Radiomics\data'
result_path = r'./results'
img_data_path = os.path.join(data_path, "scans")
excel_file_names = ["JayaFeaturesTrainPrepared"]
excel_file_names_val = ["JayaFeaturesTestPrepared"]
SELECTED_SHEET = [] # "[] for all sheets",  ["3_1"]
SELECTED_SHEET_VAL = [] # "[] for all sheets",  ["3_1"]
outcome_column = "Grade"
exclude_columns = ["Patient_ID"]
exclude_columns_val = ["Case_ID"]
categorical_columns = []


#=========================================
# set parameters
#=========================================
FEATURE_CORRELATION = True
CORR_THRESH = 0.8

FEATURE_SELECTION = True
FEATURE_SELECTION_METHOD = 'mrmr' # 'mrmr', 'pvalue', 'auc', 'composite'
min_num_features = 1
max_num_features = 20

MODEL_BUILDING = True
RESAMPLING = False
RESAMPLING_METHOD = "SMOTEENN" # "RandomOverSampler" or "SMOTEENN"
EVALUATION_METHOD = 'cross_validation' # 'train_test_split' or 'cross_validation' or 'cv_feature_selection_model_building'
TEST_SIZE = 0.3
CV_FOLDS = 5
HYPERPARAMETER_TUNING = True

EXTERNAL_VALIDATION = True

#=========================================
def save_excel_sheet(df, filepath, sheetname, index=False):
    # Create file if it does not exist
    if not os.path.exists(filepath):
        df.to_excel(filepath, sheet_name=sheetname, index=index)

    # Otherwise, add a sheet. Overwrite if there exists one with the same name.
    else:
        with pd.ExcelWriter(filepath, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer:
            df.to_excel(writer, sheet_name=sheetname, index=index)

def save_summary_results(classification_results, evaluation_method, results_dir, summary_results, sheet, num_features):

    if evaluation_method == "cross_validation":
        for classifier, result in classification_results.items():
            result_entry = {
                'Sheet': sheet,
                'Num Features': num_features,
                'Classifier': classifier,
                'AUC': result['metrics']['roc_auc'],
                'Sensitivity': result['metrics']['sensitivity'],
                'Specificity': result['metrics']['specificity'],
                'PPV': result['metrics']['ppv'],
                'NPV': result['metrics']['npv']
            }
            summary_results.append(result_entry)

    elif evaluation_method == "train_test_split":
        for dataset in classification_results.items():
            if dataset[0] == "test":
                for classifier, result in dataset[1].items():
                    result_entry = {
                        'Sheet': sheet,
                        'Num Features': num_features,
                        'Classifier': classifier,
                        'AUC': result['metrics']['roc_auc'],
                        'Sensitivity': result['metrics']['sensitivity'],
                        'Specificity': result['metrics']['specificity'],
                        'PPV': result['metrics']['ppv'],
                        'NPV': result['metrics']['npv']
                    }
                    summary_results.append(result_entry)

    # Save summary results
    summary_df = pd.DataFrame(summary_results)
    summary_file = os.path.join(results_dir, 'summary_results.xlsx')
    with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
        for sheet_name in summary_df['Sheet'].unique():
            sheet_df = summary_df[summary_df['Sheet'] == sheet_name]
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Sort all results by AUC and save to "Best Result" sheet
        best_df = summary_df.sort_values(by='AUC', ascending=False)
        best_df.to_excel(writer, sheet_name='Best Result', index=False)



#=========================================


def main():
    for excel_file_name in excel_file_names:
        features_file = os.path.join(data_path, excel_file_name + ".xlsx")
        results_dir = os.path.join(result_path, excel_file_name)
        os.makedirs(results_dir, exist_ok=True)
        xls = pd.ExcelFile(features_file)
        summary_results = []
        summary_results_val = []

        selected_sheets = xls.sheet_names if len(SELECTED_SHEET) == 0 else SELECTED_SHEET

        for sheet in selected_sheets:
            result_dir = os.path.join(results_dir, sheet)
            os.makedirs(result_dir, exist_ok=True)

            df = pd.read_excel(xls, sheet_name=sheet)
            df = df.fillna(0)

            # =========================================
            # Feature selection
            # =========================================
            if FEATURE_CORRELATION:
                # # select significant features first
                # p_values_df = calculate_p_values(df, outcome_column, categorical_columns, exclude_columns)
                # p_values_df = p_values_df[p_values_df['P_Value'] <= 0.05]
                # selected_features = p_values_df['Feature'].tolist()
                # df = df[exclude_columns + selected_features + [outcome_column]]


                print("\n======================================================================")
                print(f"Removing correlated features for sheet {sheet}")
                print("======================================================================")
                df = remove_collinear_features(df, CORR_THRESH)

            if FEATURE_SELECTION:
                print("\n======================================================================")
                print(f"Performing feature analysis for sheet {sheet}")
                print("======================================================================")
                p_values_df = calculate_p_values(df, outcome_column, categorical_columns, exclude_columns)
                auc_values_df = calculate_auc_values_CV(df, outcome_column, categorical_columns, exclude_columns)
                mrmr_df = MRMR_feature_count(df, outcome_column, categorical_columns, exclude_columns,
                                             max_num_features, CV_FOLDS)
                # lasso_df = lasso_feature_selection(df, outcome_column, exclude_columns)
                # composite_df = calculate_feature_scores(p_values_df, auc_values_df, mrmr_df, lasso_df, result_dir)
                composite_df = calculate_feature_scores(p_values_df, auc_values_df, mrmr_df, result_dir)

                output_file = os.path.join(results_dir, f'mrmr_features.xlsx')
                save_excel_sheet(mrmr_df, output_file, sheetname=sheet)

                save_feature_analysis(p_values_df, auc_values_df, mrmr_df, composite_df, result_dir)

                df_copy = df.copy()
                # global max_num_features
                # max_num_features = min(max_num_features, len(df.columns) - 2)
                # print("max_num_features: ", max_num_features)

                for num_features in range(min_num_features, max_num_features + 1):
                    print("\n======================================================================")
                    print(f"Selecting {num_features} significant features for sheet {sheet}")
                    print("======================================================================")

                    selected_features = []
                    if FEATURE_SELECTION_METHOD == 'mrmr':
                        selected_features = mrmr_df['Feature'][:num_features].tolist()
                        print(f"{num_features} features were selected by using MRMR method")
                    elif FEATURE_SELECTION_METHOD == 'pvalue':
                        selected_features = p_values_df['Feature'][:num_features].tolist()
                        print(f"{num_features} features were selected by using pvalue method")
                    elif FEATURE_SELECTION_METHOD == 'auc':
                        selected_features = auc_values_df['Feature'][:num_features].tolist()
                        print(f"{num_features} features were selected by using auc method")
                    # elif FEATURE_SELECTION_METHOD == 'lasso':
                    #     selected_features = lasso_df['Feature'][:num_features].tolist()
                    #     print(f"{num_features} features were selected by using lasso method")
                    elif FEATURE_SELECTION_METHOD == 'composite':
                        selected_features = composite_df['Feature'][:num_features].tolist()
                        print(f"{num_features} features were selected by a composite of p_value, AUC, and MRMR method")
                    else:
                        raise ValueError(
                            "FEATURE_SELECTION_METHOD is not correct. It should be 'mrmr', 'pvalue', 'auc', or 'composite'")

                    df = df_copy[exclude_columns + selected_features + [outcome_column]]

                    # =========================================
                    # Model building and evaluation
                    # =========================================
                    if MODEL_BUILDING:
                        resampling_method_ra = None
                        if RESAMPLING:
                            if RESAMPLING_METHOD == "RandomOverSampler":
                                resampling_method_ra = RandomOverSampler(random_state=42)
                            elif RESAMPLING_METHOD == "SMOTEENN":
                                resampling_method_ra = SMOTEENN(random_state=42)

                        eval_kwargs = None
                        if EVALUATION_METHOD == 'train_test_split':
                            eval_kwargs = {'test_size': TEST_SIZE,
                                           'random_state': 42,
                                           'result_path': result_dir,
                                           'num_features': num_features,
                                           'tuning': HYPERPARAMETER_TUNING,
                                           'resampling_method': resampling_method_ra}
                        elif EVALUATION_METHOD == 'cross_validation':
                            eval_kwargs = {'cv_folds': CV_FOLDS,
                                            'result_path': result_dir,
                                            'num_features': num_features,
                                            'tuning': HYPERPARAMETER_TUNING,
                                            'resampling_method': resampling_method_ra}
                        else:
                            eval_kwargs = {'cv_folds': CV_FOLDS,
                                           'result_path': result_dir,
                                           'num_features': num_features,
                                           'tuning': HYPERPARAMETER_TUNING,
                                           'resampling_method': resampling_method_ra}



                        print("\n======================================================================")
                        print(f"Training and evaluating classification models for {num_features} feature(s) in sheet {sheet}")
                        print("======================================================================")
                        if EVALUATION_METHOD == 'cv_feature_selection_model_building':
                            df = df_copy
                        X = df.loc[:, ~df.columns.isin(exclude_columns + [outcome_column])]
                        y = df[outcome_column]

                        classification_results = evaluate_models(X, y, method=EVALUATION_METHOD, **eval_kwargs)

                        classification_results_file = os.path.join(result_dir, 'model_evaluation_results.xlsx')
                        save_classification_results(classification_results, classification_results_file, num_features, method=EVALUATION_METHOD)

                        if EVALUATION_METHOD == "cross_validation" or EVALUATION_METHOD == 'cv_feature_selection_model_building':
                            for classifier, result in classification_results.items():
                                result_entry = {
                                    'Sheet': sheet,
                                    'Num Features': num_features,
                                    'Classifier': classifier,
                                    'AUC': result['metrics']['roc_auc'],
                                    'Sensitivity': result['metrics']['sensitivity'],
                                    'Specificity': result['metrics']['specificity'],
                                    'PPV': result['metrics']['ppv'],
                                    'NPV': result['metrics']['npv']
                                }
                                summary_results.append(result_entry)
                        elif EVALUATION_METHOD == "train_test_split":
                            for dataset in classification_results.items():
                                if dataset[0] == "test":
                                    for classifier, result in dataset[1].items():
                                        result_entry = {
                                            'Sheet': sheet,
                                            'Num Features': num_features,
                                            'Classifier': classifier,
                                            'AUC': result['metrics']['roc_auc'],
                                            'Sensitivity': result['metrics']['sensitivity'],
                                            'Specificity': result['metrics']['specificity'],
                                            'PPV': result['metrics']['ppv'],
                                            'NPV': result['metrics']['npv']
                                        }
                                        summary_results.append(result_entry)


                    # Save summary results
                    summary_df = pd.DataFrame(summary_results)
                    summary_file = os.path.join(results_dir, 'summary_results.xlsx')
                    with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
                        for sheet_name in summary_df['Sheet'].unique():
                            sheet_df = summary_df[summary_df['Sheet'] == sheet_name]
                            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

                        # Sort all results by AUC and save to "Best Result" sheet
                        best_df = summary_df.sort_values(by='AUC', ascending=False)
                        best_df.to_excel(writer, sheet_name='Best Result', index=False)


                    if EXTERNAL_VALIDATION:
                        roc_data_path_val = os.path.join(result_path, excel_file_name, sheet, "ROC_data_val")
                        if not os.path.exists(roc_data_path_val):
                            os.makedirs(roc_data_path_val)
                        for excel_file_name_val in excel_file_names_val:
                            features_file_val = os.path.join(data_path, excel_file_name_val + ".xlsx")
                            xls_val = pd.ExcelFile(features_file_val)
                            selected_sheets_val = xls_val.sheet_names if len(SELECTED_SHEET_VAL) == 0 else SELECTED_SHEET_VAL

                            out_dict = {}
                            for sh in selected_sheets_val:
                                rows = []
                                validation_df = pd.read_excel(features_file_val, sheet_name=sh)
                                validation_df = validation_df.fillna(0)

                                models = ['RandomForest', 'SVM', 'LogisticRegression', 'NaiveBayes']
                                for model_name in models:
                                    model_file = os.path.join(result_path, excel_file_name, sheet, "Saved_Models",
                                                              model_name + '_' + str(num_features) + '_features.pkl')

                                    model = joblib.load(model_file)
                                    f_names = model.feature_names

                                    X_validation = validation_df[f_names]
                                    y_validation = validation_df[outcome_column]

                                    predicted_probs = model.predict_proba(X_validation[f_names])[:, 1]

                                    # Compute TPR and FPR for ROC curve
                                    fpr, tpr, thresholds = roc_curve(y_validation, predicted_probs)

                                    # Compute AUC for ROC curve
                                    roc_auc = auc(fpr, tpr)

                                    # Create a DataFrame with TPR, FPR, and AUC
                                    roc_df = pd.DataFrame({
                                        'fpr': fpr,
                                        'tpr': tpr,
                                        'thresholds': thresholds
                                    })

                                    # Add the AUC value to a separate column (you can also store it separately if needed)
                                    roc_df['AUC'] = roc_auc

                                    # Define the ROC file path and save it
                                    roc_file = os.path.join(roc_data_path_val, f'ROC_{model_name}_{sh}_{num_features}.xlsx')
                                    roc_df.to_excel(roc_file, index=False)

                                    metrics, ci = compute_metrics(y_validation, model.predict(X_validation), predicted_probs, 0.8)

                                    row = [
                                        model_name,
                                        f"{metrics.get('roc_auc', 'N/A'):.2f} ({ci.get('roc_auc', ['N/A', 'N/A'])[0]:.2f}, {ci.get('roc_auc', ['N/A', 'N/A'])[1]:.2f})",
                                        f"{metrics.get('sensitivity', 'N/A'):.2f} ({ci.get('sensitivity', ['N/A', 'N/A'])[0]:.2f}, {ci.get('sensitivity', ['N/A', 'N/A'])[1]:.2f})",
                                        f"{metrics.get('specificity', 'N/A'):.2f} ({ci.get('specificity', ['N/A', 'N/A'])[0]:.2f}, {ci.get('specificity', ['N/A', 'N/A'])[1]:.2f})",
                                        f"{metrics.get('ppv', 'N/A'):.2f} ({ci.get('ppv', ['N/A', 'N/A'])[0]:.2f}, {ci.get('ppv', ['N/A', 'N/A'])[1]:.2f})",
                                        f"{metrics.get('npv', 'N/A'):.2f} ({ci.get('npv', ['N/A', 'N/A'])[0]:.2f}, {ci.get('npv', ['N/A', 'N/A'])[1]:.2f})",
                                    ]
                                    rows.append(row)

                                    result_entry_val = {
                                        'Sheet': sheet,
                                        'Val_Sheet': sh,
                                        'Num Features': num_features,
                                        'Classifier': model_name,
                                        'AUC': metrics.get('roc_auc', 'N/A'),
                                        'Sensitivity': metrics.get('sensitivity', 'N/A'),
                                        'Specificity': metrics.get('specificity', 'N/A'),
                                        'PPV': metrics.get('ppv', 'N/A'),
                                        'NPV': metrics.get('npv', 'N/A')
                                    }
                                    summary_results_val.append(result_entry_val)

                                    # Save summary results
                                    summary_df_val = pd.DataFrame(summary_results_val)
                                    summary_file_val = os.path.join(results_dir, 'summary_results_val.xlsx')
                                    with pd.ExcelWriter(summary_file_val, engine='openpyxl') as writer:
                                        best_df_val = summary_df_val.sort_values(by='AUC', ascending=False)
                                        best_df_val.to_excel(writer, sheet_name='Best Result Val', index=False)

                                        #===================

                                df = pd.DataFrame(rows, columns=['Classifier', 'AUC (95% CI)', 'Sensitivity (95% CI)',
                                                                 'Specificity (95% CI)',
                                                                 'PPV (95% CI)', 'NPV (95% CI)'])

                                output_excel_path = os.path.join(result_path, excel_file_name, sheet, f"{sh}_validation_results.xlsx")
                                save_excel_sheet(df, output_excel_path, str(num_features) + "_features", False)




if __name__ == '__main__':
    main()

