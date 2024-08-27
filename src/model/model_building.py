import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score, roc_curve)
from scipy import stats
import seaborn as sns
from openpyxl import load_workbook
import xlsxwriter
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from collections import Counter
import shap
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, roc_auc_score, recall_score, confusion_matrix
import joblib



# def get_classifiers():
#     """
#     Returns a dictionary of classifiers with their hyperparameter grids to be evaluated.
#     """
#     return {
#         'RandomForest': (RandomForestClassifier(class_weight='balanced'), {
#             'n_estimators': [100, 200, 300, 400, 500],
#             'max_features': ['auto', 'sqrt', 'log2'],
#             'max_depth': [4, 6, 8, 10, 12],
#             'criterion': ['gini', 'entropy']
#         }),
#         'SVM': (SVC(probability=True, class_weight='balanced'), {
#             'C': [0.1, 1, 10, 100, 1000],
#             'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#             'kernel': ['rbf', 'poly', 'sigmoid']
#         }),
#         'LogisticRegression': (LogisticRegression(class_weight='balanced'), {
#             'penalty': ['l1', 'l2'],
#             'C': [0.01, 0.1, 1, 10, 100]
#         }),
#         'NaiveBayes': (GaussianNB(), {})
#     }


def get_classifiers_simple():
    """
    Returns a dictionary of classifiers with their hyperparameter grids to be evaluated.
    """
    return {
        'RandomForest': RandomForestClassifier(class_weight='balanced'),
        'SVM': SVC(probability=True, class_weight='balanced'),
        'LogisticRegression': LogisticRegression(class_weight='balanced'),
        'NaiveBayes': GaussianNB()
    }


def get_classifiers():
    """
    Returns a dictionary of classifiers with their hyperparameter grids to be evaluated.
    """
    return {
        'RandomForest': (RandomForestClassifier(class_weight='balanced', random_state=42), {
            'n_estimators': [25, 50],
            'max_features': ['sqrt'],
            'max_depth': [2, 3, 4],
            'min_samples_split': [10, 15, 20],
            'min_samples_leaf': [4, 6, 8],
            'criterion': ['gini'],
            'bootstrap': [True]
        }),
        'SVM': (Pipeline([
                ('scaler', StandardScaler()),
                ('svc', SVC(probability=True, class_weight='balanced', kernel='rbf', random_state=42))
            ]), {
            'svc__C': [0.01, 0.1, 1, 10, 100],
            'svc__gamma': [0.001, 0.01, 0.1, 1]
        }),
        'LogisticRegression': (Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(class_weight='balanced', random_state=42, solver='liblinear'))
            ]), {
            'lr__penalty': ['l1', 'l2'],
            'lr__C': [0.01, 0.1, 1, 10]
        }),
        'NaiveBayes': (GaussianNB(var_smoothing=1e-9), {
        })
    }



def compute_metrics(y_true, y_pred, y_pred_prob):
    """
    Compute evaluation metrics and their confidence intervals.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    y_pred_prob (array-like): Predicted probabilities.

    Returns:
    dict: Evaluation metrics and their confidence intervals.
    """

    # Compute Youden's Index to find the optimal threshold
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    youdens_index = tpr - fpr
    optimal_idx = np.argmax(youdens_index)
    optimal_threshold = thresholds[optimal_idx]

    # Use the optimal threshold for new predictions
    y_pred_optimal = (y_pred_prob >= optimal_threshold).astype(int)

    # Compute metrics using optimal threshold predictions
    accuracy = accuracy_score(y_true, y_pred_optimal)
    roc_auc = roc_auc_score(y_true, y_pred_prob) if y_pred_prob is not None else None
    cm = confusion_matrix(y_true, y_pred_optimal)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    ppv = precision_score(y_true, y_pred_optimal)
    npv = tn / (tn + fn) if (tn + fn) else 0
    f1 = f1_score(y_true, y_pred_optimal)

    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'ppv': ppv,
        'npv': npv,
        'f1_score': f1
    }

    ci = {}
    for metric, value in metrics.items():
        if value is not None:
            ci[metric] = compute_confidence_interval(value, y_true.size)

    return metrics, ci


def compute_confidence_interval(metric_value, n, z=1.96):
    """
    Computes the confidence interval for a given metric value.
    """
    se = np.sqrt((metric_value * (1 - metric_value)) / n)
    ci_lower = metric_value - z * se
    ci_upper = metric_value + z * se
    return (ci_lower, ci_upper)


# def get_classifiers():
#     """
#     Returns a dictionary of classifiers with their hyperparameter grids to be evaluated.
#     """
#     return {
#         'RandomForest': (RandomForestClassifier(class_weight='balanced'), {
#             'n_estimators': [100, 200, 300, 400, 500],
#             'max_features': ['auto', 'sqrt', 'log2'],
#             'max_depth': [4, 6, 8, 10, 12],
#             'criterion': ['gini', 'entropy']
#         }),
#         'SVM': (SVC(probability=True, class_weight='balanced', kernel='rbf'), {
#             'C': [0.1, 1, 10],
#             'gamma': [0.1, 0.01],
#             'kernel': ['rbf']
#         }),
#         # 'SVM': (SVC(probability=True, class_weight='balanced'), {
#         #     'C': [0.1, 1, 10, 100, 1000],
#         #     'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#         #     'kernel': ['rbf', 'poly', 'sigmoid']
#         # }),
#         'LogisticRegression': (LogisticRegression(class_weight='balanced'), {
#             'penalty': ['l1', 'l2'],
#             'C': [0.01, 0.1, 1, 10, 100]
#         }),
#         'NaiveBayes': (GaussianNB(), {})
#     }


# def compute_metrics(y_true, y_pred, y_pred_prob):
#     """
#     Compute evaluation metrics and their confidence intervals.
#
#     Parameters:
#     y_true (array-like): True labels.
#     y_pred (array-like): Predicted labels.
#     y_pred_prob (array-like): Predicted probabilities.
#
#     Returns:
#     dict: Evaluation metrics and their confidence intervals.
#     """
#
#     accuracy = accuracy_score(y_true, y_pred)
#     roc_auc = roc_auc_score(y_true, y_pred_prob) if y_pred_prob is not None else None
#     cm = confusion_matrix(y_true, y_pred)
#     tn, fp, fn, tp = cm.ravel()
#     specificity = tn / (tn + fp) if (tn + fp) else 0
#     sensitivity = recall_score(y_true, y_pred)
#     ppv = precision_score(y_true, y_pred)
#     npv = tn / (tn + fn) if (tn + fn) else 0
#     f1 = f1_score(y_true, y_pred)
#
#     metrics = {
#         'accuracy': accuracy,
#         'roc_auc': roc_auc,
#         'specificity': specificity,
#         'sensitivity': sensitivity,
#         'ppv': ppv,
#         'npv': npv,
#         'f1_score': f1
#     }
#
#     ci = {}
#     for metric, value in metrics.items():
#         if value is not None:
#             ci[metric] = compute_confidence_interval(value, y_true.size)
#
#     return metrics, ci
#
#
# def compute_confidence_interval(metric, n, alpha=0.95):
#     """
#     Compute confidence interval for a metric.
#
#     Parameters:
#     metric (float): Metric value.
#     n (int): Sample size.
#     alpha (float): Confidence level.
#
#     Returns:
#     tuple: Lower and upper bounds of the confidence interval.
#     """
#     se = np.sqrt((metric * (1 - metric)) / n)
#     h = se * stats.norm.ppf((1 + alpha) / 2)
#     return metric - h, metric + h






def hyperparameter_tuning(clf, param_grid, X_train, y_train, name):
    """
    Perform hyperparameter tuning using GridSearchCV.

    Parameters:
    clf: The classifier to tune.
    param_grid (dict): The parameter grid to search over.
    X_train (pd.DataFrame): The training feature matrix.
    y_train (pd.Series): The training target vector.
    name (str): The name of the classifier.

    Returns:
    The best estimator found by GridSearchCV.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)


    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=skf, n_jobs=-1, scoring='roc_auc')
    #print("grid_search done")
    grid_search.fit(X_train, y_train)
    #print("grid_search fit done")
    #print("Best parameters found by GridSearchCV:")
    print("best_params: ", grid_search.best_params_)

    # Save grid search results
    #results_df = pd.DataFrame(grid_search.cv_results_)
    #results_df.to_csv(f'{name}_grid_search_results.csv', index=False)

    #print(f"Best parameters for {name}: {grid_search.best_params_}")
    #print(f"Best cross-validation score for {name}: {grid_search.best_score_}")

    return grid_search.best_estimator_

    #-----------------------------------------

    # random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, n_iter=20, cv=skf, n_jobs=-1,
    #                                    scoring='roc_auc', random_state=42)
    # print("random_search done")
    # random_search.fit(X_train, y_train)
    # print("grid_search fit done")
    #
    # return random_search.best_estimator_






def train_test_split_evaluation(X, y,
                                test_size=0.3,
                                random_state=42,
                                tuning=False,
                                result_path="./results",
                                num_features=10,
                                resampling_method=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    classifiers = get_classifiers()
    results = {}
    train_results = {}
    test_results = {}

    roc_data_path = os.path.join(result_path, "ROC_data")
    ensure_directory_exists(roc_data_path)
    roc_path = os.path.join(result_path, "ROC_curves")
    ensure_directory_exists(roc_path)
    calibration_path = os.path.join(result_path, "Calibration_plots")
    dca_path = os.path.join(result_path, "DCA_curves")
    shap_path = os.path.join(result_path, "Shapley_plots")
    importance_path = os.path.join(result_path, "Feature_Importance_plots")

    plt.figure(figsize=(10, 8))
    plt.title('ROC Curves', fontsize=20)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2)

    for classifier_name, (clf, param_grid) in classifiers.items():
        #if tuning and classifier_name == "SVM":
        if tuning:
            print(f"Hyperparameter_tuning for {classifier_name} classifier")
            clf = hyperparameter_tuning(clf, param_grid, X_train, y_train, classifier_name)

        if resampling_method:
            resampling_method = SMOTE(random_state=42)
            X_train, y_train = resampling_method.fit_resample(X_train, y_train)

        clf.fit(X_train, y_train)

        y_pred_train = clf.predict(X_train)
        y_pred_prob_train = clf.predict_proba(X_train)[:, 1] if hasattr(clf, "predict_proba") else None
        metrics_train, ci_train = compute_metrics(y_train, y_pred_train, y_pred_prob_train)
        train_results[classifier_name] = {
            'metrics': metrics_train,
            'confidence_intervals': ci_train
        }

        fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_pred_prob_train)
        roc_data_train_df = pd.DataFrame({'fpr': fpr_train, 'tpr': tpr_train, 'thresholds': thresholds_train})
        roc_data_train_df.to_excel(
            os.path.join(roc_data_path, f'{classifier_name}_train_roc_data_{num_features}_features.xlsx'), index=False)

        y_pred_test = clf.predict(X_test)
        y_pred_prob_test = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        metrics_test, ci_test = compute_metrics(y_test, y_pred_test, y_pred_prob_test)
        test_results[classifier_name] = {
            'metrics': metrics_test,
            'confidence_intervals': ci_test
        }

        fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_prob_test)
        roc_data_test_df = pd.DataFrame({'fpr': fpr_test, 'tpr': tpr_test, 'thresholds': thresholds_test})
        roc_data_test_df.to_excel(
            os.path.join(roc_data_path, f'{classifier_name}_test_roc_data_{num_features}_features.xlsx'), index=False)


        plot_calibration_curve(y_test, y_pred_prob_test, classifier_name, num_features, output_dir=calibration_path)
        plot_dca(y_test, y_pred_prob_test, classifier_name, num_features, output_dir=dca_path)

        # Plot ROC curve for the test set in a single figure
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob_test)
        roc_auc = roc_auc_score(y_test, y_pred_prob_test)
        plt.plot(fpr, tpr, lw=2, label=f'{classifier_name} (AUC = {roc_auc:.2f})')

        # Save individual ROC curve for this classifier
        plot_roc_curve(fpr, tpr, roc_auc, f'{classifier_name} ROC Curve',
                       filename=f'{classifier_name}_roc_curve_{num_features}_features.png',
                       output_dir=os.path.join(roc_path, classifier_name))

        # Plot feature importance for tree-based models
        if hasattr(clf, 'feature_importances_'):
            plot_feature_importance(clf.feature_importances_, X.columns, 'Feature Importance',
                                    f'{classifier_name}_featureImportance_{num_features}_features.png', output_dir=importance_path)

        # Plot Shapley values
        if num_features > 1 and classifier_name in ['RandomForest', 'XGBoost', 'LightGBM']:  # Add other models if necessary
        #if num_features > 1:  # Add other models if necessary
            plot_shap_values(clf, X_test, X.columns, f'{classifier_name}_shap_values_{num_features}_features.png', output_dir=shap_path)

    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.legend(loc='lower right', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    filepath = os.path.join(roc_path, f'roc_curve_{num_features}_features.png')
    plt.savefig(filepath, dpi=300)
    # plt.show()

    results['train'] = train_results
    results['test'] = test_results

    return results

# def train_test_split_evaluation(X, y,
#                                 test_size=0.3,
#                                 random_state=42,
#                                 tuning=1,
#                                 result_path="./results",
#                                 num_features=10,
#                                 resampling_method=None):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
#     classifiers = get_classifiers()
#     results = {}
#     train_results = {}
#     test_results = {}
#
#     roc_path = os.path.join(result_path, "ROC_curves")
#     ensure_directory_exists(roc_path)
#     calibration_path = os.path.join(result_path, "Calibration_plots")
#     dca_path = os.path.join(result_path, "DCA_curves")
#
#     plt.figure(figsize=(10, 8))
#     plt.title('ROC Curves', fontsize=16)
#     plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2)
#
#     for classifier_name, (clf, param_grid) in classifiers.items():
#         if tuning and classifier_name == "SVM":
#             print(f"hyperparameter_tuning for {classifier_name} classifier")
#             clf = hyperparameter_tuning(clf, param_grid, X_train, y_train, classifier_name)
#
#         if resampling_method:
#             resampling_method = SMOTE()
#             X_train, y_train = resampling_method.fit_resample(X_train, y_train)
#
#         clf.fit(X_train, y_train)
#
#         y_pred_train = clf.predict(X_train)
#         y_pred_prob_train = clf.predict_proba(X_train)[:, 1] if hasattr(clf, "predict_proba") else None
#         metrics_train, ci_train = compute_metrics(y_train, y_pred_train, y_pred_prob_train)
#         train_results[classifier_name] = {
#             'metrics': metrics_train,
#             'confidence_intervals': ci_train
#         }
#
#         y_pred_test = clf.predict(X_test)
#         y_pred_prob_test = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
#         metrics_test, ci_test = compute_metrics(y_test, y_pred_test, y_pred_prob_test)
#         test_results[classifier_name] = {
#             'metrics': metrics_test,
#             'confidence_intervals': ci_test
#         }
#
#         plot_calibration_curve(y_test, y_pred_prob_test, classifier_name, num_features, output_dir=calibration_path)
#         plot_dca(y_test, y_pred_prob_test, classifier_name, num_features, output_dir=dca_path)
#
#         # Plot ROC curve for the test set in a single figure
#         fpr, tpr, _ = roc_curve(y_test, y_pred_prob_test)
#         roc_auc = roc_auc_score(y_test, y_pred_prob_test)
#         plt.plot(fpr, tpr, lw=2, label=f'{classifier_name} (AUC = {roc_auc:.2f})')
#
#         # Save individual ROC curve for this classifier
#         plot_roc_curve(fpr, tpr, roc_auc, f'{classifier_name} ROC Curve',
#                        filename=f'{classifier_name}_roc_curve_{num_features}_features.png',
#                        output_dir=os.path.join(roc_path, classifier_name))
#
#         # Plot feature importance for tree-based models
#         if classifier_name == 'RandomForest':
#             plot_feature_importance(clf.feature_importances_, X.columns, 'Feature Importance',
#                                     f'{classifier_name}_feature_importance.png')
#
#     plt.xlabel('False Positive Rate', fontsize=14)
#     plt.ylabel('True Positive Rate', fontsize=14)
#     plt.legend(loc='lower right', fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     filepath = os.path.join(roc_path, f'roc_curve_{num_features}_features.png')
#     plt.savefig(filepath, dpi=300)
#     # plt.show()
#
#     results['train'] = train_results
#     results['test'] = test_results
#
#     return results






def cross_validation_evaluation(X, y, cv_folds=5, tuning=False, result_path="./results",
                                num_features=10, resampling_method=None):
    classifiers = get_classifiers()
    results = {}

    roc_data_path = os.path.join(result_path, "ROC_data")
    ensure_directory_exists(roc_data_path)
    prob_data_path = os.path.join(result_path, "Prob_data")
    ensure_directory_exists(prob_data_path)
    roc_path = os.path.join(result_path, "ROC_curves")
    ensure_directory_exists(roc_path)
    calibration_path = os.path.join(result_path, "Calibration_plots")
    dca_path = os.path.join(result_path, "DCA_curves")
    shap_path = os.path.join(result_path, "Shapley_plots")
    importance_path = os.path.join(result_path, "Feature_Importance_plots")
    model_path = os.path.join(result_path, "Saved_Models")
    ensure_directory_exists(model_path)

    plt.figure(figsize=(10, 8))
    plt.title('ROC Curves', fontsize=20)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=17)

    for name, (clf, param_grid) in classifiers.items():
        metrics_list = []
        fpr_list = []
        tpr_list = []
        thresholds_list = []
        auc_list = []
        y_pred_prob_all_folds = []

        if tuning:
            print(f"Hyperparameter_tuning for {name} classifier")
            clf = hyperparameter_tuning(clf, param_grid, X, y, name)

        for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            if resampling_method:
                #print(f"Resampling data for {name} classifier")
                X_train, y_train = resampling_method.fit_resample(X_train, y_train)

            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            y_pred_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
            y_pred_prob_all_folds.append(y_pred_prob)
            metrics, _ = compute_metrics(y_test, y_pred, y_pred_prob)
            metrics_list.append(metrics)

            # Collect data for ROC plotting
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
            roc_auc = roc_auc_score(y_test, y_pred_prob)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            thresholds_list.append(thresholds)
            auc_list.append(roc_auc)

            # # Plot individual ROC curve for each fold
            # plot_roc_curve(fpr, tpr, roc_auc, f'{name} ROC Curve - Fold {fold}',
            #                filename=f'{name}_roc_curve_fold_{fold}_{num_features}_features.png',
            #                output_dir=os.path.join(roc_path, name))

            # # Plot calibration curve for each fold
            # plot_calibration_curve(y_test, y_pred_prob, f'{name} - Fold {fold}', num_features, output_dir=os.path.join(calibration_path, name))

            # # Plot DCA curve for each fold
            # plot_dca(y_test, y_pred_prob, f'{name} - Fold {fold}', num_features, output_dir=os.path.join(dca_path, name))

            # Plot feature importance for tree-based models
            if hasattr(clf, 'feature_importances_'):
                plot_feature_importance(clf.feature_importances_, X.columns, 'Feature Importance',
                                        f'{name}_featureImportance_{num_features}_features.png',
                                        output_dir=importance_path)

            # Plot Shapley values
            if num_features > 1 and name in ['RandomForest', 'XGBoost', 'LightGBM']:
                plot_shap_values(clf, X_test, X.columns, f'{name}_shap_values_{num_features}_features.png', output_dir=shap_path)

        # Average metrics and confidence intervals across folds
        averaged_metrics = {metric: np.mean([m[metric] for m in metrics_list if m[metric] is not None]) for metric in metrics_list[0]}
        ci = {metric: compute_confidence_interval(averaged_metrics[metric], y.size) for metric in averaged_metrics}

        results[name] = {
            'metrics': averaged_metrics,
            'confidence_intervals': ci
        }

        # Plot and save averaged ROC curve across folds
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fpr_list, tpr_list)], axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(auc_list)

        plt.plot(mean_fpr, mean_tpr, lw=2, label=f'{name} (AUC = {mean_auc:.2f})')

        plot_roc_curve(mean_fpr, mean_tpr, mean_auc, f'{name} Averaged ROC Curve',
                       filename=f'{name}_averaged_roc_curve_{num_features}_features.png',
                       output_dir=os.path.join(roc_path, name))

        # Align the predicted probabilities across folds by stacking them vertically
        all_probs = np.concatenate(y_pred_prob_all_folds)
        probs_df = pd.DataFrame({'y_pred_prob': all_probs})
        new_excel_path = os.path.join(prob_data_path, f'{name}_predicted_probs_{num_features}_features.xlsx')
        probs_df.to_excel(new_excel_path, index=False)

        # Save the averaged ROC data
        roc_data_avg_df = pd.DataFrame({
            'mean_fpr': mean_fpr,
            'mean_tpr': mean_tpr,
        })
        roc_data_avg_df.to_excel(
            os.path.join(roc_data_path, f'{name}_averaged_roc_data_{num_features}_features.xlsx'), index=False)

        # Train the final model on the entire dataset and save it
        final_model = clf.fit(X, y)
        final_model_filename = os.path.join(model_path, f'{name}_{num_features}_features.pkl')
        joblib.dump(final_model, final_model_filename)

    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.legend(loc='lower right', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    filepath = os.path.join(roc_path, f'averaged_roc_curve_{num_features}_features.png')
    plt.savefig(filepath, dpi=300)

    return results






def evaluate_models(X, y, method='train_test_split', **kwargs):
    """
    Evaluate models using the specified method.
    Parameters:
    X (pd.DataFrame): Feature matrix.
    y (pd.Series): Target vector.
    method (str): Evaluation method ('train_test_split' or 'cross_validation').
    kwargs: Additional arguments for the evaluation methods.
    Returns:
    dict: Evaluation results for each classifier.
    """
    if method == 'train_test_split':
        return train_test_split_evaluation(X, y, **kwargs)
    elif method == 'cross_validation':
        return cross_validation_evaluation(X, y, **kwargs)
    else:
        raise ValueError("Invalid method. Choose 'train_test_split' or 'cross_validation'.")


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, auc, title, filename, output_dir='./plots'):
    ensure_directory_exists(output_dir)
    filepath = os.path.join(output_dir, filename)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(title, fontsize=20)
    plt.legend(loc="lower right", fontsize=16)
    plt.savefig(filepath)
    plt.close()





def plot_calibration_curve(y_true, y_pred_prob, classifier_name, num_features, num_bins=10, output_dir='./plots'):
    ensure_directory_exists(output_dir)
    filepath = os.path.join(output_dir, f'calibration_curve_{classifier_name}_{num_features}_features.png')

    plt.figure()
    prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=num_bins)
    plt.plot(prob_pred, prob_true, marker='o', label=f'{classifier_name}')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title(f'Calibration Curve - {classifier_name}')
    plt.legend(loc='upper left', fontsize=16)
    plt.savefig(filepath)
    plt.close()


def plot_dca(y_true, y_pred_prob, classifier_name, num_features, output_dir='./plots'):
    ensure_directory_exists(output_dir)
    filepath = os.path.join(output_dir, f'dca_{classifier_name}_{num_features}_features.png')

    thresholds = np.linspace(0.01, 0.99, 99)
    net_benefits = []
    treat_all = []

    for threshold in thresholds:
        tp = ((y_pred_prob >= threshold) & (y_true == 1)).sum()
        fp = ((y_pred_prob >= threshold) & (y_true == 0)).sum()
        tn = ((y_pred_prob < threshold) & (y_true == 0)).sum()
        fn = ((y_pred_prob < threshold) & (y_true == 1)).sum()

        # Net benefit calculation
        if len(y_true) > 0:
            treat_all_net_benefit = tp / len(y_true) - fp / len(y_true) * (threshold / (1 - threshold))
            net_benefit = (tp / len(y_true)) - (fp / len(y_true)) * (threshold / (1 - threshold))
        else:
            treat_all_net_benefit = 0
            net_benefit = 0

        treat_all.append(treat_all_net_benefit)
        net_benefits.append(net_benefit)

    # Plot DCA curve
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, net_benefits, label='Model', color='blue', linewidth=2)
    plt.fill_between(thresholds, 0, net_benefits, color='red', alpha=0.2)
    plt.plot(thresholds, treat_all, label='Treat all', color='black', linestyle='-', linewidth=2)
    plt.axhline(y=0, color='gray', linestyle='--', label='Treat none', linewidth=2)

    plt.xlabel('Threshold Probability', fontsize=14)
    plt.ylabel('Net Benefit', fontsize=14)
    plt.title(f'Model {classifier_name} DCA', fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(filepath)
    plt.close()




# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title, filename, output_dir='./plots'):
    ensure_directory_exists(output_dir)
    filepath = os.path.join(output_dir, filename)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(filepath)
    plt.close()


# Function to plot precision-recall curve
def plot_precision_recall_curve(y_true, y_pred_prob, title, filename, output_dir='./plots'):
    ensure_directory_exists(output_dir)
    filepath = os.path.join(output_dir, filename)

    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    average_precision = average_precision_score(y_true, y_pred_prob)

    plt.figure()
    plt.step(recall, precision, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'{title} - AP: {average_precision:.2f}')
    plt.savefig(filepath)
    plt.close()


# Function to plot feature importance
def plot_feature_importance(importances, feature_names, title, filename, output_dir='./plots'):
    """
    Plot feature importance as a horizontal bar chart, similar to the R code.

    Parameters:
    importances (array-like): Feature importance scores.
    feature_names (array-like): Names of the features.
    title (str): Title of the plot.
    filename (str): Name of the file to save the plot.
    output_dir (str): Directory where the plot will be saved.
    """
    ensure_directory_exists(output_dir)
    filepath = os.path.join(output_dir, filename)

    # Create a DataFrame for feature importances
    importance_data = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Order the features by absolute importance value for better visualization
    importance_data = importance_data.sort_values(by='Importance', key=abs, ascending=False)

    # Create a horizontal bar plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_data, palette='Blues_d')
    plt.title(title)
    plt.xlabel('Coefficient')
    plt.ylabel('Feature')
    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig(filepath, dpi=300)
    plt.close()





def plot_shap_values(model, X, feature_names, filename, output_dir='./plots'):
    """
    Plot Shapley values for a given model and dataset.

    Parameters:
    model (object): The trained model (RF, SVM, LR, NB).
    X (DataFrame): The dataset (features) to compute Shapley values.
    feature_names (array-like): List of feature names.
    filename (str): Name of the file to save the plot.
    output_dir (str): Directory where the plot will be saved.
    """
    ensure_directory_exists(output_dir)
    filepath = os.path.join(output_dir, filename)

    try:
        # Select the appropriate SHAP explainer based on the model type
        if isinstance(model, RandomForestClassifier):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X, check_additivity=False)  # Only apply check_additivity for tree-based models
        elif isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, X)
            shap_values = explainer(X)
        elif isinstance(model, (SVC, GaussianNB)):
            explainer = shap.KernelExplainer(model.predict, X)
            shap_values = explainer.shap_values(X)
        else:
            raise ValueError("Unsupported model type for SHAP value computation.")

        print("shap_values.shape: ", shap_values.shape)

        # If shap_values has multiple outputs, select the first one (e.g., binary classification or regression)
        if isinstance(shap_values, list) or hasattr(shap_values, "values") and shap_values.values.ndim > 1:
            shap_values_to_plot = shap_values[..., 1]  # Select the SHAP values for one output (e.g., for class 1)
        else:
            shap_values_to_plot = shap_values

        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(shap_values_to_plot, show=False)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

    except Exception as e:
        print(f"Error while plotting SHAP for {filename}: {e}")
        # You can log the error or take other actions here if needed




def plot_learning_curve(estimator, X, y, title, filename, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters:
    estimator (object): An estimator object implementing `fit` and `predict`.
    X (array-like): Training vector.
    y (array-like): Target vector relative to X.
    title (str): Title of the plot.
    filename (str): Filename to save the plot.
    cv (int, cross-validation generator or an iterable): Determines the cross-validation splitting strategy.
    n_jobs (int): Number of jobs to run in parallel (default: -1).
    train_sizes (array-like): Relative or absolute numbers of training examples to be used to generate the learning curve.
    """
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(filename)
    plt.close()



def save_excel_sheet(df, filepath, sheetname, index=False):
    # Create file if it does not exist
    if not os.path.exists(filepath):
        df.to_excel(filepath, sheet_name=sheetname, index=index)

    # Otherwise, add a sheet. Overwrite if there exists one with the same name.
    else:
        with pd.ExcelWriter(filepath, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer:
            df.to_excel(writer, sheet_name=sheetname, index=index)



def save_classification_results(results, output_file, num_features, method='train_test_split'):
    """
    Save evaluation results to an Excel file.

    Parameters:
    results (dict): Evaluation results for each classifier.
    output_file (str): Path to save the Excel file.
    num_features (int): Number of features used in the classification.
    method (str): Method used for evaluation ('train_test_split' or 'cross_validation').
    """
    print(f"Saving evaluation results to {output_file} using method '{method}' with {num_features} features.")

    if method == 'train_test_split':
        rows = []
        for dataset, classification_results in results.items():
            for classifier, data in classification_results.items():
                metrics = data.get('metrics', {})
                ci = data.get('confidence_intervals', {})
                row = [
                    dataset.capitalize(),
                    classifier,
                    f"{metrics.get('roc_auc', 'N/A'):.2f} ({ci.get('roc_auc', ['N/A', 'N/A'])[0]:.2f}, {ci.get('roc_auc', ['N/A', 'N/A'])[1]:.2f})",
                    f"{metrics.get('sensitivity', 'N/A'):.2f} ({ci.get('sensitivity', ['N/A', 'N/A'])[0]:.2f}, {ci.get('sensitivity', ['N/A', 'N/A'])[1]:.2f})",
                    f"{metrics.get('specificity', 'N/A'):.2f} ({ci.get('specificity', ['N/A', 'N/A'])[0]:.2f}, {ci.get('specificity', ['N/A', 'N/A'])[1]:.2f})",
                    f"{metrics.get('ppv', 'N/A'):.2f} ({ci.get('ppv', ['N/A', 'N/A'])[0]:.2f}, {ci.get('ppv', ['N/A', 'N/A'])[1]:.2f})",
                    f"{metrics.get('npv', 'N/A'):.2f} ({ci.get('npv', ['N/A', 'N/A'])[0]:.2f}, {ci.get('npv', ['N/A', 'N/A'])[1]:.2f})",
                ]
                rows.append(row)

        df = pd.DataFrame(rows, columns=['Dataset', 'Classifier', 'AUC (95% CI)', 'Sensitivity (95% CI)',
                                         'Specificity (95% CI)',
                                         'PPV (95% CI)', 'NPV (95% CI)'])

    elif method == 'cross_validation':
        rows = []
        for classifier, data in results.items():
            metrics = data.get('metrics', {})
            ci = data.get('confidence_intervals', {})
            row = [
                classifier,
                f"{metrics.get('roc_auc', 'N/A'):.2f} ({ci.get('roc_auc', ['N/A', 'N/A'])[0]:.2f}, {ci.get('roc_auc', ['N/A', 'N/A'])[1]:.2f})",
                f"{metrics.get('sensitivity', 'N/A'):.2f} ({ci.get('sensitivity', ['N/A', 'N/A'])[0]:.2f}, {ci.get('sensitivity', ['N/A', 'N/A'])[1]:.2f})",
                f"{metrics.get('specificity', 'N/A'):.2f} ({ci.get('specificity', ['N/A', 'N/A'])[0]:.2f}, {ci.get('specificity', ['N/A', 'N/A'])[1]:.2f})",
                f"{metrics.get('ppv', 'N/A'):.2f} ({ci.get('ppv', ['N/A', 'N/A'])[0]:.2f}, {ci.get('ppv', ['N/A', 'N/A'])[1]:.2f})",
                f"{metrics.get('npv', 'N/A'):.2f} ({ci.get('npv', ['N/A', 'N/A'])[0]:.2f}, {ci.get('npv', ['N/A', 'N/A'])[1]:.2f})",
            ]
            rows.append(row)

        df = pd.DataFrame(rows, columns=['Classifier', 'AUC (95% CI)', 'Sensitivity (95% CI)', 'Specificity (95% CI)',
                                         'PPV (95% CI)', 'NPV (95% CI)'])

    else:
        raise ValueError("Invalid method. Choose 'train_test_split' or 'cross_validation'.")

    # Save results
    sheetname = str(num_features) + "_features"
    save_excel_sheet(df, output_file, sheetname)


