import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns

def plot_boxplot(data, x, y, hue=None, title=None, figsize=(10, 6), save_path=None):
    """Plot and save a boxplot."""
    plt.figure(figsize=figsize)
    sns.boxplot(data=data, x=x, y=y, hue=hue)
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_facet_grid_boxplot(data, x, y, hue=None, col=None, col_wrap=4, order=None, hue_order=None, title=None, save_path=None):
    """Plot and save a FacetGrid of boxplots."""
    g = sns.FacetGrid(data, col=col, col_wrap=col_wrap, height=4, aspect=1.2)
    g.map(sns.boxplot, x, y, hue, order=order, hue_order=hue_order)
    g.add_legend()
    g.set_titles("{col_name}")
    if title:
        g.fig.suptitle(title, y=1.05)
    if save_path:
        g.savefig(save_path)
    plt.close()


def plot_auc_with_ci(y_true, y_pred, n_bootstraps=1000, alpha=0.95):
    bootstrapped_scores = []
    rng = np.random.RandomState(42)

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue

        score = auc(*roc_curve(y_true[indices], y_pred[indices])[:2])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    lower_bound = sorted_scores[int((1.0 - alpha) / 2 * len(sorted_scores))]
    upper_bound = sorted_scores[int((1.0 + alpha) / 2 * len(sorted_scores))]

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.fill_between(fpr, tpr, alpha=0.2, label=f'{alpha * 100:.1f}% CI [{lower_bound:0.2f} - {upper_bound:0.2f}]')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_kaplan_meier(time, status, save_path=None):
    """
    Plot Kaplan-Meier survival curve.

    Parameters:
    time (array-like): Survival time data
    status (array-like): Event occurrence data (1 if event occurred, 0 if censored)
    """
    # Sort the data by time
    sorted_indices = np.argsort(time)
    time = np.array(time)[sorted_indices]
    status = np.array(status)[sorted_indices]

    # Number of subjects at time 0
    n = len(time)

    # Initialize the survival probability and times
    survival_prob = np.ones(n + 1)
    survival_times = np.concatenate(([0], time))

    # Calculate survival probability at each time point
    for i in range(1, n + 1):
        survival_prob[i] = survival_prob[i - 1] * (1 - status[i - 1] / n)
        n -= 1

    # Plot the Kaplan-Meier curve
    plt.step(survival_times, survival_prob, where="post")
    plt.title("Kaplan-Meier Curve")
    plt.xlabel("Time (Months)")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=600)
    #plt.show()