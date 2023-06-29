
from sklearn.metrics import (f1_score,
                             precision_score,
                             recall_score,
                             confusion_matrix
                            )
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def eval_metrics(y, t, threshold=0.5):
    t = t.get_label()
    y_bin = (y > threshold).astype(int)
    return [('f1',f1_score(t,y_binç)), ('recall', recall_score(t,y_bin)) , ('precision', precision_score(t,y_bin))]

def f1_metric(preds, train_data):

    labels = train_data.get_label()

    return 'f1', f1_score(labels, preds), True

##Optimize Threshold
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

def plot_confusion_matrix(y_true,
                          y_predicted,
                          labels = ["negative","positive"],
                          save = False,
                          save_path = None
                          ):
    plt.figure()
    cm  = confusion_matrix(y_true, y_predicted)
    # Get the per-class normalized value for each cell
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # We color each cell according to its normalized value, annotate with exact counts.
    ax = sns.heatmap(cm_norm, annot=cm, fmt="d")
    ax.set(xticklabels=labels, yticklabels=labels)
    ax.set_ylim([0,len(labels)])
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('Real Classes')
    ax.set_xlabel('Predicted Classes')

    if save==True:
        ax.figure.savefig(os.path.join(save_path,"confusion_matrix.png"))
        plt.close()
    else:
        ax.show()

