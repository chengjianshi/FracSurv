import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import itertools

from tqdm import tqdm
from typing import Callable, Union, List, Any, Dict
from sklearn.model_selection import cross_validate, GridSearchCV, KFold
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix

def plot_confusion_matrix(cm: np.ndarray, 
                          classes: List[Union[int, str]], 
                          title: str, 
                          cmap = plt.cm.Blues):
    """
    plot_confusion_matrix prints and plots the cm 
    confusion matrix received in input.
    """
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    return

def predict_ensemble(ensemble: List[Any], X: Union[np.ndarray, pd.DataFrame]):
    
    """
    predict_ensemble runs the X data set through
    each classifier in the ensemble list to get predicted
    probabilities.
    Those are then averaged out across all classifiers. 
    """
    
    probs = [r.predict_proba(X)[:,1] for r in ensemble]
    
    return np.vstack(probs).mean(axis=0)

def adjusted_classes(y_scores: List[float], t: float):
    
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    
    return [1 if y >= t else 0 for y in y_scores]

def print_report(model: Any, 
                 X_valid: Union[np.ndarray, pd.DataFrame], 
                 y_valid: Union[np.ndarray, pd.DataFrame, pd.Series], 
                 X_train: Union[np.ndarray, pd.DataFrame] = None, 
                 y_train: Union[np.ndarray, pd.DataFrame, pd.Series] = None, 
                 t: float = 0.5,
                 plot_cm: bool = False, 
                 plot_cm_title: str = "Confusion matrix validation set"
                 ):
    
    """
    print_report prints a comprehensive classification report
    on both validation and training set (if provided).
    The metrics returned are AUC, F1, Precision, Recall and 
    Confusion Matrix.
    It accepts both single classifiers and ensembles.
    Results are dependent on the probability threshold t 
    applied to individual predictions.
    """
    
    if isinstance(X_valid, pd.DataFrame):
        X_valid = X_valid.values
    if X_train is not None and isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    
    if isinstance(model, list):
        proba_valid = predict_ensemble(model, X_valid)
        y_val_hat = adjusted_classes(proba_valid, t)
        
        if X_train is not None:
            proba_train = predict_ensemble(model, X_train)    
            y_train_hat = adjusted_classes(proba_train, t)
    
    else:
        proba_valid = model.predict_proba(X_valid)[:,1]
        y_val_hat = adjusted_classes(proba_valid, t)
        if X_train is not None:
            proba_train = model.predict_proba(X_train)[:,1]
            y_train_hat = adjusted_classes(proba_train, t)
    
    res = {"auc_valid": None, 
           "f1_valid": None, 
           "auc_train": None, 
           "f1_train": None, 
           "cm_valid": None}
    
    res["auc_valid"] = roc_auc_score(y_valid, proba_valid)
    res["f1_valid"] = f1_score(y_valid, y_val_hat)
    res["cm_valid"] = confusion_matrix(y_valid, y_val_hat)
    report = f"AUC valid: {res['auc_valid']:.5f} \nF1 valid: {res['f1_valid']:.5f} "
    
    if X_train is not None:
        res['auc_train'] = roc_auc_score(y_train, proba_train)
        res['f1_train'] = f1_score(y_train, y_train_hat)
        report += f"\nAUC train: {res['auc_train']:.5f} \nF1 train: {res['f1_train']:.5f}"
         
    print(report)
    print(classification_report(y_valid, y_val_hat))
    if plot_cm:
        plot_confusion_matrix(res['cm_valid'], classes = ["Negative", "Positive"], title = plot_cm_title)
    
    return res 

def model_selection_cv(model: Any, 
                       grid: Dict[str, List[Any]],
                       X_train: Union[np.ndarray, pd.DataFrame], 
                       y_train: Union[np.ndarray, pd.Series, pd.DataFrame],
                       scoring: str = "roc_auc", 
                       cv: int = 5,
                       engine: Any = GridSearchCV,
                       name: str = "classifer"
                       ):

    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    
    cv_res = engine(model, grid, cv = cv, scoring=scoring)
    cv_res.fit(X_train, y_train)
    
    result = {"model_name": name, 
            "best_model": cv_res.best_estimator_, 
            "best_params": cv_res.best_params_, 
            "score": cv_res.best_score_
            }
    
    return result

def optim_t(y_true, pred_proba):
    thresholds = np.arange(0.0, 1.0, 0.0001)
    fscore = np.zeros(shape=(len(thresholds)))
    for index, elem in enumerate(thresholds):
        y_pred_prob = (pred_proba > elem).astype('int')
        fscore[index] = f1_score(y_true, y_pred_prob)
    
    opt_th = thresholds[np.argmax(fscore)]
    opt_fscore = np.max(fscore)
    print(f"Best Threshold: {opt_th:.3f} with F-Score: {opt_fscore:.3f}'")    
    
    return opt_th



def hparam_tunning(search_space: List[np.float], 
                   fixed_params: Dict[str, np.float], 
                   estimator_engine: Callable, 
                   name: str, 
                   X_train: Union[pd.DataFrame, np.ndarray], 
                   y_train: Union[pd.DataFrame, np.ndarray]):

    kf = KFold(n_splits = 5, shuffle = True)    
    scores = {name: [], 
           'score': [], 
           'std': [], 
           'elapse': []}

    for x in tqdm(search_space, desc = f"{name} tunning: "):
        
        start = time.time()
        fixed_params[name] = x
        estimator = estimator_engine(**fixed_params)
        cv_res = cross_validate(estimator, X_train, y_train, cv = kf, error_score=0.5, n_jobs=-1)
    
        scores[name].append(round(x, 5))
        scores['score'].append(np.mean(cv_res['test_score']))
        scores['std'].append(np.std(cv_res['test_score']))

        end = time.time()
        scores['elapse'].append(end - start)
        
    scores = pd.DataFrame.from_dict(scores)
    
    fig, ax = plt.subplots(nrows = 1,ncols = 2, figsize=(10, 5))
    ax[0].plot(scores[name], scores['score'])
    ax[0].fill_between(scores[name], scores['score'] - scores['std'], scores['score'] + scores['std'], alpha=0.15, color = "red")
    ax[0].set_ylabel("concordance index")
    ax[0].set_xlabel(name)
    ax[0].grid(True)

    ax[1].plot(scores[name], scores['elapse'])
    
    return scores







