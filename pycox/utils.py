import h5py
import torch
import configparser
import numpy as np
import pandas as pd

from torch import Tensor
# from pycox.models import LogisticHazard, logistic_hazard
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.evaluation import EvalSurv

from lifelines.utils import concordance_index


def read_config(ini_file):
    ''' Performs read config file and parses it.

    :param ini_file: (String) the path of a .ini file.
    :return config: (dict) the dictionary of information in ini_file.
    '''
    def _build_dict(items):
        return {item[0]: eval(item[1]) for item in items}
    # create configparser object
    cf = configparser.ConfigParser()
    # read .ini file
    cf.read(ini_file)
    config = {sec: _build_dict(cf.items(sec)) for sec in cf.sections()}
    return config

def read_h5_file(file_dir: str, is_train: bool):
        
        split = "train" if is_train else "test"
        with h5py.File(file_dir, 'r') as f:
            X = f[split]['x'][:]
            e = f[split]['e'][:]
            t = f[split]['t'][:]
        
        return (X, e, t)
    
def preprocess_data(data_path, num_durations = 10):
    
    train = read_h5_file(data_path, is_train = True)
    test = read_h5_file(data_path, is_train = False)
    
    x_train, e_train, t_train = train
    x_test, e_test, t_test = test
    
    preprocess = lambda data: tuple(d.astype('int') for d in data)
    trans = lambda data: tuple(torch.from_numpy(d) for d in data)
    
    labtrans = LabTransDiscreteTime(num_durations)
    
    t_train, e_train = labtrans.fit_transform(*preprocess((t_train, e_train)))
    t_test, e_test = labtrans.fit_transform(*preprocess((t_test, e_test)))
    
    train = trans((x_train, t_train, e_train))
    test = trans((x_test, t_test, e_test))
    
    return train, test, labtrans

def hazard2surv(hazard: Tensor, epsilon: float = 1e-7):
    """Transform discrete hazards to discrete survival estimates.
    Ref: LogisticHazard
    """
    return (1 - hazard).add(epsilon).log().cumsum(1).exp()

def output2hazard(output: Tensor):
    """Transform a network output tensor to discrete hazards. This just calls the sigmoid function
    Ref: LogisticHazard
    """
    return output.sigmoid()

def output2surv(output: Tensor, epsilon: float = 1e-7):
    """Transform a network output tensor to discrete survival estimates.
    Ref: LogisticHazard
    """
    hazards = output2hazard(output)
    return hazard2surv(hazards, epsilon)

def discrete_c_score(output, t, e, labtrans):
    trans = lambda x: x.detach().numpy() if type(x) == torch.Tensor else x
    t = trans(t)
    e = trans(e)
    surv = output2surv(output)
    surv_df = pd.DataFrame(surv.detach().numpy().transpose(), labtrans.cuts)
    ev = EvalSurv(surv_df, t, e, censor_surv='km')
    return ev.concordance_td()

def c_score(risk_pred, y, e, labtrans):
    return discrete_c_score(risk_pred, y, e, labtrans)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss