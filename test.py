
import sys
import torch
import time
import numpy as np 

from tqdm import tqdm 
from typing import Union
from pathlib import Path 
from torch.utils.data import DataLoader, TensorDataset
from utils import read_config, read_h5_file, EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from deepFrac import FracDeepSurv
from loss import CoxPHLoss
from sksurv.metrics import concordance_index_censored

SEED = 1234

np.random.seed(SEED)
_ = torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_data(data_path):
    
    train = read_h5_file(data_path, is_train = True)
    test = read_h5_file(data_path, is_train = False)
    
    trans = lambda data: tuple(torch.from_numpy(d) for d in data)
    
    train = trans(train)
    test = trans(test)
    
    return train, test

def c_score(risk_pred, time, event):
    
    trans = lambda x: x.detach().numpy() if type(x) == torch.Tensor else x
    risk_pred = trans(risk_pred)
    event = trans(event)
    time = trans(time)
    
    risk_pred = risk_pred.flatten()
    event = event.flatten().astype(np.bool_)
    time = time.flatten()
    
    cindex = concordance_index_censored(event, time, risk_pred)
    
    return cindex[0]

def train(config_file: Union[str, Path], log_dir: Union[str, Path]):
    
    # load config
    config = read_config(config_file)   
    train_config = config['train']
    model_config = config['model']
    log_dir = Path(log_dir) if (type(log_dir) == str) else log_dir

    model_path = log_dir / 'best_model.pth'
    
    # preprocess data 
    train, test = preprocess_data(train_config['data_path'])
    
    # discrete time
    x_train, y_train_duration, y_train_event = train
    x_test, y_test_duration, y_test_event = test
    
    train_dataset = TensorDataset(x_train, y_train_duration, y_train_event)
    
    in_features = x_train.shape[1]
    out_features = 1
    
    # build model dims & loss func
    model_config['dims'] = [in_features] + model_config['dims'] + [out_features]
    loss_func = CoxPHLoss()
    net = FracDeepSurv(model_config).to(device)
        
    # train 
    epochs = train_config['num_epochs']
    batch_size = train_config['batch_size'] if train_config['batch_size'] > 0 else x_train.shape[0]
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    optimizer = torch.optim.Adam(net.parameters(), weight_decay = train_config['weight_decay'], lr = train_config['learning_rate'])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    early_stopping = EarlyStopping(delta = 0, patience=20, verbose=False, path = model_path)
    
    # tensorboard
    writer = SummaryWriter(log_dir = log_dir)
    
    data_iter = iter(train_dataloader)
    X,_,_ = next(data_iter)
    writer.add_graph(net, X)
    
    # run 
    epoch_pbar = tqdm(range(epochs), total = epochs, leave = False)
    l1_reg = lambda net: train_config["l1_reg"] * sum(torch.linalg.norm(p, 1) for p in net.parameters())
    
    for epoch in epoch_pbar:
        
        train_loss = 0
        train_c_score = 0
        
        for _, (x, t, e) in enumerate(train_dataloader):
            
            x.to(device)
            t.to(device)
            e.to(device)
            
            optimizer.zero_grad()
            output = net(x)
                        
            loss = loss_func(output, t, e) + l1_reg(net)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_c_score += c_score(output, t, e)
        
        train_loss /= train_dataloader.__len__()
        train_c_score /= train_dataloader.__len__()
        
        test_loss = 0
        test_c_score = 0

        net.eval()
        with torch.no_grad():
            output = net(x_test)
            loss = loss_func(output, y_test_duration, y_test_event) + l1_reg(net)
            test_loss = loss.item()
            test_c_score = c_score(output, y_test_duration, y_test_event)
        
        lr_scheduler.step(test_loss)
        epoch_pbar.set_description(f"Epoch [{epoch}/{epochs}]")
        epoch_pbar.set_postfix(lr = optimizer.param_groups[0]['lr'],
                               train_c = train_c_score, 
                               test_c = test_c_score,
                               train_loss = train_loss, 
                               test_loss = test_loss)
        
        writer.add_scalars("loss", {"train": train_loss, "test": test_loss}, epoch)
        writer.add_scalars("c-score", {"train": train_c_score, "test": test_c_score}, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        
        early_stopping(test_loss, net)
        if early_stopping.early_stop:
            break
    
    # metrics
    best_model = FracDeepSurv(model_config).to(device)
    best_model.load_state_dict(torch.load(model_path))
    best_model.eval()
    with torch.no_grad():
        output_train = best_model(x_train)
        output_test = best_model(x_test)
        train_c_score = c_score(output_train, y_train_duration, y_train_event)
        test_c_score = c_score(output_test, y_test_duration, y_test_event)

    for key in model_config:
        if ('dims' in key):
            model_config[key] = ','.join([str(elem) for elem in model_config[key]])
    
    writer.add_hparams(
            hparam_dict = {**model_config, **train_config}, 
            metric_dict = { 
                            'train_c_score': train_c_score, 
                            'test_c_score': test_c_score}
                       )
    
    print(f"Metrics \n\
        Best model metric: \n \
        --> train c score: {train_c_score} \n \
        --> test c score: {test_c_score} \n"
        )

    epoch_pbar.close()
    writer.close()
    return 

if __name__ == "__main__":
    
    config_file = Path(sys.argv[1])
    log_dir = Path(sys.argv[2])
    
    curr_date = time.strftime("%Y%m%d")
    curr_time = time.strftime("%H%M%S")
    log_dir = log_dir / curr_date / curr_time
    
    if not config_file.exists():
        raise FileExistsError(f"not found config file {config_file}")
    
    log_dir.mkdir(parents=True, exist_ok=False)    
    train(config_file, log_dir)