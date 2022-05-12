from turtle import forward
import torch.nn as nn 
from typing import Dict, List, Union

class FracDeepSurv(nn.Module):
    
    def __init__(self, config: Dict[str, Union[int, float, str, List[int]]]):
        super(FracDeepSurv, self).__init__()
        
        self.activation = config['activation']
        self.dropout = config['dropout']
        self.dims = config['dims']
        self.batch_norm = config['batch_norm']
        self.num_layers = len(self.dims)
        
        self.model = self._build_model()
        
    def _build_model(self):
        model = []
        for i in range(self.num_layers - 1):    
            block = self._build_block(self.dims[i], self.dims[i+1])
            model += block
        return nn.Sequential(*model)
    
    def _build_block(self, input_dim, output_dim):
        block = []
        if (self.dropout is not None):
            block.append(nn.Dropout(self.dropout))
        block.append(nn.Linear(input_dim, output_dim))
        if (self.batch_norm is not None):
            block.append(nn.BatchNorm1d(output_dim))
        if (self.activation is not None):
            block.append(eval('nn.{}()'.format(self.activation)))
        return block
    
    def forward(self, x):
        return self.model(x)


class FracAESurv(nn.Module):
    
    def __init__(self, config: Dict[str, Union[int, float, str, List[int]]]):
        
        super(FracAESurv, self).__init__()
        
        self.latent_feature = config['latent_feature']
        self.encoder_dims = config['encoder_dims']
        self.decoder_dims = config['decoder_dims']
        self.survnet_dims =  config['survnet_dims']
        
        assert self.encoder_dims[0] == self.decoder_dims[-1], \
            f"input dims {self.encoder_dims[0]} != output dims {self.decoder_dims[-1]}"
        
        self.encoder_activation = config['encoder_activation']
        self.decoder_activation = config['decoder_activation']
        self.survnet_activation = config['survnet_activation']
        
        self.dropout = config['dropout']
        self.batch_norm = config['batch_norm']
        
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.survnet = self._build_model()
        
    def _build_encoder(self):
        
        encoder = []
        
        for i in range(len(self.encoder_dims) - 1):
            encoder.append(nn.Linear(self.encoder_dims[i], self.encoder_dims[i+1]))
            encoder.append(eval('nn.{}()'.format(self.encoder_activation)))
        
        encoder.append(nn.Linear(self.encoder_dims[-1], self.latent_feature))
        
        return nn.Sequential(*encoder)
    
    def _build_decoder(self):
        
        decoder = []
        
        decoder.append(nn.Linear(self.latent_feature, self.decoder_dims[0]))
        decoder.append(eval('nn.{}()'.format(self.decoder_activation)))
        
        for i in range(len(self.decoder_dims) - 1):
            decoder.append(nn.Linear(self.decoder_dims[i], self.decoder_dims[i+1]))
            decoder.append(eval('nn.{}()'.format(self.decoder_activation)))
            
        return nn.Sequential(*decoder)
        
    def _build_model(self):
        
        model = []
        
        model.append(nn.Linear(self.latent_feature, self.survnet_dims[0]))
        model.append(eval('nn.{}()'.format(self.survnet_activation)))
        
        for i in range(len(self.survnet_dims) - 1):    
            block = self._build_block(self.survnet_dims[i], self.survnet_dims[i+1])
            model += block
            
        return nn.Sequential(*model)
    
    def _build_block(self, input_dim, output_dim):
        block = []
        if (self.dropout is not None):
            block.append(nn.Dropout(self.dropout))
        block.append(nn.Linear(input_dim, output_dim))
        if (self.batch_norm is not None):
            block.append(nn.BatchNorm1d(output_dim))
        if (self.survnet_activation is not None):
            block.append(eval('nn.{}()'.format(self.survnet_activation)))
        return block
    
    def forward(self, x):
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        phi = self.survnet(encoded)
        
        return (phi, decoded)
    
    def predict(self, x):
        
        return self.survnet(self.encoder(x))