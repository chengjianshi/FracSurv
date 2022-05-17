from matplotlib.pyplot import axis
import torch
import numpy as np
from torch import nn
from pycox.models.loss import NLLLogistiHazardLoss
    
class LossAELogHaz(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        assert (alpha >= 0) and (alpha <= 1), 'Need `alpha` in [0, 1].'
        self.alpha = alpha
        self.loss_surv = NLLLogistiHazardLoss()
        self.loss_ae = nn.MSELoss()
        
    def forward(self, phi, decoded, target_loghaz, target_ae):
        idx_durations, events = target_loghaz
        loss_surv = self.loss_surv(phi, idx_durations, events)
        loss_ae = self.loss_ae(decoded, target_ae)
        return self.alpha * loss_surv + (1 - self.alpha) * loss_ae
    
    
class CoxPHLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def _make_riskset(self, time):
        
        o = np.argsort(-time, kind = 'mergesort')
        n_samples = len(time)
        risk_set = np.zeros((n_samples, n_samples), dtype=np.bool)
        
        for i_org, i_sort in enumerate(o):
            ti = time[i_sort]
            k = i_org
            while k < n_samples and ti == time[o[k]]:
                k += 1
            risk_set[i_sort, o[:k]] = True 
            
        return risk_set
    
    def _safe_normalize(self, x):
        
        x_min, _ = torch.min(x, dim = 0)
        c = torch.zeros_like(x_min)
        norm = torch.where(x_min < 0, -x_min, c)
        
        return x + norm
    
    def _logsumexp_masked(self, risk_pred, risk_set):
        
        risk_set = risk_set.type(risk_pred.dtype)
        risk_score_masked = torch.mul(risk_pred, risk_set)
        amax = torch.max(risk_score_masked, dim=1, keepdim=True)
        risk_score_shift = risk_score_masked - amax
        
        exp_masked = torch.mul(torch.exp(risk_score_shift), risk_set)
        exp_sum = torch.sum(exp_masked, dim=1, keepdim=True)
        
        output = amax + torch.log(exp_sum)
        
        return output
    
    def forward(self, event, time, risk_pred):
        risk_set = self._make_riskset(time)
        risk_pred = self._safe_normalize(risk_pred)
        
        rr = self._logsumexp_masked(risk_pred, risk_set)
        losses = torch.mul(event, rr - risk_pred)
       
        return torch.mean(losses)
            
            
        