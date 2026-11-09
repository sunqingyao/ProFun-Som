from copy import deepcopy

from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple, Sequence, Union
import numpy as np
from numpy import ndarray 
import torch
import torch.nn as nn
import torch.nn.parallel as dp
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from functools import partial, reduce

from scipy.stats import wilcoxon, mannwhitneyu


def AuPRC_score(targs: np.ndarray, preds: np.ndarray):
    pr_ary, rc_ary, _ = precision_recall_curve(targs.flatten(), preds.flatten())
    AuPRC_value = auc(rc_ary, pr_ary)
    return 100 * AuPRC_value

def fmax_score(targs: ndarray, preds: ndarray,
               no_empty_labels: bool = True):
    n = 100
    if no_empty_labels:
        idx = np.where(targs.sum(1))[0]
        targs = targs[idx]
        preds = preds[idx]
    fmax_value = 0.

    for t in range(n+1):
        thres = t / n
        preds_label = np.where(preds > thres, 1, 0)
        true_positive = np.where(np.logical_and(preds_label, targs), 1, 0)
        tp = true_positive.sum(axis=1)
        tp_and_fp = preds_label.sum(axis=1)
        tp_and_fn = targs.sum(axis=1)

        idx = np.where(tp_and_fp != 0)[0]
        tp_and_fp1 = tp_and_fp[idx]
        tp1 = tp[idx]
        avgprs = tp1 / tp_and_fp1 if tp_and_fp1.shape[0] != 0 else np.array([0.])

        # when no_empty_label is False
        idx = np.where(tp_and_fn != 0)[0]
        tp_and_fn1 = tp_and_fn[idx]
        tp1 = tp[idx]
        avgrcs = tp1 / tp_and_fn1 if tp_and_fn1.shape[0] != 0 else np.array([0.])

        avgpr: float = np.mean(avgprs)
        avgrc: float = np.mean(avgrcs)
        fscore = (2 * avgpr * avgrc) / (avgpr + avgrc) if avgpr != 0 or avgrc != 0 else 0.
        if fscore > fmax_value:
            fmax_value = fscore
    # fmax_value = reduce(_calculate_fmax, np.arange(n+1))
    # fmax_value = max(map(_calculate_fmax, np.arange(n+1)))
    fmax_value *= 100

    return fmax_value


def fmax_tp(tp: np.ndarray, no_empty_labels: bool = True):
    """
    tp: 2 x 1 x n_classes
    """
    t, p = tp # both are 1 x n_classes
    return fmax_score(t,p, no_empty_labels)

def AuPRC_tp(tp: ndarray):
    t, p = tp
    return AuPRC_score(t,p)

def fmax_pvalue(pvaluefunc: Callable[[ndarray,ndarray], Any]):
    def _wrapper(targs_preds0, targs_preds1, 
                 alternative: str = "two_sided",
                 no_empty_labels:bool=False):
        vfmax_score = np.vectorize(fmax_tp,signature="(i,j,k),()->()")
        fm_ary0 = vfmax_score(targs_preds0, no_empty_labels)
        fm_ary1 = vfmax_score(targs_preds1, no_empty_labels)

        _, pvalue = pvaluefunc(fm_ary0, fm_ary1, alternative)
        return pvalue
    return _wrapper

def AuPRC_pvalue(pvaluefunc: Callable[[ndarray,ndarray], Any]):
    def _wrapper(targs_preds0, targs_preds1,
                 alternative: str = "two_sided"):
        vAuPRC_score = np.vectorize(AuPRC_tp, signature="(i,j,k)->()")
        
        au_ary0 = vAuPRC_score(targs_preds0)
        au_ary1 = vAuPRC_score(targs_preds1)

        _, pvalue = pvaluefunc(au_ary0, au_ary1, alternative)
        return pvalue
    return _wrapper

@fmax_pvalue
def fmax_wilcoxon(m0: ndarray,m1: ndarray, alternative: str = "two_sided"):
    return wilcoxon(m0,m1, alternative=alternative)

@AuPRC_pvalue
def AuPRC_wilcoxon(m0: ndarray, m1: ndarray, alternative: str = "two_sided"):
    return wilcoxon(m0, m1, alternative=alternative)

@fmax_pvalue
def fmax_mannwhitneyu(m0: ndarray, m1: ndarray, alternative: str = "two_sided"):
    return mannwhitneyu(m0, m1, alternative=alternative)

@AuPRC_pvalue
def AuPRC_mannwhitneyu(m0: ndarray, m1:ndarray, alternative: str = "two_sided"):
    return mannwhitneyu(m0, m1, alternative=alternative)

def load_by_module_name(cpu_model: nn.Module,
                        gpu_model: nn.Module,
                        r: None, 
                        module_name: str):
    """
    Using:
    partial_loading = partial(load_by_module_name, cpu_model, gpu_model)
    reduce(partial_loading, module_names, None)
    """
    gpu_module = gpu_model.get_submodule(module_name)
    if isinstance(gpu_module, dp.DataParallel):
        gpu_module = gpu_module.module
    else:
        pass
    
    cpu_model.get_submodule(module_name).load_state_dict(gpu_module.state_dict())
    
    return r
    
def loading_with_cpu(cpu_model: nn.Module, gpu_model: nn.Module, weights: OrderedDict,
                     module_names: List[str]):
    """
    loading weights that trained on gpu into cpu
    """
    gpu_model.load_state_dict(weights)
    partial_loading = partial(load_by_module_name, cpu_model, gpu_model)
    reduce(partial_loading, module_names, None)

def check_model_weights(model0: nn.Module, model1: nn.Module):
    return all([torch.equal(w0,w1) 
                for w0, w1 in zip(model0.state_dict().values(),
                                  model1.state_dict().values())])

class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


if __name__ == "__main__":
    """
    """
    targs = np.array([[0, 1, 1, 0], [1, 0, 1, 0]])
    preds = np.array([[0.2, 0.3, 0.3, 0.5], [0.2, 0.3, 0.5, 0.3]])
    f0 = fmax_score(targs, preds)
    a0 = AuPRC_score(targs, preds) 
    print("fmax {:.4f}, AuPRC {:.4f}".format(f0,a0))
