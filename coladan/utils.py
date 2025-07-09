import torch
import numpy as np
import pandas as pd
from scipy.stats import rankdata

def _as_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=torch.float32, device="cpu")
    if isinstance(x, pd.DataFrame):
        return torch.from_numpy(x.values).to(dtype=torch.float32)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=torch.float32)
    raise TypeError(f"Unsupported type: {type(x)}")

def _rowwise_pearson(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    a_mean = a.mean(dim=1, keepdim=True)
    b_mean = b.mean(dim=1, keepdim=True)
    a_c = a - a_mean
    b_c = b - b_mean
    num   = (a_c * b_c).sum(dim=1)
    denom = torch.sqrt((a_c**2).sum(dim=1)) * torch.sqrt((b_c**2).sum(dim=1))
    return num / (denom + eps)

def _rowwise_spearman(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    a_rank = torch.from_numpy(
        np.vstack([rankdata(row, method="average") for row in a.numpy()])
    ).to(dtype=torch.float32)
    b_rank = torch.from_numpy(
        np.vstack([rankdata(row, method="average") for row in b.numpy()])
    ).to(dtype=torch.float32)
    return _rowwise_pearson(a_rank, b_rank, eps)

def calculate_pearson_and_spearman(ori, predict):
    ori_t  = _as_tensor(ori)
    pred_t = _as_tensor(predict)

    if ori_t.shape != pred_t.shape:
        raise ValueError(f"Shape mismatch: ori {ori_t.shape} vs predict {pred_t.shape}")

    pearson_vec  = _rowwise_pearson(ori_t, pred_t)     # [N_cells]
    spearman_vec = _rowwise_spearman(ori_t, pred_t)

    mean_pearson  = pearson_vec.mean().item()
    mean_spearman = spearman_vec.mean().item()
    return mean_pearson, mean_spearman
