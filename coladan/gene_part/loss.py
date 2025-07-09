import torch
import torch.nn.functional as F


def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss((input * mask).float(), (target * mask).float(), reduction="sum")
    return loss / mask.sum()


def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()


def masked_gaussian_nll_loss(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked Gaussian Negative Log-Likelihood:
      NLL = log(sigma) + (target - mu)^2 / (2 * sigma^2)
    """

    mask = mask.float()

    sigma = torch.exp(log_sigma) 
    sigma = torch.clamp(sigma, min=1e-6)
    nll = (torch.log(sigma) + 0.5 * ((target - mu)**2 / (sigma**2))).float()

    nll_masked_sum = (nll * mask).sum()
    denom = mask.sum().clamp_min(1.0) 
    loss = nll_masked_sum / denom

    return loss
    
    

