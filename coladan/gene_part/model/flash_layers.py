from functools import lru_cache
import math
from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.transformer import _get_clones

from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.flash_attention import FlashAttention
from flash_attn.modules.mha import FlashCrossAttention
from .layers import MultiheadAttention


class FlashscGPTMHA(nn.Module):
    """
    Custom MHA layer for scGPT. This takes two separate forward passes on the pect
    genes, and on the gen genes.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        batch_first=True,
        attention_dropout=0.0,
        causal=False,
        device=None,
        dtype=None,
    ) -> None:
        assert batch_first
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.self_attn = FlashAttention(attention_dropout=attention_dropout)
        #self.cross_attn = MultiheadAttention(
        #    embed_dim,
        #    num_heads,
        #    dropout=attention_dropout,
        #    batch_first=batch_first,
        #    **factory_kwargs,
        #)
        self.flash_cross_attn = FlashCrossAttention(attention_dropout=attention_dropout)
        # for cross attetion, launch multiple queries in parallel, each query is just
        # a single gen gene. Then each kv is the entire set of pect genes plus this gen
        # gene together.
        # In practice, we can simply put these queries in the batch dimension, and then
        # they can be processed in parallel.
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
    
    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Tensor,
        pcpt_key_padding_mask: Optional[Tensor] = None,
        gen_key_padding_mask: Optional[Tensor] = None,
        need_weights=False,
        attn_non_mask_position: Optional[Tensor] = None,
    ):
        # Self attention for pcpt genes
        B = pcpt_total_embs.shape[0]
        pcpt_len = pcpt_total_embs.shape[1]
        pcpt_qkv = self.Wqkv(pcpt_total_embs)
        pcpt_qkv = rearrange(pcpt_qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads)
        
        pcpt_context, pcpt_attn_weights = self.self_attn(
            pcpt_qkv,
            key_padding_mask=pcpt_key_padding_mask,
            need_weights=need_weights,
            causal=self.causal,
        )
        pcpt_context = self.out_proj(rearrange(pcpt_context, "b s h d -> b s (h d)"))
    
        if gen_total_embs is None:
            return (pcpt_context, None), (pcpt_attn_weights, None)
        
        gen_len = gen_total_embs.shape[1]
        gen_qkv = self.Wqkv(gen_total_embs)
        gen_qkv = rearrange(
            gen_qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads
        )
        
        cross_q = gen_qkv[:, :, 0, :, :]
        cross_kv = torch.cat([pcpt_qkv[:, :, 1:, :, :], gen_qkv[:, :, 1:, :, :]], dim=1)
        
        ###
        pcpt_seq = pcpt_qkv.shape[1] 
        gen_seq = gen_qkv.shape[1]
        cross_kv_pcpt = cross_kv[:, :pcpt_seq, :, :, :]  # (B, pcpt_seq, 2, nheads, head_dim)
        cross_kv_gen  = cross_kv[:, pcpt_seq:, :, :, :]   # (B, gen_seq, 2, nheads, head_dim)
        
        actual_gen_kv_len = gen_seq 
        diag_mask = torch.eye(actual_gen_kv_len, device=cross_kv_gen.device).unsqueeze(0)  
        diag_mask = diag_mask.expand(cross_kv_gen.shape[0], -1, -1)
        diag_mask_exp = diag_mask.diagonal(dim1=1, dim2=2)
        diag_mask_exp = diag_mask_exp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        if attn_non_mask_position is not None:
            attn_mask = attn_non_mask_position.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).type_as(cross_kv_gen)
            final_mask = diag_mask_exp * attn_mask  # (B, gen_seq, 1, 1, 1)
        else:
            final_mask = diag_mask_exp  # (B, gen_seq, 1, 1, 1)
            
        cross_kv_gen = cross_kv_gen * final_mask
        cross_kv = torch.cat([cross_kv_pcpt, cross_kv_gen], dim=1)
        ###
        

        if pcpt_key_padding_mask is None:
            pcpt_key_padding_mask = torch.zeros(
                (B, pcpt_len), dtype=torch.bool, device=pcpt_total_embs.device
            )
        if gen_key_padding_mask is None:
            gen_key_padding_mask = torch.zeros(
                (B, gen_len), dtype=torch.bool, device=gen_total_embs.device
            )
        
        q_key_mask = ~gen_key_padding_mask
        kv_key_mask = torch.cat([~pcpt_key_padding_mask, ~gen_key_padding_mask], dim=1)
        #cross_q_ = cross_q.unsqueeze(2)
        q_unpadded, q_indices, q_cu_seqlens, q_max_seqlen = unpad_input(cross_q, q_key_mask)
        kv_unpadded, kv_indices, kv_cu_seqlens, kv_max_seqlen = unpad_input(cross_kv,kv_key_mask)
        
        kv_unpadded = kv_unpadded.to(q_unpadded.dtype)
        cross_ctx_unpadded = self.flash_cross_attn(
            q_unpadded,
            kv_unpadded,
            cu_seqlens=q_cu_seqlens,
            cu_seqlens_k=kv_cu_seqlens,
            max_seqlen=q_max_seqlen,
            max_seqlen_k=kv_max_seqlen,
            causal=False
        )
        
        cross_ctx = pad_input(
            cross_ctx_unpadded,
            q_indices,
            B,
            gen_len
        )
        cross_ctx = cross_ctx.squeeze(2)
        
        gen_context = rearrange(cross_ctx, "b s h d -> b s (h d)")
        gen_context = self.out_proj(gen_context)
        
        gen_attn_weights = None
        
        return (pcpt_context, gen_context), (pcpt_attn_weights, gen_attn_weights)



class AddAuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.requires_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.requires_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss
        

class MOEGate(nn.Module):
    def __init__(self, hidden_size, n_routed_experts, top_k, scoring_func="softmax",
                 alpha=0.2, seq_aux=True, norm_topk_prob=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.alpha = alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob

        #  shape: (n_routed_experts, hidden_size)
        self.weight = nn.Parameter(torch.empty((n_routed_experts, hidden_size)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_flat = hidden_states.view(-1, h)  # [bsz*seq_len, hidden_size]
        logits = F.linear(hidden_flat, self.weight, None)  # [bsz*seq_len, n_routed_experts]
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f"Unsupported scoring function: {self.scoring_func}")
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)  # shape: [bsz, seq_len * top_k]
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class RoutedExpert(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=512, dropout=0.1, activation=F.relu):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = activation
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        
        
class SharedExpert(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048 ,output_dim=512, dropout=0.1, activation=F.relu):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = activation
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        
        
class FFNMoE(nn.Module):
    def __init__(self, hidden_size=512, dropout=0.1, activation=F.relu,
                 n_routed_experts=8, top_k=2, use_shared_expert=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = top_k
        self.gate = MOEGate(hidden_size, n_routed_experts, top_k,
                             scoring_func="softmax", alpha=0.1, seq_aux=True, norm_topk_prob=True)
        #dim_feedforward = 512 in Bo Wang 's work scGPT
        self.experts = nn.ModuleList([
            RoutedExpert(input_dim=hidden_size, hidden_dim=128, output_dim=hidden_size,
                         dropout=dropout, activation=activation)
            for i in range(n_routed_experts)
        ])
        self.use_shared_expert = use_shared_expert
        self.shared_expert = SharedExpert(input_dim=hidden_size, hidden_dim=512 ,output_dim=hidden_size,
                                          dropout=dropout, activation=activation) if use_shared_expert else None

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
          active_experts = torch.unique(flat_topk_idx, sorted=True)
          expert_outputs = []
          for expert_id in active_experts:
            mask = (flat_topk_idx == expert_id)
            inputs = hidden_states[mask]
            expert = self.experts[expert_id]
            expert_out = expert(inputs)
            if expert_out.dtype != hidden_states.dtype:  # 统一以hidden_states的dtype为基准
              expert_out = expert_out.to(hidden_states.dtype)
            expert_outputs.append((mask, expert_out))
          y = torch.zeros_like(hidden_states)
          for mask, out in expert_outputs:
            y[mask] = out.to(y.dtype) 
          y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
          y = y.view(*orig_shape)
          y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.use_shared_expert is not None:
            y = y + self.shared_expert(identity)
        return y
        
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x, device=x.device, dtype=x.dtype)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out = expert_out.to(x.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]].to(x.dtype))
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache
        
        
        
class FlashscGPTLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    The class is modified from torch.nn.TransformerEncoderLayer to support the
    FlashAttention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """

    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        device=None,
        dtype=None,
        norm_scheme="post",  # "pre" or "post"
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = FlashscGPTMHA(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=batch_first,
            attention_dropout=dropout,
            **factory_kwargs,
        )
        self.moe = FFNMoE(hidden_size=d_model, dropout=dropout, activation=F.relu,
                          n_routed_experts=8, top_k=1, use_shared_expert=True)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if norm_scheme not in ["pre", "post"]:
            raise ValueError("norm_scheme must be either pre or post")
        
        
        
    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def _reverse_key_padding_mask(self, src_key_padding_mask):
        """
        Reverse the true false values of the key padding mask. This is because
        we follow pytorch rule that the mask is True for padded tokens, but
        in the inner flash MHA, it assumes the mask is False for padded tokens.
        """
        if src_key_padding_mask is None:
            return None

        if not src_key_padding_mask.any().item():
            # no padding tokens in src
            return None
        return ~src_key_padding_mask

    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Tensor,
        pcpt_key_padding_mask: Optional[Tensor] = None,
        gen_key_padding_mask: Optional[Tensor] = None,
        attn_non_mask_position : Optional[Tensor] = None,
    ) -> Tensor:
        pcpt_key_padding_mask_ = self._reverse_key_padding_mask(pcpt_key_padding_mask)
        gen_key_padding_mask_ = self._reverse_key_padding_mask(gen_key_padding_mask)

        if self.norm_scheme == "pre":
            pcpt_total_embs = self.norm1(pcpt_total_embs)
            if gen_total_embs is not None:
                gen_total_embs = self.norm1(gen_total_embs)
            pcpt_total_embs2, gen_total_embs2 = self.self_attn(
                pcpt_total_embs,
                gen_total_embs,
                pcpt_key_padding_mask=pcpt_key_padding_mask_,
                gen_key_padding_mask=gen_key_padding_mask_,
                attn_non_mask_position=attn_non_mask_position,
            )[0]
            pcpt_total_embs = pcpt_total_embs + self.dropout1(pcpt_total_embs2)
            pcpt_total_embs = self.norm2(pcpt_total_embs)
            pcpt_total_embs2 = self.moe(pcpt_total_embs)
            pcpt_total_embs = pcpt_total_embs + self.dropout2(pcpt_total_embs2)

            if gen_total_embs is not None:
                gen_total_embs = gen_total_embs + self.dropout1(gen_total_embs2)
                gen_total_embs = self.norm2(gen_total_embs)
                gen_total_embs2 = self.moe(gen_total_embs)
                gen_total_embs = gen_total_embs + self.dropout2(gen_total_embs2)
        else:
            pcpt_total_embs2, gen_total_embs2 = self.self_attn(
                pcpt_total_embs,
                gen_total_embs,
                pcpt_key_padding_mask=pcpt_key_padding_mask_,
                gen_key_padding_mask=gen_key_padding_mask_,
                attn_non_mask_position=attn_non_mask_position,
            )[0]
            pcpt_total_embs = pcpt_total_embs + self.dropout1(pcpt_total_embs2)
            pcpt_total_embs = self.norm1(pcpt_total_embs)
            pcpt_total_embs2 = self.moe(pcpt_total_embs)
            pcpt_total_embs = pcpt_total_embs + self.dropout2(pcpt_total_embs2)
            pcpt_total_embs = self.norm2(pcpt_total_embs)

            if gen_total_embs is not None:
                gen_total_embs = gen_total_embs + self.dropout1(gen_total_embs2)
                gen_total_embs = self.norm1(gen_total_embs)
                gen_total_embs2 = self.moe(gen_total_embs)
                gen_total_embs = gen_total_embs + self.dropout2(gen_total_embs2)
                gen_total_embs = self.norm2(gen_total_embs)

        return pcpt_total_embs, gen_total_embs




class FlashscGPTGenerator(nn.Module):
    # takes in the set of different inputs in an mapping
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    __constants__ = ["norm"]

    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        mask_check=True,
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.mask_check = mask_check

    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Tensor,
        pcpt_key_padding_mask: Optional[Tensor] = None,
        gen_key_padding_mask: Optional[Tensor] = None,
        attn_non_mask_position : Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if pcpt_key_padding_mask is not None:
            _skpm_dtype = pcpt_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(
                pcpt_key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )
        
        if attn_non_mask_position is None:
          for mod in self.layers:
            pcpt_total_embs, gen_total_embs = mod(
                pcpt_total_embs,
                gen_total_embs,
                pcpt_key_padding_mask,
                gen_key_padding_mask,
            )
        else:
          for mod in self.layers:
            pcpt_total_embs, gen_total_embs = mod(
                pcpt_total_embs,
                gen_total_embs,
                pcpt_key_padding_mask,
                gen_key_padding_mask,
                attn_non_mask_position,
            )

        if self.norm is not None:
            pcpt_total_embs = self.norm(pcpt_total_embs)
            gen_total_embs = self.norm(gen_total_embs)

        return pcpt_total_embs, gen_total_embs
